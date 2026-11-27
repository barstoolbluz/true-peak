//! Public native-path regressions against independent MP references.
use true_peak::loudness::{
    ChannelRole, IntegratedLoudness, LoudnessMeter, LoudnessMetricDemand, LoudnessProfile,
    LoudnessRange, LoudnessSimdBackend, LoudnessSummary, ShortTermMeter,
    DEFAULT_LOUDNESS_STORAGE_LIMIT_BYTES,
};

const UPPER: u32 = 2_822_400;
const MONO: [ChannelRole; 1] = [ChannelRole::Mono];

fn stream_segment(
    rate: u32,
    frames: usize,
    amplitude: f64,
    dc: f64,
    chunk: usize,
    mut consume: impl FnMut(&[f64]),
) {
    let mut pcm = vec![0.0; chunk];
    let mut offset = 0;
    while offset < frames {
        let n = (frames - offset).min(chunk);
        for (i, x) in pcm[..n].iter_mut().enumerate() {
            let phase = std::f64::consts::TAU
                * (((offset + i) as u64 * 1000) % u64::from(rate)) as f64
                / f64::from(rate);
            *x = dc + amplitude * phase.sin();
        }
        consume(&pcm[..n]);
        offset += n;
    }
}

// This test runs unchanged against the previous candidate. Its default
// native constructor already accepts UPPER, so a constructor rejection is
// not a valid failing-first result. The legacy recurrence admits spurious
// integrated observations after the real DC-step transient has decayed.
#[test]
fn native_default_dc_gate_at_upper_boundary() {
    let mut meter = LoudnessMeter::new(UPPER, 1).unwrap();
    stream_segment(UPPER, 4 * UPPER as usize, 0.0, 0.1, 4093, |pcm| {
        meter.push_interleaved(pcm).unwrap()
    });
    let result = meter.finalize().unwrap().summary;
    eprintln!(
        "native DC absolute observations = {}",
        result.absolute_integrated_observations
    );
    assert_eq!(
        result.absolute_integrated_observations, 1,
        "unexpected gated DC residual"
    );
    assert!((result.integrated.finite_lufs().unwrap() + 46.31859365864308).abs() < 1e-7);
    assert_eq!(result.absolute_lra_observations, 16);
    assert_eq!(result.range.observations(), 16);
    assert!(result.range.finite_lu().unwrap().abs() < 1e-7);
}

fn assert_level(actual: Option<f64>, expected: &str) {
    if expected == "NA" {
        assert!(actual.is_none());
    } else {
        let expected: f64 = expected.parse().unwrap();
        let actual = actual.unwrap();
        assert!((actual - expected).abs() < 1e-7, "{actual} != {expected}");
    }
}

fn check_summary(summary: &LoudnessSummary, fields: &[&str], total: u64) {
    let count: Vec<usize> = fields[5..9].iter().map(|s| s.parse().unwrap()).collect();
    assert_eq!(summary.real_frames, total);
    assert_level(summary.integrated.finite_lufs(), fields[3]);
    assert_level(summary.range.finite_lu(), fields[4]);
    assert_eq!(summary.absolute_integrated_observations, count[0]);
    assert_eq!(summary.absolute_lra_observations, count[2]);
    assert_eq!(summary.range.observations(), count[3]);
    match &summary.integrated {
        IntegratedLoudness::Finite {
            relative_observations,
            ..
        } => assert_eq!(*relative_observations, count[1]),
        IntegratedLoudness::BelowAbsoluteGate => assert_eq!(count[0], 0),
        other => panic!("unexpected integrated outcome: {other:?}"),
    }
    if count[2] == 0 {
        assert_eq!(
            summary.range,
            LoudnessRange::Unavailable { observations: 0 }
        );
    }
}

#[test]
fn full_range_gating_and_continuous_programme() {
    for line in include_str!("fixtures/k_weighting_full_gates.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let f: Vec<_> = line.split_whitespace().collect();
        let rate: u32 = f[0].parse().unwrap();
        let mut meter =
            LoudnessMeter::with_roles(rate, &MONO, LoudnessProfile::NativeEbu2023).unwrap();
        let mut total = 0;
        for seg in f[2].split(',') {
            let v: Vec<_> = seg.split(':').collect();
            let length: usize = v[0].parse().unwrap();
            let amp: f64 = v[1].parse().unwrap();
            let dc: f64 = v[2].parse().unwrap();
            // One meter across ordered, off-hop track boundaries: no resets,
            // no virtual silence between segments, one finalization.
            stream_segment(rate, length, amp, dc, 701, |pcm| {
                meter.push_interleaved(pcm).unwrap()
            });
            total += length as u64;
        }
        check_summary(&meter.finalize().unwrap().summary, &f, total);
    }
}

fn last_short(rate: u32, amp: f64, dc: f64, chunk: usize, explicit: bool) -> f64 {
    let mut meter = if explicit {
        ShortTermMeter::new_high_fidelity(rate, &MONO).unwrap()
    } else {
        ShortTermMeter::new(rate, &MONO).unwrap()
    };
    let mut last = None;
    stream_segment(rate, 4 * rate as usize, amp, dc, chunk, |pcm| {
        meter.push_interleaved(pcm, |p| last = Some(p)).unwrap()
    });
    let p = last.unwrap();
    assert_eq!(p.end_frame, 4 * u64::from(rate));
    p.lufs
}

#[test]
fn native_short_term_physical_tail_and_signal_resolution() {
    for line in include_str!("fixtures/k_weighting_full_dc.tsv")
        .lines()
        .filter(|s| !s.starts_with('#'))
    {
        let f: Vec<_> = line.split_whitespace().collect();
        let rate: u32 = f[0].parse().unwrap();
        let dc: f64 = f[1].parse().unwrap();
        let tone: f64 = f[2].parse().unwrap();
        assert!(
            (last_short(rate, 0.0, 0.1, 4093, false) - dc).abs() < 1e-5,
            "DC {rate}"
        );
        let mixed = last_short(rate, 1e-6, 0.01, 1009, false);
        assert!((mixed - tone).abs() < 1e-7, "mixed {rate}");
        assert!(
            (last_short(rate, 1e-12, 0.0, 8192, true) - (tone - 120.0)).abs() < 1e-7,
            "tiny {rate}"
        );
        if rate == UPPER {
            assert_eq!(
                mixed.to_bits(),
                last_short(rate, -1e-6, -0.01, 8192, true).to_bits()
            );
            // Do not claim infinite rejection from a finite step tail.
            let mut m = ShortTermMeter::new(rate, &MONO).unwrap();
            let mut last = None;
            let constant = [-0.1; 4093];
            let mut left = 9 * rate as usize;
            while left > 0 {
                let n = left.min(constant.len());
                m.push_interleaved(&constant[..n], |p| last = Some(p))
                    .unwrap();
                left -= n;
            }
            assert_eq!(last.unwrap().lufs, f64::NEG_INFINITY);
            // A large programme-to-quiet transition must recover the tiny
            // tone after the real K-weighted transient has decayed.
            let mut recovery = ShortTermMeter::new(rate, &MONO).unwrap();
            let mut recovered = None;
            stream_segment(rate, rate as usize / 2, 0.5, 0.1, 4093, |pcm| {
                recovery
                    .push_interleaved(pcm, |p| recovered = Some(p))
                    .unwrap()
            });
            stream_segment(rate, 4 * rate as usize, 1e-12, 0.0, 1009, |pcm| {
                recovery
                    .push_interleaved(pcm, |p| recovered = Some(p))
                    .unwrap()
            });
            assert!((recovered.unwrap().lufs - (tone - 120.0)).abs() < 1e-7);
        }
    }
}

#[test]
fn default_and_explicit_native_backends_agree_at_high_rate() {
    use ChannelRole::*;
    // Integrated-only storage for the six-channel upper-bound case stays
    // within the original default resource limit. No LRA is requested.
    for rate in [1_536_000, UPPER] {
        for roles in [
            &[Left, Right][..],
            &[Left, Right, Center, Lfe, LeftSurround, RightSurround][..],
        ] {
            let run = |backend: Option<LoudnessSimdBackend>, explicit: bool, chunk: usize| {
                let mut meter = if explicit {
                    LoudnessMeter::with_roles_high_fidelity_and_metric_demand_and_limit(
                        rate,
                        roles,
                        LoudnessMetricDemand::IntegratedOnly,
                        DEFAULT_LOUDNESS_STORAGE_LIMIT_BYTES,
                    )
                    .unwrap()
                } else {
                    LoudnessMeter::with_roles_and_metric_demand(
                        rate,
                        roles,
                        LoudnessProfile::NativeEbu2023,
                        LoudnessMetricDemand::IntegratedOnly,
                    )
                    .unwrap()
                };
                if let Some(b) = backend {
                    if meter.force_simd_backend_for_qualification(b).is_err() {
                        eprintln!("backend unavailable: {b:?}");
                        return None;
                    }
                }
                let mut interleaved = Vec::with_capacity(chunk * roles.len());
                stream_segment(rate, rate as usize / 2 + 37, 0.05, 0.01, chunk, |pcm| {
                    interleaved.clear();
                    for &x in pcm {
                        for &role in roles {
                            interleaved.push(if role == Lfe { 0.8 } else { x });
                        }
                    }
                    meter.push_interleaved(&interleaved).unwrap();
                });
                Some(meter.finalize().unwrap().summary)
            };
            let scalar = run(Some(LoudnessSimdBackend::Scalar), true, 4093).unwrap();
            // MP programme oracle for the mono segment, plus independent
            // role weighting. LFE is excluded even with a large DC value.
            let mono = if rate == UPPER {
                -29.06036543100152_f64
            } else {
                -29.05975265582444_f64
            };
            let weights = if roles.len() == 2 {
                2.0_f64
            } else {
                3.0 + 2.0 * 1.41
            };
            assert!(
                (scalar.integrated.finite_lufs().unwrap() - (mono + 10.0 * weights.log10())).abs()
                    < 1e-7
            );
            assert_eq!(scalar.absolute_integrated_observations, 2);
            for backend in [
                None,
                Some(LoudnessSimdBackend::Sse2),
                Some(LoudnessSimdBackend::Avx),
            ] {
                if let Some(actual) = run(backend, false, 1009) {
                    assert_eq!(actual.integrated, scalar.integrated);
                    assert_eq!(actual.range, scalar.range);
                    assert_eq!(
                        actual.absolute_integrated_observations,
                        scalar.absolute_integrated_observations
                    );
                }
            }
        }
    }
}

#[test]
fn public_rate_boundaries_and_compatibility_admission() {
    // Owner-authorized native admission change; compatibility retains its
    // historical input range. See k_weighting_admission.rs for all constructors.
    assert!(LoudnessMeter::new(16, 1).is_err());
    assert!(LoudnessMeter::new(3_363, 1).is_err());
    assert!(LoudnessMeter::with_roles_high_fidelity(3_363, &MONO).is_err());
    for rate in [3_364, 7_999, 768_001, 1_411_200, UPPER] {
        assert!(LoudnessMeter::with_roles_high_fidelity(rate, &MONO).is_ok());
    }
    assert!(LoudnessMeter::libebur128_126_default_layout(UPPER, 1).is_ok());
    assert!(LoudnessMeter::new(UPPER + 1, 1).is_err());
    assert!(ShortTermMeter::new(UPPER + 1, &MONO).is_err());
    assert!(ShortTermMeter::new(7_999, &MONO).is_err());
    assert!(ShortTermMeter::new(UPPER, &MONO).is_ok());
}
