use true_peak::loudness::{ChannelRole, LoudnessMeter, LoudnessMetricDemand, LoudnessProfile,
    LoudnessSimdBackend, ShortTermMeter, DEFAULT_LOUDNESS_STORAGE_LIMIT_BYTES};

fn response_at(rate: u32, freq: f64) -> (f64, f64) {
    for line in include_str!("fixtures/k_weighting_reference.tsv").lines().filter(|s| !s.starts_with('#')) {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields[0].parse::<u32>().unwrap() == rate && fields[1].parse::<f64>().unwrap() == freq {
            return (fields[2].parse().unwrap(), fields[3].parse().unwrap());
        }
    }
    panic!("missing independent reference {rate} {freq}");
}

fn signal(rate: u32, frames: usize, amp: f64, dc: f64) -> Vec<f64> {
    (0..frames).map(|i| dc + amp * (std::f64::consts::TAU * ((i as u64 * 1000) % u64::from(rate)) as f64 / f64::from(rate)).sin()).collect()
}

#[test]
fn short_term_reference_resolution_across_both_rate_families() {
    for rate in [8_000, 16_000, 22_050, 24_000, 32_000, 44_100, 44_101, 48_000,
        88_200, 96_000, 176_400, 192_000, 352_800, 384_000, 705_600, 768_000] {
        let (re, im) = response_at(rate, 1000.0);
        // Below 32-bit PCM's quantization floor, with no DC to mask the tone.
        let amplitude = 1e-12;
        let pcm = signal(rate, 4 * rate as usize, amplitude, 0.0);
        let expected = -0.691 + 10.0 * (0.5 * amplitude * amplitude * (re * re + im * im)).log10();
        // At the deliberately odd clock, a 1 kHz tone still has exactly 3000
        // periods in the last 3 s. No rational-hop rounding of frame count.
        let mut short = ShortTermMeter::new_high_fidelity(rate, &[ChannelRole::Mono]).unwrap();
        let mut last = None;
        for block in pcm.chunks(4093) { short.push_interleaved(block, |p| last = Some(p)).unwrap(); }
        assert!((last.unwrap().lufs - expected).abs() < 1e-7, "rate {rate}");
    }
}

#[test]
fn integrated_channel_weighting_and_scalar_simd_parity() {
    let rate = 48_000;
    let roles = [ChannelRole::Left, ChannelRole::Right, ChannelRole::Center,
        ChannelRole::Lfe, ChannelRole::LeftSurround, ChannelRole::RightSurround];
    let mono = signal(rate, 7 * rate as usize + 37, 0.1, 0.0);
    let pcm: Vec<f64> = mono.iter().enumerate().flat_map(|(i, &x)| {
        let level = if i < 3 * rate as usize { 0.1 } else { 1.0 };
        let x = x * level;
        [x, x, x, 0.7 * x, x, x]
    }).collect();
    let run = |backend| {
        let mut m = LoudnessMeter::with_roles_high_fidelity(rate, &roles).unwrap();
        if m.force_simd_backend_for_qualification(backend).is_err() { return None; }
        for c in pcm.chunks(701 * roles.len()) { m.push_interleaved(c).unwrap(); }
        Some(m.finalize().unwrap().summary)
    };
    let baseline = run(LoudnessSimdBackend::Scalar).unwrap();
    assert_eq!(baseline.profile, LoudnessProfile::NativeEbu2023);
    for backend in [LoudnessSimdBackend::Sse2, LoudnessSimdBackend::Avx] {
        if let Some(s) = run(backend) {
            assert_eq!(baseline.integrated, s.integrated);
            assert_eq!(baseline.range, s.range);
            assert_eq!(baseline.absolute_lra_observations, s.absolute_lra_observations);
        }
    }
    // Independent stationary transfer and role weights, above the gate.
    let mono = signal(rate, 4 * rate as usize, 0.1, 0.0);
    let pcm: Vec<f64> = mono.iter().flat_map(|&x| [x,x,x,0.9,x,x]).collect();
    let mut m = LoudnessMeter::with_roles_high_fidelity_and_metric_demand_and_limit(
        rate,&roles,LoudnessMetricDemand::IntegratedOnly,DEFAULT_LOUDNESS_STORAGE_LIMIT_BYTES).unwrap();
    m.push_interleaved(&pcm).unwrap();
    let actual = m.finalize().unwrap().summary.integrated.finite_lufs().unwrap();
    let (re,im) = response_at(rate,1000.0);
    let expected = -0.691 + 10.0 * (0.005 * (3.0 + 2.0 * 1.41) * (re*re+im*im)).log10();
    // The integrated calculation includes the real onset transient, not a
    // virtual warmup; its effect for this long stationary tone is small.
    assert!((actual-expected).abs()<0.002, "{actual} != {expected}");
    let mut short = ShortTermMeter::new_high_fidelity(rate, &roles).unwrap();
    let mut last = None;
    short.push_interleaved(&pcm, |p| last=Some(p)).unwrap();
    assert!((last.unwrap().lufs-expected).abs()<1e-7);
}

#[test]
fn high_fidelity_storage_admission_and_validation() {
    let roles = [ChannelRole::Mono];
    let mut m = LoudnessMeter::with_roles_high_fidelity_and_metric_demand_and_limit(
        768_000,&roles,LoudnessMetricDemand::IntegratedOnly,DEFAULT_LOUDNESS_STORAGE_LIMIT_BYTES).unwrap();
    let bytes = m.retained_storage_bytes();
    assert!(LoudnessMeter::with_roles_high_fidelity_and_metric_demand_and_limit(
        768_000,&roles,LoudnessMetricDemand::IntegratedOnly,bytes-1).is_err());
    assert!(m.push_interleaved(&[0.0,f64::NAN]).is_err());
    assert_eq!(m.real_frames(),0);
    assert!(LoudnessMeter::with_roles_high_fidelity(2_822_401,&roles).is_err());
    assert!(ShortTermMeter::new_high_fidelity(7_999,&roles).is_err());
}

#[test]
fn actual_measurement_paths_match_independent_programme_windows() {
    use true_peak::loudness::IntegratedLoudness;
    for line in include_str!("fixtures/k_weighting_programme.tsv").lines().filter(|s| !s.starts_with('#')) {
        let f: Vec<_> = line.split_whitespace().collect();
        let rate: u32 = f[0].parse().unwrap();
        let first: usize = f[1].parse().unwrap();
        let second: usize = f[2].parse().unwrap();
        let expected: Vec<f64> = f[3..7].iter().map(|s| s.parse().unwrap()).collect();
        let counts: Vec<usize> = f[7..].iter().map(|s| s.parse().unwrap()).collect();
        let roles = [ChannelRole::Mono];
        let mut full = LoudnessMeter::with_roles_high_fidelity(rate,&roles).unwrap();
        let mut short = ShortTermMeter::new_high_fidelity(rate,&roles).unwrap();
        let mut observations = Vec::new();
        for (length,amp,dc) in [(first,0.05,0.01),(second,0.5,-0.02)] {
            let samples = signal(rate,length,amp,dc);
            for c in samples.chunks(701) {
                full.push_interleaved(c).unwrap();
                short.push_interleaved(c,|p| observations.push(p)).unwrap();
            }
        }
        let summary = full.finalize().unwrap().summary;
        let IntegratedLoudness::Finite { lufs, absolute_observations, relative_observations } = summary.integrated else { panic!("expected finite integrated loudness"); };
        let actual = [lufs,summary.range.finite_lu().unwrap(),observations.iter().map(|p|p.lufs).fold(f64::NEG_INFINITY,f64::max),observations.last().unwrap().lufs];
        for (a,b) in actual.into_iter().zip(expected) {
            assert!((a-b).abs()<1e-7,"rate {rate}: {a} != {b}");
        }
        assert_eq!(counts,vec![absolute_observations,relative_observations,summary.absolute_lra_observations,summary.range.observations(),observations.len()]);
        assert_eq!(summary.real_frames,(first+second) as u64);
    }
}

#[test]
fn physical_dc_step_tail_matches_arbitrary_precision() {
    for line in include_str!("fixtures/k_weighting_dc.tsv").lines().filter(|s| !s.starts_with('#')) {
        let f: Vec<_> = line.split_whitespace().collect();
        let rate: u32 = f[0].parse().unwrap();
        let expected: f64 = f[1].parse().unwrap();
        for offset in [0.1,-0.1] {
            let mut short = ShortTermMeter::new_high_fidelity(rate,&[ChannelRole::Mono]).unwrap();
            let mut last = None;
            let pcm = vec![offset;4093];
            let mut remaining = 4*rate as usize;
            while remaining > 0 {
                let n = remaining.min(pcm.len());
                short.push_interleaved(&pcm[..n],|p|last=Some(p)).unwrap();
                remaining -= n;
            }
            let actual = last.unwrap().lufs;
            assert!((actual-expected).abs()<1e-5,"rate {rate}: {actual} != {expected}");
            // The homogeneous state eventually clears below representable
            // squared energy. A subsequent nonzero change must excite it.
            remaining=5*rate as usize;
            while remaining>0 {
                let n=remaining.min(pcm.len());
                short.push_interleaved(&pcm[..n],|p|last=Some(p)).unwrap();
                remaining-=n;
            }
            assert_eq!(last.unwrap().lufs,f64::NEG_INFINITY);
            short.push_interleaved(&vec![-offset;rate as usize/10],|p|last=Some(p)).unwrap();
            assert!(last.unwrap().lufs.is_finite());
        }
    }
}
