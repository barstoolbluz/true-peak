//! Production-path qualification. Exact prefiltered fixtures are independent
//! of the partition algorithm; PCM comparisons use the original native ring.
use super::*;

const CODES: [(u32, u32); 9] = [
    (0, 0),
    (310, 40),
    (6400, 3200),
    (90000, 14000),
    (200, 100),
    (24000, 18000),
    (1000, 300),
    (150000, 60000),
    (500, 0),
];

fn bits(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).unwrap())
}

#[test]
fn production_cells_exact_windows_gates_and_percentiles() {
    let fixture = include_str!("../../tests/fixtures/energy_cells_reference.tsv");
    for rate in [
        3364, 8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009, 44101,
    ] {
        let rate64 = u64::from(rate);
        let roles = [
            ChannelRole::Left,
            ChannelRole::LeftSurround,
            ChannelRole::Lfe,
        ];
        let mut meter = LoudnessMeter::with_roles_high_fidelity(rate, &roles).unwrap();
        let real = 8 * rate64 + 17;
        let target = real + (3 * rate64).div_ceil(2);
        let mut windows = Vec::new();
        let mut summary = None;
        for line in fixture.lines().filter(|line| !line.starts_with('#')) {
            let f: Vec<_> = line.split_whitespace().collect();
            if f[1].parse::<u32>().unwrap() != rate {
                continue;
            }
            if f[0] == "W" {
                windows.push((
                    f[2].parse::<u64>().unwrap(),
                    f[3].parse::<usize>().unwrap(),
                    bits(f[4]),
                    f[5] == "1",
                ));
            } else {
                summary = Some((
                    f[2].parse::<usize>().unwrap(),
                    f[3].parse::<usize>().unwrap(),
                    f[4].parse::<usize>().unwrap(),
                    f[5].parse::<usize>().unwrap(),
                    bits(f[6]),
                    bits(f[7]),
                ));
            }
        }
        let mut at = 0;
        let mut kept = [0; 2];
        meter.preflight_history(real, true, true).unwrap();
        for end in 1..=target {
            let is_real = end <= real;
            if end == real + 1 {
                meter.preflight_history(target, false, true).unwrap();
            }
            let (l, r) = if is_real {
                CODES[((end - 1) / rate64) as usize]
            } else {
                (0, 0)
            };
            // Test production observation integration with exactly specified
            // prefiltered values. Huge finite LFE values must not be squared.
            meter.ring.copy_from_slice(&[
                f64::from(l) / 1_048_576.0,
                f64::from(r) / 1_048_576.0,
                f64::MAX,
            ]);
            meter.finish_frame(is_real, true).unwrap();
            while at < windows.len() && windows[at].0 == end {
                let (_, kind, expected, above) = windows[at];
                let len = if kind == 0 {
                    meter.clock.integrated_window_frames()
                } else {
                    meter.clock.lra_window_frames()
                };
                let actual = meter.window_energy(len).unwrap();
                // Two levels of compensated sums of nonnegative binary64
                // terms plus weight multiplication and division. For these
                // normal exact-dyadic inputs, allow 16 unit roundoffs; the
                // oracle is exact before final rounding, not the old ring.
                assert!(
                    (actual - expected).abs() <= 8.0 * f64::EPSILON * expected,
                    "{rate} {end} {kind}: {actual:e} != {expected:e}"
                );
                let gate = if kind == 0 {
                    actual > ABSOLUTE_GATE_ENERGY
                } else {
                    actual >= ABSOLUTE_GATE_ENERGY
                };
                assert_eq!(gate, above, "{rate} {end} {kind}");
                kept[kind] += above as usize;
                at += 1;
            }
            assert_eq!(meter.integrated_energies.len(), kept[0]);
            assert_eq!(meter.lra_energies.len(), kept[1]);
        }
        assert_eq!(at, windows.len());
        let (ia, ir, la, lr, lufs, lu) = summary.unwrap();
        assert_eq!(kept, [ia, la]);
        let IntegratedLoudness::Finite {
            lufs: actual,
            absolute_observations,
            relative_observations,
        } = integrated_from_absolute_energies(
            &meter.integrated_energies,
            meter.profile,
            real,
            meter.clock.integrated_window_frames(),
        )
        else {
            panic!("expected finite integrated loudness")
        };
        assert_eq!((absolute_observations, relative_observations), (ia, ir));
        assert!((actual - lufs).abs() < 1e-12, "{rate}: {actual} != {lufs}");
        sort_lra_energies(&mut meter.lra_energies);
        let LoudnessRange::Finite {
            lu: actual,
            observations,
        } = lra_from_sorted_absolute_energies(&meter.lra_energies, meter.profile)
        else {
            panic!("expected finite LRA")
        };
        assert_eq!(observations, lr);
        assert!((actual - lu).abs() < 1e-12, "{rate}: {actual} != {lu}");
    }
}

#[test]
fn production_cells_exact_absolute_gate_and_range_failure() {
    // Independently generated rounded binary64 input patterns immediately
    // surrounding the gate, retained from the checkpoint-2 exact study.
    for (bits, above) in [
        (0x3f3670bbcee4aad4, false),
        (0x3f3670bbcee4aad5, false),
        (0x3f3670bbcee4aad6, true),
        (0x3f3670bbcee4aad7, true),
    ] {
        let mut m = LoudnessMeter::new(8001, 1).unwrap();
        let end = 3 * 8001;
        m.preflight_history(end, true, true).unwrap();
        for _ in 0..end {
            m.ring[0] = f64::from_bits(bits);
            m.finish_frame(true, true).unwrap();
        }
        assert_eq!(m.integrated_energies.len(), if above { 27 } else { 0 });
        assert_eq!(m.lra_energies.len(), above as usize);
    }
    // A finite filtered sample whose square overflows is relevant only to a
    // complete requested window. Preserve the reference ring's timing of the
    // error; short recordings with no observation remain insufficient.
    for reference in [false, true] {
        let mut m = LoudnessMeter::with_roles_and_metric_demand(
            8001,
            &[ChannelRole::Mono],
            LoudnessProfile::NativeEbu2023,
            LoudnessMetricDemand::IntegratedOnly,
        )
        .unwrap();
        if reference {
            m.use_native_reference_ring();
        }
        let end = m.clock.integrated_window_frames();
        m.preflight_history(end, true, false).unwrap();
        for n in 1..=end {
            m.ring[m.write_frame] = if n == 1 { f64::MAX } else { 0.0 };
            let result = m.finish_frame(true, false);
            assert_eq!(
                result,
                if n == end {
                    Err(LoudnessError::NumericalRange)
                } else {
                    Ok(())
                }
            );
        }
    }
    // Sum of finite squares can overflow even when each product is finite.
    let mut cells = EnergyCells::new(8001, 1, false).unwrap();
    for n in 1..=3200 {
        cells.push(n, &[1e154], &[1.0]).unwrap();
    }
    assert_eq!(
        cells.energy(3200, 3200, &[1.0]),
        Err(LoudnessError::NumericalRange)
    );
    // Subnormal positive energies remain finite, rather than being mistaken
    // for a numerical-range error (default floating-point environment).
    let mut cells = EnergyCells::new(8000, 1, false).unwrap();
    let x = 2.0_f64.powi(-530);
    for n in 1..=3200 {
        cells.push(n, &[x], &[1.0]).unwrap();
    }
    assert_eq!(
        cells.energy(3200, 3200, &[1.0]).unwrap().to_bits(),
        (x * x).to_bits()
    );
}

fn compare_pcm(rate: u32, roles: &[ChannelRole], frames: usize, backend: simd::Backend) {
    let mut native = LoudnessMeter::with_roles_high_fidelity(rate, roles).unwrap();
    let mut ring = LoudnessMeter::with_roles_high_fidelity(rate, roles).unwrap();
    ring.use_native_reference_ring();
    let mut reduced = LoudnessMeter::with_roles_and_metric_demand(
        rate,
        roles,
        LoudnessProfile::NativeEbu2023,
        LoudnessMetricDemand::IntegratedOnly,
    )
    .unwrap();
    let mut range_only = LoudnessRangeMeter::new(rate, roles).unwrap();
    range_only.inner.simd_backend = backend;
    native.simd_backend = backend;
    ring.simd_backend = backend;
    reduced.simd_backend = backend;
    let mut pcm = Vec::with_capacity(frames * roles.len());
    let mut seed = 0xa9ab_87f9_ba42_2e27_u64;
    for n in 0..frames {
        let amplitude = match n / rate as usize % 4 {
            0 => 0.03,
            1 => 1e-9,
            2 => 0.13,
            _ => 0.002,
        };
        for _ in roles {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            pcm.push(((seed >> 11) as f64 / (1_u64 << 53) as f64 - 0.5) * amplitude);
        }
    }
    for p in pcm.chunks(997 * roles.len()) {
        native.push_interleaved(p).unwrap();
    }
    for p in pcm.chunks(701 * roles.len()) {
        ring.push_interleaved(p).unwrap();
    }
    for p in pcm.chunks(4093 * roles.len()) {
        reduced.push_interleaved(p).unwrap();
    }
    for p in pcm.chunks(503 * roles.len()) {
        range_only.push_interleaved(p).unwrap();
    }
    assert_eq!(range_only.inner.integrated_energies.capacity(), 0);
    assert_eq!(range_only.inner.integrated_energies.len(), 0);
    let range = range_only.finalize().unwrap();
    let a = native.finalize().unwrap();
    assert_eq!(range, a.summary.range);
    let b = ring.finalize().unwrap();
    let c = reduced.finalize().unwrap();
    assert_eq!(a.summary.real_frames, frames as u64);
    assert_eq!(a.summary.integrated, c.summary.integrated);
    assert_eq!(
        a.statistics.integrated_energies,
        c.statistics.integrated_energies
    );
    assert_eq!(c.summary.range, LoudnessRange::NotRequested);
    assert!(c.statistics.lra_energies.is_empty());
    for (actual, expected, window) in [
        (
            &a.statistics.integrated_energies,
            &b.statistics.integrated_energies,
            (2 * u64::from(rate) + 2) / 5,
        ),
        (
            &a.statistics.lra_energies,
            &b.statistics.lra_energies,
            3 * u64::from(rate),
        ),
    ] {
        assert_eq!(actual.len(), expected.len(), "{rate} {frames} {window}");
        // Conservative ring forward-error envelope on ordinary signals.
        let nu = (window as f64 + 2.0 * roles.len() as f64 + 128.0) * f64::EPSILON / 2.0;
        for (&a, &b) in actual.iter().zip(expected) {
            assert!(
                (a - b).abs() <= 2.0 * nu / (1.0 - nu) * b,
                "{rate} {frames} {window}: {a:e} {b:e}"
            );
        }
    }
    match (a.summary.integrated, b.summary.integrated) {
        (
            IntegratedLoudness::Finite {
                lufs: a,
                absolute_observations: aa,
                relative_observations: ar,
            },
            IntegratedLoudness::Finite {
                lufs: b,
                absolute_observations: ba,
                relative_observations: br,
            },
        ) => {
            assert_eq!((aa, ar), (ba, br));
            assert!((a - b).abs() < 1e-8);
        }
        (a, b) => assert_eq!(a, b),
    }
    match (a.summary.range, b.summary.range) {
        (
            LoudnessRange::Finite {
                lu: a,
                observations: aa,
            },
            LoudnessRange::Finite {
                lu: b,
                observations: ba,
            },
        ) => {
            assert_eq!(aa, ba);
            assert!((a - b).abs() < 1e-8);
        }
        (a, b) => assert_eq!(a, b),
    }
}

#[test]
fn production_cells_pcm_chunking_demand_eof_and_ring_reference() {
    use ChannelRole::*;
    let layouts: [&[ChannelRole]; 4] = [
        &[Mono],
        &[Left, Right],
        &[Left, Center, Lfe],
        &[Left, Right, Center, Lfe, LeftSurround, RightSurround],
    ];
    for rate in 8000..=8009 {
        for roles in layouts {
            compare_pcm(rate, roles, 5 * rate as usize + 17, simd::Backend::scalar());
        }
    }
    let rate = 8001;
    let iw = (2 * rate + 2) / 5;
    for frames in [
        0,
        1,
        iw - 1,
        iw,
        iw + 1,
        3 * rate - 1,
        3 * rate,
        3 * rate + 1,
        60 * rate + 7,
    ] {
        compare_pcm(rate as u32, layouts[1], frames, simd::Backend::scalar());
    }
    #[cfg(target_arch = "x86_64")]
    for backend in [
        simd::Backend::sse2_if_available(),
        simd::Backend::avx_if_available(),
    ]
    .into_iter()
    .flatten()
    {
        for roles in layouts {
            compare_pcm(8003, roles, 5 * 8003 + 17, backend);
        }
    }
    compare_pcm(3364, layouts[1], 5 * 3364 + 17, simd::Backend::scalar());
    compare_pcm(
        44101,
        layouts[1],
        5 * 44101 + 17,
        simd::Backend::production(2),
    );
}

#[test]
fn production_cells_bounded_storage_and_admission() {
    let roles = [
        ChannelRole::Left,
        ChannelRole::Right,
        ChannelRole::Center,
        ChannelRole::Lfe,
        ChannelRole::LeftSurround,
        ChannelRole::RightSurround,
    ];
    let build = |rate, limit| {
        LoudnessMeter::with_roles_and_limit(rate, &roles, LoudnessProfile::NativeEbu2023, limit)
    };
    let low = build(3364, 1_000_000).unwrap();
    let high = build(2_822_400, 1_000_000).unwrap();
    assert_eq!(low.retained_storage_bytes(), high.retained_storage_bytes());
    let bytes = high.retained_storage_bytes();
    assert!(bytes < 16_384);
    assert!(build(2_822_400, bytes).is_ok());
    assert!(matches!(
        build(2_822_400, bytes - 1),
        Err(LoudnessError::ResourceLimit { .. })
    ));
    // Actual storage limits still apply before history growth or DSP mutation.
    let mut constrained = build(8001, bytes).unwrap();
    assert!(matches!(
        constrained.push_interleaved(&vec![0.0; 3200 * roles.len()]),
        Err(LoudnessError::ResourceLimit { .. })
    ));
    assert_eq!(constrained.frames_seen, 0);
    assert_eq!(constrained.real_frames(), 0);
    constrained.push_interleaved(&[0.0; 6]).unwrap();
    assert_eq!(constrained.real_frames(), 1);
}
