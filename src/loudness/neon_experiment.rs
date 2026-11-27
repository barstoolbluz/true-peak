//! Opt-in AArch64 conditioned-filter candidate; default production unchanged.
//! Native ARM differential and performance qualification are required.
use super::super::fidelity::{Coefficients, TINY};
use super::fidelity_scalar;
use std::arch::aarch64::*;

// SAFETY: caller has checked NEON availability and exact buffer geometry.
// Each packed load/store stays within its state plane or frame; scalar tail
// handles an odd channel count. No FMA, changed coefficients or rounding graph.
#[target_feature(enable = "neon")]
#[inline(never)]
pub(super) unsafe fn conditioned_neon(
    frame: &[f64],
    states: &mut [f64],
    c: Coefficients,
    output: &mut [f64],
) {
    let n = frame.len();
    let mut ch = 0;
    let zero = vdupq_n_f64(0.0);
    let tiny = vdupq_n_f64(TINY);
    let two = vdupq_n_f64(2.0);
    let cc = vdupq_n_f64(c.c);
    let cs = vdupq_n_f64(c.s);
    let vh = vdupq_n_f64(c.vh);
    let ct = vdupq_n_f64(c.t);
    let rg = vdupq_n_f64(c.rg);
    let k2 = vdupq_n_f64(c.k2);
    let bm = vdupq_n_f64(c.bp_mix);
    let lm = vdupq_n_f64(c.lp_mix);
    while ch + 2 <= n {
        // SAFETY: ch+2 <= n, every state plane is exactly n elements.
        unsafe {
            let p = states.as_mut_ptr().add(ch);
            let x = vld1q_f64(frame.as_ptr().add(ch));
            let previous = vld1q_f64(p);
            let y = vld1q_f64(p.add(n));
            let q = vld1q_f64(p.add(2 * n));
            let a = vld1q_f64(p.add(3 * n));
            let b = vld1q_f64(p.add(4 * n));
            let dx = vsubq_f64(x, previous);
            let mut yn = vaddq_f64(
                y,
                vsubq_f64(vsubq_f64(dx, vmulq_f64(cc, y)), vmulq_f64(cs, q)),
            );
            let mut qn = vaddq_f64(q, vmulq_f64(cs, y));
            let clear = vandq_u64(
                vceqq_f64(dx, zero),
                vandq_u64(
                    vcltq_f64(vabsq_f64(yn), tiny),
                    vcltq_f64(vabsq_f64(qn), tiny),
                ),
            );
            yn = vbslq_f64(clear, zero, yn);
            qn = vbslq_f64(clear, zero, qn);
            let t = vsubq_f64(vsubq_f64(yn, b), vmulq_f64(rg, a));
            let inc = vmulq_f64(ct, t);
            let bp = vaddq_f64(a, inc);
            let mut out = vaddq_f64(
                vaddq_f64(vmulq_f64(vh, yn), vmulq_f64(bm, bp)),
                vmulq_f64(lm, b),
            );
            let mut an = vaddq_f64(a, vmulq_f64(two, inc));
            let mut bn = vaddq_f64(b, vmulq_f64(k2, bp));
            let clear = vandq_u64(
                vceqq_f64(yn, zero),
                vandq_u64(
                    vcltq_f64(vabsq_f64(an), tiny),
                    vcltq_f64(vabsq_f64(bn), tiny),
                ),
            );
            an = vbslq_f64(clear, zero, an);
            bn = vbslq_f64(clear, zero, bn);
            out = vbslq_f64(clear, zero, out);
            vst1q_f64(p, x);
            vst1q_f64(p.add(n), yn);
            vst1q_f64(p.add(2 * n), qn);
            vst1q_f64(p.add(3 * n), an);
            vst1q_f64(p.add(4 * n), bn);
            vst1q_f64(output.as_mut_ptr().add(ch), out);
        }
        ch += 2;
    }
    fidelity_scalar(frame, states, c, output, ch);
}

#[test]
fn neon_experimental_parity() {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        eprintln!("SKIP: NEON unavailable");
        return;
    }
    let mut seed = 0x5ca791335d_u64;
    let mut cases = 0;
    for rate in [
        3364, 8000, 44100, 44101, 48000, 192000, 768000, 1536000, 2822400,
    ] {
        let c = Coefficients::for_rate(rate).unwrap();
        for n in [1, 2, 3, 4, 5, 6, 7, 8, 63, 64] {
            // Offset the real slices by one f64, not vector alignment.
            let mut input = vec![0.0; n + 1];
            let mut a = vec![0.0; 5 * n + 1];
            let mut b = a.clone();
            let mut x = vec![0.0; n + 1];
            let mut y = x.clone();
            for i in 0..4093 {
                for s in &mut input[1..] {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    let random = (seed >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
                    *s = match i {
                        0..=8 => 0.0,
                        9 => -0.0,
                        10 => 1e-220,
                        11 => f64::from_bits(1),
                        12..=100 => 0.0,
                        101..=1000 => 0.25 + random * 1e-10,
                        1001..=2000 => random * 0.25,
                        _ => 0.0,
                    };
                }
                fidelity_scalar(&input[1..], &mut a[1..], c, &mut x[1..], 0);
                // SAFETY: runtime NEON check above; slices satisfy geometry.
                unsafe {
                    conditioned_neon(&input[1..], &mut b[1..], c, &mut y[1..]);
                }
                for (expected, actual) in x.iter().zip(&y).chain(a.iter().zip(&b)) {
                    assert_eq!(
                        expected.to_bits(),
                        actual.to_bits(),
                        "rate={rate} channels={n} frame={i}"
                    );
                }
            }
            cases += 1;
        }
    }
    eprintln!("NEON_KERNEL_EXECUTED geometries={cases}");
}

#[cfg(feature = "experimental-loudness-neon")]
#[test]
#[ignore = "native ARM kernel timing; opt in on a quiet host"]
fn neon_experimental_timing() {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        eprintln!("SKIP: NEON unavailable");
        return;
    }
    use super::{filter_fidelity_frame, Backend};
    use std::{hint::black_box, time::Instant};
    let neon = Backend::neon_if_available().unwrap();
    for rate in [48000, 192000, 768000, 2822400] {
        let c = Coefficients::for_rate(rate).unwrap();
        for n in [2, 3, 6, 8] {
            let frames = 262144;
            let input: Vec<f64> = (0..frames * n)
                .map(|i| 0.2 * ((i / n) as f64 * 0.043 + (i % n) as f64).sin())
                .collect();
            for rep in 0..8 {
                for k in 0..2 {
                    let mode = (rep + k) % 2;
                    let backend = if mode == 0 { Backend::scalar() } else { neon };
                    let mut states = vec![0.0; 5 * n];
                    let mut out = vec![0.0; n];
                    let start = Instant::now();
                    for frame in input.chunks_exact(n) {
                        filter_fidelity_frame(backend, frame, &mut states, c, &mut out);
                        black_box(&out);
                    }
                    let elapsed = start.elapsed().as_nanos();
                    black_box(states);
                    eprintln!("NEON_TIMING\t{rate}\t{n}\t{rep}\t{mode}\t{elapsed}");
                }
            }
        }
    }
}

#[cfg(feature = "experimental-loudness-neon")]
#[test]
fn neon_dispatch_and_native_meter_parity() {
    use super::super::{ChannelRole::*, LoudnessMeter, LoudnessProfile, LoudnessSimdBackend};
    use super::{filter_fidelity_frame, Backend};
    let Some(neon) = Backend::neon_if_available() else {
        eprintln!("NEON_UNAVAILABLE");
        return;
    };
    assert!(Backend::production(1).is_scalar());
    for n in [2, 3, 6, 8, 63, 64] {
        assert_eq!(Backend::production(n), neon);
        let c = Coefficients::for_rate(44_101).unwrap();
        let mut a = vec![0.0; 5 * n];
        let mut b = a.clone();
        let mut x = vec![0.0; n];
        let mut y = x.clone();
        let mut frame = vec![0.0; n];
        for i in 0..8193 {
            for (ch, sample) in frame.iter_mut().enumerate() {
                *sample = if i > 4000 {
                    0.0
                } else {
                    ((i + ch * 7) % 101) as f64 / 1024.0 - 0.05
                };
            }
            filter_fidelity_frame(Backend::scalar(), &frame, &mut a, c, &mut x);
            filter_fidelity_frame(Backend::production(n), &frame, &mut b, c, &mut y);
            for (lhs, rhs) in x.iter().zip(&y).chain(a.iter().zip(&b)) {
                assert_eq!(lhs.to_bits(), rhs.to_bits(), "n={n} frame={i}");
            }
        }
    }
    for (rate, roles) in [
        (3364, vec![Left, Right]),
        (44101, vec![Left, Right, Center]),
        (
            48000,
            vec![Left, Right, Center, Lfe, LeftSurround, RightSurround],
        ),
        (768000, vec![Left, Right]),
    ] {
        let mut scalar =
            LoudnessMeter::with_roles(rate, &roles, LoudnessProfile::NativeEbu2023).unwrap();
        let mut selected =
            LoudnessMeter::with_roles(rate, &roles, LoudnessProfile::NativeEbu2023).unwrap();
        scalar
            .force_simd_backend_for_qualification(LoudnessSimdBackend::Scalar)
            .unwrap();
        assert_eq!(selected.simd_backend, neon);
        let n = roles.len();
        let block: Vec<_> = (0..rate as usize * n)
            .map(|i| {
                let ch = i % n;
                if roles[ch] == Lfe {
                    0.8
                } else {
                    ((i / n + ch * 7) % 97) as f64 / 1024.0 - 0.05
                }
            })
            .collect();
        for gain in [1.0, 0.001, 0.25, 1.0, 0.0625] {
            let pcm: Vec<_> = block.iter().map(|x| x * gain).collect();
            for chunk in pcm.chunks(1009 * n) {
                scalar.push_interleaved(chunk).unwrap();
            }
            for chunk in pcm.chunks(4093 * n) {
                selected.push_interleaved(chunk).unwrap();
            }
        }
        let a = scalar.finalize().unwrap();
        let b = selected.finalize().unwrap();
        // Allocation capacity can differ with chunk boundaries; the summary
        // measurements and all retained energy bits must not.
        let mut a_summary = a.summary.clone();
        let mut b_summary = b.summary.clone();
        a_summary.retained_storage_bytes = 0;
        b_summary.retained_storage_bytes = 0;
        assert_eq!(a_summary, b_summary, "rate={rate} roles={roles:?}");
        for (lhs, rhs) in [
            (
                &a.statistics.integrated_energies,
                &b.statistics.integrated_energies,
            ),
            (&a.statistics.lra_energies, &b.statistics.lra_energies),
        ] {
            assert_eq!(lhs.len(), rhs.len());
            for (x, y) in lhs.iter().zip(rhs) {
                assert_eq!(x.to_bits(), y.to_bits(), "rate={rate} roles={roles:?}");
            }
        }
    }
    let mut compat =
        LoudnessMeter::with_roles(48000, &[Left, Right], LoudnessProfile::Libebur128126).unwrap();
    assert!(compat.simd_backend.is_scalar());
    assert!(compat
        .force_simd_backend_for_qualification(LoudnessSimdBackend::Neon)
        .is_err());
    assert!(compat.simd_backend.is_scalar());
    eprintln!("NEON_DISPATCH_EXECUTED geometries=6 meters=4 compatibility=scalar");
}
