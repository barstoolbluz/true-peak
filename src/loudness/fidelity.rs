//! Conditioned, binary64 realization of the native BS.1770 transfer function.
//!
//! RLB first: A(z) = (1-r)^2 + c*r*(1-r) + s^2*r^2, r=z^-1.
//! A difference of the input drives a scaled two-state recurrence. No
//! near-unity pole coefficients or subtraction of large DC filter states.
//! The shelf uses trapezoidal-integrator states and a direct/residual mix.
//! See K-WEIGHTING-REPORT.md for derivation and numerical evidence.
use std::f64::consts::PI;

use super::{simd, LoudnessError};

// Clearing a homogeneous tail below this amplitude prevents persistent
// subnormal states. Its squared energy is below binary64's smallest nonzero
// value; no input is cleared, including tiny changes atop an offset.
pub(super) const TINY: f64 = 1e-200;

#[derive(Debug, Clone, Copy)]
pub(super) struct Coefficients {
    pub(super) c: f64,
    pub(super) s: f64,
    pub(super) vh: f64,
    pub(super) t: f64,
    pub(super) rg: f64,
    pub(super) k2: f64,
    pub(super) bp_mix: f64,
    pub(super) lp_mix: f64,
}

impl Coefficients {
    pub(super) fn for_rate(rate: u32) -> Result<Self, LoudnessError> {
        if rate == 0 {
            return Err(LoudnessError::InvalidSampleRate);
        }
        // Require both prototypes to lie strictly below Nyquist.
        if !(super::MIN_FIDELITY_RATE_HZ..=super::MAX_SAMPLE_RATE_HZ).contains(&rate) {
            return Err(LoudnessError::UnsupportedSampleRate {
                sample_rate_hz: rate,
            });
        }
        let r = f64::from(rate);
        let k = (PI * 38.13547087602444 / r).tan();
        let q_inv = 1.0 / 0.5003270373238773;
        let d = 1.0 + k * q_inv + k * k;
        let c = (2.0 * k * q_inv + 4.0 * k * k) / d;
        let s = 2.0 * k / d.sqrt();

        let k = (PI * 1681.974450955533 / r).tan();
        let q_inv = 1.0 / 0.7071752369554196;
        let vh = 10.0_f64.powf(3.999843853973347 / 20.0);
        let vb = vh.powf(0.4996667741545416);
        let h = 1.0 / (1.0 + k * q_inv + k * k);
        let lp_mix = 1.0 - vh;
        let result = Self {
            c,
            s,
            vh,
            t: k * h,
            rg: q_inv + k,
            k2: 2.0 * k,
            bp_mix: (vb - vh) * q_inv + lp_mix * k,
            lp_mix,
        };
        // Strict Jury conditions for the actual rounded state-space
        // coefficients, evaluated without subtracting near-unity poles.
        let s2 = result.s * result.s;
        if ![
            result.c,
            result.s,
            result.vh,
            result.t,
            result.rg,
            result.k2,
            result.bp_mix,
            result.lp_mix,
        ]
        .iter()
        .all(|v| v.is_finite())
            || !(s2 > 0.0 && result.c > s2 && 4.0 - 2.0 * result.c + s2 > 0.0)
            || !(result.t * result.k2 > 0.0
                && result.t * result.rg < 1.0
                && 2.0 * result.rg > result.k2)
        {
            return Err(LoudnessError::NumericalRange);
        }
        Ok(result)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct State {
    // Previous input, HP output, scaled HP integral, two shelf states.
    pub(super) delay: [f64; 5],
}

impl State {
    #[inline]
    pub(super) fn process(&mut self, x: f64, c: Coefficients) -> f64 {
        let [previous, y, q, a, b] = self.delay;
        let dx = x - previous;
        let mut y_next = y + ((dx - c.c * y) - c.s * q);
        let mut q_next = q + c.s * y;
        if dx == 0.0 && y_next.abs() < TINY && q_next.abs() < TINY {
            y_next = 0.0;
            q_next = 0.0;
        }

        let t = (y_next - b) - c.rg * a;
        let inc = c.t * t;
        let bp = a + inc;
        let mut out = (c.vh * y_next + c.bp_mix * bp) + c.lp_mix * b;
        let mut a_next = a + 2.0 * inc;
        let mut b_next = b + c.k2 * bp;
        if y_next == 0.0 && a_next.abs() < TINY && b_next.abs() < TINY {
            a_next = 0.0;
            b_next = 0.0;
            out = 0.0;
        }
        self.delay = [x, y_next, q_next, a_next, b_next];
        out
    }
}

#[derive(Debug)]
pub(super) struct Filter {
    pub(super) coefficients: Coefficients,
    // State-major (five contiguous channel planes). This permits direct
    // SSE2/AVX loads and stores without gathering or transposing states.
    pub(super) states: Vec<f64>,
}

impl Filter {
    pub(super) fn new(rate: u32, channels: usize) -> Result<Self, LoudnessError> {
        let coefficients = Coefficients::for_rate(rate)?;
        let mut states = Vec::new();
        let samples = channels.checked_mul(5).ok_or(LoudnessError::InputTooLong)?;
        states
            .try_reserve_exact(samples)
            .map_err(|_| LoudnessError::AllocationFailed)?;
        states.resize(samples, 0.0);
        Ok(Self {
            coefficients,
            states,
        })
    }

    #[inline]
    pub(super) fn frame(&mut self, backend: simd::Backend, frame: &[f64], out: &mut [f64]) {
        simd::filter_fidelity_frame(backend, frame, &mut self.states, self.coefficients, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounded_coefficients_admit_the_entire_pcm_rate_envelope() {
        for rate in super::super::MIN_FIDELITY_RATE_HZ..=super::super::MAX_SAMPLE_RATE_HZ {
            Coefficients::for_rate(rate).unwrap_or_else(|e| panic!("{rate}: {e:?}"));
        }
        assert!(Coefficients::for_rate(3_363).is_err());
        assert!(Coefficients::for_rate(2_822_401).is_err());
    }

    #[test]
    fn report_rounded_coefficients_for_qualification() {
        for rate in [
            8_000, 16_000, 22_050, 24_000, 32_000, 44_100, 44_101, 48_000, 88_200, 96_000, 176_400,
            192_000, 352_800, 384_000, 705_600, 768_000, 3_364, 4_000, 7_999, 768_001, 1_411_200,
            1_536_000, 2_000_003, 2_822_399, 2_822_400,
        ] {
            let c = Coefficients::for_rate(rate).unwrap();
            eprint!("KWEIGHT_COEFF {rate}");
            for v in [c.c, c.s, c.vh, c.t, c.rg, c.k2, c.bp_mix, c.lp_mix] {
                eprint!(" {:016x}", v.to_bits());
            }
            eprintln!();
        }
    }

    #[test]
    fn independently_derived_complex_frequency_response() {
        for line in include_str!("../../tests/fixtures/k_weighting_reference.tsv")
            .lines()
            .chain(include_str!("../../tests/fixtures/k_weighting_full_range.tsv").lines())
        {
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            let fields: Vec<_> = line.split_whitespace().collect();
            let rate: u32 = fields[0].parse().unwrap();
            let frequency: f64 = fields[1].parse().unwrap();
            let re: f64 = fields[2].parse().unwrap();
            let im: f64 = fields[3].parse().unwrap();
            let coefficients = Coefficients::for_rate(rate).unwrap();
            let mut state = State::default();
            let mut error = 0.0;
            let mut power = 0.0;
            // At 3364 Hz the shelf is almost at Nyquist. Its physical
            // settling time is seconds rather than milliseconds.
            let frames = if rate == 3_364 { 600 * rate } else { rate / 2 };
            for n in 0..frames {
                let phase = std::f64::consts::TAU * (f64::from(n) * frequency % f64::from(rate))
                    / f64::from(rate);
                let (sin, cos) = phase.sin_cos();
                let input = 0.1 * sin;
                let actual = state.process(input, coefficients);
                let expected = 0.1 * (re * sin + im * cos);
                if n >= frames / 2 {
                    error += (actual - expected) * (actual - expected);
                    power += expected * expected;
                }
            }
            assert!(
                (error / power).sqrt() < 2e-6,
                "{rate} Hz, {frequency} Hz: relative RMS error {}",
                (error / power).sqrt()
            );
        }
    }

    #[test]
    fn independent_transient_samples() {
        let mut state = State::default();
        let mut current = 0;
        let mut coefficients = Coefficients::for_rate(8_000).unwrap();
        for line in include_str!("../../tests/fixtures/k_weighting_transient.tsv")
            .lines()
            .chain(include_str!("../../tests/fixtures/k_weighting_full_transient.tsv").lines())
        {
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            let fields: Vec<_> = line.split_whitespace().collect();
            let rate: u32 = fields[0].parse().unwrap();
            if current != rate {
                current = rate;
                coefficients = Coefficients::for_rate(rate).unwrap();
                state = State::default();
            }
            let input: f64 = fields[1].parse().unwrap();
            let expected: f64 = fields[2].parse().unwrap();
            let actual = state.process(input, coefficients);
            // Only the new 3364 Hz case (shelf nearly at Nyquist) has a
            // separate tolerance; all historical assertions are retained.
            let limit = if rate == 3_364 { 2e-10 } else { 2e-13 };
            assert!(
                (actual - expected).abs() < limit,
                "{rate}: {actual:e} != {expected:e}"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn scalar_and_existing_vector_widths_are_bitwise_equal() {
        for backend in [
            simd::Backend::sse2_if_available(),
            simd::Backend::avx_if_available(),
        ]
        .into_iter()
        .flatten()
        {
            for rate in [
                3_364, 8_000, 44_101, 48_000, 768_000, 1_411_200, 1_536_000, 2_822_400,
            ] {
                let c = Coefficients::for_rate(rate).unwrap();
                for channels in [1, 2, 3, 4, 5, 7, 8, 64] {
                    let mut scalar = vec![0.0; 5 * channels];
                    let mut packed = scalar.clone();
                    let mut a = vec![0.0; channels];
                    let mut b = a.clone();
                    let mut frame = vec![0.0; channels];
                    for n in 0..2_003 {
                        for (ch, x) in frame.iter_mut().enumerate() {
                            *x = match n {
                                0 => 1e-220 * (ch as f64 + 1.0),
                                1..=7 => 0.0,
                                _ => match (n + ch) % 11 {
                                    0 => -0.0,
                                    1 => f64::from_bits(1),
                                    2 => 0.01,
                                    _ => 1.75 * ((n * (ch + 3)) as f64 * 0.013).sin(),
                                },
                            };
                        }
                        simd::filter_fidelity_frame(
                            simd::Backend::scalar(),
                            &frame,
                            &mut scalar,
                            c,
                            &mut a,
                        );
                        simd::filter_fidelity_frame(backend, &frame, &mut packed, c, &mut b);
                        for ch in 0..channels {
                            assert_eq!(
                                a[ch].to_bits(),
                                b[ch].to_bits(),
                                "rate {rate}, frame {n}, ch {ch}"
                            );
                            for j in 0..5 {
                                assert_eq!(
                                    scalar[j * channels + ch].to_bits(),
                                    packed[j * channels + ch].to_bits()
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
