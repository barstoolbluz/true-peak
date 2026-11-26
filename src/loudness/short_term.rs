//! Ungated native short-term loudness, without integrated or LRA work.
use super::{ChannelRole, LoudnessError, LoudnessProfile, validate_roles};
use super::statistics::energy_to_lufs;

/// A complete, ungated three-second observation. Time is the exclusive end
/// frame on the input clock. Exact silence has a level of negative infinity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShortTermObservation {
    /// Exclusive end frame, counted from the start of the stream.
    pub end_frame: u64,
    /// K-weighted loudness in LUFS, including values below the LRA gate.
    pub lufs: f64,
}

/// Native BS.1770 short-term meter. Emits full three-second windows at
/// rational 100 ms intervals; no artificial EOF windows or gating.
/// Uses the existing native K-weighting coefficient and channel-role core.
/// Memory is independent of stream length. Unlike the full loudness meter,
/// this meter does not compute integrated loudness or retain observations.
pub struct ShortTermMeter {
    rate: u32,
    fidelity: super::fidelity::Filter,
    backend: super::simd::Backend,
    weights: Vec<f64>,
    blocks: [f64; 30],
    block_sum: f64,
    frames: u64,
    blocks_seen: u64,
    next_end: u64,
    failed: bool,
}

impl ShortTermMeter {
    /// Create a conditioned native meter for a PCM rate of 8 kHz through
    /// 2,822,400 Hz and explicit roles.
    pub fn new(rate: u32, roles: &[ChannelRole]) -> Result<Self, LoudnessError> {
        if !(8_000..=super::MAX_SAMPLE_RATE_HZ).contains(&rate) {
            return Err(LoudnessError::UnsupportedSampleRate { sample_rate_hz: rate });
        }
        if roles.is_empty() || roles.len() > super::MAX_CHANNELS {
            return Err(LoudnessError::InvalidChannelCount);
        }
        validate_roles(roles, LoudnessProfile::NativeEbu2023)?;
        Ok(Self {
            rate,
            fidelity: super::fidelity::Filter::new(rate, roles.len())?,
            backend: super::simd::Backend::production(roles.len()),
            weights: roles.iter().map(|r| r.energy_weight()).collect(),
            blocks: [0.0; 30], block_sum: 0.0, frames: 0, blocks_seen: 0,
            next_end: u64::from(rate) / 10, failed: false,
        })
    }

    /// Explicit spelling of the conditioned native meter; equivalent to `new`.
    pub fn new_high_fidelity(rate: u32, roles: &[ChannelRole]) -> Result<Self, LoudnessError> {
        Self::new(rate, roles)
    }

    /// Push complete interleaved frames. Each observation is delivered once
    /// in chronological order. Invalid input is rejected before DSP mutation;
    /// a numerical failure is terminal. No observations are buffered here.
    pub fn push_interleaved(&mut self, samples: &[f64], emit: impl FnMut(ShortTermObservation)) -> Result<(), LoudnessError> {
        if self.failed { return Err(LoudnessError::MeterFailed); }
        let channels = self.weights.len();
        if samples.len() % channels != 0 {
            return Err(LoudnessError::IncompleteFrame { samples: samples.len(), channels });
        }
        if let Some(sample_index) = samples.iter().position(|x| !x.is_finite()) {
            return Err(LoudnessError::NonFiniteSample { sample_index });
        }
        let end = self.frames.checked_add((samples.len() / channels) as u64).ok_or(LoudnessError::InputTooLong)?;
        if end > u64::MAX - u64::from(self.rate) { return Err(LoudnessError::InputTooLong); }
        self.consume(samples, emit)
    }

    fn consume(
        &mut self, samples: &[f64], mut emit: impl FnMut(ShortTermObservation),
    ) -> Result<(), LoudnessError> {
        let channels = self.weights.len();
        let mut filtered = [0.0; super::MAX_CHANNELS];
        for frame in samples.chunks_exact(channels) {
            self.fidelity.frame(self.backend, frame, &mut filtered[..channels]);
            for (&y, &weight) in filtered[..channels].iter().zip(&self.weights) {
                self.block_sum += weight * y * y;
            }
            if !self.block_sum.is_finite() {
                self.failed = true;
                return Err(LoudnessError::NumericalRange);
            }
            self.frames += 1;
            if self.frames == self.next_end {
                self.blocks[(self.blocks_seen % 30) as usize] = self.block_sum;
                self.block_sum = 0.0;
                self.blocks_seen += 1;
                if self.blocks_seen >= 30 {
                    // Resum 30 nonnegative block energies; no subtractive
                    // cancellation when a loud passage leaves the window.
                    let energy = self.blocks.iter().sum::<f64>() / (3.0 * f64::from(self.rate));
                    if !energy.is_finite() {
                        self.failed = true;
                        return Err(LoudnessError::NumericalRange);
                    }
                    emit(ShortTermObservation { end_frame: self.frames, lufs: energy_to_lufs(energy, LoudnessProfile::NativeEbu2023) });
                }
                self.next_end = ((u128::from(self.blocks_seen) + 1) * u128::from(self.rate) / 10) as u64;
            }
        }
        Ok(())
    }
}
