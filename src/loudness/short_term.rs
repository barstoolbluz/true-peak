//! Ungated native short-term loudness, without integrated or LRA work.
use super::statistics::energy_to_lufs;
use super::{validate_roles, ChannelRole, LoudnessError, LoudnessProfile};

/// A complete, ungated three-second observation. Time is the exclusive end
/// frame on the input clock. Exact silence has a level of negative infinity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShortTermObservation {
    /// Exclusive end frame, counted from the start of the stream.
    pub end_frame: u64,
    /// K-weighted loudness in LUFS, including values below the LRA gate.
    pub lufs: f64,
}

// The observer is independent of filter ownership. Both the standalone
// meter and the combined range/short-term meter use the same arithmetic,
// physical ring order and real-frame clock. No EOF samples enter here.
#[derive(Debug)]
pub(super) struct ShortTermObservations {
    rate: u32,
    blocks: [f64; 30],
    block_sum: f64,
    frames: u64,
    blocks_seen: u64,
    next_end: u64,
}

impl ShortTermObservations {
    pub(super) fn new(rate: u32) -> Result<Self, LoudnessError> {
        if !(8_000..=super::MAX_SAMPLE_RATE_HZ).contains(&rate) {
            return Err(LoudnessError::UnsupportedSampleRate {
                sample_rate_hz: rate,
            });
        }
        Ok(Self {
            rate,
            blocks: [0.0; 30],
            block_sum: 0.0,
            frames: 0,
            blocks_seen: 0,
            next_end: u64::from(rate) / 10,
        })
    }

    pub(super) fn preflight(&self, frames: u64) -> Result<(), LoudnessError> {
        let end = self
            .frames
            .checked_add(frames)
            .ok_or(LoudnessError::InputTooLong)?;
        if end > u64::MAX - u64::from(self.rate) {
            return Err(LoudnessError::InputTooLong);
        }
        Ok(())
    }

    // This is called once per PCM frame by both consumers. A plain
    // #[inline] left an out-of-line call in the qualified CP4 binary.
    #[inline(always)]
    pub(super) fn observe(
        &mut self,
        filtered: &[f64],
        weights: &[f64],
    ) -> Result<Option<ShortTermObservation>, LoudnessError> {
        debug_assert_eq!(filtered.len(), weights.len());
        for (&y, &weight) in filtered.iter().zip(weights) {
            // Retain the standalone meter's multiply grouping and order.
            self.block_sum += weight * y * y;
        }
        if !self.block_sum.is_finite() {
            return Err(LoudnessError::NumericalRange);
        }
        self.frames += 1;
        if self.frames != self.next_end {
            return Ok(None);
        }
        self.finish_block()
    }

    // Only reached on the rational 100 ms clock. Keep the ring reduction,
    // wide clock arithmetic and logarithm out of the per-frame hot path.
    #[inline(never)]
    fn finish_block(&mut self) -> Result<Option<ShortTermObservation>, LoudnessError> {
        self.blocks[(self.blocks_seen % 30) as usize] = self.block_sum;
        self.block_sum = 0.0;
        self.blocks_seen += 1;
        self.next_end = ((u128::from(self.blocks_seen) + 1) * u128::from(self.rate) / 10) as u64;
        if self.blocks_seen < 30 {
            return Ok(None);
        }
        // Resum 30 nonnegative block energies; no subtractive
        // cancellation when a loud passage leaves the window.
        let energy = self.blocks.iter().sum::<f64>() / (3.0 * f64::from(self.rate));
        if !energy.is_finite() {
            return Err(LoudnessError::NumericalRange);
        }
        Ok(Some(ShortTermObservation {
            end_frame: self.frames,
            lufs: energy_to_lufs(energy, LoudnessProfile::NativeEbu2023),
        }))
    }
}

/// Native BS.1770 short-term meter. Emits full three-second windows at
/// rational 100 ms intervals; no artificial EOF windows or gating.
/// Uses the existing native K-weighting coefficient and channel-role core.
/// Memory is independent of stream length. Unlike the full loudness meter,
/// this meter does not compute integrated loudness or retain observations.
pub struct ShortTermMeter {
    fidelity: super::fidelity::Filter,
    backend: super::simd::Backend,
    weights: Vec<f64>,
    observations: ShortTermObservations,
    failed: bool,
}

impl ShortTermMeter {
    /// Create a conditioned native meter for a PCM rate of 8 kHz through
    /// 2,822,400 Hz and explicit roles.
    pub fn new(rate: u32, roles: &[ChannelRole]) -> Result<Self, LoudnessError> {
        let observations = ShortTermObservations::new(rate)?;
        if roles.is_empty() || roles.len() > super::MAX_CHANNELS {
            return Err(LoudnessError::InvalidChannelCount);
        }
        validate_roles(roles, LoudnessProfile::NativeEbu2023)?;
        Ok(Self {
            fidelity: super::fidelity::Filter::new(rate, roles.len())?,
            backend: super::simd::Backend::production(roles.len()),
            weights: roles.iter().map(|r| r.energy_weight()).collect(),
            observations,
            failed: false,
        })
    }

    /// Explicit spelling of the conditioned native meter; equivalent to `new`.
    pub fn new_high_fidelity(rate: u32, roles: &[ChannelRole]) -> Result<Self, LoudnessError> {
        Self::new(rate, roles)
    }

    /// Push complete interleaved frames. Each observation is delivered once
    /// in chronological order. Invalid input is rejected before DSP mutation;
    /// a numerical failure is terminal. No observations are buffered here.
    pub fn push_interleaved(
        &mut self,
        samples: &[f64],
        mut emit: impl FnMut(ShortTermObservation),
    ) -> Result<(), LoudnessError> {
        if self.failed {
            return Err(LoudnessError::MeterFailed);
        }
        let channels = self.weights.len();
        if !samples.len().is_multiple_of(channels) {
            return Err(LoudnessError::IncompleteFrame {
                samples: samples.len(),
                channels,
            });
        }
        if let Some(sample_index) = samples.iter().position(|x| !x.is_finite()) {
            return Err(LoudnessError::NonFiniteSample { sample_index });
        }
        self.observations
            .preflight((samples.len() / channels) as u64)?;
        let mut filtered = [0.0; super::MAX_CHANNELS];
        for frame in samples.chunks_exact(channels) {
            self.fidelity
                .frame(self.backend, frame, &mut filtered[..channels]);
            match self
                .observations
                .observe(&filtered[..channels], &self.weights)
            {
                Ok(Some(point)) => emit(point),
                Ok(None) => (),
                Err(error) => {
                    self.failed = true;
                    return Err(error);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "shared_experiment.rs"]
mod shared_experiment;

#[cfg(test)]
mod observer_tests {
    use super::*;

    #[test]
    fn short_observer_split_preserves_boundaries_and_range_errors() {
        // Exact dyadic filtered samples and integer energy units supply an
        // independent reference for the hot/cold boundary, including ring wrap.
        // Exercise all ten residues and a zero-weight channel.
        for rate in 8_000..8_010 {
            let mut observer = ShortTermObservations::new(rate).unwrap();
            let mut end = 0_u64;
            let mut cells = Vec::<u64>::new();
            for block in 1..=35_u64 {
                let next = block * u64::from(rate) / 10;
                let k = block % 4 + 1;
                let filtered = [k as f64 / 8.0, 0.75, -0.25];
                let weights = [1.0, 0.0, 1.0];
                cells.push((next - end) * (k * k + 4));
                for frame in end + 1..=next {
                    let point = observer.observe(&filtered, &weights).unwrap();
                    if frame == next && block >= 30 {
                        let point = point.unwrap();
                        let units: u64 = cells[cells.len() - 30..].iter().sum();
                        let energy = (units as f64 / 64.0) / (3.0 * f64::from(rate));
                        assert_eq!(point.end_frame, next);
                        assert_eq!(
                            point.lufs.to_bits(),
                            energy_to_lufs(energy, LoudnessProfile::NativeEbu2023).to_bits()
                        );
                    } else {
                        assert!(point.is_none());
                    }
                }
                end = next;
            }
        }

        // Overflow in frame accumulation fails before advancing the clock.
        let mut observer = ShortTermObservations::new(8_001).unwrap();
        assert!(matches!(
            observer.observe(&[1e200], &[1.0]),
            Err(LoudnessError::NumericalRange)
        ));
        assert_eq!(observer.frames, 0);

        // Finite cells may overflow the sum only on a full-window boundary.
        // Preserve the previous terminal numerical-range behavior.
        let mut observer = ShortTermObservations::new(8_001).unwrap();
        observer.blocks[..29].fill(f64::MAX / 2.0);
        observer.blocks_seen = 29;
        observer.next_end = 3 * 8_001;
        observer.frames = observer.next_end - 1;
        assert!(matches!(
            observer.observe(&[0.0], &[1.0]),
            Err(LoudnessError::NumericalRange)
        ));
        assert_eq!(observer.frames, 3 * 8_001);
        assert_eq!(observer.blocks_seen, 30);
    }
}
