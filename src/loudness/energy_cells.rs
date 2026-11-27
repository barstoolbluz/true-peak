//! Native full-meter positive energy cells. The union of the rational start
//! grid and the integrated-end grid partitions every requested observation.
//! Store per-channel rounded square sums, not filtered PCM or a subtractive
//! running total. The compatibility meter retains its chronological ring.
use super::{LoudnessError, MAX_CHANNELS};

#[derive(Debug, Clone, Copy, Default)]
struct Kahan {
    sum: f64,
    correction: f64,
}

impl Kahan {
    #[inline]
    fn add(&mut self, x: f64) {
        // Nonnegative squares and weighted totals may overflow. Keep an
        // infinity marker local to this cell/window, rather than poisoning
        // unrelated windows or rejecting an incomplete observation early.
        let y = x - self.correction;
        let t = self.sum + y;
        if t.is_finite() {
            self.correction = (t - self.sum) - y;
            self.sum = t;
        } else {
            self.sum = f64::INFINITY;
            self.correction = 0.0;
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct Span {
    start: u64,
    end: u64,
}

/// Fixed-capacity circular queue. No allocation occurs after construction.
/// Each of the two grids has exactly ten boundaries in a rate-sized period.
/// In a 3 s half-open interval there are at most 60 distinct ends; in the
/// rounded 400 ms interval each grid has at most five ends. Coalescing
/// coincident boundaries gives the same cells for both demand modes.
#[derive(Debug)]
pub(super) struct EnergyCells {
    rate: u64,
    integrated_window: u64,
    horizon: u64,
    next_start_index: u64,
    next_end_index: u64,
    // Use u128 so the successor after the last representable frame cannot
    // introduce a new, premature u64 deadline overflow.
    next_start: u128,
    next_end: u128,
    cell_start: u64,
    head: usize,
    len: usize,
    spans: Vec<Span>,
    sums: Vec<f64>,
    current: Vec<Kahan>,
}

impl EnergyCells {
    const fn capacity(include_range: bool) -> usize {
        if include_range {
            64
        } else {
            10
        }
    }

    pub(super) fn requested_bytes(channels: usize, include_range: bool) -> usize {
        let capacity = Self::capacity(include_range);
        // channels <= MAX_CHANNELS has already been validated by the meter.
        capacity * (std::mem::size_of::<Span>() + channels * std::mem::size_of::<f64>())
            + channels * std::mem::size_of::<Kahan>()
    }

    pub(super) fn new(
        rate: u32,
        channels: usize,
        include_range: bool,
    ) -> Result<Self, LoudnessError> {
        let rate = u64::from(rate);
        let integrated_window = (2 * rate + 2) / 5;
        let capacity = Self::capacity(include_range);
        let mut spans = Vec::new();
        spans
            .try_reserve_exact(capacity)
            .map_err(|_| LoudnessError::AllocationFailed)?;
        spans.resize(capacity, Span::default());
        let mut sums = Vec::new();
        sums.try_reserve_exact(capacity * channels)
            .map_err(|_| LoudnessError::AllocationFailed)?;
        sums.resize(capacity * channels, 0.0);
        let mut current = Vec::new();
        current
            .try_reserve_exact(channels)
            .map_err(|_| LoudnessError::AllocationFailed)?;
        current.resize(channels, Kahan::default());
        Ok(Self {
            rate,
            integrated_window,
            horizon: if include_range {
                3 * rate
            } else {
                integrated_window
            },
            next_start_index: 1,
            next_end_index: 0,
            next_start: u128::from(rate / 10),
            next_end: u128::from(integrated_window),
            cell_start: 0,
            head: 0,
            len: 0,
            spans,
            sums,
            current,
        })
    }

    pub(super) fn retained_bytes(&self) -> Result<usize, LoudnessError> {
        self.spans
            .capacity()
            .checked_mul(std::mem::size_of::<Span>())
            .and_then(|n| {
                self.sums
                    .capacity()
                    .checked_mul(std::mem::size_of::<f64>())
                    .and_then(|s| n.checked_add(s))
            })
            .and_then(|n| {
                self.current
                    .capacity()
                    .checked_mul(std::mem::size_of::<Kahan>())
                    .and_then(|s| n.checked_add(s))
            })
            .ok_or(LoudnessError::InputTooLong)
    }

    #[inline]
    fn offset(&self, index: u64) -> u128 {
        u128::from(index) * u128::from(self.rate) / 10
    }

    #[inline]
    pub(super) fn push(
        &mut self,
        end: u64,
        frame: &[f64],
        weights: &[f64],
    ) -> Result<(), LoudnessError> {
        debug_assert_eq!(frame.len(), self.current.len());
        debug_assert_eq!(weights.len(), frame.len());
        for ((sum, &x), &weight) in self.current.iter_mut().zip(frame).zip(weights) {
            if weight != 0.0 {
                sum.add(x * x);
            }
        }
        let deadline = u128::from(end);
        if deadline != self.next_start && deadline != self.next_end {
            return Ok(());
        }
        self.close_cell(end)?;
        if deadline == self.next_start {
            self.next_start_index += 1;
            self.next_start = self.offset(self.next_start_index);
        }
        if deadline == self.next_end {
            self.next_end_index += 1;
            self.next_end = u128::from(self.integrated_window) + self.offset(self.next_end_index);
        }
        Ok(())
    }

    #[cold]
    fn close_cell(&mut self, end: u64) -> Result<(), LoudnessError> {
        let capacity = self.spans.len();
        let cutoff = end.saturating_sub(self.horizon);
        while self.len != 0 && self.spans[self.head].end <= cutoff {
            self.head = (self.head + 1) % capacity;
            self.len -= 1;
        }
        if self.len == capacity || end <= self.cell_start {
            // Impossible for sequential admitted frames and the two grids.
            return Err(LoudnessError::InputTooLong);
        }
        let slot = (self.head + self.len) % capacity;
        self.spans[slot] = Span {
            start: self.cell_start,
            end,
        };
        let channels = self.current.len();
        for (dst, src) in self.sums[slot * channels..(slot + 1) * channels]
            .iter_mut()
            .zip(&mut self.current)
        {
            *dst = src.sum;
            *src = Kahan::default();
        }
        self.cell_start = end;
        self.len += 1;
        Ok(())
    }

    pub(super) fn energy(
        &self,
        end: u64,
        window: u64,
        weights: &[f64],
    ) -> Result<f64, LoudnessError> {
        if window == 0 || window > self.horizon || end != self.cell_start {
            return Err(LoudnessError::InputTooLong);
        }
        let start = end.checked_sub(window).ok_or(LoudnessError::InputTooLong)?;
        let channels = self.current.len();
        let mut sums = [Kahan::default(); MAX_CHANNELS];
        let mut at = start;
        for index in 0..self.len {
            let slot = (self.head + index) % self.spans.len();
            let span = self.spans[slot];
            if span.end <= start {
                continue;
            }
            if span.start != at || span.end > end {
                return Err(LoudnessError::InputTooLong);
            }
            at = span.end;
            for (sum, &value) in sums
                .iter_mut()
                .zip(&self.sums[slot * channels..(slot + 1) * channels])
            {
                sum.add(value);
            }
        }
        if at != end {
            return Err(LoudnessError::InputTooLong);
        }
        let mut total = Kahan::default();
        for (sum, &weight) in sums.iter().zip(weights) {
            if weight != 0.0 {
                total.add(weight * sum.sum);
            }
        }
        let energy = total.sum / window as f64;
        if energy.is_finite() && energy >= 0.0 {
            Ok(energy)
        } else {
            Err(LoudnessError::NumericalRange)
        }
    }
}
