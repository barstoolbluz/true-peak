//! Native experiment: share the existing full meter's filtered frame with a
//! short-term consumer without changing either observation graph. Not a
//! public API or a production path; native measurements gate any integration.
use super::super::{
    simd, statistics::energy_to_lufs, ChannelRole, LoudnessError, LoudnessMeasurement,
    LoudnessMeter, LoudnessProfile, LoudnessRange, LoudnessRangeAndShortTermMeter,
    LoudnessRangeMeter,
};
use super::{ShortTermMeter, ShortTermObservation};

// State of a short-only observation consumer; no second filter or ring.
// The weighted accumulation and physical 30-slot traversal deliberately
// match ShortTermMeter (including multiply grouping and wrapped slot order).
struct SharedObservations {
    blocks: [f64; 30],
    sum: f64,
    seen: u64,
    next_end: u64,
}
impl SharedObservations {
    fn new(rate: u32) -> Self {
        Self {
            blocks: [0.0; 30],
            sum: 0.0,
            seen: 0,
            next_end: u64::from(rate) / 10,
        }
    }
    fn observe(
        &mut self,
        end: u64,
        rate: u32,
        frame: &[f64],
        weights: &[f64],
    ) -> Result<Option<ShortTermObservation>, LoudnessError> {
        for (&y, &w) in frame.iter().zip(weights) {
            self.sum += w * y * y;
        }
        if !self.sum.is_finite() {
            return Err(LoudnessError::NumericalRange);
        }
        if end != self.next_end {
            return Ok(None);
        }
        self.blocks[(self.seen % 30) as usize] = self.sum;
        self.sum = 0.0;
        self.seen += 1;
        self.next_end = u64::try_from((u128::from(self.seen) + 1) * u128::from(rate) / 10)
            .map_err(|_| LoudnessError::InputTooLong)?;
        if self.seen < 30 {
            return Ok(None);
        }
        let energy = self.blocks.iter().sum::<f64>() / (3.0 * f64::from(rate));
        if !energy.is_finite() {
            return Err(LoudnessError::NumericalRange);
        }
        Ok(Some(ShortTermObservation {
            end_frame: end,
            lufs: energy_to_lufs(energy, LoudnessProfile::NativeEbu2023),
        }))
    }
}

struct Combined {
    full: LoudnessMeter,
    short: SharedObservations,
}
impl Combined {
    fn new(rate: u32, roles: &[ChannelRole], backend: simd::Backend) -> Self {
        let mut full = LoudnessMeter::with_roles_high_fidelity(rate, roles).unwrap();
        full.simd_backend = backend;
        Self {
            full,
            short: SharedObservations::new(rate),
        }
    }
    // Experimental valid-input path. Preflight and terminal error rules
    // follow the two original consumers. Keep the full meter's EOF path
    // unchanged: synthetic continuation never feeds short-term observations.
    fn push(
        &mut self,
        pcm: &[f64],
        mut emit: impl FnMut(ShortTermObservation),
    ) -> Result<(), LoudnessError> {
        let m = &mut self.full;
        if m.failed {
            return Err(LoudnessError::MeterFailed);
        }
        if !pcm.len().is_multiple_of(m.channels) {
            return Err(LoudnessError::IncompleteFrame {
                samples: pcm.len(),
                channels: m.channels,
            });
        }
        if let Some(sample_index) = pcm.iter().position(|x| !x.is_finite()) {
            return Err(LoudnessError::NonFiniteSample { sample_index });
        }
        if pcm.is_empty() {
            return Ok(());
        }
        let frames =
            u64::try_from(pcm.len() / m.channels).map_err(|_| LoudnessError::InputTooLong)?;
        let target = m
            .frames_seen
            .checked_add(frames)
            .ok_or(LoudnessError::InputTooLong)?;
        if target > u64::MAX - u64::from(m.sample_rate_hz) {
            return Err(LoudnessError::InputTooLong);
        }
        let include_integrated = m.include_integrated;
        m.preflight_history(target, include_integrated, true)?;
        for frame in pcm.chunks_exact(m.channels) {
            let base = m.write_frame * m.channels;
            if let Err(e) = m.process_frame_fidelity(frame, include_integrated, true) {
                m.failed = true;
                return Err(e);
            }
            // The ring cell just written is still present even when write_frame
            // wraps to zero. No extra filtered buffer or DSP pass.
            match self.short.observe(
                m.frames_seen,
                m.sample_rate_hz,
                &m.ring[base..base + m.channels],
                &m.weights,
            ) {
                Ok(Some(point)) => emit(point),
                Ok(None) => (),
                Err(e) => {
                    m.failed = true;
                    return Err(e);
                }
            }
        }
        m.real_frames = m.real_frames.checked_add(frames).ok_or_else(|| {
            m.failed = true;
            LoudnessError::InputTooLong
        })?;
        Ok(())
    }
}

fn layout(channels: usize) -> Vec<ChannelRole> {
    use ChannelRole::*;
    if channels == 1 {
        return vec![Mono];
    }
    [Left, Right, Center, Lfe, LeftSurround, RightSurround][..channels].to_vec()
}
fn pcm(rate: u32, channels: usize, frames: usize) -> Vec<f64> {
    let mut seed = 0x357ba6e2aa_u64;
    (0..frames * channels)
        .map(|i| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let x = (seed >> 11) as f64 / (1_u64 << 53) as f64 - 0.5;
            let j = i / channels;
            let gain = if j < rate as usize / 3 {
                0.2
            } else if j < 2 * rate as usize {
                1e-9
            } else if j < 3 * rate as usize {
                0.05
            } else {
                0.0
            };
            gain * (x + 0.25 * (i % channels % 3) as f64)
        })
        .collect()
}
fn backends() -> Vec<simd::Backend> {
    #[cfg(target_arch = "x86_64")]
    {
        let mut b = vec![simd::Backend::scalar()];
        b.extend(simd::Backend::sse2_if_available());
        b.extend(simd::Backend::avx_if_available());
        b
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        vec![simd::Backend::scalar()]
    }
}
fn observation_bits(points: &[ShortTermObservation]) -> Vec<(u64, u64)> {
    points
        .iter()
        .map(|p| (p.end_frame, p.lufs.to_bits()))
        .collect()
}
fn energy_bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|x| x.to_bits()).collect()
}

fn check(rate: u32, channels: usize, frames: usize, backend: simd::Backend) {
    let roles = layout(channels);
    let data = pcm(rate, channels, frames);
    let mut full = LoudnessMeter::with_roles_high_fidelity(rate, &roles).unwrap();
    full.simd_backend = backend;
    let mut short = ShortTermMeter::new(rate, &roles).unwrap();
    short.backend = backend;
    let mut combined = Combined::new(rate, &roles, backend);
    let mut expected = Vec::new();
    let mut actual = Vec::new();
    // Empty and malformed input does not mutate the test-only candidate.
    combined
        .push(&[], |_| panic!("empty push emitted"))
        .unwrap();
    assert!(matches!(
        combined.push(&vec![f64::NAN; channels], |_| panic!()),
        Err(LoudnessError::NonFiniteSample { .. })
    ));
    if channels > 1 {
        assert!(matches!(
            combined.push(&[0.0], |_| panic!()),
            Err(LoudnessError::IncompleteFrame { .. })
        ));
    }
    assert_eq!(combined.full.real_frames, 0);
    for p in data.chunks(713 * channels) {
        full.push_interleaved(p).unwrap();
        short
            .push_interleaved(p, |point| expected.push(point))
            .unwrap();
    }
    for p in data.chunks(997 * channels) {
        combined.push(p, |point| actual.push(point)).unwrap();
    }
    assert_eq!(observation_bits(&expected), observation_bits(&actual));
    assert_eq!(
        energy_bits(&full.integrated_energies),
        energy_bits(&combined.full.integrated_energies)
    );
    assert_eq!(
        energy_bits(&full.lra_energies),
        energy_bits(&combined.full.lra_energies)
    );
    assert_eq!(energy_bits(&full.ring), energy_bits(&combined.full.ring));
    assert_eq!(
        energy_bits(&full.fidelity.as_ref().unwrap().states),
        energy_bits(&combined.full.fidelity.as_ref().unwrap().states)
    );
    assert_eq!(full.real_frames, combined.full.real_frames);
    let f = full.finalize().unwrap();
    let g = combined.full.finalize().unwrap();
    assert_eq!(
        energy_bits(&f.statistics.integrated_energies),
        energy_bits(&g.statistics.integrated_energies)
    );
    assert_eq!(
        energy_bits(&f.statistics.lra_energies),
        energy_bits(&g.statistics.lra_energies)
    );
    assert_eq!(f.summary.integrated, g.summary.integrated);
    assert_eq!(f.summary.range, g.summary.range);
    assert_eq!(f.summary.real_frames, g.summary.real_frames);
    // The short-term path must not receive EOF-continuation observations.
    assert_eq!(
        actual.last().map(|p| p.end_frame),
        expected.last().map(|p| p.end_frame)
    );
    assert!(actual.iter().all(|p| p.end_frame <= frames as u64));
}

#[test]
fn shared_filter_experimental_parity() {
    for backend in backends() {
        for rate in 8_000..=8_009 {
            for channels in [1, 2, 3, 6] {
                check(rate, channels, 3 * rate as usize + 17, backend);
            }
        }
        for frames in [0, 1, 3 * 8_001 - 1, 3 * 8_001, 3 * 8_001 + 1] {
            check(8_001, 2, frames, backend);
        }
        check(44_101, 2, 4 * 44_101 + 17, backend);
    }
}

fn run_timed(
    rate: u32,
    roles: &[ChannelRole],
    data: &[f64],
    shared: bool,
) -> (LoudnessMeasurement, Vec<ShortTermObservation>) {
    let mut points = Vec::new();
    let backend = simd::Backend::production(roles.len());
    if shared {
        let mut m = Combined::new(rate, roles, backend);
        for p in data.chunks(4093 * roles.len()) {
            m.push(p, |x| points.push(x)).unwrap();
        }
        (m.full.finalize().unwrap(), points)
    } else {
        let mut full = LoudnessMeter::with_roles_high_fidelity(rate, roles).unwrap();
        let mut short = ShortTermMeter::new(rate, roles).unwrap();
        for p in data.chunks(4093 * roles.len()) {
            full.push_interleaved(p).unwrap();
            short.push_interleaved(p, |x| points.push(x)).unwrap();
        }
        (full.finalize().unwrap(), points)
    }
}

#[test]
#[ignore = "experimental paired native timing; opt in after pristine baseline qualification"]
fn shared_filter_experimental_timing() {
    use std::{hint::black_box, time::Instant};
    for rate in [44_100, 44_101, 48_000, 96_000, 192_000, 384_000, 768_000] {
        for channels in [1, 2, 6] {
            let roles = layout(channels);
            let data = pcm(rate, channels, 4 * rate as usize + 17);
            let mut times = [Vec::new(), Vec::new()];
            let reference = run_timed(rate, &roles, &data, false);
            for rep in 0..5 {
                for k in 0..2 {
                    let mode = (rep + k) % 2;
                    let start = Instant::now();
                    let result = black_box(run_timed(rate, &roles, &data, mode == 1));
                    times[mode].push(start.elapsed().as_secs_f64());
                    assert_eq!(observation_bits(&reference.1), observation_bits(&result.1));
                    assert_eq!(
                        energy_bits(&reference.0.statistics.integrated_energies),
                        energy_bits(&result.0.statistics.integrated_energies)
                    );
                    assert_eq!(
                        energy_bits(&reference.0.statistics.lra_energies),
                        energy_bits(&result.0.statistics.lra_energies)
                    );
                    assert_eq!(reference.0.summary.integrated, result.0.summary.integrated);
                    assert_eq!(reference.0.summary.range, result.0.summary.range);
                }
            }
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            eprintln!(
                "SHARED_TIMING {rate} {channels} separate_s={:.9} shared_s={:.9} speedup={:.5}",
                times[0][2],
                times[1][2],
                times[0][2] / times[1][2]
            );
        }
    }
}

// Compare actual production dispatch, not just the earlier prototype. The
// separate consumers have independent filters and retain the exact same
// observation graphs; the older prototype still checks the short-term graph
// independently in shared_filter_experimental_parity.
fn check_production_shared(rate: u32, channels: usize, frames: usize, backend: simd::Backend) {
    let roles = layout(channels);
    let data = pcm(rate, channels, frames);
    let mut range = LoudnessRangeMeter::new(rate, &roles).unwrap();
    let mut short = ShortTermMeter::new(rate, &roles).unwrap();
    let mut shared = LoudnessRangeAndShortTermMeter::new(rate, &roles).unwrap();
    range.inner.simd_backend = backend;
    short.backend = backend;
    shared.range.inner.simd_backend = backend;
    let mut expected = Vec::new();
    let mut actual = Vec::new();
    for chunk in data.chunks(713 * channels) {
        range.push_interleaved(chunk).unwrap();
        short
            .push_interleaved(chunk, |point| expected.push(point))
            .unwrap();
    }
    for chunk in data.chunks(997 * channels) {
        // Malformed input remains retryable even after real input has been
        // consumed. It must not change either clock, filter or accumulator.
        let before = shared.range.inner.real_frames;
        shared
            .push_interleaved(&[], |_| panic!("empty push emitted"))
            .unwrap();
        assert!(matches!(
            shared.push_interleaved(&vec![f64::NAN; channels], |_| panic!()),
            Err(LoudnessError::NonFiniteSample { sample_index: 0 })
        ));
        if channels > 1 {
            assert!(matches!(
                shared.push_interleaved(&[0.0], |_| panic!()),
                Err(LoudnessError::IncompleteFrame { .. })
            ));
        }
        assert_eq!(shared.range.inner.real_frames, before);
        assert_eq!(shared.short.frames, before);
        shared
            .push_interleaved(chunk, |point| actual.push(point))
            .unwrap();
    }
    assert_eq!(shared.range.inner.real_frames, frames as u64);
    assert_eq!(shared.short.frames, frames as u64);
    assert_eq!(observation_bits(&actual), observation_bits(&expected));
    assert!(actual.iter().all(|point| point.end_frame <= frames as u64));
    assert_eq!(
        energy_bits(&range.inner.lra_energies),
        energy_bits(&shared.range.inner.lra_energies)
    );
    assert_eq!(
        energy_bits(&range.inner.fidelity.as_ref().unwrap().states),
        energy_bits(&shared.range.inner.fidelity.as_ref().unwrap().states)
    );
    assert!(!shared.range.inner.include_integrated);
    assert_eq!(shared.range.inner.integrated_energies.capacity(), 0);
    let a = range.finalize().unwrap();
    let b = shared.finalize().unwrap();
    assert_eq!(a, b);
    assert_eq!(
        a.finite_lu().map(f64::to_bits),
        b.finite_lu().map(f64::to_bits)
    );
}

#[test]
fn production_shared_range_and_short_parity() {
    for backend in backends() {
        for rate in 8_000..=8_009 {
            for channels in [1, 2, 3, 6] {
                check_production_shared(rate, channels, 4 * rate as usize + 17, backend);
            }
        }
        for frames in [0, 1, 3 * 8_001 - 1, 3 * 8_001, 3 * 8_001 + 1] {
            check_production_shared(8_001, 2, frames, backend);
        }
        check_production_shared(44_101, 2, 4 * 44_101 + 17, backend);
        check_production_shared(768_000, 2, 3 * 768_000 + 17, backend);
    }
}

#[test]
fn production_shared_validation_and_terminal_errors() {
    let roles = [ChannelRole::Left, ChannelRole::Right];
    assert!(matches!(
        LoudnessRangeAndShortTermMeter::new(7_999, &roles),
        Err(LoudnessError::UnsupportedSampleRate { .. })
    ));
    assert!(matches!(
        LoudnessRangeAndShortTermMeter::new(2_822_401, &roles),
        Err(LoudnessError::UnsupportedSampleRate { .. })
    ));
    assert!(LoudnessRangeAndShortTermMeter::new(8_000, &[]).is_err());
    assert!(
        LoudnessRangeAndShortTermMeter::new(8_000, &[ChannelRole::Left, ChannelRole::Left])
            .is_err()
    );
    assert!(LoudnessRangeAndShortTermMeter::new(8_000, &[ChannelRole::Lfe]).is_err());
    let mut meter = LoudnessRangeAndShortTermMeter::new(8_000, &roles).unwrap();
    // An over-budget push fails before either filter/observer consumes data.
    meter.range.inner.storage_limit_bytes = meter.range.inner.fixed_storage_bytes;
    assert!(matches!(
        meter.push_interleaved(&vec![0.1; 3 * 8_000 * 2], |_| panic!()),
        Err(LoudnessError::ResourceLimit { .. })
    ));
    assert_eq!(meter.range.inner.real_frames, 0);
    assert_eq!(meter.short.frames, 0);
    assert!(!meter.range.inner.failed);
    meter.push_interleaved(&[0.1, 0.2], |_| panic!()).unwrap();
    assert_eq!(meter.range.inner.real_frames, 1);
    assert_eq!(meter.short.frames, 1);
    // A finite input can overflow the short-term energy accumulation well
    // before the first range window. The combined meter must be terminal.
    assert!(matches!(
        meter.push_interleaved(&[1e200, -1e200], |_| panic!()),
        Err(LoudnessError::NumericalRange)
    ));
    assert!(matches!(
        meter.push_interleaved(&[], |_| panic!()),
        Err(LoudnessError::MeterFailed)
    ));
    assert!(matches!(meter.finalize(), Err(LoudnessError::MeterFailed)));
}

// Time the actual production range-plus-short-term path against the two
// standalone consumers. Keep the independent full-meter prototype above.
fn range_timed(
    rate: u32,
    roles: &[ChannelRole],
    data: &[f64],
    shared: bool,
) -> (LoudnessRange, Vec<ShortTermObservation>) {
    let mut points = Vec::new();
    if shared {
        let mut m = LoudnessRangeAndShortTermMeter::new(rate, roles).unwrap();
        for p in data.chunks(4093 * roles.len()) {
            m.push_interleaved(p, |x| points.push(x)).unwrap();
        }
        assert!(m.range.inner.integrated_energies.is_empty());
        (m.finalize().unwrap(), points)
    } else {
        let mut full = LoudnessRangeMeter::new(rate, roles).unwrap();
        let mut short = ShortTermMeter::new(rate, roles).unwrap();
        for p in data.chunks(4093 * roles.len()) {
            full.push_interleaved(p).unwrap();
            short.push_interleaved(p, |x| points.push(x)).unwrap();
        }
        (full.finalize().unwrap(), points)
    }
}

#[test]
#[ignore = "production shared range and short-term paired native timing"]
fn production_shared_range_timing() {
    use std::{hint::black_box, time::Instant};
    for rate in [44_101, 48_000, 768_000] {
        for channels in [1, 2, 6] {
            let roles = layout(channels);
            let data = pcm(rate, channels, 4 * rate as usize + 17);
            let reference = range_timed(rate, &roles, &data, false);
            // Warm both paths; raw alternating measured pairs remain visible.
            let warm = range_timed(rate, &roles, &data, true);
            assert_eq!(reference.0, warm.0);
            assert_eq!(observation_bits(&reference.1), observation_bits(&warm.1));
            for rep in 0..5 {
                let mut times = [0.0; 2];
                for k in 0..2 {
                    let mode = (rep + k) % 2;
                    let start = Instant::now();
                    let result = black_box(range_timed(rate, &roles, &data, mode == 1));
                    times[mode] = start.elapsed().as_secs_f64();
                    assert_eq!(reference.0, result.0);
                    assert_eq!(observation_bits(&reference.1), observation_bits(&result.1));
                    assert!(result
                        .1
                        .iter()
                        .all(|p| p.end_frame <= (data.len() / channels) as u64));
                }
                eprintln!(
                    "RANGE_SHARED_PAIR {rate} {channels} {rep} separate_s={:.9} shared_s={:.9}",
                    times[0], times[1]
                );
            }
        }
    }
}
