//! Public-path failing-first regressions. See the baseline runner in the
//! delivery's kweight-qualification directory. The pre-fidelity R5 base
//! uses the old arithmetic; both native constructors now use fidelity.
use true_peak::loudness::{ChannelRole, ShortTermMeter};

fn last(rate: u32, dc: f64, amplitude: f64, chunk: usize) -> f64 {
    let mut meter = ShortTermMeter::new_high_fidelity(rate, &[ChannelRole::Mono]).unwrap();
    let mut observation = None;
    let mut position = 0;
    let mut pcm = vec![0.0; chunk];
    while position < 4 * rate as usize {
        let count = chunk.min(4 * rate as usize - position);
        for (n, x) in pcm[..count].iter_mut().enumerate() {
            let phase = std::f64::consts::TAU * (((position + n) * 1000) % rate as usize) as f64 / f64::from(rate);
            *x = dc + amplitude * phase.sin();
        }
        meter.push_interleaved(&pcm[..count], |p| observation = Some(p)).unwrap();
        position += count;
    }
    let result = observation.unwrap();
    assert_eq!(result.end_frame, 4 * u64::from(rate));
    result.lufs
}

#[test]
fn high_rate_dc_rejection() {
    // The actual filter transient after one second is approximately -2106 LUFS.
    // This requirement separates it from the old -107 LUFS arithmetic residual.
    for dc in [0.1, -0.1] {
        let value = last(768_000, dc, 0.0, 4093);
        eprintln!("768 kHz DC {dc}: {value:.12} LUFS");
        assert!(value < -1500.0, "arithmetic residual: {value:.12} LUFS");
    }
}

#[test]
fn high_rate_low_level_on_dc() {
    // Arbitrary-precision bilinear transform at 1 kHz, 768 kHz. This is the
    // last three seconds, not the first window containing the DC step.
    let expected = -123.03842522169084;
    for (dc, amplitude) in [(0.01, 1e-6), (-0.01, -1e-6)] {
        let value = last(768_000, dc, amplitude, 1009);
        eprintln!("768 kHz mixed: {value:.12} LUFS, error {:.12} LU", value - expected);
        assert!((value - expected).abs() < 1e-7, "{value} != {expected}");
        assert_eq!(value.to_bits(), last(768_000, dc, amplitude, 8192).to_bits());
    }
}
