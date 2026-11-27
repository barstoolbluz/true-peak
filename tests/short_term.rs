use true_peak::loudness::{ChannelRole, LoudnessError, ShortTermMeter};

#[test]
fn exact_rational_clock_and_push_boundary_invariance() {
    let rate = 44_101_u32;
    let pcm: Vec<_> = (0..4 * rate)
        .map(|i| 0.1 * (std::f64::consts::TAU * 1000.0 * f64::from(i) / f64::from(rate)).sin())
        .collect();
    let mut a = ShortTermMeter::new(rate, &[ChannelRole::Mono]).unwrap();
    let mut whole = Vec::new();
    a.push_interleaved(&pcm, |p| whole.push(p)).unwrap();
    let mut b = ShortTermMeter::new(rate, &[ChannelRole::Mono]).unwrap();
    let mut split = Vec::new();
    for c in pcm.chunks(713) {
        b.push_interleaved(c, |p| split.push(p)).unwrap();
    }
    assert_eq!(whole, split);
    assert_eq!(whole.len(), 11);
    for (i, p) in whole.iter().enumerate() {
        assert_eq!(
            p.end_frame,
            3 * u64::from(rate) + (i as u64 * u64::from(rate)) / 10
        );
    }
    assert!(whole.iter().all(|p| p.lufs.is_finite()));
}

#[test]
fn ungated_silence_and_validation_before_mutation() {
    let rate = 48_000;
    let mut m = ShortTermMeter::new(rate, &[ChannelRole::Left, ChannelRole::Right]).unwrap();
    assert!(matches!(
        m.push_interleaved(&[0.0], |_| panic!()),
        Err(LoudnessError::IncompleteFrame { .. })
    ));
    assert!(matches!(
        m.push_interleaved(&[0.0, f64::NAN], |_| panic!()),
        Err(LoudnessError::NonFiniteSample { .. })
    ));
    let mut p = Vec::new();
    m.push_interleaved(&vec![0.0; rate as usize * 3 * 2 - 2], |x| p.push(x))
        .unwrap();
    assert!(p.is_empty());
    m.push_interleaved(&[0.0, 0.0], |x| p.push(x)).unwrap();
    assert_eq!(p.len(), 1);
    assert_eq!(p[0].lufs, f64::NEG_INFINITY);
    assert_eq!(p[0].end_frame, 3 * u64::from(rate));
}
