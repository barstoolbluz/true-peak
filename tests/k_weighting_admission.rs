//! Regression for the owner-authorized native lower-rate admission change.
use true_peak::loudness::{
    ChannelRole, LoudnessError, LoudnessMeter, LoudnessMetricDemand, LoudnessProfile,
    ShortTermMeter, DEFAULT_LOUDNESS_STORAGE_LIMIT_BYTES,
};

const MONO: [ChannelRole; 1] = [ChannelRole::Mono];
const LIMIT: usize = DEFAULT_LOUDNESS_STORAGE_LIMIT_BYTES;
const NATIVE: LoudnessProfile = LoudnessProfile::NativeEbu2023;
const COMPAT: LoudnessProfile = LoudnessProfile::Libebur128126;
const FULL: LoudnessMetricDemand = LoudnessMetricDemand::IntegratedAndRange;
const INTEGRATED: LoudnessMetricDemand = LoudnessMetricDemand::IntegratedOnly;
const UPPER: u32 = 2_822_400;

fn assert_unsupported<T>(result: Result<T, LoudnessError>, rate: u32) {
    assert!(
        matches!(result, Err(LoudnessError::UnsupportedSampleRate { sample_rate_hz }) if sample_rate_hz == rate),
        "native sub-Nyquist rate {rate} was not rejected with UnsupportedSampleRate"
    );
}

#[test]
fn ordinary_native_rejects_sub_nyquist_rates() {
    for rate in [16, 3_000, 3_363] {
        assert_unsupported(LoudnessMeter::new(rate, 1), rate);
        assert_unsupported(LoudnessMeter::with_roles(rate, &MONO, NATIVE), rate);
        assert_unsupported(LoudnessMeter::with_roles_and_limit(rate, &MONO, NATIVE, LIMIT), rate);
        for demand in [FULL, INTEGRATED] {
            assert_unsupported(LoudnessMeter::with_roles_and_metric_demand(rate, &MONO, NATIVE, demand), rate);
            assert_unsupported(LoudnessMeter::with_roles_and_metric_demand_and_limit(rate, &MONO, NATIVE, demand, LIMIT), rate);
            assert_unsupported(LoudnessMeter::with_roles_high_fidelity_and_metric_demand_and_limit(rate, &MONO, demand, LIMIT), rate);
        }
        assert_unsupported(LoudnessMeter::with_roles_high_fidelity(rate, &MONO), rate);
    }
}

#[test]
fn accepted_native_and_compatibility_boundaries() {
    for rate in [3_364, 4_000, 7_999, 8_000, 44_101, 48_000, 768_001, 1_536_000, UPPER] {
        assert!(LoudnessMeter::new(rate, 1).is_ok(), "native rate {rate}");
        assert!(LoudnessMeter::with_roles(rate, &MONO, NATIVE).is_ok());
        assert!(LoudnessMeter::with_roles_and_metric_demand(rate, &MONO, NATIVE, INTEGRATED).is_ok());
        assert!(LoudnessMeter::with_roles_high_fidelity(rate, &MONO).is_ok());
    }
    for rate in [16, 3_000, 3_363, 3_364, UPPER] {
        assert!(LoudnessMeter::libebur128_126_default_layout(rate, 1).is_ok(), "compatibility rate {rate}");
        assert!(LoudnessMeter::with_roles(rate, &MONO, COMPAT).is_ok());
    }
    assert_unsupported(LoudnessMeter::new(UPPER + 1, 1), UPPER + 1);
    assert_unsupported(LoudnessMeter::libebur128_126_default_layout(15, 1), 15);
    assert_unsupported(LoudnessMeter::libebur128_126_default_layout(UPPER + 1, 1), UPPER + 1);
    assert!(matches!(LoudnessMeter::new(0, 1), Err(LoudnessError::InvalidSampleRate)));
    assert!(matches!(LoudnessMeter::with_roles_high_fidelity(0, &MONO), Err(LoudnessError::InvalidSampleRate)));
    for rate in [3_364, 7_999, UPPER + 1] {
        assert_unsupported(ShortTermMeter::new(rate, &MONO), rate);
        assert_unsupported(ShortTermMeter::new_high_fidelity(rate, &MONO), rate);
    }
    for rate in [8_000, UPPER] {
        assert!(ShortTermMeter::new(rate, &MONO).is_ok());
        assert!(ShortTermMeter::new_high_fidelity(rate, &MONO).is_ok());
    }
}
