//! Test-only compensated positive-cell reduction. This is not a replacement
//! meter, a storage-admission change, or a production runtime path.
use super::*;
use std::collections::VecDeque;

#[derive(Clone, Copy, Default)]
struct Kahan {
    sum: f64,
    correction: f64,
}
impl Kahan {
    fn add(&mut self, x: f64) {
        let y = x - self.correction;
        let t = self.sum + y;
        self.correction = (t - self.sum) - y;
        self.sum = t;
    }
}
struct Cell {
    start: u64,
    end: u64,
    sums: [Kahan; MAX_CHANNELS],
}
struct Cells {
    rate: u64,
    iw: u64,
    lw: u64,
    frames: u64,
    start: u64,
    bidx: u64,
    eidx: u64,
    be: u64,
    ee: u64,
    cells: VecDeque<Cell>,
    current: [Kahan; MAX_CHANNELS],
}
impl Cells {
    fn new(rate: u32) -> Self {
        let rate = u64::from(rate);
        let iw = (2 * rate + 2) / 5;
        Self {
            rate,
            iw,
            lw: 3 * rate,
            frames: 0,
            start: 0,
            bidx: 1,
            eidx: 0,
            be: rate / 10,
            ee: iw,
            cells: VecDeque::with_capacity(64),
            current: [Kahan::default(); MAX_CHANNELS],
        }
    }
    fn offset(&self, index: u64) -> u64 {
        u64::try_from(u128::from(index) * u128::from(self.rate) / 10).unwrap()
    }
    fn reduce(&self, start: u64, end: u64, weights: &[f64]) -> f64 {
        let mut sums = [Kahan::default(); MAX_CHANNELS];
        let mut at = start;
        for cell in &self.cells {
            if cell.end <= start {
                continue;
            }
            assert_eq!(cell.start, at);
            assert!(cell.end <= end);
            at = cell.end;
            for (s, c) in sums.iter_mut().zip(&cell.sums).take(weights.len()) {
                s.add(c.sum);
            }
        }
        assert_eq!(at, end);
        let mut total = Kahan::default();
        for (s, &w) in sums.iter().zip(weights) {
            if w != 0.0 {
                total.add(w * s.sum);
            }
        }
        total.sum / (end - start) as f64
    }
    // Input is a finite already-filtered frame. Do not feed this experimental
    // object arbitrary public data: range/error semantics remain a later gate.
    fn push(&mut self, frame: &[f64], weights: &[f64], real: bool) -> [Option<f64>; 2] {
        assert_eq!(frame.len(), weights.len());
        self.frames += 1;
        let end = self.frames;
        for ((s, &x), &w) in self.current.iter_mut().zip(frame).zip(weights) {
            if w != 0.0 {
                s.add(x * x);
            }
        }
        if end == self.be || end == self.ee {
            let cutoff = end.saturating_sub(self.lw);
            while self.cells.front().is_some_and(|c| c.end <= cutoff) {
                self.cells.pop_front();
            }
            assert!(self.cells.len() < 64);
            self.cells.push_back(Cell {
                start: self.start,
                end,
                sums: self.current,
            });
            self.current.fill(Kahan::default());
            self.start = end;
        }
        let mut out = [None; 2];
        if end == self.ee {
            if real {
                out[0] = Some(self.reduce(end - self.iw, end, weights));
            }
            self.eidx += 1;
            self.ee = self.iw + self.offset(self.eidx);
        }
        if end == self.be {
            if self.bidx >= 30 {
                out[1] = Some(self.reduce(end - self.lw, end, weights));
            }
            self.bidx += 1;
            self.be = self.offset(self.bidx);
        }
        out
    }
}

fn compare_meter(rate: u32, roles: &[ChannelRole], backend: simd::Backend) {
    let mut m = LoudnessMeter::with_roles_high_fidelity(rate, roles).unwrap();
    m.simd_backend = backend;
    m.use_native_reference_ring();
    let mut cells = Cells::new(rate);
    let mut clock = WindowClock::new(rate, LoudnessProfile::NativeEbu2023).unwrap();
    let real = 3 * u64::from(rate) + 17;
    let tail = (3 * u64::from(rate)).div_ceil(2);
    let mut seed = 0x98ab_53f6_a497_u64;
    let mut input = vec![0.0; roles.len()];
    let mut counts = [0; 2];
    m.preflight_history(real, true, true).unwrap();
    for end in 1..=real + tail {
        let is_real = end <= real;
        if end == real + 1 {
            m.preflight_history(real + tail, false, true).unwrap();
        }
        for x in &mut input {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let amplitude = if end < u64::from(rate) / 3 {
                0.2
            } else if end < 2 * u64::from(rate) {
                1e-9
            } else if is_real {
                0.03
            } else {
                0.0
            };
            *x = ((seed >> 11) as f64 / (1_u64 << 53) as f64 - 0.5) * amplitude;
        }
        let base = m.write_frame * roles.len();
        m.process_frame_fidelity(&input, is_real, true).unwrap();
        let output = cells.push(&m.ring[base..base + roles.len()], &m.weights, is_real);
        let due = [
            is_real && clock.take_integrated_if_due(end).unwrap(),
            clock.take_lra_if_due(end).unwrap(),
        ];
        let lengths = [clock.integrated_window_frames(), clock.lra_window_frames()];
        for (k, result) in output.into_iter().enumerate() {
            assert_eq!(result.is_some(), due[k], "{rate} {end} {k}");
            if let Some(energy) = result {
                let expected = m.window_energy(lengths[k]).unwrap();
                // A conservative normal positive-sum forward-error envelope;
                // no bitwise requirement when addition grouping changes.
                let n = (lengths[k] + 2 * roles.len() as u64 + 128) as f64;
                let u = f64::EPSILON / 2.0;
                let tolerance = 2.0 * n * u / (1.0 - n * u);
                assert!(
                    (energy - expected).abs() <= tolerance * expected,
                    "{rate} {end} {k} {energy:e} {expected:e}"
                );
                // This ordinary signal fixture stays away from absolute gates.
                let gate = |e| {
                    if k == 0 {
                        e > ABSOLUTE_GATE_ENERGY
                    } else {
                        e >= ABSOLUTE_GATE_ENERGY
                    }
                };
                assert_eq!(gate(energy), gate(expected));
                if gate(energy) {
                    counts[k] += 1;
                }
            }
        }
    }
    assert_eq!(counts, [m.integrated_energies.len(), m.lra_energies.len()]);
}

#[test]
fn compensated_cells_experimental_clock_and_filter_parity() {
    use ChannelRole::*;
    let layouts: [&[ChannelRole]; 4] = [
        &[Mono],
        &[Left, Right],
        &[Left, Right, Center],
        &[Left, Right, Center, Lfe, LeftSurround, RightSurround],
    ];
    for rate in 8000..=8009 {
        for roles in layouts {
            compare_meter(rate, roles, simd::Backend::scalar());
        }
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
            compare_meter(8001, roles, backend);
        }
    }
}

#[test]
fn compensated_cells_experimental_exact_gate_fixture() {
    // Exact prefiltered constant samples from the adversarial MP/rational
    // experiment. The specified f64 gate threshold is the comparison target.
    // Input bit patterns and exact classifications are generated independently.
    for &(bits, above) in GATE_CASES {
        let x = f64::from_bits(bits);
        let rate = 8001;
        let mut cells = Cells::new(rate);
        let mut count = [0; 2];
        for _ in 0..3 * rate + 1 {
            for (k, result) in cells.push(&[x], &[1.0], true).into_iter().enumerate() {
                if let Some(e) = result {
                    assert_eq!(
                        if k == 0 {
                            e > ABSOLUTE_GATE_ENERGY
                        } else {
                            e >= ABSOLUTE_GATE_ENERGY
                        },
                        above
                    );
                    count[k] += 1;
                }
            }
        }
        assert_eq!(count, [27, 1]);
    }
}

const GATE_CASES: &[(u64, bool)] = &[
    (0x3f3670bbcee4aacd, false),
    (0x3f3670bbcee4aace, false),
    (0x3f3670bbcee4aacf, false),
    (0x3f3670bbcee4aad0, false),
    (0x3f3670bbcee4aad1, false),
    (0x3f3670bbcee4aad2, false),
    (0x3f3670bbcee4aad3, false),
    (0x3f3670bbcee4aad4, false),
    (0x3f3670bbcee4aad5, false),
    (0x3f3670bbcee4aad6, true),
    (0x3f3670bbcee4aad7, true),
    (0x3f3670bbcee4aad8, true),
    (0x3f3670bbcee4aad9, true),
    (0x3f3670bbcee4aada, true),
    (0x3f3670bbcee4aadb, true),
    (0x3f3670bbcee4aadc, true),
    (0x3f3670bbcee4aadd, true),
];
