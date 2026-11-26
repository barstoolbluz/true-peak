//! Native release qualification: default and explicit constructors, same
//! PCM and window demand. Copy unchanged to the preceding candidate for
//! cross-release comparisons (default there is the historical recurrence).
use std::hint::black_box;
use std::time::Instant;
use true_peak::loudness::{ChannelRole, LoudnessMeter, LoudnessProfile,
    LoudnessMetricDemand, LoudnessSimdBackend, ShortTermMeter};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(4..=5).contains(&args.len()) { return Err("usage: k_weighting_fidelity_bench RATE CHANNELS SECONDS [default|explicit|both]".into()); }
    let mode = args.get(4).map_or("both",String::as_str);
    if !["default","explicit","both"].contains(&mode) {return Err("invalid constructor selection".into());}
    let rate: u32 = args[1].parse()?;
    let channels: usize = args[2].parse()?;
    let seconds: usize = args[3].parse()?;
    use ChannelRole::*;
    let roles = match channels {
        1 => vec![Mono],
        2 => vec![Left,Right],
        6 => vec![Left,Right,Center,Lfe,LeftSurround,RightSurround],
        _ => return Err("use 1, 2 or 6 channels".into()),
    };
    if seconds < 4 || !(8_000..=2_822_400).contains(&rate) { return Err("at least four seconds; rate 8–2822.4 kHz".into()); }
    // Explicit benchmark budget only: default library storage admission is unchanged.
    let limit = 1024usize * 1024 * 1024;
    let pcm: Vec<_> = (0..rate).flat_map(|i| (0..channels).map(move |ch| {
        0.17 * (std::f64::consts::TAU * f64::from(i) * (997+17*ch) as f64 / f64::from(rate)).sin() + 0.01
    })).collect();
    for fidelity in [false,true] {
        let name = if fidelity {"explicit"} else {"default"};
        if mode != "both" && mode != name {continue;}
        for backend in [None,Some(LoudnessSimdBackend::Scalar),Some(LoudnessSimdBackend::Sse2),Some(LoudnessSimdBackend::Avx)] {
            let mut timings=Vec::new();
            let backend_name=backend.map_or("production",|b|b.qualification_name());
            for _ in 0..3 {
                let mut meter=if fidelity {LoudnessMeter::with_roles_high_fidelity_and_metric_demand_and_limit(rate,&roles,LoudnessMetricDemand::IntegratedAndRange,limit)?} else {LoudnessMeter::with_roles_and_limit(rate,&roles,LoudnessProfile::NativeEbu2023,limit)?};
                if let Some(b)=backend {
                    if meter.force_simd_backend_for_qualification(b).is_err() {
                        println!("UNAVAILABLE {name} {backend_name} rate={rate} ch={channels}");
                        break;
                    }
                }
                let start=Instant::now();
                for _ in 0..seconds { for c in pcm.chunks(4096*channels) {meter.push_interleaved(black_box(c))?;} }
                let result=meter.finalize()?;
                let elapsed=start.elapsed().as_secs_f64();
                black_box(result);
                timings.push(elapsed);
            }
            if !timings.is_empty() {
                timings.sort_by(f64::total_cmp);
                println!("full {name} {backend_name} rate={rate} ch={channels} median_s={:.9} realtime={:.3}",timings[timings.len()/2],seconds as f64/timings[timings.len()/2]);
            }
        }
        let mut times=Vec::new();
        for _ in 0..3 {
            let meter=if fidelity {ShortTermMeter::new_high_fidelity(rate,&roles)} else {ShortTermMeter::new(rate,&roles)};
            // The preceding release capped both short constructors at 768k.
            let mut meter=match meter {
                Ok(m)=>m,
                Err(true_peak::loudness::LoudnessError::UnsupportedSampleRate{..}) if rate>768_000=>{
                    println!("UNAVAILABLE short {name} rate={rate} ch={channels}"); break;
                }
                Err(e)=>return Err(e.into()),
            };
            let start=Instant::now();
            for _ in 0..seconds {for c in pcm.chunks(4096*channels) {meter.push_interleaved(black_box(c),|p|{black_box(p);})?;}}
            times.push(start.elapsed().as_secs_f64());
        }
        if !times.is_empty() {
            times.sort_by(f64::total_cmp);
            println!("short {name} production rate={rate} ch={channels} median_s={:.9} realtime={:.3}",times[1],seconds as f64/times[1]);
        }
    }
    Ok(())
}
