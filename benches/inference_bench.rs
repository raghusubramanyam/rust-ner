//! Latency benchmarks for the two-tier PII/NER pipeline.
//!
//! Run with:
//!   cargo bench
//!
//! The NER model benchmarks are gated behind the existence of model files.
//! Regex benchmarks always run.

use aegis_ner::{regex_scanner::RegexScanner, Scanner, ScanOptions};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

// ── Sample prompts ────────────────────────────────────────────────────────────

const CLEAN_PROMPT: &str =
    "The quarterly review meeting is scheduled for Thursday at 2pm in the main boardroom.";

const REGEX_HEAVY_PROMPT: &str = concat!(
    "Please process the following: SSN 123-45-6789, ",
    "email alice@example.com, credit card 4111111111111111, ",
    "IP 192.168.1.100, DOB 01/15/1990, AWS key AKIAIOSFODNN7EXAMPLE."
);

const NER_HEAVY_PROMPT: &str =
    "Alice Johnson, the CFO of Acme Corporation, visited Paris last week \
     before flying to New York for meetings with Goldman Sachs executives.";

// ── Regex (Tier 1) benchmarks ─────────────────────────────────────────────────

fn bench_regex_clean(c: &mut Criterion) {
    c.bench_function("regex/clean_prompt", |b| {
        b.iter(|| RegexScanner::scan(black_box(CLEAN_PROMPT)))
    });
}

fn bench_regex_pii(c: &mut Criterion) {
    c.bench_function("regex/pii_heavy_prompt", |b| {
        b.iter(|| RegexScanner::scan(black_box(REGEX_HEAVY_PROMPT)))
    });
}

fn bench_regex_prompt_lengths(c: &mut Criterion) {
    let mut group = c.benchmark_group("regex/by_length");
    for word_count in [10, 50, 100, 500] {
        let prompt = format!("email test@example.com {}", "word ".repeat(word_count));
        group.bench_with_input(
            BenchmarkId::from_parameter(word_count),
            &prompt,
            |b, p| b.iter(|| RegexScanner::scan(black_box(p.as_str()))),
        );
    }
    group.finish();
}

// ── Scanner (Tier 1 + Tier 2) benchmarks — require model files ────────────────

fn bench_scanner_regex_only(c: &mut Criterion) {
    let scanner = Scanner::regex_only();
    let opts = ScanOptions::default();

    c.bench_function("scanner/regex_only_pii", |b| {
        b.iter(|| {
            scanner
                .scan_with_options(black_box(REGEX_HEAVY_PROMPT), &opts)
                .unwrap()
        })
    });
}

fn bench_scanner_full_pipeline(c: &mut Criterion) {
    // Skip if model files are not present
    if !std::path::Path::new("models/model_quantized.onnx").exists() {
        eprintln!("Skipping full-pipeline benchmark — model files not found");
        return;
    }

    let scanner = Scanner::new("models/model_quantized.onnx", "models/tokenizer.json")
        .expect("Failed to load model for benchmark");
    let opts = ScanOptions::default();

    let mut group = c.benchmark_group("scanner/full_pipeline");
    group.sample_size(20); // Fewer samples for slow NER inference

    group.bench_function("clean_prompt", |b| {
        b.iter(|| {
            scanner
                .scan_with_options(black_box(CLEAN_PROMPT), &opts)
                .unwrap()
        })
    });

    group.bench_function("ner_heavy_prompt", |b| {
        b.iter(|| {
            scanner
                .scan_with_options(black_box(NER_HEAVY_PROMPT), &opts)
                .unwrap()
        })
    });

    group.bench_function("pii_heavy_prompt", |b| {
        b.iter(|| {
            scanner
                .scan_with_options(black_box(REGEX_HEAVY_PROMPT), &opts)
                .unwrap()
        })
    });

    group.finish();
}

criterion_group!(
    regex_benches,
    bench_regex_clean,
    bench_regex_pii,
    bench_regex_prompt_lengths,
    bench_scanner_regex_only,
);

criterion_group!(model_benches, bench_scanner_full_pipeline);

criterion_main!(regex_benches, model_benches);
