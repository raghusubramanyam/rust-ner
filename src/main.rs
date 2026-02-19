//! AEGIS NER — standalone CLI binary.
//!
//! Reads text from stdin (one prompt per line) or from a `--text` argument,
//! runs the two-tier PII/NER detection pipeline, and prints a JSON report to
//! stdout.
//!
//! # Usage
//!
//! ```sh
//! # Single prompt via flag
//! aegis-ner --text "Alice emailed alice@example.com" --model models/model_quantized.onnx --tokenizer models/tokenizer.json
//!
//! # Regex-only mode (no model files required)
//! aegis-ner --text "SSN: 123-45-6789" --regex-only
//!
//! # Pipe from stdin
//! echo "Contact bob@company.io" | aegis-ner --regex-only
//!
//! # Redact output
//! aegis-ner --text "Call me at 555-867-5309" --regex-only --redact
//! ```

use std::io::{self, BufRead};

use aegis_ner::{Scanner, ScanOptions};
use anyhow::{Context, Result};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

fn main() -> Result<()> {
    // ── Logging ───────────────────────────────────────────────────────────────
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    // ── CLI argument parsing ──────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let cli = Cli::parse_args(&args)?;

    // ── Build scanner ─────────────────────────────────────────────────────────
    let scanner = if cli.regex_only {
        info!("Running in regex-only mode (Tier 1 only)");
        Scanner::regex_only()
    } else {
        let model_path = cli
            .model_path
            .as_deref()
            .unwrap_or("models/model_quantized.onnx");
        let tokenizer_path = cli
            .tokenizer_path
            .as_deref()
            .unwrap_or("models/tokenizer.json");

        info!("Loading NER model from '{}'", model_path);
        Scanner::new(model_path, tokenizer_path)
            .with_context(|| format!("Failed to load model from '{}'", model_path))?
    };

    let scan_opts = ScanOptions {
        redact: cli.redact,
        confidence_threshold: cli.confidence,
        ..Default::default()
    };

    // ── Collect input lines ───────────────────────────────────────────────────
    let lines: Vec<String> = if let Some(text) = cli.text {
        vec![text]
    } else {
        let stdin = io::stdin();
        stdin.lock().lines().filter_map(|l| l.ok()).collect()
    };

    // ── Scan each line ────────────────────────────────────────────────────────
    let mut exit_code = 0i32;

    for (i, line) in lines.iter().enumerate() {
        if line.trim().is_empty() {
            continue;
        }

        match scanner.scan_with_options(line, &scan_opts) {
            Ok(result) => {
                let json = serde_json::to_string_pretty(&result)
                    .context("failed to serialise scan result")?;
                println!("{}", json);

                if result.has_entities() {
                    info!(
                        "Line {}: {} entities detected in {}μs",
                        i + 1,
                        result.entities.len(),
                        result.scan_duration_us,
                    );
                }
            }
            Err(e) => {
                error!("Line {}: scan failed — {}", i + 1, e);
                exit_code = 1;
            }
        }
    }

    std::process::exit(exit_code);
}

// ── Minimal arg parser ────────────────────────────────────────────────────────

struct Cli {
    text: Option<String>,
    model_path: Option<String>,
    tokenizer_path: Option<String>,
    regex_only: bool,
    redact: bool,
    confidence: f32,
}

impl Cli {
    fn parse_args(args: &[String]) -> Result<Self> {
        let mut cli = Cli {
            text: None,
            model_path: None,
            tokenizer_path: None,
            regex_only: false,
            redact: false,
            confidence: 0.75,
        };

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--text" | "-t" => {
                    i += 1;
                    cli.text = args.get(i).cloned();
                }
                "--model" | "-m" => {
                    i += 1;
                    cli.model_path = args.get(i).cloned();
                }
                "--tokenizer" => {
                    i += 1;
                    cli.tokenizer_path = args.get(i).cloned();
                }
                "--regex-only" | "-r" => {
                    cli.regex_only = true;
                }
                "--redact" => {
                    cli.redact = true;
                }
                "--confidence" | "-c" => {
                    i += 1;
                    if let Some(v) = args.get(i) {
                        cli.confidence = v
                            .parse::<f32>()
                            .context("--confidence must be a float between 0 and 1")?;
                    }
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                unknown => {
                    anyhow::bail!("Unknown argument: '{}'. Run with --help.", unknown);
                }
            }
            i += 1;
        }

        Ok(cli)
    }
}

fn print_help() {
    eprintln!(
        r#"aegis-ner — Two-tier PII/NER detection (regex + ONNX DistilBERT)

USAGE:
    aegis-ner [OPTIONS]

OPTIONS:
    -t, --text <TEXT>           Text to scan (otherwise reads stdin line-by-line)
    -m, --model <PATH>          Path to model_quantized.onnx [default: models/model_quantized.onnx]
        --tokenizer <PATH>      Path to tokenizer.json        [default: models/tokenizer.json]
    -r, --regex-only            Tier 1 only; skip ONNX inference (no model needed)
        --redact                Replace detected entities with [TYPE_REDACTED] placeholders
    -c, --confidence <FLOAT>    Minimum NER confidence threshold [default: 0.75]
    -h, --help                  Print this help message

ENVIRONMENT:
    RUST_LOG=debug              Enable verbose tracing output

EXAMPLES:
    aegis-ner --text "Alice's SSN is 123-45-6789" --regex-only
    echo "bob@corp.io" | aegis-ner --regex-only --redact
    aegis-ner --text "John works at Acme" --model models/model_quantized.onnx
"#
    );
}
