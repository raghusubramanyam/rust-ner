# AEGIS NER — CLAUDE.md

Developer guide for working on this codebase with Claude Code.

## Project overview

`aegis-ner` is a pure-Rust two-tier PII and Named Entity Recognition service.

| Tier | Technology | Latency | Detects |
|------|-----------|---------|---------|
| 1 | Regex (`once_cell` + `regex`) | ~1 μs | SSN, credit cards, emails, phone numbers, IP addresses, AWS/API keys, passports, DOBs, IBANs |
| 2 | ONNX DistilBERT (`ort` crate) | 5–15 ms | PERSON, ORGANIZATION, LOCATION, MISCELLANEOUS |

## Repository layout

```
aegis-ner/
├── Cargo.toml                  # Workspace manifest and all dependencies
├── CLAUDE.md                   # This file
├── models/                     # Model artifacts (not committed — see below)
│   ├── model_quantized.onnx    # INT8 quantised DistilBERT (~65.8 MB)
│   ├── tokenizer.json          # HuggingFace WordPiece tokenizer config
│   └── config.json             # Label map and model metadata
├── scripts/
│   └── download_model.sh       # Downloads + verifies model files from HuggingFace
├── src/
│   ├── lib.rs                  # Public crate API — Scanner, ScanOptions, re-exports
│   ├── main.rs                 # CLI binary (reads stdin or --text flag)
│   ├── entity.rs               # EntityType, RiskLevel, DetectedEntity, ScanResult
│   ├── regex_scanner.rs        # Tier 1: compiled RegexSet + per-type validators
│   ├── tokenizer.rs            # NERTokenizer wrapper (load, encode, truncate)
│   └── ner_model.rs            # Tier 2: ONNX session + BIO tag decoder
├── tests/
│   ├── regex_tests.rs          # Unit tests for Tier 1 — no model files needed
│   ├── ner_tests.rs            # Integration tests for Tier 2 — require model files
│   └── integration_tests.rs    # End-to-end pipeline tests
└── benches/
    └── inference_bench.rs      # Criterion latency benchmarks
```

## Build commands

```sh
# Check the project compiles (fast, no linking)
cargo check

# Build in debug mode
cargo build

# Build optimised release binary
cargo build --release

# Run the CLI (regex-only, no model needed)
cargo run -- --text "Alice's SSN is 123-45-6789" --regex-only

# Run the CLI with the full NER pipeline
cargo run -- --text "John works at Acme Corp in Paris" \
  --model models/model_quantized.onnx
```

## Test commands

```sh
# Run all tests that do NOT require model files (fast)
cargo test

# Run a specific test file
cargo test --test regex_tests
cargo test --test integration_tests

# Run model-dependent tests (requires ./scripts/download_model.sh first)
cargo test -- --ignored

# Run model tests with output visible
cargo test -- --ignored --nocapture

# Run unit tests inside src/ modules
cargo test --lib
```

## Benchmarks

```sh
# Run all benchmarks (regex tier always runs; NER tier requires model files)
cargo bench

# Run only regex benchmarks
cargo bench --bench inference_bench regex

# Open the HTML report
open target/criterion/report/index.html
```

## Getting model files

The ONNX model and tokenizer are **not committed to the repository** due to file size.

```sh
./scripts/download_model.sh
```

This downloads three files into `models/`:
- `model_quantized.onnx` — INT8 quantised DistilBERT (~65.8 MB)
- `tokenizer.json`       — HuggingFace WordPiece tokenizer
- `config.json`          — Model label configuration

**Model source:** `onnx-community/distilbert-base-cased-finetuned-conll03-english-ONNX` on HuggingFace Hub.

## Architecture decisions

### Why two tiers?

- **Tier 1 (regex)** handles structured PII deterministically with near-zero latency.
  Regex patterns are compiled once at process start via `once_cell::sync::Lazy`.
- **Tier 2 (ONNX NER)** handles unstructured named entities that regex cannot capture
  (person names, organisations, locations) using a fine-tuned DistilBERT model.
  The model runs entirely in-process with no Python or libtorch dependency.

### Why `ort` (ONNX Runtime)?

`ort` provides pure-Rust bindings to the ONNX Runtime C library.  The
`load-dynamic` feature means the `.so`/`.dll` is loaded at runtime, so you
only need to have ONNX Runtime installed on the target system (or bundle it).

### Confidence threshold

The default NER confidence threshold is `0.75` (configurable via `ScanOptions`
or `NERModel.confidence_threshold`).  Tokens below this threshold are silently
discarded.  Lower values increase recall; higher values increase precision.

### Luhn validation

Credit card candidates are validated with the Luhn algorithm before being
surfaced as detections.  This significantly reduces false positives from
sequences of 13–19 digits that are not real card numbers.

## Key source locations

| Concept | File | Notes |
|---------|------|-------|
| All entity/PII types | `src/entity.rs` | `EntityType` enum |
| Regex patterns | `src/regex_scanner.rs:PATTERNS` | `const &[&str]` array |
| Luhn check | `src/regex_scanner.rs:validate_credit_card` | |
| Redaction | `src/regex_scanner.rs:redact` | standalone function |
| ONNX session init | `src/ner_model.rs:NERModel::load` | |
| BIO tag decoding | `src/ner_model.rs:NERModel::predict` | |
| Public API | `src/lib.rs:Scanner` | `scan()` / `scan_with_options()` |
| CLI entry point | `src/main.rs` | minimal hand-rolled arg parser |

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `RUST_LOG` | `info` | Tracing log level (`debug`, `info`, `warn`, `error`) |
| `ORT_DYLIB_PATH` | auto | Override ONNX Runtime shared library path |
| `SKIP_CHECKSUM` | `0` | Set to `1` in `download_model.sh` to skip SHA-256 verification |

## Adding a new PII pattern

1. Add a new variant to `EntityType` in `src/entity.rs` and update
   `default_risk_level()` and the `Display` impl.
2. Add the regex string to the `PATTERNS` array in `src/regex_scanner.rs`
   and a corresponding `IDX_*` constant.
3. Add the `(EntityType, RiskLevel)` mapping entry to `ENTITY_MAP` at the
   same index.
4. If the pattern needs custom validation, add a branch to `RegexScanner::validate`.
5. Add tests in `tests/regex_tests.rs`.

## Adding a new NER label

The NER model is fine-tuned on CoNLL-2003 and only supports PER/ORG/LOC/MISC.
To add new entity classes you would need to fine-tune or swap the model.  Update
`LABEL_MAP` in `src/ner_model.rs` and `label_to_entity_type` accordingly.
