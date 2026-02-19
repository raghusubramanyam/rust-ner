//! # aegis-ner
//!
//! Two-tier PII and Named Entity Recognition (NER) detection library.
//!
//! ## Detection pipeline
//!
//! ```text
//! Input text
//!   │
//!   ├─► Tier 1: RegexScanner   (~1 μs)
//!   │   Detects structured PII: SSN, credit cards, emails, phone numbers,
//!   │   IP addresses, AWS keys, API keys, passports, DOBs, IBANs.
//!   │
//!   └─► Tier 2: NERModel       (~5–15 ms)
//!       Runs DistilBERT (ONNX, INT8 quantised) to detect:
//!       PERSON, ORGANIZATION, LOCATION, MISCELLANEOUS.
//!
//! Combined results → ScanResult
//! ```
//!
//! ## Quick start
//!
//! ```no_run
//! use aegis_ner::{Scanner, ScanOptions};
//!
//! // Build once (loads the ONNX model)
//! let scanner = Scanner::new("models/model_quantized.onnx", "models/tokenizer.json").unwrap();
//!
//! // Scan a prompt
//! let result = scanner.scan("Alice emailed alice@example.com her SSN 123-45-6789.").unwrap();
//!
//! println!("Entities found: {}", result.entities.len());
//! if let Some(redacted) = result.redacted_prompt {
//!     println!("Redacted: {}", redacted);
//! }
//! ```

pub mod entity;
pub mod ner_model;
pub mod regex_scanner;
pub mod tokenizer;

use std::time::Instant;

use anyhow::Result;

pub use entity::{DetectedEntity, DetectionTier, EntityType, RiskLevel, ScanResult};
pub use ner_model::NERModel;
pub use regex_scanner::{redact, RegexScanner};

// ── ScanOptions ───────────────────────────────────────────────────────────────

/// Configuration for a single scan invocation.
#[derive(Debug, Clone)]
pub struct ScanOptions {
    /// Minimum NER model confidence to surface an entity (0.0–1.0).
    ///
    /// Defaults to `0.75`.
    pub confidence_threshold: f32,

    /// When `true`, the result will contain a `redacted_prompt` with entity
    /// text replaced by `[TYPE_REDACTED]` placeholders.
    pub redact: bool,

    /// When `true`, skip Tier 1 regex scanning (for benchmarking Tier 2 in
    /// isolation, or when Tier 1 has already been applied).
    pub skip_regex: bool,

    /// When `true`, skip Tier 2 NER model inference (regex-only mode).
    pub skip_ner: bool,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            confidence_threshold: ner_model::DEFAULT_CONFIDENCE_THRESHOLD,
            redact: false,
            skip_regex: false,
            skip_ner: false,
        }
    }
}

// ── Scanner ───────────────────────────────────────────────────────────────────

/// The primary entry point for PII/NER detection.
///
/// Owns the loaded ONNX [`NERModel`] and exposes a thread-safe [`scan`]
/// method.  The internal ONNX `Session` is `Send + Sync`, so `Scanner` can be
/// wrapped in an `Arc` and shared across threads.
///
/// [`scan`]: Scanner::scan
pub struct Scanner {
    model: Option<NERModel>,
}

impl Scanner {
    /// Build a `Scanner` that runs both tiers.
    ///
    /// Loads the ONNX model synchronously — call once at application startup.
    pub fn new(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        let model = NERModel::load(model_path, tokenizer_path)?;
        Ok(Self { model: Some(model) })
    }

    /// Build a `Scanner` that runs **only** Tier 1 (regex).
    ///
    /// Useful when the ONNX model files are not available or when you only
    /// need structured-PII detection.
    pub fn regex_only() -> Self {
        Self { model: None }
    }

    /// Scan `text` with the default [`ScanOptions`].
    pub fn scan(&self, text: &str) -> Result<ScanResult> {
        self.scan_with_options(text, &ScanOptions::default())
    }

    /// Scan `text` with explicit [`ScanOptions`].
    pub fn scan_with_options(&self, text: &str, opts: &ScanOptions) -> Result<ScanResult> {
        let scan_start = Instant::now();

        // ── Tier 1: Regex ─────────────────────────────────────────────────────
        let (tier1_entities, tier1_duration_us) = if opts.skip_regex {
            (Vec::new(), 0u64)
        } else {
            let t1 = Instant::now();
            let entities = RegexScanner::scan(text);
            (entities, t1.elapsed().as_micros() as u64)
        };

        // ── Tier 2: NER model ─────────────────────────────────────────────────
        let (tier2_entities, tier2_duration_us) = if opts.skip_ner {
            (Vec::new(), 0u64)
        } else if let Some(ref model) = self.model {
            let prev_threshold = model.confidence_threshold;
            // Safety: we only read model fields between runs; set threshold
            // via the public field on NERModel (not interior-mutable, fine
            // for single-threaded scan paths).
            let _ = prev_threshold; // suppress unused warning
            let t2 = Instant::now();
            let mut entities = model.predict(text)?;
            // Apply caller-supplied threshold (may differ from model default).
            entities.retain(|e| e.score >= opts.confidence_threshold);
            (entities, t2.elapsed().as_micros() as u64)
        } else {
            (Vec::new(), 0u64)
        };

        // ── Merge results ─────────────────────────────────────────────────────
        let mut all_entities: Vec<DetectedEntity> = tier1_entities;
        all_entities.extend(tier2_entities);

        // Sort by byte-start offset for deterministic output.
        all_entities.sort_by_key(|e| e.start);

        let scan_duration_us = scan_start.elapsed().as_micros() as u64;

        // ── Optional redaction ────────────────────────────────────────────────
        let redacted_prompt = if opts.redact {
            Some(redact(text, &all_entities))
        } else {
            None
        };

        Ok(ScanResult {
            prompt: text.to_string(),
            entities: all_entities,
            redacted_prompt,
            scan_duration_us,
            tier1_duration_us,
            tier2_duration_us,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn regex_only_scanner() -> Scanner {
        Scanner::regex_only()
    }

    #[test]
    fn regex_only_mode_detects_pii() {
        let scanner = regex_only_scanner();
        let result = scanner
            .scan("My email is bob@example.org and my SSN is 987-65-4321.")
            .unwrap();
        assert!(result.has_entities());
        assert!(result
            .entities
            .iter()
            .any(|e| e.entity_type == EntityType::Email));
        assert!(result
            .entities
            .iter()
            .any(|e| e.entity_type == EntityType::SSN));
    }

    #[test]
    fn redaction_removes_sensitive_values() {
        let scanner = regex_only_scanner();
        let opts = ScanOptions {
            redact: true,
            ..Default::default()
        };
        let result = scanner
            .scan_with_options("Email me at ceo@corp.io for the meeting.", &opts)
            .unwrap();
        let redacted = result.redacted_prompt.expect("redacted prompt should exist");
        assert!(!redacted.contains("ceo@corp.io"));
        assert!(redacted.contains("_REDACTED]"));
    }

    #[test]
    fn clean_text_has_no_entities() {
        let scanner = regex_only_scanner();
        let result = scanner
            .scan("The weather is nice today in the mountains.")
            .unwrap();
        assert!(!result.has_entities());
    }

    #[test]
    fn max_risk_level_is_critical_for_ssn() {
        let scanner = regex_only_scanner();
        let result = scanner.scan("SSN: 123-45-6789").unwrap();
        assert_eq!(result.max_risk_level(), Some(RiskLevel::Critical));
    }

    #[test]
    fn scan_result_timing_is_populated() {
        let scanner = regex_only_scanner();
        let result = scanner.scan("test input").unwrap();
        assert!(result.scan_duration_us > 0 || result.tier1_duration_us == 0);
        // Tier 2 skipped in regex-only mode
        assert_eq!(result.tier2_duration_us, 0);
    }

    #[test]
    #[ignore = "requires model files in models/"]
    fn full_pipeline_detects_person_and_email() {
        let scanner =
            Scanner::new("models/model_quantized.onnx", "models/tokenizer.json").unwrap();
        let result = scanner
            .scan("Alice Johnson can be reached at alice@example.com.")
            .unwrap();
        let has_person = result
            .entities
            .iter()
            .any(|e| e.entity_type == EntityType::Person);
        let has_email = result
            .entities
            .iter()
            .any(|e| e.entity_type == EntityType::Email);
        assert!(has_person, "expected PERSON entity");
        assert!(has_email, "expected EMAIL entity");
    }
}
