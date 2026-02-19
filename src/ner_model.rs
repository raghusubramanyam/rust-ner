//! Tier 2: ONNX NER inference using a quantised DistilBERT model.
//!
//! # Model details
//! - Source: `onnx-community/distilbert-base-cased-finetuned-conll03-english-ONNX`
//! - File:   `models/model_quantized.onnx` (INT8, ~65.8 MB)
//! - Labels: CoNLL-2003 BIO scheme — O, B-PER, I-PER, B-ORG, I-ORG,
//!           B-LOC, I-LOC, B-MISC, I-MISC
//!
//! # Usage
//! ```no_run
//! use aegis_ner::ner_model::NERModel;
//!
//! let model = NERModel::load("models/model_quantized.onnx", "models/tokenizer.json").unwrap();
//! let entities = model.predict("Alice works at OpenAI in San Francisco.").unwrap();
//! ```

use std::sync::Mutex;

use anyhow::{Context, Result};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};

use crate::entity::{DetectedEntity, DetectionTier, EntityType};
use crate::tokenizer::NERTokenizer;

// ── Label map ────────────────────────────────────────────────────────────────

/// BIO label index → label string (CoNLL-2003 order).
const LABEL_MAP: &[&str] = &[
    "O",      // 0
    "B-PER",  // 1
    "I-PER",  // 2
    "B-ORG",  // 3
    "I-ORG",  // 4
    "B-LOC",  // 5
    "I-LOC",  // 6
    "B-MISC", // 7
    "I-MISC", // 8
];

/// Minimum softmax probability for a NER detection to be surfaced.
pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.75;

// ── Core structs ──────────────────────────────────────────────────────────────

/// Loaded NER model.  Intended to be created once at startup and reused.
///
/// The ONNX `Session::run` requires `&mut Session`, so we wrap it in a
/// `Mutex` to allow `predict` to take `&self` — making `NERModel` safe to
/// share across threads via `Arc<NERModel>`.
pub struct NERModel {
    session: Mutex<Session>,
    tokenizer: NERTokenizer,
    /// Detections with a softmax score below this are discarded.
    pub confidence_threshold: f32,
}

impl NERModel {
    /// Load the ONNX session and tokenizer from disk.
    ///
    /// This is an expensive, blocking operation — call it once at startup.
    ///
    /// # Parameters
    /// - `model_path`     — path to `model_quantized.onnx`
    /// - `tokenizer_path` — path to `tokenizer.json`
    pub fn load(model_path: &str, tokenizer_path: &str) -> Result<Self> {
        let session = Session::builder()
            .context("failed to create ONNX session builder")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .context("failed to set optimisation level")?
            .with_intra_threads(4)
            .context("failed to set intra-op threads")?
            .commit_from_file(model_path)
            .with_context(|| format!("failed to load ONNX model from '{}'", model_path))?;

        let tokenizer = NERTokenizer::from_file(tokenizer_path)?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
        })
    }

    /// Run NER inference on `text`.
    ///
    /// Returns entities ordered by byte-start position.  Only detections at or
    /// above `self.confidence_threshold` are included.
    pub fn predict(&self, text: &str) -> Result<Vec<DetectedEntity>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        // ── 1. Tokenize ───────────────────────────────────────────────────────
        let encoding = self.tokenizer.encode(text)?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();

        let seq_len = ids.len();
        if seq_len == 0 {
            return Ok(Vec::new());
        }

        // ── 2. Build input tensors (shape [1, seq_len]) ───────────────────────
        let input_ids_tensor =
            Tensor::<i64>::from_array((vec![1i64, seq_len as i64], ids))
                .context("failed to build input_ids tensor")?;
        let attention_mask_tensor =
            Tensor::<i64>::from_array((vec![1i64, seq_len as i64], mask))
                .context("failed to build attention_mask tensor")?;

        // ── 3. Run inference ──────────────────────────────────────────────────
        // Bind the MutexGuard to a named variable so it lives long enough for
        // the returned `SessionOutputs` (which borrows the session).
        let mut session_guard = self
            .session
            .lock()
            .expect("NERModel session mutex poisoned");
        let outputs = session_guard
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
            ])
            .context("ONNX session run failed")?;

        // ── 4. Extract logits: shape [1, seq_len, num_labels] ─────────────────
        let (logit_shape, logit_data) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .context("failed to extract logits tensor")?;

        // logit_shape: [1, seq_len, num_labels]
        let num_labels = logit_shape
            .get(2)
            .copied()
            .unwrap_or(LABEL_MAP.len() as i64) as usize;

        // ── 5. Decode BIO tags → entities ────────────────────────────────────
        let tokens = encoding.get_tokens();
        let offsets = encoding.get_offsets();

        let mut entities: Vec<DetectedEntity> = Vec::new();
        let mut builder: Option<EntityBuilder> = None;

        for token_idx in 0..seq_len {
            // Skip [CLS] (index 0) and [SEP] (last index).
            if token_idx == 0 || token_idx == seq_len - 1 {
                continue;
            }

            let token = &tokens[token_idx];

            // Sub-word continuation tokens start with "##".  They extend the
            // current entity span but don't start a new one.
            if token.starts_with("##") {
                if let Some(ref mut b) = builder {
                    b.extend_end(offsets[token_idx].1);
                }
                continue;
            }

            // Per-token logits slice from the flat [1, seq_len, num_labels] buffer.
            let slice_start = token_idx * num_labels;
            let slice_end = slice_start + num_labels;
            if slice_end > logit_data.len() {
                break;
            }
            let token_logits = &logit_data[slice_start..slice_end];

            // Argmax → predicted label index
            let (label_idx, _) = token_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();

            let confidence = softmax_confidence(token_logits, label_idx);
            let label = LABEL_MAP.get(label_idx).copied().unwrap_or("O");

            if label.starts_with("B-") {
                // Flush any open entity before starting a new one.
                if let Some(b) = builder.take() {
                    if b.score >= self.confidence_threshold {
                        entities.push(b.build(text));
                    }
                }
                builder = Some(EntityBuilder::new(
                    label_to_entity_type(label),
                    offsets[token_idx].0,
                    offsets[token_idx].1,
                    confidence,
                ));
            } else if label.starts_with("I-") {
                // Continuation: update end and blend score (take minimum).
                if let Some(ref mut b) = builder {
                    b.extend_end(offsets[token_idx].1);
                    b.score = b.score.min(confidence);
                }
                // Stray I- tags without a preceding B- are ignored.
            } else {
                // "O" — flush any open entity.
                if let Some(b) = builder.take() {
                    if b.score >= self.confidence_threshold {
                        entities.push(b.build(text));
                    }
                }
            }
        }

        // Flush a trailing entity.
        if let Some(b) = builder.take() {
            if b.score >= self.confidence_threshold {
                entities.push(b.build(text));
            }
        }

        Ok(entities)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Convert the raw logit slice to a softmax probability for `target_idx`.
fn softmax_confidence(logits: &[f32], target_idx: usize) -> f32 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return 0.0;
    }
    exps[target_idx] / sum
}

/// Map a BIO label string to our `EntityType`.
fn label_to_entity_type(label: &str) -> EntityType {
    match label {
        "B-PER" | "I-PER" => EntityType::Person,
        "B-ORG" | "I-ORG" => EntityType::Organization,
        "B-LOC" | "I-LOC" => EntityType::Location,
        "B-MISC" | "I-MISC" => EntityType::Miscellaneous,
        _ => EntityType::Unknown,
    }
}

// ── Entity builder ────────────────────────────────────────────────────────────

struct EntityBuilder {
    entity_type: EntityType,
    start: usize,
    end: usize,
    score: f32,
}

impl EntityBuilder {
    fn new(entity_type: EntityType, start: usize, end: usize, score: f32) -> Self {
        Self {
            entity_type,
            start,
            end,
            score,
        }
    }

    /// Extend the span end to cover a subsequent sub-word or I-* token.
    fn extend_end(&mut self, new_end: usize) {
        if new_end > self.end {
            self.end = new_end;
        }
    }

    /// Materialise into a [`DetectedEntity`], slicing `text` for the matched span.
    fn build(self, text: &str) -> DetectedEntity {
        // Guard against tokenizer offset edge cases (e.g. trailing whitespace).
        let end = self.end.min(text.len());
        let matched_text = text.get(self.start..end).unwrap_or("").to_string();
        let risk_level = self.entity_type.default_risk_level();

        DetectedEntity {
            entity_type: self.entity_type,
            text: matched_text,
            score: self.score,
            start: self.start,
            end,
            tier: DetectionTier::NERModel,
            risk_level,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![1.0f32, 2.0, 3.0, 0.5, -1.0, 0.0, 0.0, 0.0, 0.0];
        let total: f32 = (0..logits.len())
            .map(|i| softmax_confidence(&logits, i))
            .sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn label_mapping_is_correct() {
        assert_eq!(label_to_entity_type("B-PER"), EntityType::Person);
        assert_eq!(label_to_entity_type("I-ORG"), EntityType::Organization);
        assert_eq!(label_to_entity_type("B-LOC"), EntityType::Location);
        assert_eq!(label_to_entity_type("B-MISC"), EntityType::Miscellaneous);
        assert_eq!(label_to_entity_type("O"), EntityType::Unknown);
    }

    #[test]
    #[ignore = "requires model files in models/"]
    fn predict_person_entity() {
        let model =
            NERModel::load("models/model_quantized.onnx", "models/tokenizer.json").unwrap();
        let entities = model
            .predict("Alice Johnson works at Acme Corp in New York.")
            .unwrap();

        let persons: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == EntityType::Person)
            .collect();
        assert!(!persons.is_empty(), "expected at least one PERSON entity");
    }

    #[test]
    #[ignore = "requires model files in models/"]
    fn empty_text_returns_no_entities() {
        let model =
            NERModel::load("models/model_quantized.onnx", "models/tokenizer.json").unwrap();
        let entities = model.predict("").unwrap();
        assert!(entities.is_empty());
    }
}
