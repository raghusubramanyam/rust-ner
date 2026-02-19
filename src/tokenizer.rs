//! Tokenizer wrapper around HuggingFace `tokenizers` (pure Rust).
//!
//! Loads a WordPiece tokenizer from a `tokenizer.json` file (the standard
//! HuggingFace serialisation format) and exposes the encoding primitives
//! needed by the NER model.

use tokenizers::{Encoding, Tokenizer as HFTokenizer};

use anyhow::{Context, Result};

/// Maximum token sequence length accepted by the DistilBERT model.
pub const MAX_SEQ_LEN: usize = 512;

/// Thin wrapper around a loaded HuggingFace tokenizer.
///
/// Cheap to clone (inner tokenizer is `Arc`-wrapped by the `tokenizers` crate).
pub struct NERTokenizer {
    inner: HFTokenizer,
}

impl NERTokenizer {
    /// Load a tokenizer from a `tokenizer.json` file on disk.
    pub fn from_file(path: &str) -> Result<Self> {
        let inner = HFTokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from '{}': {}", path, e))
            .context("tokenizer load")?;
        Ok(Self { inner })
    }

    /// Tokenize `text` and return the full `Encoding`.
    ///
    /// Adds special tokens ([CLS] / [SEP]) and truncates to [`MAX_SEQ_LEN`].
    /// The encoding carries:
    /// - `get_ids()`           — token IDs (i32) for the model input tensor
    /// - `get_attention_mask()` — 1 for real tokens, 0 for padding
    /// - `get_tokens()`        — string form of each token (for BIO decoding)
    /// - `get_offsets()`       — `(start, end)` byte offsets in the original text
    pub fn encode(&self, text: &str) -> Result<Encoding> {
        let mut encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        // Truncate to MAX_SEQ_LEN if necessary (keeps [CLS] + content + [SEP]).
        if encoding.get_ids().len() > MAX_SEQ_LEN {
            encoding.truncate(MAX_SEQ_LEN, 0, tokenizers::TruncationDirection::Right);
        }

        Ok(encoding)
    }

    /// Number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

#[cfg(test)]
mod tests {
    // These tests require the model files to be present; they are skipped in CI
    // unless AEGIS_MODEL_DIR is set.
    use super::*;

    #[test]
    #[ignore = "requires tokenizer.json in models/"]
    fn loads_and_encodes() {
        let tok = NERTokenizer::from_file("models/tokenizer.json").unwrap();
        let enc = tok.encode("John Smith works at Acme Corp.").unwrap();
        assert!(!enc.get_ids().is_empty());
        // [CLS] and [SEP] are always present
        assert!(enc.get_ids().len() >= 2);
    }

    #[test]
    #[ignore = "requires tokenizer.json in models/"]
    fn truncates_long_input() {
        let tok = NERTokenizer::from_file("models/tokenizer.json").unwrap();
        let long = "word ".repeat(600);
        let enc = tok.encode(&long).unwrap();
        assert!(enc.get_ids().len() <= MAX_SEQ_LEN);
    }
}
