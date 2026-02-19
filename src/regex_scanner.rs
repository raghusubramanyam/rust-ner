//! Tier 1: Regex-based structured PII scanner.
//!
//! Uses a `once_cell`-initialised `RegexSet` for a single-pass match over the
//! input, then individual `Regex` objects to extract byte offsets and validate
//! matches (e.g. Luhn check for credit cards).
//!
//! Target latency: < 10 μs per prompt.

use std::collections::BTreeMap;

use once_cell::sync::Lazy;
use regex::{Regex, RegexSet};

use crate::entity::{DetectedEntity, DetectionTier, EntityType, RiskLevel};

// ── Pattern index constants ──────────────────────────────────────────────────
// These constants name each slot in the PATTERNS array.  Only the ones used
// in per-pattern validation branches need to be referenced; the rest document
// the array layout and may be used by callers.

#[allow(dead_code)]
const IDX_SSN: usize = 0;
const IDX_CREDIT_CARD: usize = 1;
#[allow(dead_code)]
const IDX_EMAIL: usize = 2;
const IDX_PHONE_US: usize = 3;
const IDX_PHONE_INTL: usize = 4;
#[allow(dead_code)]
const IDX_IP: usize = 5;
#[allow(dead_code)]
const IDX_AWS_KEY: usize = 6;
#[allow(dead_code)]
const IDX_AWS_SECRET: usize = 7;
#[allow(dead_code)]
const IDX_API_KEY: usize = 8;
const IDX_PASSPORT: usize = 9;
#[allow(dead_code)]
const IDX_DOB: usize = 10;
#[allow(dead_code)]
const IDX_IBAN: usize = 11;

/// Ordered list of raw regex patterns.  Index must stay in sync with `IDX_*`.
const PATTERNS: &[&str] = &[
    // 0: SSN  — XXX-XX-XXXX
    r"\b\d{3}-\d{2}-\d{4}\b",
    // 1: Credit card — 13-19 contiguous or dash/space-separated digits
    r"\b(?:\d[ \-]?){13,19}\b",
    // 2: Email
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    // 3: US phone — various formats, no leading +1 required.
    // No leading \b because phones can start with '(' which is a non-word char.
    // Trailing \b anchors after the last digit.
    r"(?:\+?1[\s.\-]?)?(?:\(\d{3}\)|\d{3})[\s.\-]?\d{3}[\s.\-]?\d{4}\b",
    // 4: International phone — must have country code prefix
    r"\+[1-9]\d{0,2}[\s.\-]?\(?\d{1,4}\)?[\s.\-]?\d{1,4}[\s.\-]?\d{1,9}",
    // 5: IPv4
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
    // 6: AWS access key ID
    r"\b(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}\b",
    // 7: AWS secret key
    r"(?i)(?:aws_secret(?:_access)?_key|secret_key)\s*[=:]\s*[A-Za-z0-9/+=]{40}",
    // 8: Generic API key — OpenAI sk-*, Anthropic sk-ant-*, generic key-*
    r"\b(?:sk-(?:ant-)?[a-zA-Z0-9\-_]{20,}|key-[a-zA-Z0-9]{32,})\b",
    // 9: US passport — 9 contiguous digits (high false-positive; validated below)
    r"\b[0-9]{9}\b",
    // 10: Date of birth — MM/DD/YYYY or MM-DD-YYYY
    r"\b(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b",
    // 11: IBAN
    r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]{0,16})?\b",
];

/// Mapping from pattern index → `(EntityType, RiskLevel)`.
static ENTITY_MAP: &[(EntityType, RiskLevel)] = &[
    (EntityType::SSN, RiskLevel::Critical),
    (EntityType::CreditCard, RiskLevel::Critical),
    (EntityType::Email, RiskLevel::High),
    (EntityType::PhoneUS, RiskLevel::High),
    (EntityType::PhoneIntl, RiskLevel::High),
    (EntityType::IPAddress, RiskLevel::Medium),
    (EntityType::AWSAccessKey, RiskLevel::Critical),
    (EntityType::AWSSecretKey, RiskLevel::Critical),
    (EntityType::APIKey, RiskLevel::Critical),
    (EntityType::PassportUS, RiskLevel::High),
    (EntityType::DateOfBirth, RiskLevel::Medium),
    (EntityType::IBAN, RiskLevel::Critical),
];

// ── Compiled statics ─────────────────────────────────────────────────────────

/// `RegexSet` for a single-pass detection of which pattern categories match.
static PII_SET: Lazy<RegexSet> =
    Lazy::new(|| RegexSet::new(PATTERNS).expect("invalid PII regex patterns"));

/// Individual compiled regexes for extracting byte offsets.
static PII_REGEXES: Lazy<Vec<Regex>> = Lazy::new(|| {
    PATTERNS
        .iter()
        .map(|p| Regex::new(p).expect("invalid PII regex"))
        .collect()
});

// ── Public interface ──────────────────────────────────────────────────────────

/// Tier 1 regex-based PII scanner.
///
/// Instantiation is zero-cost — all regex compilation happens at the first
/// call to [`RegexScanner::scan`] via `once_cell::sync::Lazy`.
pub struct RegexScanner;

impl RegexScanner {
    /// Scan `text` for structured PII using compiled regex patterns.
    ///
    /// Returns a `Vec<DetectedEntity>` ordered by byte offset (start position).
    /// Overlapping matches are deduplicated; longer spans win.
    pub fn scan(text: &str) -> Vec<DetectedEntity> {
        // Fast path: which pattern categories matched at all?
        let matched_indices: Vec<usize> = PII_SET.matches(text).into_iter().collect();
        if matched_indices.is_empty() {
            return Vec::new();
        }

        // Collect all raw span matches, keyed by start offset so we can merge.
        // BTreeMap keeps them sorted by start for dedup.
        let mut span_map: BTreeMap<usize, DetectedEntity> = BTreeMap::new();

        for idx in matched_indices {
            let regex = &PII_REGEXES[idx];
            let (entity_type, risk_level) = &ENTITY_MAP[idx];

            for m in regex.find_iter(text) {
                let matched_text = m.as_str().to_string();

                // Entity-specific validation / filtering
                if !Self::validate(idx, &matched_text) {
                    continue;
                }

                let entity = DetectedEntity {
                    entity_type: entity_type.clone(),
                    text: matched_text,
                    score: 1.0,
                    start: m.start(),
                    end: m.end(),
                    tier: DetectionTier::Regex,
                    risk_level: *risk_level,
                };

                // Dedup: if two patterns match the same start, keep the one
                // that covers more characters (larger span).
                let span = entity.end - entity.start;
                if let Some(existing) = span_map.get(&entity.start) {
                    if (existing.end - existing.start) >= span {
                        continue;
                    }
                }
                span_map.insert(entity.start, entity);
            }
        }

        // Remove overlapping entries: if an entity is fully contained within
        // a preceding one, drop it.
        let mut result: Vec<DetectedEntity> = Vec::with_capacity(span_map.len());
        let mut last_end: usize = 0;
        for (_, entity) in span_map {
            if entity.start >= last_end {
                last_end = entity.end;
                result.push(entity);
            }
        }

        result
    }

    // ── Per-pattern validators ────────────────────────────────────────────────

    fn validate(idx: usize, text: &str) -> bool {
        match idx {
            IDX_CREDIT_CARD => Self::validate_credit_card(text),
            IDX_PASSPORT => Self::validate_passport_context(text),
            IDX_PHONE_US | IDX_PHONE_INTL => Self::validate_phone(text),
            _ => true,
        }
    }

    /// Luhn algorithm check for credit card numbers.
    ///
    /// Strips spaces and dashes before checking.
    fn validate_credit_card(s: &str) -> bool {
        let digits: Vec<u32> = s
            .chars()
            .filter(|c| c.is_ascii_digit())
            .filter_map(|c| c.to_digit(10))
            .collect();

        let n = digits.len();
        if !(13..=19).contains(&n) {
            return false;
        }

        // Luhn
        let sum: u32 = digits
            .iter()
            .rev()
            .enumerate()
            .map(|(i, &d)| {
                if i % 2 == 1 {
                    let doubled = d * 2;
                    if doubled > 9 { doubled - 9 } else { doubled }
                } else {
                    d
                }
            })
            .sum();

        sum % 10 == 0
    }

    /// Reject 9-digit sequences that are too common to be passports.
    ///
    /// The passport regex is intentionally broad; we skip all-same-digit
    /// sequences and sequences that look like zip+4 codes or SSNs.
    fn validate_passport_context(s: &str) -> bool {
        // Already matched as SSN (different format), skip.
        if s.contains('-') {
            return false;
        }
        // Reject trivially repeated digits (e.g. 000000000, 111111111)
        let first = s.chars().next().unwrap_or('0');
        if s.chars().all(|c| c == first) {
            return false;
        }
        // Reject sequences starting with 0 (unlikely passport)
        if s.starts_with('0') {
            return false;
        }
        true
    }

    /// Ensure phone numbers have a minimum of 7 digits (after stripping
    /// formatting chars) to suppress false positives.
    fn validate_phone(s: &str) -> bool {
        let digit_count = s.chars().filter(|c| c.is_ascii_digit()).count();
        digit_count >= 7
    }
}

// ── Redaction helper ──────────────────────────────────────────────────────────

/// Replace every detected entity span in `text` with its placeholder string.
///
/// Replacements are applied in reverse byte-offset order so that earlier
/// replacements do not shift the offsets of later ones.
pub fn redact(text: &str, entities: &[DetectedEntity]) -> String {
    // Sort by start, then process in reverse to preserve offsets.
    let mut sorted: Vec<&DetectedEntity> = entities.iter().collect();
    sorted.sort_by_key(|e| e.start);

    let mut result = text.to_string();
    for entity in sorted.iter().rev() {
        let placeholder = format!("[{}_REDACTED]", entity.entity_type);
        result.replace_range(entity.start..entity.end, &placeholder);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_ssn() {
        let hits = RegexScanner::scan("Patient SSN is 123-45-6789 in the record.");
        assert!(hits.iter().any(|e| e.entity_type == EntityType::SSN));
    }

    #[test]
    fn detects_email() {
        let hits = RegexScanner::scan("Contact me at user@example.com please.");
        assert!(hits.iter().any(|e| e.entity_type == EntityType::Email));
    }

    #[test]
    fn detects_valid_credit_card() {
        // Visa test number — passes Luhn
        let hits = RegexScanner::scan("Card: 4111111111111111");
        assert!(hits.iter().any(|e| e.entity_type == EntityType::CreditCard));
    }

    #[test]
    fn rejects_invalid_credit_card() {
        // Fails Luhn
        let hits = RegexScanner::scan("Bad card: 4111111111111112");
        assert!(!hits.iter().any(|e| e.entity_type == EntityType::CreditCard));
    }

    #[test]
    fn detects_aws_key() {
        let hits = RegexScanner::scan("key=AKIAIOSFODNN7EXAMPLE rest of line");
        assert!(hits.iter().any(|e| e.entity_type == EntityType::AWSAccessKey));
    }

    #[test]
    fn detects_ipv4() {
        let hits = RegexScanner::scan("Server is at 192.168.1.100 port 8080");
        assert!(hits.iter().any(|e| e.entity_type == EntityType::IPAddress));
    }

    #[test]
    fn detects_dob() {
        let hits = RegexScanner::scan("Born on 01/15/1990.");
        assert!(hits.iter().any(|e| e.entity_type == EntityType::DateOfBirth));
    }

    #[test]
    fn redact_replaces_spans() {
        let text = "Email user@example.com today";
        let entities = RegexScanner::scan(text);
        let redacted = redact(text, &entities);
        assert!(!redacted.contains("user@example.com"));
        assert!(redacted.contains("[EMAIL_REDACTED]"));
    }

    #[test]
    fn no_false_positive_on_clean_text() {
        let hits = RegexScanner::scan("Hello world, this is a normal sentence.");
        assert!(hits.is_empty());
    }

    #[test]
    fn luhn_check_passes_for_mastercard_test_number() {
        // Mastercard test: 5500005555555559
        let hits = RegexScanner::scan("5500005555555559");
        assert!(hits.iter().any(|e| e.entity_type == EntityType::CreditCard));
    }
}
