//! End-to-end integration tests for the full two-tier pipeline via `Scanner`.
//!
//! Tests that do not require model files run unconditionally.
//! Tests that exercise the NER tier are gated with `#[ignore]`.

use aegis_ner::{
    entity::{EntityType, RiskLevel},
    Scanner, ScanOptions,
};

// ── Regex-only pipeline ───────────────────────────────────────────────────────

fn scanner() -> Scanner {
    Scanner::regex_only()
}

#[test]
fn scan_detects_ssn() {
    let result = scanner().scan("SSN: 123-45-6789").unwrap();
    assert!(result.has_entities());
    assert!(result
        .entities
        .iter()
        .any(|e| e.entity_type == EntityType::SSN));
}

#[test]
fn scan_detects_email_and_ssn_together() {
    let result = scanner()
        .scan("Email alice@corp.com SSN 987-65-4321")
        .unwrap();
    let types: Vec<_> = result.entities.iter().map(|e| &e.entity_type).collect();
    assert!(types.contains(&&EntityType::Email));
    assert!(types.contains(&&EntityType::SSN));
}

#[test]
fn scan_with_redact_option() {
    let opts = ScanOptions {
        redact: true,
        ..Default::default()
    };
    let result = scanner()
        .scan_with_options("My email is test@example.com", &opts)
        .unwrap();
    let redacted = result.redacted_prompt.unwrap();
    assert!(!redacted.contains("test@example.com"));
    assert!(redacted.contains("_REDACTED]"));
}

#[test]
fn scan_clean_prompt_no_entities() {
    let result = scanner()
        .scan("The conference is scheduled for next Tuesday.")
        .unwrap();
    assert!(!result.has_entities());
    assert_eq!(result.entities.len(), 0);
}

#[test]
fn scan_result_has_timing() {
    let result = scanner().scan("test@example.com").unwrap();
    assert!(result.scan_duration_us > 0);
    assert!(result.tier1_duration_us > 0);
    assert_eq!(result.tier2_duration_us, 0); // regex-only, no Tier 2
}

#[test]
fn scan_max_risk_level_critical_for_aws_key() {
    let result = scanner()
        .scan("aws_key=AKIAIOSFODNN7EXAMPLE")
        .unwrap();
    assert_eq!(result.max_risk_level(), Some(RiskLevel::Critical));
}

#[test]
fn scan_entities_sorted_by_start_offset() {
    let result = scanner()
        .scan("Email alice@x.com SSN 987-65-4321 IP 10.0.0.1")
        .unwrap();
    let starts: Vec<usize> = result.entities.iter().map(|e| e.start).collect();
    let mut sorted = starts.clone();
    sorted.sort_unstable();
    assert_eq!(starts, sorted, "Entities should be sorted by start offset");
}

#[test]
fn scan_confidence_score_is_one_for_regex() {
    let result = scanner().scan("user@example.com").unwrap();
    assert!(result
        .entities
        .iter()
        .all(|e| (e.score - 1.0).abs() < f32::EPSILON));
}

#[test]
fn skip_regex_option_returns_no_entities_in_regex_only_scanner() {
    let opts = ScanOptions {
        skip_regex: true,
        skip_ner: true,
        ..Default::default()
    };
    let result = scanner()
        .scan_with_options("SSN 123-45-6789 email foo@bar.com", &opts)
        .unwrap();
    assert!(!result.has_entities());
}

#[test]
fn multiple_credit_cards_all_detected() {
    // Two valid Visa test numbers in the same prompt
    let result = scanner()
        .scan("Card A: 4111111111111111 Card B: 4012888888881881")
        .unwrap();
    let cards: Vec<_> = result
        .entities
        .iter()
        .filter(|e| e.entity_type == EntityType::CreditCard)
        .collect();
    assert_eq!(cards.len(), 2, "Both Visa test numbers should be detected");
}

#[test]
fn entities_at_risk_helper_filters_correctly() {
    let result = scanner()
        .scan("SSN 123-45-6789 IP 10.0.0.1")
        .unwrap();
    let critical = result.entities_at_risk(RiskLevel::Critical);
    assert!(critical
        .iter()
        .all(|e| e.risk_level >= RiskLevel::Critical));
    let medium_or_above = result.entities_at_risk(RiskLevel::Medium);
    assert!(medium_or_above.len() >= critical.len());
}

// ── Full pipeline (requires model) ────────────────────────────────────────────

#[test]
#[ignore = "requires model files in models/"]
fn full_pipeline_ssn_plus_person() {
    let scanner =
        Scanner::new("models/model_quantized.onnx", "models/tokenizer.json").unwrap();
    let result = scanner
        .scan("Alice Smith's SSN is 123-45-6789.")
        .unwrap();

    let has_ssn = result
        .entities
        .iter()
        .any(|e| e.entity_type == EntityType::SSN);
    let has_person = result
        .entities
        .iter()
        .any(|e| e.entity_type == EntityType::Person);

    assert!(has_ssn, "SSN should be detected by Tier 1");
    assert!(has_person, "Person name should be detected by Tier 2");
    assert!(result.tier1_duration_us > 0);
    assert!(result.tier2_duration_us > 0);
}

#[test]
#[ignore = "requires model files in models/"]
fn full_pipeline_redacts_everything() {
    let scanner =
        Scanner::new("models/model_quantized.onnx", "models/tokenizer.json").unwrap();
    let opts = ScanOptions {
        redact: true,
        ..Default::default()
    };
    let result = scanner
        .scan_with_options(
            "John Doe's email is john.doe@acme.com and his SSN is 987-65-4321.",
            &opts,
        )
        .unwrap();

    let redacted = result.redacted_prompt.unwrap();
    assert!(!redacted.contains("john.doe@acme.com"));
    assert!(!redacted.contains("987-65-4321"));
}
