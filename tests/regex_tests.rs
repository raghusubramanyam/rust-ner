//! Integration tests for the Tier 1 regex PII scanner.
//!
//! These run without any model files and test the full regex pipeline
//! including Luhn validation, deduplication, and redaction.

use aegis_ner::{
    entity::{EntityType, RiskLevel},
    regex_scanner::{redact, RegexScanner},
};

// ── SSN ───────────────────────────────────────────────────────────────────────

#[test]
fn ssn_standard_format() {
    let entities = RegexScanner::scan("Patient SSN: 123-45-6789.");
    assert!(entities.iter().any(|e| e.entity_type == EntityType::SSN));
}

#[test]
fn ssn_no_match_without_dashes() {
    // Nine contiguous digits can match the passport pattern, not SSN pattern
    let entities = RegexScanner::scan("ID 123456789");
    assert!(!entities.iter().any(|e| e.entity_type == EntityType::SSN));
}

#[test]
fn ssn_risk_level_is_critical() {
    let entities = RegexScanner::scan("SSN 987-65-4321");
    let ssn = entities.iter().find(|e| e.entity_type == EntityType::SSN);
    assert_eq!(ssn.map(|e| e.risk_level), Some(RiskLevel::Critical));
}

// ── Credit card ───────────────────────────────────────────────────────────────

#[test]
fn visa_test_number_passes_luhn() {
    // 4111111111111111 is the canonical Visa test number
    let entities = RegexScanner::scan("Card number: 4111111111111111");
    assert!(entities
        .iter()
        .any(|e| e.entity_type == EntityType::CreditCard));
}

#[test]
fn mastercard_test_number_passes_luhn() {
    let entities = RegexScanner::scan("MC: 5500005555555559");
    assert!(entities
        .iter()
        .any(|e| e.entity_type == EntityType::CreditCard));
}

#[test]
fn invalid_credit_card_rejected_by_luhn() {
    let entities = RegexScanner::scan("Fake card 4111111111111112");
    assert!(!entities
        .iter()
        .any(|e| e.entity_type == EntityType::CreditCard));
}

#[test]
fn credit_card_with_spaces() {
    // 4111 1111 1111 1111
    let entities = RegexScanner::scan("Card: 4111 1111 1111 1111");
    assert!(entities
        .iter()
        .any(|e| e.entity_type == EntityType::CreditCard));
}

#[test]
fn credit_card_with_dashes() {
    let entities = RegexScanner::scan("Card: 4111-1111-1111-1111");
    assert!(entities
        .iter()
        .any(|e| e.entity_type == EntityType::CreditCard));
}

// ── Email ─────────────────────────────────────────────────────────────────────

#[test]
fn email_simple() {
    let entities = RegexScanner::scan("Send to alice@example.com");
    assert!(entities.iter().any(|e| e.entity_type == EntityType::Email));
}

#[test]
fn email_with_plus() {
    let entities = RegexScanner::scan("Address: user+tag@company.org");
    assert!(entities.iter().any(|e| e.entity_type == EntityType::Email));
}

#[test]
fn email_risk_is_high() {
    let entities = RegexScanner::scan("bob@corp.io");
    let e = entities.iter().find(|e| e.entity_type == EntityType::Email);
    assert_eq!(e.map(|e| e.risk_level), Some(RiskLevel::High));
}

// ── Phone ─────────────────────────────────────────────────────────────────────

#[test]
fn us_phone_parentheses_format() {
    let entities = RegexScanner::scan("Call (555) 867-5309");
    assert!(entities
        .iter()
        .any(|e| e.entity_type == EntityType::PhoneUS));
}

#[test]
fn us_phone_dotted_format() {
    let entities = RegexScanner::scan("Phone: 555.123.4567");
    assert!(entities
        .iter()
        .any(|e| e.entity_type == EntityType::PhoneUS));
}

#[test]
fn intl_phone_with_country_code() {
    let entities = RegexScanner::scan("Call +44 20 7946 0958 for details");
    assert!(entities
        .iter()
        .any(|e| e.entity_type == EntityType::PhoneIntl));
}

// ── IPv4 ──────────────────────────────────────────────────────────────────────

#[test]
fn ipv4_standard() {
    let entities = RegexScanner::scan("Server at 192.168.1.100");
    assert!(entities
        .iter()
        .any(|e| e.entity_type == EntityType::IPAddress));
}

#[test]
fn ipv4_max_octets() {
    let entities = RegexScanner::scan("255.255.255.255 is broadcast");
    assert!(entities
        .iter()
        .any(|e| e.entity_type == EntityType::IPAddress));
}

#[test]
fn invalid_octet_not_matched() {
    let entities = RegexScanner::scan("999.999.999.999 is invalid");
    assert!(!entities
        .iter()
        .any(|e| e.entity_type == EntityType::IPAddress));
}

// ── AWS keys ──────────────────────────────────────────────────────────────────

#[test]
fn aws_access_key_detected() {
    let entities = RegexScanner::scan("Key: AKIAIOSFODNN7EXAMPLE");
    assert!(entities
        .iter()
        .any(|e| e.entity_type == EntityType::AWSAccessKey));
}

#[test]
fn aws_access_key_risk_is_critical() {
    let entities = RegexScanner::scan("AKIAIOSFODNN7EXAMPLE");
    let e = entities
        .iter()
        .find(|e| e.entity_type == EntityType::AWSAccessKey);
    assert_eq!(e.map(|e| e.risk_level), Some(RiskLevel::Critical));
}

#[test]
fn aws_asia_prefix_detected() {
    let entities = RegexScanner::scan("Temp key: ASIAIOSFODNN7EXAMPLE");
    assert!(entities
        .iter()
        .any(|e| e.entity_type == EntityType::AWSAccessKey));
}

// ── API keys ──────────────────────────────────────────────────────────────────

#[test]
fn openai_style_api_key() {
    let entities = RegexScanner::scan("OPENAI_KEY=sk-proj-abcdefghijklmnopqrstu");
    assert!(entities
        .iter()
        .any(|e| e.entity_type == EntityType::APIKey));
}

// ── Date of birth ─────────────────────────────────────────────────────────────

#[test]
fn dob_slash_format() {
    let entities = RegexScanner::scan("Born 01/15/1990");
    assert!(entities
        .iter()
        .any(|e| e.entity_type == EntityType::DateOfBirth));
}

#[test]
fn dob_dash_format() {
    let entities = RegexScanner::scan("DOB: 12-31-2000");
    assert!(entities
        .iter()
        .any(|e| e.entity_type == EntityType::DateOfBirth));
}

#[test]
fn dob_invalid_month_not_matched() {
    let entities = RegexScanner::scan("Date 13/01/1990 is invalid month");
    assert!(!entities
        .iter()
        .any(|e| e.entity_type == EntityType::DateOfBirth));
}

// ── Redaction ─────────────────────────────────────────────────────────────────

#[test]
fn redact_email_in_sentence() {
    let text = "Contact alice@example.com for info.";
    let entities = RegexScanner::scan(text);
    let redacted = redact(text, &entities);
    assert!(!redacted.contains("alice@example.com"));
    assert!(redacted.contains("[EMAIL_REDACTED]"));
    assert!(redacted.contains("Contact"));
    assert!(redacted.contains("for info."));
}

#[test]
fn redact_multiple_entities() {
    let text = "SSN 123-45-6789 and email bob@corp.com";
    let entities = RegexScanner::scan(text);
    let redacted = redact(text, &entities);
    assert!(!redacted.contains("123-45-6789"));
    assert!(!redacted.contains("bob@corp.com"));
}

#[test]
fn redact_preserves_non_pii_text() {
    let text = "Hello world, please call 555-123-4567";
    let entities = RegexScanner::scan(text);
    let redacted = redact(text, &entities);
    assert!(redacted.starts_with("Hello world"));
}

// ── Clean text ────────────────────────────────────────────────────────────────

#[test]
fn clean_sentence_no_entities() {
    let entities = RegexScanner::scan("The quick brown fox jumps over the lazy dog.");
    assert!(entities.is_empty());
}

#[test]
fn clean_number_no_entities() {
    // Short numbers should not trigger patterns
    let entities = RegexScanner::scan("I have 42 apples and 7 oranges.");
    // Should not match anything (no SSN, card, etc.)
    assert!(!entities
        .iter()
        .any(|e| e.entity_type == EntityType::CreditCard));
    assert!(!entities
        .iter()
        .any(|e| e.entity_type == EntityType::SSN));
}

// ── Score ─────────────────────────────────────────────────────────────────────

#[test]
fn regex_match_score_is_one() {
    let entities = RegexScanner::scan("user@test.com");
    let e = entities.first().unwrap();
    assert!((e.score - 1.0).abs() < f32::EPSILON);
}

// ── Deduplication ─────────────────────────────────────────────────────────────

#[test]
fn no_duplicate_entities_for_same_span() {
    // An email should appear exactly once even if multiple patterns could match
    let text = "user@example.com";
    let entities = RegexScanner::scan(text);
    // Should only be one entity at offset 0
    let at_start: Vec<_> = entities.iter().filter(|e| e.start == 0).collect();
    assert_eq!(at_start.len(), 1);
}
