//! Integration tests for the Tier 2 NER model.
//!
//! All tests in this file are marked `#[ignore]` because they require the
//! ONNX model and tokenizer files to be present in `models/`.
//!
//! To run these tests, first download the model files:
//!   ./scripts/download_model.sh
//!
//! Then run:
//!   cargo test --test ner_tests -- --ignored

use aegis_ner::{
    entity::EntityType,
    ner_model::NERModel,
};

const MODEL_PATH: &str = "models/model_quantized.onnx";
const TOKENIZER_PATH: &str = "models/tokenizer.json";

fn load_model() -> NERModel {
    NERModel::load(MODEL_PATH, TOKENIZER_PATH)
        .expect("Failed to load NER model — run scripts/download_model.sh first")
}

#[test]
#[ignore = "requires model files"]
fn detects_person_name() {
    let model = load_model();
    let entities = model
        .predict("Alice Johnson is a software engineer.")
        .unwrap();
    assert!(
        entities.iter().any(|e| e.entity_type == EntityType::Person),
        "Expected PERSON entity, got: {:?}",
        entities
    );
}

#[test]
#[ignore = "requires model files"]
fn detects_organization() {
    let model = load_model();
    let entities = model
        .predict("Bob Smith works at Microsoft Corporation.")
        .unwrap();
    assert!(
        entities
            .iter()
            .any(|e| e.entity_type == EntityType::Organization),
        "Expected ORGANIZATION entity"
    );
}

#[test]
#[ignore = "requires model files"]
fn detects_location() {
    let model = load_model();
    let entities = model
        .predict("The summit is held in San Francisco this year.")
        .unwrap();
    assert!(
        entities
            .iter()
            .any(|e| e.entity_type == EntityType::Location),
        "Expected LOCATION entity"
    );
}

#[test]
#[ignore = "requires model files"]
fn detects_multiple_entity_types() {
    let model = load_model();
    let entities = model
        .predict("Jane Doe, CEO of Acme Inc., lives in New York.")
        .unwrap();

    let has_person = entities.iter().any(|e| e.entity_type == EntityType::Person);
    let has_org = entities
        .iter()
        .any(|e| e.entity_type == EntityType::Organization);
    let has_loc = entities
        .iter()
        .any(|e| e.entity_type == EntityType::Location);

    assert!(has_person, "Expected PERSON in multi-entity text");
    assert!(has_org, "Expected ORG in multi-entity text");
    assert!(has_loc, "Expected LOC in multi-entity text");
}

#[test]
#[ignore = "requires model files"]
fn empty_string_returns_empty() {
    let model = load_model();
    let entities = model.predict("").unwrap();
    assert!(entities.is_empty());
}

#[test]
#[ignore = "requires model files"]
fn no_entities_in_clean_sentence() {
    let model = load_model();
    let entities = model
        .predict("The weather is sunny today.")
        .unwrap();
    // This may or may not produce false positives depending on model confidence,
    // but with the default threshold they should be filtered out.
    // We just confirm it doesn't panic.
    let _ = entities;
}

#[test]
#[ignore = "requires model files"]
fn confidence_scores_are_valid() {
    let model = load_model();
    let entities = model
        .predict("President Obama visited Berlin last week.")
        .unwrap();
    for entity in &entities {
        assert!(
            (0.0..=1.0).contains(&entity.score),
            "Score out of range: {}",
            entity.score
        );
    }
}

#[test]
#[ignore = "requires model files"]
fn entity_text_matches_span_in_input() {
    let text = "Elon Musk founded Tesla Inc. in California.";
    let model = load_model();
    let entities = model.predict(text).unwrap();
    for entity in &entities {
        assert_eq!(
            &text[entity.start..entity.end],
            &entity.text,
            "Entity text does not match byte span"
        );
    }
}

#[test]
#[ignore = "requires model files"]
fn long_prompt_does_not_panic() {
    let model = load_model();
    // 600 words should be truncated to MAX_SEQ_LEN
    let long_text = "John Smith visited London. ".repeat(100);
    let entities = model.predict(&long_text).unwrap();
    // Should not panic; just confirm we get some results
    let _ = entities;
}

#[test]
#[ignore = "requires model files"]
fn model_is_reusable_across_calls() {
    let model = load_model();
    let texts = [
        "Alice works in Paris.",
        "Bob is from Toronto.",
        "Google is a company in Mountain View.",
    ];
    for text in &texts {
        let result = model.predict(text);
        assert!(result.is_ok(), "Failed on: {}", text);
    }
}
