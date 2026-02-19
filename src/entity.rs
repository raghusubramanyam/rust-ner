use serde::{Deserialize, Serialize};

/// All entity types recognized by the two-tier detection pipeline.
///
/// Tier 2 (NER model) produces `Person`, `Organization`, `Location`, `Miscellaneous`.
/// Tier 1 (regex) produces the structured PII variants below them.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EntityType {
    // ── Tier 2: NER model entities (CoNLL-2003 schema) ──────────────────────
    Person,
    Organization,
    Location,
    Miscellaneous,

    // ── Tier 1: Structured PII detected by regex ─────────────────────────────
    SSN,
    CreditCard,
    Email,
    PhoneUS,
    PhoneIntl,
    IPAddress,
    AWSAccessKey,
    AWSSecretKey,
    APIKey,
    PassportUS,
    DateOfBirth,
    IBAN,

    Unknown,
}

impl EntityType {
    /// Default risk level associated with each entity type.
    pub fn default_risk_level(&self) -> RiskLevel {
        match self {
            EntityType::SSN
            | EntityType::CreditCard
            | EntityType::AWSAccessKey
            | EntityType::AWSSecretKey
            | EntityType::APIKey
            | EntityType::IBAN => RiskLevel::Critical,

            EntityType::Email
            | EntityType::PhoneUS
            | EntityType::PhoneIntl
            | EntityType::PassportUS
            | EntityType::Person => RiskLevel::High,

            EntityType::IPAddress
            | EntityType::DateOfBirth
            | EntityType::Organization
            | EntityType::Location
            | EntityType::Miscellaneous => RiskLevel::Medium,

            EntityType::Unknown => RiskLevel::Low,
        }
    }
}

impl std::fmt::Display for EntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            EntityType::Person => "PERSON",
            EntityType::Organization => "ORGANIZATION",
            EntityType::Location => "LOCATION",
            EntityType::Miscellaneous => "MISCELLANEOUS",
            EntityType::SSN => "SSN",
            EntityType::CreditCard => "CREDIT_CARD",
            EntityType::Email => "EMAIL",
            EntityType::PhoneUS => "PHONE_US",
            EntityType::PhoneIntl => "PHONE_INTL",
            EntityType::IPAddress => "IP_ADDRESS",
            EntityType::AWSAccessKey => "AWS_ACCESS_KEY",
            EntityType::AWSSecretKey => "AWS_SECRET_KEY",
            EntityType::APIKey => "API_KEY",
            EntityType::PassportUS => "PASSPORT_US",
            EntityType::DateOfBirth => "DATE_OF_BIRTH",
            EntityType::IBAN => "IBAN",
            EntityType::Unknown => "UNKNOWN",
        };
        write!(f, "{}", s)
    }
}

/// Risk severity for an entity detection.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl std::fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            RiskLevel::Critical => "CRITICAL",
            RiskLevel::High => "HIGH",
            RiskLevel::Medium => "MEDIUM",
            RiskLevel::Low => "LOW",
        };
        write!(f, "{}", s)
    }
}

/// Which detection tier produced this entity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DetectionTier {
    /// Tier 1: deterministic regex pattern matching (~1 μs)
    Regex,
    /// Tier 2: ML-based ONNX NER inference (~5–15 ms)
    NERModel,
}

impl std::fmt::Display for DetectionTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DetectionTier::Regex => write!(f, "REGEX"),
            DetectionTier::NERModel => write!(f, "NER_MODEL"),
        }
    }
}

/// A single detected entity span within the input text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedEntity {
    /// Semantic type of the detected entity.
    pub entity_type: EntityType,
    /// The raw text that was matched.
    pub text: String,
    /// Confidence score: `1.0` for regex matches, model softmax score for NER.
    pub score: f32,
    /// Byte offset of the start of the match within the original input.
    pub start: usize,
    /// Byte offset (exclusive) of the end of the match within the original input.
    pub end: usize,
    /// Which detection tier produced this entity.
    pub tier: DetectionTier,
    /// Risk severity level.
    pub risk_level: RiskLevel,
}

impl DetectedEntity {
    /// Returns `true` if the entity has a confidence score above `threshold`.
    pub fn above_threshold(&self, threshold: f32) -> bool {
        self.score >= threshold
    }
}

impl std::fmt::Display for DetectedEntity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{} | {} | score={:.3} | risk={} | tier={} | {}..{}]",
            self.entity_type,
            self.text,
            self.score,
            self.risk_level,
            self.tier,
            self.start,
            self.end,
        )
    }
}

/// Aggregated result of scanning a single prompt through both tiers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanResult {
    /// The original input prompt.
    pub prompt: String,
    /// All detected entities (combined Tier 1 + Tier 2).
    pub entities: Vec<DetectedEntity>,
    /// Prompt with entity text replaced by type placeholders, if redaction was
    /// requested (e.g. `"[EMAIL_REDACTED]"`). `None` when not redacted.
    pub redacted_prompt: Option<String>,
    /// Total wall-clock time for the full scan, in microseconds.
    pub scan_duration_us: u64,
    /// Wall-clock time for Tier 1 regex scan, in microseconds.
    pub tier1_duration_us: u64,
    /// Wall-clock time for Tier 2 NER model inference, in microseconds.
    pub tier2_duration_us: u64,
}

impl ScanResult {
    /// Returns `true` when at least one entity was detected.
    pub fn has_entities(&self) -> bool {
        !self.entities.is_empty()
    }

    /// Returns the highest risk level among all detected entities, or `None`
    /// when no entities are present.
    pub fn max_risk_level(&self) -> Option<RiskLevel> {
        self.entities.iter().map(|e| e.risk_level).max()
    }

    /// Returns all entities at or above the given risk level.
    pub fn entities_at_risk(&self, min_risk: RiskLevel) -> Vec<&DetectedEntity> {
        self.entities
            .iter()
            .filter(|e| e.risk_level >= min_risk)
            .collect()
    }
}
