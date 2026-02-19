#!/usr/bin/env bash
# scripts/download_model.sh
#
# Downloads the quantised DistilBERT NER model and tokenizer files from
# HuggingFace Hub into the ./models directory.
#
# Requirements: curl, sha256sum (or shasum on macOS)
#
# Usage:
#   chmod +x scripts/download_model.sh
#   ./scripts/download_model.sh
#
# To skip checksum verification (e.g. during development):
#   SKIP_CHECKSUM=1 ./scripts/download_model.sh

set -euo pipefail

MODEL_REPO="onnx-community/distilbert-base-cased-finetuned-conll03-english-ONNX"
HF_BASE="https://huggingface.co/${MODEL_REPO}/resolve/main"
MODEL_DIR="./models"

# Expected SHA-256 checksums
MODEL_SHA256="2ff638639abe90e83ea079443393df9d2d2e1e04b0904946c4578c0cabd0f7c4"
# tokenizer.json and config.json checksums — update after first download
TOKENIZER_SHA256=""  # set to empty to skip tokenizer checksum
CONFIG_SHA256=""     # set to empty to skip config checksum

SKIP_CHECKSUM="${SKIP_CHECKSUM:-0}"

# ── Helpers ───────────────────────────────────────────────────────────────────

log()  { echo "[download_model] $*"; }
warn() { echo "[download_model] WARN: $*" >&2; }
die()  { echo "[download_model] ERROR: $*" >&2; exit 1; }

sha256_file() {
    local file="$1"
    if command -v sha256sum &>/dev/null; then
        sha256sum "$file" | awk '{print $1}'
    elif command -v shasum &>/dev/null; then
        shasum -a 256 "$file" | awk '{print $1}'
    else
        die "Neither sha256sum nor shasum found — cannot verify checksums"
    fi
}

verify_checksum() {
    local file="$1"
    local expected="$2"
    local label="$3"

    if [[ -z "$expected" ]]; then
        warn "No checksum configured for ${label}, skipping verification."
        return 0
    fi

    local actual
    actual=$(sha256_file "$file")
    if [[ "$actual" != "$expected" ]]; then
        die "Checksum mismatch for ${label}!\n  expected: ${expected}\n  actual:   ${actual}"
    fi
    log "  ✓ checksum OK for ${label}"
}

download_file() {
    local url="$1"
    local dest="$2"
    local label="$3"

    if [[ -f "$dest" ]]; then
        log "${label} already exists at ${dest}, skipping download."
        return 0
    fi

    log "Downloading ${label} ..."
    curl -fL --progress-bar "$url" -o "$dest" \
        || die "Failed to download ${label} from ${url}"
    log "  Saved to ${dest}"
}

# ── Main ──────────────────────────────────────────────────────────────────────

log "Creating model directory: ${MODEL_DIR}"
mkdir -p "$MODEL_DIR"

# 1. Quantised ONNX model (~65.8 MB)
download_file \
    "${HF_BASE}/onnx/model_quantized.onnx" \
    "${MODEL_DIR}/model_quantized.onnx" \
    "model_quantized.onnx"

# 2. HuggingFace tokenizer config
download_file \
    "${HF_BASE}/tokenizer.json" \
    "${MODEL_DIR}/tokenizer.json" \
    "tokenizer.json"

# 3. Model label config
download_file \
    "${HF_BASE}/config.json" \
    "${MODEL_DIR}/config.json" \
    "config.json"

# ── Checksum verification ─────────────────────────────────────────────────────

if [[ "$SKIP_CHECKSUM" == "1" ]]; then
    warn "SKIP_CHECKSUM=1 — skipping all checksum verification."
else
    log "Verifying checksums..."
    verify_checksum "${MODEL_DIR}/model_quantized.onnx" "$MODEL_SHA256" "model_quantized.onnx"
    verify_checksum "${MODEL_DIR}/tokenizer.json"       "$TOKENIZER_SHA256" "tokenizer.json"
    verify_checksum "${MODEL_DIR}/config.json"          "$CONFIG_SHA256"    "config.json"
fi

# ── Summary ───────────────────────────────────────────────────────────────────

log ""
log "Model files ready:"
ls -lh "$MODEL_DIR/"
log ""
log "Run tests that require model files with:"
log "  cargo test -- --ignored"
log ""
log "Run the binary:"
log "  cargo run -- --text 'Alice works at Acme Corp in London' --model ${MODEL_DIR}/model_quantized.onnx"
