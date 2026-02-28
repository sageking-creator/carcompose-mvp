#!/usr/bin/env bash

set -euo pipefail

BASE_URL=""
PASSCODE=""
JOB_ID=""
OUT_DIR=""
WAIT_MODE="false"
POLL_SECONDS=5
MAX_POLLS=240

usage() {
  cat <<'EOF'
Usage:
  bash scripts/fetch-debug-artifacts.sh \
    --base-url https://your-app.vercel.app \
    --passcode YOUR_PASSCODE \
    --job-id UUID \
    [--out-dir /absolute/path] \
    [--wait] \
    [--poll-seconds 5] \
    [--max-polls 240]

Description:
  Calls GET /api/status/<jobId>, stores raw JSON, and downloads output + debug artifact URLs (if present).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL="${2:-}"
      shift 2
      ;;
    --passcode)
      PASSCODE="${2:-}"
      shift 2
      ;;
    --job-id)
      JOB_ID="${2:-}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --wait)
      WAIT_MODE="true"
      shift
      ;;
    --poll-seconds)
      POLL_SECONDS="${2:-5}"
      shift 2
      ;;
    --max-polls)
      MAX_POLLS="${2:-240}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$BASE_URL" || -z "$PASSCODE" || -z "$JOB_ID" ]]; then
  echo "Missing required args." >&2
  usage
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required." >&2
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required." >&2
  exit 1
fi

BASE_URL="${BASE_URL%/}"

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$(pwd)/tmp/debug-jobs/${JOB_ID}"
fi
mkdir -p "$OUT_DIR"

STATUS_URL="${BASE_URL}/api/status/${JOB_ID}"
STATUS_FILE="${OUT_DIR}/status.json"

fetch_status() {
  local tmp_file="${STATUS_FILE}.tmp"
  curl --fail --silent --show-error \
    -H "x-carcompose-passcode: ${PASSCODE}" \
    "${STATUS_URL}" >"${tmp_file}"
  mv "${tmp_file}" "${STATUS_FILE}"
}

echo "==> Fetching status for job ${JOB_ID}"
fetch_status
STATUS_VALUE="$(jq -r '.status // empty' "${STATUS_FILE}")"

if [[ "$WAIT_MODE" == "true" ]]; then
  poll_count=0
  while [[ "$STATUS_VALUE" == "processing" && "$poll_count" -lt "$MAX_POLLS" ]]; do
    poll_count=$((poll_count + 1))
    echo "   [${poll_count}/${MAX_POLLS}] status=processing; waiting ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
    fetch_status
    STATUS_VALUE="$(jq -r '.status // empty' "${STATUS_FILE}")"
  done
fi

echo "==> Final status: ${STATUS_VALUE:-unknown}"

EXIT_CODE=0
if [[ "$STATUS_VALUE" == "error" ]]; then
  echo "Worker/API error: $(jq -r '.message // "unknown error"' "${STATUS_FILE}")" >&2
  EXIT_CODE=2
elif [[ "$STATUS_VALUE" == "rejected" ]]; then
  echo "Job rejected: $(jq -r '.score // empty' "${STATUS_FILE}")" >&2
  EXIT_CODE=3
elif [[ "$STATUS_VALUE" != "success" ]]; then
  echo "Job not in success state. Raw status saved to ${STATUS_FILE}"
fi

OUTPUT_URL="$(jq -r '.outputUrl // empty' "${STATUS_FILE}")"
if [[ -n "$OUTPUT_URL" ]]; then
  echo "==> Downloading output image"
  curl --fail --silent --show-error "$OUTPUT_URL" -o "${OUT_DIR}/output.jpg"
fi

DEBUG_COUNT="$(jq -r '.debugUrls | if type=="object" then (keys|length) else 0 end' "${STATUS_FILE}")"
if [[ "$DEBUG_COUNT" -gt 0 ]]; then
  echo "==> Downloading ${DEBUG_COUNT} debug artifacts"
  while IFS=$'\t' read -r artifact_name artifact_url; do
    [[ -z "$artifact_name" || -z "$artifact_url" ]] && continue
    extension="jpg"
    if [[ "$artifact_name" == *"_png" ]]; then
      extension="png"
    fi
    if ! curl --fail --silent --show-error "$artifact_url" -o "${OUT_DIR}/${artifact_name}.${extension}"; then
      echo "WARN: failed to download debug artifact '${artifact_name}' (may be missing for this worker build)" >&2
      continue
    fi
  done < <(jq -r '.debugUrls | to_entries[] | [.key, .value] | @tsv' "${STATUS_FILE}")
fi

echo "==> Saved artifacts to ${OUT_DIR}"
echo "==> status.json: ${STATUS_FILE}"
exit "$EXIT_CODE"
