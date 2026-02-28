#!/usr/bin/env bash

set -euo pipefail

BASE_URL=""
PASSCODE=""
CAR_PATH="$(pwd)/test-images/car_image.jpg"
BG_PATH="$(pwd)/test-images/background.png"
CLEANUP="false"
CLEANUP_ALL="false"
POLL_SECONDS=5
MAX_POLLS=240

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run-test-job.sh \
    --base-url https://your-app.vercel.app \
    --passcode YOUR_PASSCODE \
    [--car /absolute/path/to/car.jpg] \
    [--background /absolute/path/to/background.png] \
    [--wait] \
    [--poll-seconds 5] \
    [--max-polls 240] \
    [--cleanup] \
    [--cleanup-all]

Description:
  End-to-end smoke test using the repo's `test-images/` pair:
  - presign uploads
  - PUT car+background to R2
  - submit composite job
  - poll status and download output/debug artifacts into `tmp/debug-jobs/<jobId>/`

Notes:
  - Debug artifacts require `DEBUG_ARTIFACTS=true` in Vercel env.
  - `--cleanup` deletes debug/uploads/masks/jobs for this job from R2 (requires R2_* env vars locally).
  - `--cleanup-all` also deletes outputs/ for this job.
EOF
}

WAIT_MODE="true"

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
    --car)
      CAR_PATH="${2:-}"
      shift 2
      ;;
    --background)
      BG_PATH="${2:-}"
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
    --cleanup)
      CLEANUP="true"
      shift
      ;;
    --cleanup-all)
      CLEANUP="true"
      CLEANUP_ALL="true"
      shift
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

if [[ -z "$BASE_URL" || -z "$PASSCODE" ]]; then
  echo "Missing required args." >&2
  usage
  exit 1
fi

if [[ ! -f "$CAR_PATH" ]]; then
  echo "Car image not found: $CAR_PATH" >&2
  exit 1
fi
if [[ ! -f "$BG_PATH" ]]; then
  echo "Background image not found: $BG_PATH" >&2
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

echo "==> Ensuring system is ready"
READY_URL="${BASE_URL}/api/ready"
ready_json="$(curl --fail --silent --show-error -H "x-carcompose-passcode: ${PASSCODE}" "${READY_URL}")"
ready_val="$(echo "$ready_json" | jq -r '.ready // false')"
phase_val="$(echo "$ready_json" | jq -r '.phase // empty')"
msg_val="$(echo "$ready_json" | jq -r '.message // empty')"
echo "   [0/${MAX_POLLS}] ready=${ready_val} phase=${phase_val} msg=${msg_val}"
if [[ "${ready_val}" != "true" ]]; then
  for ((i=1; i<=MAX_POLLS; i++)); do
    sleep "${POLL_SECONDS}"
    ready_json="$(curl --fail --silent --show-error -H "x-carcompose-passcode: ${PASSCODE}" "${READY_URL}")"
    ready_val="$(echo "$ready_json" | jq -r '.ready // false')"
    phase_val="$(echo "$ready_json" | jq -r '.phase // empty')"
    msg_val="$(echo "$ready_json" | jq -r '.message // empty')"
    echo "   [${i}/${MAX_POLLS}] ready=${ready_val} phase=${phase_val} msg=${msg_val}"
    if [[ "${ready_val}" == "true" ]]; then
      break
    fi
  done
fi

if [[ "${ready_val}" != "true" ]]; then
  echo "System not ready after ${MAX_POLLS} polls." >&2
  exit 2
fi

echo "==> Presigning uploads"
PRESIGN_JSON="$(curl --fail --silent --show-error \
  -X POST \
  -H "x-carcompose-passcode: ${PASSCODE}" \
  -H "content-type: application/json" \
  "${BASE_URL}/api/uploads/presign" \
  -d '{"contentTypes":{"car":"image/jpeg","background":"image/png"}}')"

JOB_ID="$(echo "$PRESIGN_JSON" | jq -r '.jobId')"
CAR_PUT="$(echo "$PRESIGN_JSON" | jq -r '.car.putUrl')"
BG_PUT="$(echo "$PRESIGN_JSON" | jq -r '.background.putUrl')"

if [[ -z "$JOB_ID" || "$JOB_ID" == "null" ]]; then
  echo "Failed to get jobId from presign response." >&2
  echo "$PRESIGN_JSON" >&2
  exit 1
fi

echo "==> Uploading car"
curl --fail --silent --show-error \
  -X PUT \
  -H "content-type: image/jpeg" \
  --data-binary @"${CAR_PATH}" \
  "${CAR_PUT}" >/dev/null

echo "==> Uploading background"
curl --fail --silent --show-error \
  -X PUT \
  -H "content-type: image/png" \
  --data-binary @"${BG_PATH}" \
  "${BG_PUT}" >/dev/null

echo "==> Submitting composite job jobId=${JOB_ID}"
curl --fail --silent --show-error \
  -X POST \
  -H "x-carcompose-passcode: ${PASSCODE}" \
  -H "content-type: application/json" \
  "${BASE_URL}/api/composite" \
  -d "{\"jobId\":\"${JOB_ID}\",\"options\":{\"harmonyThreshold\":0.65,\"shadowStrength\":0.85,\"reflectionStrength\":0.60}}" \
  >/dev/null

echo "==> Fetching status + artifacts"
bash scripts/fetch-debug-artifacts.sh \
  --base-url "${BASE_URL}" \
  --passcode "${PASSCODE}" \
  --job-id "${JOB_ID}" \
  --wait \
  --poll-seconds "${POLL_SECONDS}" \
  --max-polls "${MAX_POLLS}" || true

OUT_DIR="$(pwd)/tmp/debug-jobs/${JOB_ID}"
echo "==> Done. Artifacts in: ${OUT_DIR}"

if [[ "$CLEANUP" == "true" ]]; then
  if [[ -z "${R2_ENDPOINT_URL:-}" || -z "${R2_BUCKET_NAME:-}" || -z "${R2_ACCESS_KEY_ID:-}" || -z "${R2_SECRET_ACCESS_KEY:-}" ]]; then
    echo "==> Cleanup requested but R2_* env vars are missing locally; skipping cleanup." >&2
    exit 0
  fi

  TARGETS="debug,uploads,masks,jobs"
  if [[ "$CLEANUP_ALL" == "true" ]]; then
    TARGETS="debug,uploads,masks,outputs,jobs"
  fi

  echo "==> Cleaning R2 keys for jobId=${JOB_ID} targets=${TARGETS}"
  (cd web && npx tsx scripts/r2-prune-job.ts --job-id "${JOB_ID}" --delete "${TARGETS}" --yes)
fi

echo "JOB_ID=${JOB_ID}"
