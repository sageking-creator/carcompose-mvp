#!/usr/bin/env bash

set -euo pipefail

BASE_URL=""
PASSCODE=""
ITERS=8
POLL_SECONDS=5
MAX_POLLS=240
CAR_PATH="$(pwd)/test-images/car_image.jpg"
BG_PATH="$(pwd)/test-images/background.png"
KEEP_BEST="true"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/studio-loop.sh \
    --base-url https://your-app.vercel.app \
    --passcode YOUR_PASSCODE \
    [--iters 8] \
    [--car /abs/path/car.jpg] \
    [--background /abs/path/background.png] \
    [--poll-seconds 5] \
    [--max-polls 240] \
    [--keep-best true|false]

Description:
  Iterative quality loop:
  1) Runs end-to-end composite jobs with real test images
  2) Scores each job via scripts/score-studio-job.py
  3) Prunes non-best R2 objects via POST /api/admin/prune-job
  4) Stops early after 3 consecutive passing runs
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
    --iters)
      ITERS="${2:-8}"
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
    --poll-seconds)
      POLL_SECONDS="${2:-5}"
      shift 2
      ;;
    --max-polls)
      MAX_POLLS="${2:-240}"
      shift 2
      ;;
    --keep-best)
      KEEP_BEST="${2:-true}"
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

if [[ -z "$BASE_URL" || -z "$PASSCODE" ]]; then
  echo "Missing --base-url or --passcode" >&2
  usage
  exit 1
fi

if [[ ! -f "$CAR_PATH" || ! -f "$BG_PATH" ]]; then
  echo "Missing test image files." >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required" >&2
  exit 1
fi

BASE_URL="${BASE_URL%/}"
best_job_id=""
best_score="-999999"
consecutive_passes=0

prune_job() {
  local job_id="$1"
  local targets="$2"
  curl --fail --silent --show-error \
    -X POST \
    -H "x-carcompose-passcode: ${PASSCODE}" \
    -H "content-type: application/json" \
    "${BASE_URL}/api/admin/prune-job" \
    -d "{\"jobId\":\"${job_id}\",\"targets\":[${targets}]}" >/dev/null
}

for ((i=1; i<=ITERS; i++)); do
  echo "==> Iteration ${i}/${ITERS}"

  set +e
  run_output="$(bash scripts/run-test-job.sh \
    --base-url "${BASE_URL}" \
    --passcode "${PASSCODE}" \
    --car "${CAR_PATH}" \
    --background "${BG_PATH}" \
    --wait \
    --poll-seconds "${POLL_SECONDS}" \
    --max-polls "${MAX_POLLS}" 2>&1)"
  run_exit=$?
  set -e

  echo "${run_output}"
  job_id="$(echo "${run_output}" | awk -F= '/^JOB_ID=/{print $2}' | tail -n1)"
  if [[ -z "${job_id}" ]]; then
    echo "No JOB_ID returned; aborting loop." >&2
    exit 2
  fi

  job_dir="$(pwd)/tmp/debug-jobs/${job_id}"
  score_json_path="${job_dir}/score.json"

  if [[ $run_exit -ne 0 ]]; then
    echo "{\"jobId\":\"${job_id}\",\"pass\":false,\"score\":-9999,\"metrics\":{\"runExit\":${run_exit}}}" >"${score_json_path}"
  else
    python3 scripts/score-studio-job.py --job-dir "${job_dir}" --background "${BG_PATH}" >"${score_json_path}"
  fi

  score="$(jq -r '.score // -9999' "${score_json_path}")"
  pass="$(jq -r '.pass // false' "${score_json_path}")"
  echo "Score: ${score} pass=${pass}"

  if awk "BEGIN {exit !(${score} > ${best_score})}"; then
    if [[ "$KEEP_BEST" == "true" && -n "$best_job_id" && "$best_job_id" != "$job_id" ]]; then
      prune_job "$best_job_id" "\"debug\",\"uploads\",\"masks\",\"outputs\",\"jobs\"" || true
    fi
    best_score="${score}"
    best_job_id="${job_id}"
  elif [[ "$KEEP_BEST" == "true" ]]; then
    prune_job "$job_id" "\"debug\",\"uploads\",\"masks\",\"outputs\",\"jobs\"" || true
  else
    prune_job "$job_id" "\"debug\",\"uploads\",\"masks\",\"outputs\",\"jobs\"" || true
  fi

  if [[ "$pass" == "true" ]]; then
    consecutive_passes=$((consecutive_passes + 1))
  else
    consecutive_passes=0
  fi

  if [[ $consecutive_passes -ge 3 ]]; then
    echo "Reached 3 consecutive passing runs. Stopping early."
    break
  fi
done

echo "==> Best job: ${best_job_id}"
echo "==> Best score: ${best_score}"
if [[ -n "$best_job_id" ]]; then
  echo "==> Best artifacts: $(pwd)/tmp/debug-jobs/${best_job_id}"
fi
