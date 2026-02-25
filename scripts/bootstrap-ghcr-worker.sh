#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "$ROOT_DIR/README.md" || ! -d "$ROOT_DIR/web" || ! -d "$ROOT_DIR/worker" ]]; then
  echo "Error: run this from the CarCompose repo root."
  exit 1
fi

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: missing required command: $1"
    exit 1
  fi
}

need_cmd git
need_cmd gh

if ! gh auth status -h github.com >/dev/null 2>&1; then
  echo "Error: GitHub CLI not authenticated."
  echo "Run: gh auth login"
  exit 1
fi

DEFAULT_REPO_NAME="${CARCOMPOSE_REPO_NAME:-carcompose-mvp}"
OWNER="${GITHUB_OWNER:-}"
REPO_NAME="${GITHUB_REPO_NAME:-$DEFAULT_REPO_NAME}"
VISIBILITY="${GITHUB_VISIBILITY:-public}" # public|private
REMOTE_PROTOCOL="${GIT_REMOTE_PROTOCOL:-https}" # https|ssh

if [[ -z "$OWNER" ]]; then
  OWNER="$(gh api user -q .login)"
fi

REPO_FULL="${OWNER}/${REPO_NAME}"
REPO_LC="$(echo "$REPO_NAME" | tr '[:upper:]' '[:lower:]')"
OWNER_LC="$(echo "$OWNER" | tr '[:upper:]' '[:lower:]')"
PACKAGE_NAME="${REPO_LC}-worker"
IMAGE_MAIN="ghcr.io/${OWNER_LC}/${PACKAGE_NAME}:main"

echo "==> Target GitHub repo: $REPO_FULL"
echo "==> Target worker image: $IMAGE_MAIN"

TOPLEVEL="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ "$TOPLEVEL" != "$ROOT_DIR" ]]; then
  if [[ -d "$ROOT_DIR/.git" ]]; then
    echo "Error: unexpected git state: .git exists but toplevel is $TOPLEVEL"
    exit 1
  fi

echo "==> Initializing a standalone git repo in:"
  echo "    $ROOT_DIR"
  echo "    (This avoids accidentally pushing your entire home directory.)"
  git init -b main
fi

if ! git config user.email >/dev/null 2>&1; then
  if git config --global user.email >/dev/null 2>&1; then
    git config user.email "$(git config --global user.email)"
  fi
fi
if ! git config user.name >/dev/null 2>&1; then
  if git config --global user.name >/dev/null 2>&1; then
    git config user.name "$(git config --global user.name)"
  fi
fi

if ! git config user.email >/dev/null 2>&1 || ! git config user.name >/dev/null 2>&1; then
  echo "Error: git user.name/email not configured."
  echo "Run:"
  echo "  git config --global user.name \"Your Name\""
  echo "  git config --global user.email \"you@example.com\""
  exit 1
fi

if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  echo "==> Creating initial commit..."
  git add -A
  git commit -m "Initial CarCompose MVP"
fi

if ! gh repo view "$REPO_FULL" >/dev/null 2>&1; then
  echo "==> Creating GitHub repo ($VISIBILITY)..."
  if [[ "$VISIBILITY" == "private" ]]; then
    gh repo create "$REPO_FULL" --private --confirm
  else
    gh repo create "$REPO_FULL" --public --confirm
  fi
fi

REMOTE_URL=""
if [[ "$REMOTE_PROTOCOL" == "ssh" ]]; then
  REMOTE_URL="$(gh repo view "$REPO_FULL" --json sshUrl -q .sshUrl)"
else
  # `httpsUrl` was removed from newer `gh` versions. Use the web URL and convert to clone URL.
  WEB_URL="$(gh repo view "$REPO_FULL" --json url -q .url)"
  REMOTE_URL="${WEB_URL}.git"
fi

if git remote get-url origin >/dev/null 2>&1; then
  EXISTING_ORIGIN="$(git remote get-url origin)"
  if [[ "$EXISTING_ORIGIN" != "$REMOTE_URL" ]]; then
    echo "Error: git remote 'origin' already set and does not match target repo."
    echo "  origin: $EXISTING_ORIGIN"
    echo "  target: $REMOTE_URL"
    echo "Fix by removing or updating origin, then re-run:"
    echo "  git remote remove origin"
    exit 1
  fi
else
  git remote add origin "$REMOTE_URL"
fi

echo "==> Pushing to main..."
git branch -M main
git push -u origin main

echo "==> Waiting for GitHub Actions workflow 'Worker Image'..."
RUN_ID=""
for _ in $(seq 1 90); do
  RUN_ID="$(gh run list --workflow worker-image.yml --branch main --limit 1 --json databaseId -q '.[0].databaseId' 2>/dev/null || true)"
  if [[ -n "${RUN_ID:-}" && "${RUN_ID:-null}" != "null" ]]; then
    break
  fi
  sleep 2
done

if [[ -z "${RUN_ID:-}" || "${RUN_ID:-null}" == "null" ]]; then
  echo "Warning: could not find a workflow run yet."
  echo "Check manually in GitHub Actions: $REPO_FULL → Actions → Worker Image"
else
  gh run watch "$RUN_ID" --exit-status
fi

echo "==> Best-effort: set GHCR package visibility to public (for RunPod pulls without auth)..."
OWNER_TYPE="$(gh api "users/${OWNER}" -q .type 2>/dev/null || true)"
set +e
if [[ "$OWNER_TYPE" == "Organization" ]]; then
  gh api -X PATCH "/orgs/${OWNER}/packages/container/${PACKAGE_NAME}/visibility" -f visibility=public >/dev/null
else
  gh api -X PATCH "/user/packages/container/${PACKAGE_NAME}/visibility" -f visibility=public >/dev/null
fi
VIS_RC=$?
set -e
if [[ $VIS_RC -ne 0 ]]; then
  echo "Note: could not change package visibility via API (it may already be public, or your token lacks permissions)."
  echo "You can set it manually in GitHub: Packages → ${PACKAGE_NAME} → Package settings → Change visibility."
fi

echo
echo "Done."
echo "Worker image:"
echo "  $IMAGE_MAIN"
echo
echo "Next:"
echo "  - Deploy the web app to Vercel from $REPO_FULL"
echo "  - If Vercel can't infer the image, set WORKER_IMAGE=$IMAGE_MAIN"
