# CarCompose MVP

Single-tenant, one-click car compositing pipeline:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/YOUR_ORG/YOUR_REPO&project-name=carcompose-mvp&repo-name=carcompose-mvp&env=APP_PASSCODE,RUNPOD_API_KEY,CLOUDFLARE_ACCOUNT_ID,CLOUDFLARE_API_TOKEN,R2_ACCESS_KEY_ID,R2_SECRET_ACCESS_KEY,R2_BUCKET_NAME,R2_ENDPOINT_URL&root-directory=web)

> **Note:** Replace `YOUR_ORG/YOUR_REPO` in the URL above with your actual GitHub username/organization and repository name before clicking.

- `web/`: Next.js app on Vercel (passcode-gated API)
- `worker/`: RunPod serverless worker (`download_models` + `composite` actions)
- `docker/worker/Dockerfile`: GPU worker image
- `.github/workflows/worker-image.yml`: GHCR build/push + worker smoke check

## One-click flow

1. Deploy `web/` to Vercel.
2. Set required environment variables from `.env.example`.
3. Open the app and enter `APP_PASSCODE`.
4. App calls `GET /api/ready`:
   - ensures R2 bucket exists
   - ensures RunPod volume/template/endpoint exists
   - starts model initialization (`action: download_models`)
5. Once ready, upload car + background, then process.

## Accounts + API keys you need

- **GitHub**: hosts the repo and builds the worker image in **GHCR** via GitHub Actions.
- **Vercel**: runs the `web/` Next.js app (orchestrator + API).
- **Cloudflare**: R2 bucket + S3 credentials for presigned URLs.
- **RunPod**: serverless GPU endpoint + jobs.

Required Vercel env vars are listed in `.env.example`.

## Build the worker image (GHCR)

The web app provisions RunPod resources, but it cannot build Docker images. The worker image must exist first.

### Fast path: one command bootstrap (GitHub + GHCR)

If you have the GitHub CLI installed (`gh`) and you are logged in, this script will:
- create a **new GitHub repo**
- push this code to `main`
- wait for the `Worker Image` workflow to build/push the image to GHCR
- best-effort set the GHCR package visibility to **public**

Run:

```bash
bash scripts/bootstrap-ghcr-worker.sh
```

Customize (optional):
- `GITHUB_OWNER`, `GITHUB_REPO_NAME`, `GITHUB_VISIBILITY=public|private`, `GIT_REMOTE_PROTOCOL=https|ssh`

### Option A (recommended): GitHub Actions builds it for you

This repo includes an automated workflow: `.github/workflows/worker-image.yml`.

1. Create a GitHub repo (public is simplest for MVP).
2. Push this code to the repo's `main` branch (the workflow triggers on pushes to `main`).
3. In GitHub, open the `Actions` tab and wait for `Worker Image` to finish.
4. The image is pushed to GHCR as:
   - `ghcr.io/<owner>/<repo>-worker:main`
   - `ghcr.io/<owner>/<repo>-worker:sha-<shortsha>`

If you deploy Vercel from the same GitHub repo, the app will auto-infer the image name. If inference fails, set
`WORKER_IMAGE` in Vercel explicitly.

If you keep the GHCR package private, set `GHCR_USERNAME` and `GHCR_TOKEN` (PAT with `read:packages`) in Vercel so
`/api/ready` can attach registry auth to the RunPod template.

### Option B: Build and push locally (manual)

1. Log in to GHCR.
2. Build:
   - `docker build -f docker/worker/Dockerfile -t ghcr.io/<owner>/<repo>-worker:main .`
3. Push:
   - `docker push ghcr.io/<owner>/<repo>-worker:main`



## API summary

All API routes require header: `x-carcompose-passcode: <APP_PASSCODE>`.

- `GET /api/ready`: idempotent provisioning + init status
- `POST /api/uploads/presign`: returns presigned PUT URLs for input images
- `POST /api/composite`: submits RunPod job and persists job metadata in R2
- `GET /api/status/{jobId}`: polls RunPod and returns final output URL/status

## State keys in R2

- `system/setup.json`
- `jobs/{jobId}.json`

## Local checks

```bash
cd web
npm test
npm run typecheck

cd ..
python3 -m py_compile worker/handler.py worker/settings.py worker/actions/download_models.py worker/actions/composite.py worker/utils/image.py
```
