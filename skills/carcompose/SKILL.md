---
name: carcompose
description: Workflow + invariants for CarCompose (Vercel orchestrator + R2 state + RunPod serverless GPU worker) with quality-hardening and debug-first diagnostics.
---

# CarCompose (MVP) — working rules

## Repo map (key paths)

- Web app + API: `/Users/Hikmet.Erdil/Documents/autobot/web`
  - Provisioning + readiness: `/Users/Hikmet.Erdil/Documents/autobot/web/lib/ready-service.ts`
  - RunPod admin GraphQL: `/Users/Hikmet.Erdil/Documents/autobot/web/lib/runpod-admin.ts`
  - RunPod job submit/status: `/Users/Hikmet.Erdil/Documents/autobot/web/lib/runpod-jobs.ts`
  - R2 state helpers: `/Users/Hikmet.Erdil/Documents/autobot/web/lib/r2-state.ts`
  - API routes: `/Users/Hikmet.Erdil/Documents/autobot/web/app/api/*`
- Worker (RunPod serverless): `/Users/Hikmet.Erdil/Documents/autobot/worker`
  - Action dispatch: `/Users/Hikmet.Erdil/Documents/autobot/worker/handler.py`
  - Model init job: `/Users/Hikmet.Erdil/Documents/autobot/worker/actions/download_models.py`
  - Composite job: `/Users/Hikmet.Erdil/Documents/autobot/worker/actions/composite.py`
  - Full ML pipeline: `/Users/Hikmet.Erdil/Documents/autobot/worker/pipeline.py`
- Worker image: `/Users/Hikmet.Erdil/Documents/autobot/docker/worker/Dockerfile`

## Non-negotiables (MVP constraints)

- **Single-tenant**: all API routes require `x-carcompose-passcode` matching `APP_PASSCODE`.
- **No client-triggered provisioning**: provisioning happens server-side via `/api/ready` only.
- **No Upstash/BullMQ**: RunPod provides queueing/status; job state is stored in R2 JSON.
- **No worker storage creds**: worker only uses **presigned GET/PUT** URLs from job input.
- **No fake progress bars**: UI only shows readiness + RunPod job status.
- **No silent quality regressions**: low-quality masks or halo-heavy harmonization must trigger deterministic fallback or explicit error.

## Provisioning workflow (web)

1. `/api/ready` calls `ensureReady()` which:
   - Ensures R2 bucket + lifecycle.
   - Selects a viable RunPod datacenter+GPU placement (autopick/failover).
   - Ensures RunPod volume, template, endpoint.
   - Starts (or monitors) a single `download_models` init job until it completes.
2. Provisioning state lives in R2 at `system/setup.json` via `/Users/Hikmet.Erdil/Documents/autobot/web/lib/r2-state.ts`.

### Template/endpoint versioning rule

- Any change to worker image, worker env, or endpoint scaling config must be treated as a **new template+endpoint**.
- `ensureReady()` computes a **hash** from the desired config, and uses versioned names:
  - `carcompose-worker-template-<hash>`
  - `carcompose-pipeline-<hash>`
- If the hash changes, `system/setup.json` is patched to reset:
  - `runpodTemplateId`, `runpodEndpointId`, `initJobId`, `initJobStatus`

## Worker workflow (RunPod)

### Runtime env expected (passed via template env)

- `MODEL_CACHE_DIR=/runpod-volume/models`
- `HF_HOME=/runpod-volume/hf_cache`
- `TRANSFORMERS_CACHE=/runpod-volume/hf_cache`
- `CONTROLCOM_CKPT=/runpod-volume/models/controlcom/ControlCom_blend_harm.pth`
- `CLIP_MODEL_DIR=/runpod-volume/models/controlcom/openai-clip-vit-large-patch14`
- `PIPELINE_VARIANT=core|full`
- `STUDIO_CAR_WIDTH_RATIO=0.82` (studio placement scale)
- `STUDIO_GROUND_RATIO=0.90` (studio placement ground anchor)

### Init job: `action="download_models"`

- Variant-only policy:
  - `core`: download/validate BiRefNet + ControlCom + CLIP + ckpt.
  - `full`: also validate libcom models by instantiating:
    - `ShadowGenerationModel(model_type="GPSDiffusion")` (forces correct GPSDiffusion weights)
    - `ReflectionGenerationModel`
    - `HarmonyScoreModel`
- Write sentinel only after validation passes: `/runpod-volume/.download_complete`.

### Composite job: `action="composite"`

- Core pipeline: `BiRefNet → ControlCom`
- Full pipeline (only if `PIPELINE_VARIANT=full`): `→ GPSDiffusion shadow → reflection → BargainNet QC`
- Quality controls:
  - hardened alpha + leak checks
  - detail preservation + edge halo checks
  - deterministic luminance-transfer fallback when guidance quality is poor
- Output contract:
  - `{"status":"success", ...}` or `{"status":"rejected", ...}` or `{"status":"error", ...}`

### Debugging policy

- Keep stage-level diagnostics available (mask placement, harmonization, artifact checks).
- Debug outputs should be retrievable outside the worker runtime so failures can be audited after job completion.
- Maintain a repeatable loop: enable debug -> run job -> inspect `/api/status/:jobId` debug URLs -> tune -> rerun.
- `GET /api/status/:jobId` returns `debugUrls` even for `status=error|rejected` when `DEBUG_ARTIFACTS=true`.
- Use `/Users/Hikmet.Erdil/Documents/autobot/scripts/fetch-debug-artifacts.sh` to download `status.json` + debug images into `tmp/debug-jobs/<jobId>/`.
- Prefer the repo's `test-images/` pair for consistent end-to-end debugging:
  - `/Users/Hikmet.Erdil/Documents/autobot/test-images/car_image.jpg`
  - `/Users/Hikmet.Erdil/Documents/autobot/test-images/background.png`
  - Run: `/Users/Hikmet.Erdil/Documents/autobot/scripts/run-test-job.sh` (supports `--cleanup`).
- Preserve remote debug upload contract (`debug_put_urls`) so worker failures are diagnosable without attaching to pods.
- Cost control: R2 lifecycle rules delete `debug/` after 1 day, but during rapid iteration prune per-job objects:
  - `cd /Users/Hikmet.Erdil/Documents/autobot/web && npx tsx ../scripts/r2-prune-job.ts --job-id <uuid> --delete debug,uploads,jobs --yes`

## When editing the pipeline

- Keep worker **idempotent** and side-effect free aside from uploading to `output_put_url`.
- Preserve identity details (logos, plates, trim) as a hard requirement while harmonizing.
- Studio placement uses a best-effort turntable alignment heuristic; keep env knobs so it can be tuned without code changes.
- If you add/change required worker env vars:
  - Update `getTemplateEnv()` in `/Users/Hikmet.Erdil/Documents/autobot/web/lib/ready-service.ts`
  - Ensure the provisioning hash input includes the change (so a new template/endpoint is created).

## References

- Pipeline blueprint/spec: `/Users/Hikmet.Erdil/Documents/autobot/car-composite-architecture.md`
