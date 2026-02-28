import { DeleteObjectsCommand, ListObjectsV2Command, S3Client } from "@aws-sdk/client-s3";

type DeleteTarget = "debug" | "uploads" | "masks" | "outputs" | "jobs";

function usage(): void {
  // eslint-disable-next-line no-console
  console.log(`Usage:
  npx tsx scripts/r2-prune-job.ts --job-id <uuid> [--delete debug,uploads,masks,outputs,jobs|--all] [--yes]

Description:
  Deletes R2 objects for a specific jobId (useful to control storage costs during debugging).
  Default is a dry-run; pass --yes to actually delete.

Required env vars:
  R2_ENDPOINT_URL, R2_BUCKET_NAME, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
`);
}

function getEnv(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Missing required env var: ${name}`);
  }
  return value;
}

function chunk<T>(items: T[], size: number): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < items.length; i += size) {
    out.push(items.slice(i, i + size));
  }
  return out;
}

async function listKeys(client: S3Client, bucket: string, prefix: string): Promise<string[]> {
  const keys: string[] = [];
  let token: string | undefined;
  for (;;) {
    const resp = await client.send(
      new ListObjectsV2Command({
        Bucket: bucket,
        Prefix: prefix,
        ContinuationToken: token
      })
    );
    for (const item of resp.Contents ?? []) {
      if (item.Key) {
        keys.push(item.Key);
      }
    }
    if (!resp.IsTruncated || !resp.NextContinuationToken) {
      break;
    }
    token = resp.NextContinuationToken;
  }
  return keys;
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  let jobId = "";
  let yes = false;
  const targets = new Set<DeleteTarget>();

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--job-id") {
      jobId = argv[i + 1] ?? "";
      i += 1;
      continue;
    }
    if (arg === "--delete") {
      const raw = argv[i + 1] ?? "";
      i += 1;
      for (const part of raw.split(",")) {
        const normalized = part.trim().toLowerCase();
        if (!normalized) continue;
        if (
          normalized === "debug" ||
          normalized === "uploads" ||
          normalized === "masks" ||
          normalized === "outputs" ||
          normalized === "jobs"
        ) {
          targets.add(normalized);
        } else {
          throw new Error(`Unknown delete target: ${part}`);
        }
      }
      continue;
    }
    if (arg === "--all") {
      targets.add("debug");
      targets.add("uploads");
      targets.add("masks");
      targets.add("outputs");
      targets.add("jobs");
      continue;
    }
    if (arg === "--yes") {
      yes = true;
      continue;
    }
    if (arg === "-h" || arg === "--help") {
      usage();
      return;
    }
    throw new Error(`Unknown arg: ${arg}`);
  }

  if (!jobId) {
    usage();
    process.exitCode = 2;
    return;
  }

  if (targets.size === 0) {
    targets.add("debug");
  }

  const endpoint = getEnv("R2_ENDPOINT_URL");
  const bucket = getEnv("R2_BUCKET_NAME");
  const accessKeyId = getEnv("R2_ACCESS_KEY_ID");
  const secretAccessKey = getEnv("R2_SECRET_ACCESS_KEY");

  const client = new S3Client({
    region: "auto",
    endpoint,
    requestChecksumCalculation: "WHEN_REQUIRED",
    responseChecksumValidation: "WHEN_REQUIRED",
    credentials: { accessKeyId, secretAccessKey }
  });

  const prefixes: string[] = [];
  const exactKeys: string[] = [];
  if (targets.has("debug")) prefixes.push(`debug/${jobId}/`);
  if (targets.has("uploads")) prefixes.push(`uploads/${jobId}/`);
  if (targets.has("masks")) prefixes.push(`masks/${jobId}/`);
  if (targets.has("outputs")) prefixes.push(`outputs/${jobId}/`);
  if (targets.has("jobs")) exactKeys.push(`jobs/${jobId}.json`);

  const allKeys = new Set<string>();
  for (const key of exactKeys) {
    allKeys.add(key);
  }
  for (const prefix of prefixes) {
    const keys = await listKeys(client, bucket, prefix);
    for (const key of keys) allKeys.add(key);
  }

  const keys = Array.from(allKeys).sort();
  // eslint-disable-next-line no-console
  console.log(`R2 prune jobId=${jobId}`);
  // eslint-disable-next-line no-console
  console.log(`Targets: ${Array.from(targets).join(", ")}`);
  // eslint-disable-next-line no-console
  console.log(`Matched keys: ${keys.length}`);

  if (!yes) {
    // eslint-disable-next-line no-console
    console.log(`Dry-run only. Pass --yes to delete.`);
    for (const key of keys.slice(0, 25)) {
      // eslint-disable-next-line no-console
      console.log(`  ${key}`);
    }
    if (keys.length > 25) {
      // eslint-disable-next-line no-console
      console.log(`  ... (${keys.length - 25} more)`);
    }
    return;
  }

  if (keys.length === 0) {
    // eslint-disable-next-line no-console
    console.log(`Nothing to delete.`);
    return;
  }

  let deleted = 0;
  for (const batch of chunk(keys, 1000)) {
    const resp = await client.send(
      new DeleteObjectsCommand({
        Bucket: bucket,
        Delete: {
          Objects: batch.map((Key) => ({ Key })),
          Quiet: true
        }
      })
    );
    const errors = resp.Errors?.length ?? 0;
    deleted += batch.length - errors;
    if (resp.Errors && resp.Errors.length > 0) {
      // eslint-disable-next-line no-console
      console.error(`Delete errors:`, resp.Errors);
    }
  }

  // eslint-disable-next-line no-console
  console.log(`Deleted (attempted): ${deleted}/${keys.length}`);
}

main().catch((error) => {
  // eslint-disable-next-line no-console
  console.error(String(error));
  process.exitCode = 1;
});
