import {
  DeleteObjectsCommand,
  GetObjectCommand,
  HeadObjectCommand,
  ListObjectsV2Command,
  PutBucketCorsCommand,
  PutBucketLifecycleConfigurationCommand,
  PutObjectCommand,
  S3Client
} from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { getEnv } from "@/lib/env";

let client: S3Client | null = null;

function getClient(): S3Client {
  if (client) {
    return client;
  }

  const env = getEnv();
  client = new S3Client({
    region: "auto",
    endpoint: env.R2_ENDPOINT_URL,
    // AWS SDKs started enabling CRC32 checksums by default in some configurations.
    // R2 does not support these yet, so force the legacy "only when required" behavior.
    requestChecksumCalculation: "WHEN_REQUIRED",
    responseChecksumValidation: "WHEN_REQUIRED",
    credentials: {
      accessKeyId: env.R2_ACCESS_KEY_ID,
      secretAccessKey: env.R2_SECRET_ACCESS_KEY
    }
  });

  return client;
}

export function getBucketName(): string {
  return getEnv().R2_BUCKET_NAME;
}

export async function presignPut(
  key: string,
  contentType: string,
  expiresSeconds: number
): Promise<string> {
  const command = new PutObjectCommand({
    Bucket: getBucketName(),
    Key: key,
    ContentType: contentType
  });
  return getSignedUrl(getClient(), command, { expiresIn: expiresSeconds });
}

export async function presignGet(key: string, expiresSeconds: number): Promise<string> {
  const command = new GetObjectCommand({
    Bucket: getBucketName(),
    Key: key
  });
  return getSignedUrl(getClient(), command, { expiresIn: expiresSeconds });
}

export async function putJson(key: string, value: unknown): Promise<void> {
  await getClient().send(
    new PutObjectCommand({
      Bucket: getBucketName(),
      Key: key,
      Body: JSON.stringify(value),
      ContentType: "application/json"
    })
  );
}

export async function putBytes(key: string, bytes: Uint8Array, contentType: string): Promise<void> {
  await getClient().send(
    new PutObjectCommand({
      Bucket: getBucketName(),
      Key: key,
      Body: bytes,
      ContentType: contentType
    })
  );
}

async function bodyToString(body: unknown): Promise<string> {
  if (!body) {
    return "";
  }

  if (typeof (body as { transformToString?: () => Promise<string> }).transformToString === "function") {
    return (body as { transformToString: () => Promise<string> }).transformToString();
  }

  if (body instanceof Uint8Array) {
    return Buffer.from(body).toString("utf8");
  }

  const chunks: Buffer[] = [];
  for await (const chunk of body as AsyncIterable<Uint8Array>) {
    chunks.push(Buffer.from(chunk));
  }
  return Buffer.concat(chunks).toString("utf8");
}

export async function getJson<T>(key: string): Promise<T | null> {
  try {
    const response = await getClient().send(
      new GetObjectCommand({
        Bucket: getBucketName(),
        Key: key
      })
    );

    const text = await bodyToString(response.Body);
    if (!text) {
      return null;
    }

    return JSON.parse(text) as T;
  } catch (error) {
    const message = String(error);
    if (message.includes("NoSuchKey") || message.includes("NotFound") || message.includes("404")) {
      return null;
    }

    throw error;
  }
}

export async function objectExists(key: string): Promise<boolean> {
  try {
    await getClient().send(
      new HeadObjectCommand({
        Bucket: getBucketName(),
        Key: key
      })
    );
    return true;
  } catch (error) {
    const message = String(error);
    if (message.includes("NotFound") || message.includes("404") || message.includes("NoSuchKey")) {
      return false;
    }
    throw error;
  }
}

export async function ensureLifecycleRules(bucketName: string): Promise<void> {
  await getClient().send(
    new PutBucketLifecycleConfigurationCommand({
      Bucket: bucketName,
      LifecycleConfiguration: {
        Rules: [
          {
            ID: "delete-uploads-1d",
            Status: "Enabled",
            Filter: { Prefix: "uploads/" },
            Expiration: { Days: 1 }
          },
          {
            ID: "delete-jobs-7d",
            Status: "Enabled",
            Filter: { Prefix: "jobs/" },
            Expiration: { Days: 7 }
          },
          {
            ID: "delete-outputs-7d",
            Status: "Enabled",
            Filter: { Prefix: "outputs/" },
            Expiration: { Days: 7 }
          },
          {
            ID: "delete-debug-1d",
            Status: "Enabled",
            Filter: { Prefix: "debug/" },
            Expiration: { Days: 1 }
          },
          {
            ID: "delete-masks-1d",
            Status: "Enabled",
            Filter: { Prefix: "masks/" },
            Expiration: { Days: 1 }
          }
        ]
      }
    })
  );
}

export async function ensureCorsRules(bucketName: string): Promise<void> {
  await getClient().send(
    new PutBucketCorsCommand({
      Bucket: bucketName,
      CORSConfiguration: {
        CORSRules: [
          {
            AllowedOrigins: ["*"],
            AllowedMethods: ["GET", "PUT", "HEAD"],
            AllowedHeaders: ["*"],
            ExposeHeaders: ["ETag"],
            MaxAgeSeconds: 3600
          }
        ]
      }
    })
  );
}

export async function listKeysByPrefix(prefix: string): Promise<string[]> {
  const keys: string[] = [];
  let continuationToken: string | undefined;

  for (;;) {
    const response = await getClient().send(
      new ListObjectsV2Command({
        Bucket: getBucketName(),
        Prefix: prefix,
        ContinuationToken: continuationToken
      })
    );

    for (const object of response.Contents ?? []) {
      if (object.Key) {
        keys.push(object.Key);
      }
    }

    if (!response.IsTruncated || !response.NextContinuationToken) {
      break;
    }

    continuationToken = response.NextContinuationToken;
  }

  return keys;
}

function chunk<T>(values: T[], size: number): T[][] {
  const chunks: T[][] = [];
  for (let index = 0; index < values.length; index += size) {
    chunks.push(values.slice(index, index + size));
  }
  return chunks;
}

export async function deleteKeys(keys: string[]): Promise<number> {
  if (keys.length === 0) {
    return 0;
  }

  let deleted = 0;
  for (const batch of chunk(keys, 1000)) {
    const response = await getClient().send(
      new DeleteObjectsCommand({
        Bucket: getBucketName(),
        Delete: {
          Objects: batch.map((key) => ({ Key: key })),
          Quiet: true
        }
      })
    );

    const failed = response.Errors?.length ?? 0;
    deleted += batch.length - failed;
  }

  return deleted;
}
