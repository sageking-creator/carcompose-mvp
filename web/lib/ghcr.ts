import { Buffer } from "node:buffer";

type GhcrImageRef = {
  repository: string;
  reference: string;
};

export type GhcrCredentials = {
  username: string;
  password: string;
};

const GHCR_PREFIX = "ghcr.io/";
const MANIFEST_ACCEPT =
  [
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json"
  ].join(", ");

function buildBasicAuth(credentials: GhcrCredentials): string {
  const encoded = Buffer.from(`${credentials.username}:${credentials.password}`, "utf8").toString("base64");
  return `Basic ${encoded}`;
}

export function parseGhcrImage(image: string): GhcrImageRef | null {
  const value = image.trim();
  if (!value.startsWith(GHCR_PREFIX)) {
    return null;
  }

  const remainder = value.slice(GHCR_PREFIX.length);
  if (!remainder || remainder.startsWith("/")) {
    return null;
  }

  let repository = remainder;
  let reference = "latest";

  const digestSeparator = remainder.lastIndexOf("@");
  if (digestSeparator !== -1) {
    repository = remainder.slice(0, digestSeparator);
    reference = remainder.slice(digestSeparator + 1);
  } else {
    const slashIndex = remainder.lastIndexOf("/");
    const tagSeparator = remainder.lastIndexOf(":");
    if (tagSeparator > slashIndex) {
      repository = remainder.slice(0, tagSeparator);
      reference = remainder.slice(tagSeparator + 1);
    }
  }

  if (!repository || !repository.includes("/") || !reference) {
    return null;
  }

  return {
    repository: repository.toLowerCase(),
    reference
  };
}

async function fetchGhcrToken(
  image: GhcrImageRef,
  credentials?: GhcrCredentials
): Promise<{ token: string; usedCredentials: boolean }> {
  const params = new URLSearchParams({
    service: "ghcr.io",
    scope: `repository:${image.repository}:pull`
  });
  const headers: Record<string, string> = {};
  if (credentials) {
    headers.Authorization = buildBasicAuth(credentials);
  }

  const response = await fetch(`https://ghcr.io/token?${params.toString()}`, {
    method: "GET",
    headers,
    cache: "no-store"
  });

  const bodyText = await response.text();
  if (!response.ok) {
    if (response.status === 401 || response.status === 403) {
      const prefix = credentials
        ? "GHCR credentials were rejected while requesting pull token."
        : "Worker image appears private and no GHCR credentials were provided.";
      throw new Error(`${prefix} Set GHCR_USERNAME/GHCR_TOKEN or make the package public.`);
    }
    throw new Error(`Failed to request GHCR token (${response.status}): ${bodyText || "no response body"}`);
  }

  let parsed: { token?: string; access_token?: string } = {};
  try {
    parsed = JSON.parse(bodyText) as { token?: string; access_token?: string };
  } catch {
    throw new Error("Failed to parse GHCR token response.");
  }

  const token = parsed.token ?? parsed.access_token;
  if (!token) {
    throw new Error("GHCR token response did not include a pull token.");
  }

  return {
    token,
    usedCredentials: Boolean(credentials)
  };
}

export async function assertGhcrImagePullable(
  image: string,
  credentials?: GhcrCredentials
): Promise<void> {
  await resolveGhcrImageToDigest(image, credentials);
}

export async function resolveGhcrImageToDigest(
  image: string,
  credentials?: GhcrCredentials
): Promise<string> {
  const parsedImage = parseGhcrImage(image);
  if (!parsedImage) {
    return image;
  }

  const tokenData = await fetchGhcrToken(parsedImage, credentials);
  const manifestResponse = await fetch(
    `https://ghcr.io/v2/${parsedImage.repository}/manifests/${encodeURIComponent(parsedImage.reference)}`,
    {
      method: "GET",
      headers: {
        Accept: MANIFEST_ACCEPT,
        Authorization: `Bearer ${tokenData.token}`
      },
      cache: "no-store"
    }
  );

  if (manifestResponse.ok) {
    const digest =
      manifestResponse.headers.get("docker-content-digest") ?? manifestResponse.headers.get("Docker-Content-Digest");

    // Some registries may omit the digest header for certain responses. In that case,
    // fall back to the provided image reference.
    if (!digest) {
      return image;
    }

    if (parsedImage.reference.startsWith("sha256:")) {
      return `ghcr.io/${parsedImage.repository}@${parsedImage.reference}`;
    }

    return `ghcr.io/${parsedImage.repository}@${digest}`;
  }

  const bodyText = await manifestResponse.text();
  if (manifestResponse.status === 404) {
    throw new Error(
      `Worker image not found in GHCR: ${image}. Build/push the image first or set WORKER_IMAGE correctly.`
    );
  }

  if (manifestResponse.status === 401 || manifestResponse.status === 403) {
    const authHint = tokenData.usedCredentials
      ? "Provided GHCR credentials cannot pull this package."
      : "Image is not publicly pullable.";
    throw new Error(`${authHint} Set valid GHCR_USERNAME/GHCR_TOKEN or make the package public.`);
  }

  throw new Error(
    `Failed to verify GHCR image accessibility (${manifestResponse.status}): ${bodyText || "no response body"}`
  );
}
