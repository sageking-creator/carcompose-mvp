const allowedContentTypes = new Set(["image/jpeg", "image/png", "image/webp"]);

export type UploadContentTypes = {
  car: string;
  background: string;
};

export function normalizeContentType(value: string): string {
  return value.toLowerCase().split(";")[0].trim();
}

export function isAllowedImageContentType(value: string): boolean {
  return allowedContentTypes.has(normalizeContentType(value));
}

export function buildUploadKeys(jobId: string): { carKey: string; backgroundKey: string } {
  return {
    carKey: `uploads/${jobId}/car`,
    backgroundKey: `uploads/${jobId}/background`
  };
}

export function validateUploadContentTypes(contentTypes: UploadContentTypes): void {
  if (!isAllowedImageContentType(contentTypes.car)) {
    throw new Error("Invalid car content type. Allowed: image/jpeg, image/png, image/webp");
  }
  if (!isAllowedImageContentType(contentTypes.background)) {
    throw new Error("Invalid background content type. Allowed: image/jpeg, image/png, image/webp");
  }
}
