const allowedContentTypes = new Set(["image/jpeg", "image/png", "image/webp"]);

export const DEBUG_ARTIFACT_SPECS = {
  mask_png: { filename: "01-mask.png", contentType: "image/png" },
  foreground_rgba_png: { filename: "02-foreground-rgba.png", contentType: "image/png" },
  placed_mask_png: { filename: "03-placed-mask.png", contentType: "image/png" },
  composite_raw_jpg: { filename: "04-composite-raw.jpg", contentType: "image/jpeg" },
  controlcom_guidance_jpg: { filename: "05-controlcom-guidance.jpg", contentType: "image/jpeg" },
  harmonized_jpg: { filename: "06-harmonized.jpg", contentType: "image/jpeg" },
  final_jpg: { filename: "07-final.jpg", contentType: "image/jpeg" }
} as const;

export type DebugArtifactName = keyof typeof DEBUG_ARTIFACT_SPECS;
export type DebugArtifactKeys = Record<DebugArtifactName, string>;

export type UploadContentTypes = {
  car: string;
  background: string;
};

export type CompositeRunpodOptions = {
  harmonyThreshold: number;
  shadowStrength: number;
  reflectionStrength: number;
};

export type CompositeRunpodInput = {
  action: "composite";
  job_id: string;
  car_image_url: string;
  background_image_url: string;
  output_put_url: string;
  pipeline_variant: "core" | "full";
  options: {
    harmony_threshold: number;
    shadow_strength: number;
    reflection_strength: number;
  };
  debug_put_urls?: Partial<Record<DebugArtifactName, string>>;
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

export function buildDebugArtifactKeys(jobId: string): DebugArtifactKeys {
  return {
    mask_png: `debug/${jobId}/${DEBUG_ARTIFACT_SPECS.mask_png.filename}`,
    foreground_rgba_png: `debug/${jobId}/${DEBUG_ARTIFACT_SPECS.foreground_rgba_png.filename}`,
    placed_mask_png: `debug/${jobId}/${DEBUG_ARTIFACT_SPECS.placed_mask_png.filename}`,
    composite_raw_jpg: `debug/${jobId}/${DEBUG_ARTIFACT_SPECS.composite_raw_jpg.filename}`,
    controlcom_guidance_jpg: `debug/${jobId}/${DEBUG_ARTIFACT_SPECS.controlcom_guidance_jpg.filename}`,
    harmonized_jpg: `debug/${jobId}/${DEBUG_ARTIFACT_SPECS.harmonized_jpg.filename}`,
    final_jpg: `debug/${jobId}/${DEBUG_ARTIFACT_SPECS.final_jpg.filename}`
  };
}

export function debugArtifactEntries(
  keys?: Partial<Record<DebugArtifactName, string>>
): Array<[DebugArtifactName, string]> {
  if (!keys) {
    return [];
  }

  return (Object.entries(keys) as Array<[DebugArtifactName, string | undefined]>).filter(
    (entry): entry is [DebugArtifactName, string] => typeof entry[1] === "string" && entry[1].length > 0
  );
}

export function buildCompositeRunpodInput(args: {
  jobId: string;
  carImageUrl: string;
  backgroundImageUrl: string;
  outputPutUrl: string;
  pipelineVariant: "core" | "full";
  options: CompositeRunpodOptions;
  debugPutUrls?: Partial<Record<DebugArtifactName, string>>;
}): CompositeRunpodInput {
  const payload: CompositeRunpodInput = {
    action: "composite",
    job_id: args.jobId,
    car_image_url: args.carImageUrl,
    background_image_url: args.backgroundImageUrl,
    output_put_url: args.outputPutUrl,
    pipeline_variant: args.pipelineVariant,
    options: {
      harmony_threshold: args.options.harmonyThreshold,
      shadow_strength: args.options.shadowStrength,
      reflection_strength: args.options.reflectionStrength
    }
  };

  if (debugArtifactEntries(args.debugPutUrls).length > 0) {
    payload.debug_put_urls = args.debugPutUrls;
  }

  return payload;
}

export function validateUploadContentTypes(contentTypes: UploadContentTypes): void {
  if (!isAllowedImageContentType(contentTypes.car)) {
    throw new Error("Invalid car content type. Allowed: image/jpeg, image/png, image/webp");
  }
  if (!isAllowedImageContentType(contentTypes.background)) {
    throw new Error("Invalid background content type. Allowed: image/jpeg, image/png, image/webp");
  }
}
