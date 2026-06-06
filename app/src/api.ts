import { API_BASE_URL } from "./config";

export type RenderResponse = {
  job_id: string;
  output_url: string;
  frame_count: number;
  fps: number;
  duration_s: number;
  theme?: string;
};

export type PreviewResponse = {
  job_id: string;
  preview_url: string;
  frame_index: number;
  frame_count: number;
  fps: number;
  width: number;
  height: number;
  theme?: string;
};

type Asset = {
  uri: string;
  fileName?: string | null;
  mimeType?: string | null;
};

function extractServerError(
  responseText: string,
  fallbackStatus: number,
): string {
  const trimmed = (responseText || "").trim();
  if (trimmed) {
    try {
      const parsed = JSON.parse(trimmed);
      if (parsed && typeof parsed === "object") {
        const detail =
          (parsed as Record<string, unknown>).detail ??
          (parsed as Record<string, unknown>).message ??
          (parsed as Record<string, unknown>).error;
        if (typeof detail === "string" && detail.trim()) {
          return detail.trim();
        }
      }
    } catch {
      // not JSON — fall through
    }
    if (!trimmed.startsWith("<")) {
      return trimmed.slice(0, 200);
    }
  }
  return `request failed (${fallbackStatus})`;
}

export function friendlyApiError(err: unknown): string {
  const raw = err instanceof Error ? err.message : String(err ?? "");
  const lower = raw.toLowerCase();
  if (lower.includes("no face")) {
    return "We couldn't find a face in this video. Try a clip with a clearer view of the face.";
  }
  if (lower.includes("network error") || lower.includes("failed to fetch")) {
    return "Couldn't reach the server. Check your internet connection and try again.";
  }
  if (lower.includes("aborted")) {
    return "The request was cancelled. Please try again.";
  }
  if (lower.includes("exceeds") && lower.includes("mb")) {
    return raw; // already user-friendly (e.g. "video.mp4 exceeds 100 MB")
  }
  if (lower.includes("must be one of")) {
    return raw;
  }
  // Strip leading "preview failed:" / "render failed:" noise from any
  // server message we did not specifically recognize.
  return raw.replace(/^(preview|render)\s+failed:\s*/i, "");
}

function guessFilename(asset: Asset, fallback: string): string {
  if (asset.fileName && asset.fileName.length > 0) return asset.fileName;
  const fromUri = asset.uri.split("/").pop()?.split("?")[0];
  if (fromUri && fromUri.includes(".")) return fromUri;
  return fallback;
}

function guessMime(asset: Asset, fallback: string): string {
  if (asset.mimeType && asset.mimeType.length > 0) return asset.mimeType;
  const ext = guessFilename(asset, fallback).split(".").pop()?.toLowerCase();
  switch (ext) {
    case "mp4":
      return "video/mp4";
    case "mov":
      return "video/quicktime";
    case "m4v":
      return "video/x-m4v";
    case "png":
      return "image/png";
    case "jpg":
    case "jpeg":
      return "image/jpeg";
    case "webp":
      return "image/webp";
    default:
      return fallback;
  }
}

export type RenderProgress = {
  /** 0..1 — only meaningful during upload. */
  uploadProgress: number;
  /** Once upload completes, we're waiting on the server-side render. */
  phase: "uploading" | "rendering";
};

// ---------------------------------------------------------------------------
// AvatarShield IVP — theme-based render (no avatar PNG required).
// ---------------------------------------------------------------------------
//
// Endpoint reference: api/server.py
//   GET  /ivp/themes        → { themes: string[] }
//   POST /ivp/preview       (multipart: video, theme, frame_index?)
//   POST /ivp/preview/{id}/reroll  (multipart: theme?, frame_index?)
//   POST /ivp/render/{id}   (multipart: theme?)
//
// XMLHttpRequest is used (not fetch) for any request that uploads a local
// file URI — Expo SDK 56's WinterCG fetch does not support React Native's
// ``{uri, name, type}`` multipart parts, but XHR still routes through the
// legacy RN networking path that handles them correctly.

export type IvpPreviewOptions = {
  theme?: string;
  frameIndex?: number;
  onProgress?: (p: RenderProgress) => void;
};

export async function fetchIvpThemes(): Promise<string[]> {
  const res = await fetch(`${API_BASE_URL}/ivp/themes`);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(extractServerError(body, res.status));
  }
  const data = (await res.json()) as { themes: string[] };
  return data.themes;
}

export function renderIvpPreview(
  video: Asset,
  options: IvpPreviewOptions = {},
): Promise<PreviewResponse> {
  const { theme, frameIndex, onProgress } = options;
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("video", {
      uri: video.uri,
      name: guessFilename(video, "video.mp4"),
      type: guessMime(video, "video/mp4"),
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);
    if (theme) form.append("theme", theme);
    if (typeof frameIndex === "number" && Number.isFinite(frameIndex)) {
      form.append("frame_index", String(Math.max(0, Math.floor(frameIndex))));
    }

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE_URL}/ivp/preview`);
    xhr.responseType = "text";
    xhr.timeout = 0;

    if (xhr.upload) {
      xhr.upload.onprogress = (ev) => {
        if (!onProgress) return;
        const ratio =
          ev.lengthComputable && ev.total > 0 ? ev.loaded / ev.total : 0;
        onProgress({ uploadProgress: ratio, phase: "uploading" });
      };
      xhr.upload.onload = () => {
        onProgress?.({ uploadProgress: 1, phase: "rendering" });
      };
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as PreviewResponse);
        } catch (err) {
          reject(new Error(`bad JSON from server: ${(err as Error).message}`));
        }
      } else {
        reject(new Error(extractServerError(xhr.responseText, xhr.status)));
      }
    };
    xhr.onerror = () => reject(new Error("network error reaching API"));
    xhr.onabort = () => reject(new Error("request aborted"));

    xhr.send(form);
  });
}

export async function rerollIvpPreview(
  jobId: string,
  options: Omit<IvpPreviewOptions, "onProgress"> = {},
): Promise<PreviewResponse> {
  const form = new FormData();
  if (options.theme) form.append("theme", options.theme);
  if (
    typeof options.frameIndex === "number" &&
    Number.isFinite(options.frameIndex)
  ) {
    form.append(
      "frame_index",
      String(Math.max(0, Math.floor(options.frameIndex))),
    );
  }
  const res = await fetch(
    `${API_BASE_URL}/ivp/preview/${encodeURIComponent(jobId)}/reroll`,
    { method: "POST", body: form },
  );
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(extractServerError(body, res.status));
  }
  return (await res.json()) as PreviewResponse;
}

export async function commitIvpRender(
  jobId: string,
  options: { theme?: string } = {},
): Promise<RenderResponse> {
  const form = new FormData();
  if (options.theme) form.append("theme", options.theme);
  const res = await fetch(
    `${API_BASE_URL}/ivp/render/${encodeURIComponent(jobId)}`,
    { method: "POST", body: form },
  );
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(extractServerError(body, res.status));
  }
  return (await res.json()) as RenderResponse;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
