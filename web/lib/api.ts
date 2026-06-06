import { API_BASE_URL } from './config';

export type RenderResponse = {
  job_id: string;
  output_url: string;
  frame_count: number;
  fps: number;
  duration_s: number;
};

export type RenderProgress = {
  /** 0..1 — only meaningful during upload. */
  uploadProgress: number;
  /** Once upload completes, we're waiting on the server-side render. */
  phase: 'uploading' | 'rendering';
};

/**
 * Browser-side upload helper. Uses XMLHttpRequest so we get real upload-progress
 * events (the Fetch API does not expose them without a streaming request body).
 */
export function renderAiVideo(
  video: File,
  avatar: File,
  onProgress?: (p: RenderProgress) => void,
): Promise<RenderResponse> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append('video', video, video.name);
    form.append('avatar', avatar, avatar.name);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE_URL}/render`);
    xhr.responseType = 'text';
    xhr.timeout = 0;

    xhr.upload.onprogress = (ev) => {
      if (!onProgress) return;
      const ratio = ev.lengthComputable && ev.total > 0 ? ev.loaded / ev.total : 0;
      onProgress({ uploadProgress: ratio, phase: 'uploading' });
    };
    xhr.upload.onload = () => {
      onProgress?.({ uploadProgress: 1, phase: 'rendering' });
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as RenderResponse);
        } catch (err) {
          reject(new Error(`bad JSON from server: ${(err as Error).message}`));
        }
      } else {
        reject(
          new Error(
            `render failed (${xhr.status}): ${(xhr.responseText || '').slice(0, 200)}`,
          ),
        );
      }
    };
    xhr.onerror = () => reject(new Error('network error reaching API'));
    xhr.onabort = () => reject(new Error('request aborted'));

    xhr.send(form);
  });
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
