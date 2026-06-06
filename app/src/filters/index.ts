import {
  commitIvpRender,
  renderIvpPreview,
  rerollIvpPreview,
  type PreviewResponse,
  type RenderProgress,
  type RenderResponse,
} from "../api";
import type { AvatarPreset } from "./avatars";

export type VideoInput = {
  uri: string;
  fileName?: string | null;
  mimeType?: string | null;
};

export type FilterParams = Record<string, unknown>;

export type FilterContext = {
  video: VideoInput;
  avatar: AvatarPreset | null;
  params?: FilterParams;
  onProgress?: (p: RenderProgress) => void;
};

export type Filter = {
  id: string;
  label: string;
  description: string;
  /** Swatch color shown in the filter strip thumbnail. */
  swatch: string;
  /** When true, the editor opens the avatar picker before requesting a preview. */
  needsAvatar: boolean;
  /** Whether the filter can currently be rendered. */
  available: boolean;
  /**
   * Upload inputs and render a single preview frame. The returned ``job_id``
   * is reused by ``commitRender`` / ``rerollPreview`` to avoid re-uploading.
   */
  requestPreview: (ctx: FilterContext) => Promise<PreviewResponse>;
  /** Re-render the preview at a different frame for an existing job. */
  rerollPreview: (
    jobId: string,
    frameIndex?: number,
    params?: FilterParams,
  ) => Promise<PreviewResponse>;
  /** Commit the previewed job to a full video render. */
  commitRender: (jobId: string) => Promise<RenderResponse>;
};

const unavailable = () => {
  throw new Error("Filter not available yet");
};

// ---------------------------------------------------------------------------
// AvatarShield IVP — pure-IVP501 render, no avatar PNG required. The mobile
// editor surfaces five themes from the server's /ivp/themes endpoint; the
// filter sends the chosen theme to /ivp/preview and /ivp/render. See
// spec.md §6.3 for the theme catalogue.
// ---------------------------------------------------------------------------

export const IVP_DEFAULT_THEME = "porcelain-pink";

export const IVP_THEME_META: Record<string, { label: string; swatch: string }> =
  {
    "porcelain-pink": { label: "Porcelain pink", swatch: "#f4c2c2" },
    "tan-amber": { label: "Tan amber", swatch: "#d39253" },
    "ivory-violet": { label: "Ivory violet", swatch: "#c4a8e0" },
    "bronze-teal": { label: "Bronze teal", swatch: "#4d8c8c" },
    "peach-noir": { label: "Peach noir", swatch: "#7a4e4a" },
  };

export type IvpParams = {
  theme: string;
};

function readIvpParams(params?: FilterParams): IvpParams {
  const theme =
    typeof params?.theme === "string" && params.theme
      ? (params.theme as string)
      : IVP_DEFAULT_THEME;
  return { theme };
}

export const FILTERS: Filter[] = [
  {
    id: "avatarshield",
    label: "AvatarShield",
    description: "Cel-shade with a theme palette.",
    swatch: "#1565c0",
    needsAvatar: false,
    available: true,
    requestPreview: async ({ video, params, onProgress }) => {
      const { theme } = readIvpParams(params);
      return renderIvpPreview(video, { theme, onProgress });
    },
    rerollPreview: (jobId, frameIndex, params) => {
      const { theme } = readIvpParams(params);
      return rerollIvpPreview(jobId, { theme, frameIndex });
    },
    // The server persists the theme alongside the job, so commitIvpRender
    // with no overrides simply replays the previewed configuration.
    commitRender: (jobId) => commitIvpRender(jobId),
  },
  {
    id: "blur",
    label: "Blur faces",
    description: "Coming soon — classic face-blur filter.",
    swatch: "#6b7280",
    needsAvatar: false,
    available: false,
    requestPreview: unavailable,
    rerollPreview: unavailable,
    commitRender: unavailable,
  },
  {
    id: "mosaic",
    label: "Mosaic",
    description: "Coming soon — pixelate faces.",
    swatch: "#13a36b",
    needsAvatar: false,
    available: false,
    requestPreview: unavailable,
    rerollPreview: unavailable,
    commitRender: unavailable,
  },
];

export function findFilter(id: string): Filter | null {
  return FILTERS.find((f) => f.id === id) ?? null;
}
