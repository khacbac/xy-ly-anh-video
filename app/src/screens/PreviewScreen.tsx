import { useEffect, useMemo, useRef, useState } from "react";
import {
  ActivityIndicator,
  Image,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";

import {
  friendlyApiError,
  type PreviewResponse,
  type RenderResponse,
} from "../api";
import type { AvatarPreset } from "../filters/avatars";
import type { Filter, FilterParams, IvpParams, VideoInput } from "../filters";
import { useFeed } from "../store/feed";
import { useProfile } from "../store/profile";

export type PreviewSession = {
  video: VideoInput;
  filter: Filter;
  avatar: AvatarPreset | null;
  caption: string;
  preview: PreviewResponse;
  ivpParams?: IvpParams;
};

type Props = {
  session: PreviewSession;
  headerInset: number;
  tabBarHeight: number;
  onBack: () => void;
  onPreviewUpdated: (next: PreviewResponse) => void;
  onPublished: (result: RenderResponse) => void;
};

type Phase = "idle" | "rendering" | "done";

function pickRandomFrame(count: number, exclude: number): number {
  if (count <= 1) return 0;
  let idx = Math.floor(Math.random() * count);
  if (idx === exclude) idx = (idx + 1) % count;
  return idx;
}

function rerollParamsFor(session: PreviewSession): FilterParams | undefined {
  if (session.filter.id === "avatarshield") {
    const theme = session.preview.theme ?? session.ivpParams?.theme;
    if (theme) return { theme };
  }
  return undefined;
}

export function PreviewScreen({
  session,
  headerInset,
  tabBarHeight,
  onBack,
  onPreviewUpdated,
  onPublished,
}: Props) {
  const { addItem } = useFeed();
  const { profile } = useProfile();
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [rerolling, setRerolling] = useState(false);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const { preview, filter, avatar, caption } = session;
  const busy = phase === "rendering";
  const canReroll = preview.frame_count > 1 && !busy && !rerolling;
  // The server overwrites preview.png in place, so the URL doesn't change
  // across rerolls. A timestamp query bypasses RN's image cache.
  const previewUri = useMemo(
    () => `${preview.preview_url}?f=${preview.frame_index}&t=${Date.now()}`,
    [preview.preview_url, preview.frame_index, preview.job_id],
  );

  async function onPressReroll() {
    if (!canReroll) return;
    setError(null);
    setRerolling(true);
    try {
      const frameIndex = pickRandomFrame(
        preview.frame_count,
        preview.frame_index,
      );
      const next = await filter.rerollPreview(
        preview.job_id,
        frameIndex,
        rerollParamsFor(session),
      );
      if (mounted.current) onPreviewUpdated(next);
    } catch (err) {
      if (mounted.current) setError(friendlyApiError(err));
    } finally {
      if (mounted.current) setRerolling(false);
    }
  }

  async function onContinue() {
    if (busy) return;
    setError(null);
    setPhase("rendering");
    try {
      const res = await filter.commitRender(preview.job_id);
      await addItem({
        outputUrl: res.output_url,
        avatarId: avatar?.id ?? null,
        filterId: filter.id,
        authorName: profile.displayName,
        caption: caption.trim(),
        durationS: res.duration_s,
        frameCount: res.frame_count,
        fps: res.fps,
      });
      if (mounted.current) setPhase("done");
      onPublished(res);
    } catch (err) {
      if (mounted.current) {
        setError(friendlyApiError(err));
        setPhase("idle");
      }
    }
  }

  return (
    <View style={styles.fill}>
      <Image
        source={{ uri: previewUri }}
        style={StyleSheet.absoluteFill}
        resizeMode="contain"
      />

      <Pressable
        onPress={onBack}
        hitSlop={12}
        style={[styles.backBtn, { top: headerInset + 12 }]}
        disabled={busy}
      >
        <Text style={styles.backBtnText}>‹</Text>
      </Pressable>

      <View
        pointerEvents="box-none"
        style={[styles.bottomBar, { bottom: tabBarHeight + 24 }]}
      >
        {error && (
          <View style={styles.errorBox}>
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}
        <Pressable
          onPress={onPressReroll}
          disabled={!canReroll}
          style={[styles.rerollBtn, !canReroll && styles.rerollBtnDisabled]}
        >
          {rerolling ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.rerollBtnText}>
              Shuffle frame · {preview.frame_index + 1}/{preview.frame_count}
            </Text>
          )}
        </Pressable>
        <Pressable
          onPress={onContinue}
          disabled={busy || rerolling}
          style={[
            styles.continueBtn,
            (busy || rerolling) && styles.continueBtnDisabled,
          ]}
        >
          {busy ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.continueBtnText}>Continue</Text>
          )}
        </Pressable>
      </View>

      {busy && (
        <View style={styles.renderOverlay} pointerEvents="none">
          <ActivityIndicator color="#fff" size="large" />
          <Text style={styles.renderOverlayText}>Rendering video…</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1, backgroundColor: "#000" },
  backBtn: {
    position: "absolute",
    left: 16,
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: "rgba(0,0,0,0.45)",
    alignItems: "center",
    justifyContent: "center",
  },
  backBtnText: {
    color: "#fff",
    fontSize: 24,
    fontWeight: "600",
    lineHeight: 26,
    marginTop: -2,
  },
  bottomBar: {
    position: "absolute",
    left: 24,
    right: 24,
    gap: 12,
  },
  continueBtn: {
    backgroundColor: "#1565c0",
    borderRadius: 28,
    paddingVertical: 16,
    alignItems: "center",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.35,
    shadowRadius: 12,
    elevation: 6,
  },
  continueBtnDisabled: { opacity: 0.7 },
  continueBtnText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "700",
    letterSpacing: 0.3,
  },
  rerollBtn: {
    backgroundColor: "rgba(0,0,0,0.55)",
    borderRadius: 22,
    paddingVertical: 12,
    alignItems: "center",
  },
  rerollBtnDisabled: { opacity: 0.55 },
  rerollBtnText: {
    color: "#fff",
    fontSize: 14,
    fontWeight: "600",
    letterSpacing: 0.3,
  },
  errorBox: {
    backgroundColor: "rgba(168,38,28,0.85)",
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  errorText: { color: "#fff", fontSize: 13 },
  renderOverlay: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(0,0,0,0.62)",
    alignItems: "center",
    justifyContent: "center",
    gap: 12,
    paddingHorizontal: 32,
  },
  renderOverlayText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "600",
  },
});
