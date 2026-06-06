import { useVideoPlayer, VideoView } from "expo-video";
import { useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Image,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";

import { fetchIvpThemes, friendlyApiError } from "../api";
import { AvatarPickerSheet } from "../components/AvatarPickerSheet";
import { findAvatar, type AvatarPreset } from "../filters/avatars";
import {
  FILTERS,
  IVP_DEFAULT_THEME,
  IVP_THEME_META,
  type Filter,
  type IvpParams,
  type VideoInput,
} from "../filters";
import { useProfile } from "../store/profile";
import type { PreviewSession } from "./PreviewScreen";

type Props = {
  video: VideoInput;
  initialFilterId?: string | null;
  initialAvatarId?: string | null;
  initialCaption?: string;
  headerInset: number;
  tabBarHeight: number;
  onBack: () => void;
  onPreviewReady: (session: PreviewSession) => void;
};

export function EditorScreen({
  video,
  initialFilterId,
  initialAvatarId,
  initialCaption,
  headerInset,
  tabBarHeight,
  onBack,
  onPreviewReady,
}: Props) {
  const { profile, update } = useProfile();
  const [selectedFilterId, setSelectedFilterId] = useState<string | null>(
    initialFilterId ?? FILTERS.find((f) => f.available)?.id ?? null,
  );
  const [selectedAvatar, setSelectedAvatar] = useState<AvatarPreset | null>(
    () => findAvatar(initialAvatarId ?? profile.avatarId),
  );
  const [pickingAvatar, setPickingAvatar] = useState(false);
  const [caption, setCaption] = useState(initialCaption ?? "");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ivpTheme, setIvpTheme] = useState<string>(IVP_DEFAULT_THEME);
  const [ivpThemes, setIvpThemes] = useState<string[]>(() =>
    Object.keys(IVP_THEME_META),
  );
  const pendingFilterRef = useRef<Filter | null>(null);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const themes = await fetchIvpThemes();
        if (cancelled) return;
        if (themes.length > 0) {
          setIvpThemes(themes);
          setIvpTheme((prev) => (themes.includes(prev) ? prev : themes[0]));
        }
      } catch {
        // Keep hardcoded fallbacks when the API is unreachable.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const player = useVideoPlayer(video.uri, (p) => {
    p.loop = true;
    p.muted = true;
    p.play();
  });

  useEffect(() => {
    return () => {
      try {
        player.pause();
      } catch {
        // player may already be released
      }
    };
  }, [player]);

  const selectedFilter = selectedFilterId
    ? (FILTERS.find((f) => f.id === selectedFilterId) ?? null)
    : null;
  const showIvpOptions = selectedFilter?.id === "avatarshield";

  function buildIvpParams(): IvpParams {
    return { theme: ivpTheme };
  }

  function onPressFilter(filter: Filter) {
    if (submitting) return;
    setError(null);
    if (!filter.available) {
      Alert.alert(filter.label, filter.description);
      return;
    }
    setSelectedFilterId(filter.id);
    if (filter.needsAvatar && !selectedAvatar) {
      pendingFilterRef.current = filter;
      setPickingAvatar(true);
    }
  }

  async function onSelectAvatar(preset: AvatarPreset) {
    setSelectedAvatar(preset);
    await update({ avatarId: preset.id });
  }

  async function onPressApply() {
    if (!selectedFilter || submitting) return;
    if (selectedFilter.needsAvatar && !selectedAvatar) {
      setPickingAvatar(true);
      return;
    }
    setError(null);
    setSubmitting(true);
    try {
      const ivpParams =
        selectedFilter.id === "avatarshield" ? buildIvpParams() : undefined;
      const preview = await selectedFilter.requestPreview({
        video,
        avatar: selectedAvatar,
        params: ivpParams,
      });
      onPreviewReady({
        video,
        filter: selectedFilter,
        avatar: selectedAvatar,
        caption,
        preview,
        ivpParams,
      });
    } catch (err) {
      setError(friendlyApiError(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <View style={styles.fill}>
      <View style={styles.videoLayer}>
        <VideoView
          style={StyleSheet.absoluteFill}
          player={player}
          contentFit="contain"
          nativeControls={false}
        />
      </View>

      <View
        pointerEvents="box-none"
        style={[styles.topBar, { top: headerInset + 6 }]}
      >
        <Pressable
          onPress={onBack}
          hitSlop={12}
          style={styles.iconBtn}
          disabled={submitting}
        >
          <Text style={styles.iconBtnText}>‹ Back</Text>
        </Pressable>
        <Text style={styles.topTitle}>Editor</Text>
        <View style={{ width: 60 }} />
      </View>

      <ScrollView
        style={[styles.controls, { bottom: tabBarHeight }]}
        contentContainerStyle={styles.controlsContent}
        showsVerticalScrollIndicator={false}
        keyboardShouldPersistTaps="handled"
        keyboardDismissMode="on-drag"
      >
        <Text style={styles.sectionLabel}>Filters</Text>
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.filterStrip}
        >
          {FILTERS.map((filter) => {
            const active = filter.id === selectedFilterId;
            return (
              <Pressable
                key={filter.id}
                onPress={() => onPressFilter(filter)}
                style={[styles.filterCell, active && styles.filterCellActive]}
              >
                <View
                  style={[
                    styles.filterSwatch,
                    { backgroundColor: filter.swatch },
                    !filter.available && styles.filterSwatchDisabled,
                  ]}
                >
                  {filter.needsAvatar && selectedAvatar && active ? (
                    <Image
                      source={selectedAvatar.source}
                      style={styles.filterAvatar}
                    />
                  ) : (
                    <Text style={styles.filterGlyph}>
                      {filter.available ? filter.label[0] : "…"}
                    </Text>
                  )}
                  {!filter.available && (
                    <View style={styles.lockBadge}>
                      <Text style={styles.lockText}>soon</Text>
                    </View>
                  )}
                </View>
                <Text
                  style={[
                    styles.filterLabel,
                    active && styles.filterLabelActive,
                  ]}
                  numberOfLines={1}
                >
                  {filter.label}
                </Text>
              </Pressable>
            );
          })}
        </ScrollView>

        {selectedFilter && (
          <View style={styles.detailCard}>
            <Text style={styles.detailTitle}>{selectedFilter.label}</Text>
            <Text style={styles.detailBody}>{selectedFilter.description}</Text>
            {selectedFilter.needsAvatar && (
              <Pressable
                style={styles.avatarRow}
                onPress={() => !submitting && setPickingAvatar(true)}
                disabled={submitting}
              >
                {selectedAvatar ? (
                  <Image
                    source={selectedAvatar.source}
                    style={styles.avatarThumb}
                  />
                ) : (
                  <View style={[styles.avatarThumb, styles.avatarPlaceholder]}>
                    <Text style={styles.avatarPlaceholderText}>?</Text>
                  </View>
                )}
                <View style={{ flex: 1 }}>
                  <Text style={styles.avatarRowLabel}>
                    {selectedAvatar ? selectedAvatar.label : "Pick avatar"}
                  </Text>
                  <Text style={styles.avatarRowHint}>
                    {selectedAvatar
                      ? "Tap to change"
                      : "Required by this filter"}
                  </Text>
                </View>
                <Text style={styles.avatarRowChevron}>›</Text>
              </Pressable>
            )}
          </View>
        )}

        {showIvpOptions && (
          <View style={styles.optionsBlock}>
            <Text style={styles.subsectionLabel}>Theme</Text>
            <ScrollView
              horizontal
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={styles.chipStrip}
            >
              {ivpThemes.map((id) => {
                const meta = IVP_THEME_META[id] ?? {
                  label: id.replace(/-/g, " "),
                  swatch: "#9ca3af",
                };
                const active = id === ivpTheme;
                return (
                  <Pressable
                    key={id}
                    onPress={() => !submitting && setIvpTheme(id)}
                    style={[
                      styles.chip,
                      active && {
                        borderColor: meta.swatch,
                        backgroundColor: "#fff",
                      },
                    ]}
                    disabled={submitting}
                  >
                    <View
                      style={[styles.chipDot, { backgroundColor: meta.swatch }]}
                    />
                    <Text
                      style={[
                        styles.chipLabel,
                        active && styles.chipLabelActive,
                      ]}
                    >
                      {meta.label}
                    </Text>
                  </Pressable>
                );
              })}
            </ScrollView>
          </View>
        )}

        <Text style={styles.sectionLabel}>Caption</Text>
        <TextInput
          value={caption}
          onChangeText={setCaption}
          placeholder="Say something about your clip…"
          placeholderTextColor="#888"
          editable={!submitting}
          style={styles.captionInput}
          multiline
          maxLength={120}
        />

        {error && (
          <View style={styles.errorBox}>
            <Text style={styles.errorText}>{error}</Text>
          </View>
        )}

        <Pressable
          style={[
            styles.submitBtn,
            (!selectedFilter || !selectedFilter.available || submitting) &&
              styles.submitBtnDisabled,
          ]}
          onPress={onPressApply}
          disabled={!selectedFilter || !selectedFilter.available || submitting}
        >
          {submitting ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.submitBtnText}>
              {selectedFilter
                ? `Preview ${selectedFilter.label}`
                : "Pick a filter"}
            </Text>
          )}
        </Pressable>
      </ScrollView>

      <AvatarPickerSheet
        visible={pickingAvatar}
        selectedId={selectedAvatar?.id ?? null}
        onClose={() => {
          setPickingAvatar(false);
          pendingFilterRef.current = null;
        }}
        onSelect={onSelectAvatar}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1, backgroundColor: "#000" },
  videoLayer: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    height: "45%",
    backgroundColor: "#000",
  },
  topBar: {
    position: "absolute",
    left: 0,
    right: 0,
    paddingHorizontal: 12,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  iconBtn: {
    paddingVertical: 6,
    paddingHorizontal: 10,
    backgroundColor: "rgba(0,0,0,0.45)",
    borderRadius: 16,
  },
  iconBtnText: { color: "#fff", fontSize: 14, fontWeight: "600" },
  topTitle: {
    color: "#fff",
    fontSize: 15,
    fontWeight: "700",
    textShadowColor: "rgba(0,0,0,0.6)",
    textShadowRadius: 4,
  },
  controls: {
    position: "absolute",
    top: "45%",
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "#fff",
    borderTopLeftRadius: 18,
    borderTopRightRadius: 18,
  },
  controlsContent: {
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 24,
    gap: 14,
  },
  sectionLabel: {
    fontSize: 12,
    fontWeight: "700",
    color: "#666",
    letterSpacing: 0.5,
    textTransform: "uppercase",
  },
  subsectionLabel: {
    fontSize: 11,
    fontWeight: "700",
    color: "#888",
    letterSpacing: 0.4,
    textTransform: "uppercase",
    marginTop: 4,
  },
  filterStrip: { gap: 12, paddingRight: 12 },
  filterCell: { alignItems: "center", gap: 6, width: 76 },
  filterCellActive: {},
  filterSwatch: {
    width: 64,
    height: 64,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    overflow: "hidden",
    borderWidth: 2,
    borderColor: "transparent",
  },
  filterSwatchDisabled: { opacity: 0.55 },
  filterGlyph: { color: "#fff", fontSize: 24, fontWeight: "700" },
  filterAvatar: { width: "100%", height: "100%" },
  filterLabel: { color: "#444", fontSize: 12, fontWeight: "500" },
  filterLabelActive: { color: "#1565c0", fontWeight: "700" },
  lockBadge: {
    position: "absolute",
    top: 4,
    right: 4,
    paddingHorizontal: 5,
    paddingVertical: 1,
    backgroundColor: "rgba(0,0,0,0.55)",
    borderRadius: 6,
  },
  lockText: { color: "#fff", fontSize: 9, fontWeight: "700" },
  detailCard: {
    backgroundColor: "#f6f7f9",
    borderRadius: 12,
    padding: 14,
    gap: 8,
  },
  detailTitle: { fontSize: 15, fontWeight: "700", color: "#111" },
  detailBody: { fontSize: 13, color: "#555", lineHeight: 18 },
  optionsBlock: {
    gap: 10,
    backgroundColor: "#f6f7f9",
    borderRadius: 12,
    padding: 14,
  },
  chipStrip: { gap: 8, paddingRight: 8 },
  chip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 18,
    borderWidth: 2,
    borderColor: "#e5e7eb",
    backgroundColor: "#fff",
  },
  chipDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  chipLabel: { color: "#444", fontSize: 13, fontWeight: "600" },
  chipLabelActive: { color: "#111" },
  avatarRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    paddingVertical: 6,
    marginTop: 4,
  },
  avatarThumb: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: "#e5e7eb",
  },
  avatarPlaceholder: { alignItems: "center", justifyContent: "center" },
  avatarPlaceholderText: { color: "#9ca3af", fontSize: 18, fontWeight: "700" },
  avatarRowLabel: { color: "#111", fontSize: 14, fontWeight: "600" },
  avatarRowHint: { color: "#777", fontSize: 12, marginTop: 2 },
  avatarRowChevron: { color: "#999", fontSize: 22 },
  captionInput: {
    minHeight: 60,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 10,
    backgroundColor: "#f3f4f6",
    color: "#111",
    fontSize: 14,
    textAlignVertical: "top",
  },
  submitBtn: {
    backgroundColor: "#1565c0",
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: "center",
    marginTop: 4,
  },
  submitBtnDisabled: { backgroundColor: "#9bb4cf" },
  submitBtnText: { color: "#fff", fontSize: 15, fontWeight: "600" },
  errorBox: { backgroundColor: "#fdecea", borderRadius: 10, padding: 12 },
  errorText: { color: "#a8261c", fontSize: 13 },
});
