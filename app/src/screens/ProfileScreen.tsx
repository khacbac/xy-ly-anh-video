import { useVideoPlayer, VideoView } from 'expo-video';
import { useMemo, useState } from 'react';
import {
  Alert,
  Dimensions,
  FlatList,
  Image,
  Modal,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import { AvatarPickerSheet } from '../components/AvatarPickerSheet';
import { findAvatar, type AvatarPreset } from '../filters/avatars';
import { findFilter } from '../filters';
import { useFeed, type FeedItem } from '../store/feed';
import { useProfile } from '../store/profile';

type Props = {
  headerInset: number;
  tabBarHeight: number;
  onGoCreate: () => void;
};

const GRID_COLS = 3;
const GAP = 2;
const CELL_ASPECT = 16 / 11; // portrait, slightly taller than 4:5

export function ProfileScreen({ headerInset, tabBarHeight, onGoCreate }: Props) {
  const { profile, update } = useProfile();
  const { items, removeItem } = useFeed();
  const [editing, setEditing] = useState(false);
  const [pickingAvatar, setPickingAvatar] = useState(false);
  const [draftName, setDraftName] = useState(profile.displayName);
  const [draftBio, setDraftBio] = useState(profile.bio);
  const [openItem, setOpenItem] = useState<FeedItem | null>(null);

  const profileAvatar = findAvatar(profile.avatarId);
  const mine = useMemo(
    () => items.filter((it) => it.authorName === profile.displayName),
    [items, profile.displayName],
  );

  const { width } = Dimensions.get('window');
  const cellSize = (width - GAP * (GRID_COLS - 1)) / GRID_COLS;

  async function onSelectAvatar(preset: AvatarPreset) {
    await update({ avatarId: preset.id });
  }

  function openEdit() {
    setDraftName(profile.displayName);
    setDraftBio(profile.bio);
    setEditing(true);
  }

  async function saveEdit() {
    const name = draftName.trim() || 'me';
    await update({ displayName: name, bio: draftBio.trim() });
    setEditing(false);
  }

  function confirmDelete(it: FeedItem) {
    Alert.alert('Delete video?', it.caption, [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          await removeItem(it.id);
          setOpenItem(null);
        },
      },
    ]);
  }

  return (
    <ScrollView
      style={styles.fill}
      contentContainerStyle={{
        paddingTop: headerInset + 16,
        paddingBottom: tabBarHeight + 32,
      }}
    >
      <View style={styles.header}>
        <Pressable onPress={() => setPickingAvatar(true)} hitSlop={8}>
          {profileAvatar ? (
            <Image source={profileAvatar.source} style={styles.avatar} />
          ) : (
            <View style={[styles.avatar, styles.avatarPlaceholder]}>
              <Text style={styles.avatarPlaceholderText}>+</Text>
            </View>
          )}
        </Pressable>
        <Text style={styles.displayName}>@{profile.displayName}</Text>
        {!!profile.bio && <Text style={styles.bio}>{profile.bio}</Text>}
        <View style={styles.statsRow}>
          <Stat label="videos" value={mine.length} />
          <Stat label="likes" value={mine.reduce((sum, it) => sum + it.likes, 0)} />
        </View>
        <View style={styles.actionsRow}>
          <Pressable style={styles.btnSecondary} onPress={openEdit}>
            <Text style={styles.btnSecondaryText}>Edit profile</Text>
          </Pressable>
          <Pressable style={styles.btnPrimary} onPress={onGoCreate}>
            <Text style={styles.btnPrimaryText}>+ New video</Text>
          </Pressable>
        </View>
      </View>

      <View style={styles.gridHeader}>
        <Text style={styles.gridTitle}>My videos</Text>
      </View>

      {mine.length === 0 ? (
        <View style={styles.emptyGrid}>
          <Text style={styles.emptyTitle}>No videos yet</Text>
          <Text style={styles.emptyBody}>Tap "+ New video" to create one.</Text>
        </View>
      ) : (
        <FlatList
          data={mine}
          keyExtractor={(it) => it.id}
          numColumns={GRID_COLS}
          scrollEnabled={false}
          columnWrapperStyle={{ gap: GAP }}
          contentContainerStyle={{ gap: GAP, paddingHorizontal: 0 }}
          renderItem={({ item }) => (
            <GridCell
              item={item}
              width={cellSize}
              height={cellSize * CELL_ASPECT}
              onPress={() => setOpenItem(item)}
              onLongPress={() => confirmDelete(item)}
            />
          )}
        />
      )}

      <Modal
        visible={editing}
        transparent
        animationType="fade"
        onRequestClose={() => setEditing(false)}
      >
        <Pressable style={styles.modalBackdrop} onPress={() => setEditing(false)}>
          <Pressable style={styles.modalCard} onPress={() => undefined}>
            <Text style={styles.modalTitle}>Edit profile</Text>
            <Text style={styles.modalLabel}>Display name</Text>
            <TextInput
              value={draftName}
              onChangeText={setDraftName}
              style={styles.modalInput}
              maxLength={24}
              autoCapitalize="none"
            />
            <Text style={styles.modalLabel}>Bio</Text>
            <TextInput
              value={draftBio}
              onChangeText={setDraftBio}
              style={[styles.modalInput, styles.modalTextarea]}
              maxLength={80}
              multiline
            />
            <View style={styles.modalActions}>
              <Pressable style={styles.btnSecondary} onPress={() => setEditing(false)}>
                <Text style={styles.btnSecondaryText}>Cancel</Text>
              </Pressable>
              <Pressable style={styles.btnPrimary} onPress={saveEdit}>
                <Text style={styles.btnPrimaryText}>Save</Text>
              </Pressable>
            </View>
          </Pressable>
        </Pressable>
      </Modal>

      <Modal
        visible={!!openItem}
        animationType="slide"
        onRequestClose={() => setOpenItem(null)}
      >
        {openItem && (
          <ItemModal
            item={openItem}
            onClose={() => setOpenItem(null)}
            onDelete={() => confirmDelete(openItem)}
          />
        )}
      </Modal>

      <AvatarPickerSheet
        visible={pickingAvatar}
        selectedId={profile.avatarId}
        onClose={() => setPickingAvatar(false)}
        onSelect={onSelectAvatar}
      />
    </ScrollView>
  );
}

function Stat({ label, value }: { label: string; value: number }) {
  return (
    <View style={styles.stat}>
      <Text style={styles.statValue}>{value}</Text>
      <Text style={styles.statLabel}>{label}</Text>
    </View>
  );
}

function GridCell({
  item,
  width,
  height,
  onPress,
  onLongPress,
}: {
  item: FeedItem;
  width: number;
  height: number;
  onPress: () => void;
  onLongPress: () => void;
}) {
  const itemAvatar = findAvatar(item.avatarId);
  const filter = item.filterId ? findFilter(item.filterId) : null;
  const player = useVideoPlayer(item.outputUrl, (p) => {
    p.muted = true;
    p.loop = false;
    p.pause();
  });
  const duration = formatDuration(item.durationS);

  return (
    <Pressable
      onPress={onPress}
      onLongPress={onLongPress}
      style={[styles.gridCell, { width, height }]}
    >
      {itemAvatar ? (
        <Image
          source={itemAvatar.source}
          style={StyleSheet.absoluteFill}
          resizeMode="cover"
          blurRadius={8}
        />
      ) : null}
      <VideoView
        style={StyleSheet.absoluteFill}
        player={player}
        contentFit="cover"
        nativeControls={false}
        pointerEvents="none"
      />
      <View style={styles.gridBottomGradient} pointerEvents="none" />
      <View style={styles.gridGlyphWrap} pointerEvents="none">
        <View style={styles.gridGlyphBubble}>
          <Text style={styles.gridGlyph}>▶</Text>
        </View>
      </View>
      {filter && (
        <View style={styles.gridFilterTag}>
          <Text style={styles.gridFilterTagText}>{filter.label}</Text>
        </View>
      )}
      {duration && (
        <View style={styles.gridDurationTag}>
          <Text style={styles.gridDurationTagText}>{duration}</Text>
        </View>
      )}
      <View style={styles.gridOverlay}>
        <Text style={styles.gridLikes}>♥ {item.likes}</Text>
      </View>
    </Pressable>
  );
}

function formatDuration(seconds?: number): string | null {
  if (!seconds || !Number.isFinite(seconds)) return null;
  const total = Math.max(0, Math.round(seconds));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function ItemModal({
  item,
  onClose,
  onDelete,
}: {
  item: FeedItem;
  onClose: () => void;
  onDelete: () => void;
}) {
  const player = useVideoPlayer(item.outputUrl, (p) => {
    p.loop = true;
    p.play();
  });
  return (
    <View style={styles.itemModal}>
      <VideoView
        style={StyleSheet.absoluteFill}
        player={player}
        contentFit="contain"
        nativeControls
      />
      <View style={styles.itemModalTop}>
        <Pressable onPress={onClose} hitSlop={12}>
          <Text style={styles.itemModalBtn}>Close</Text>
        </Pressable>
        <Pressable onPress={onDelete} hitSlop={12}>
          <Text style={[styles.itemModalBtn, { color: '#ff5a5f' }]}>Delete</Text>
        </Pressable>
      </View>
      <View style={styles.itemModalBottom}>
        <Text style={styles.itemModalCaption}>{item.caption}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  fill: { flex: 1, backgroundColor: '#fff' },
  header: { alignItems: 'center', paddingHorizontal: 20, gap: 8 },
  avatar: {
    width: 96,
    height: 96,
    borderRadius: 48,
    backgroundColor: '#eee',
  },
  avatarPlaceholder: { alignItems: 'center', justifyContent: 'center' },
  avatarPlaceholderText: { fontSize: 36, color: '#999' },
  displayName: { fontSize: 18, fontWeight: '700', color: '#111', marginTop: 8 },
  bio: { fontSize: 13, color: '#555', textAlign: 'center', maxWidth: 280 },
  statsRow: { flexDirection: 'row', gap: 24, marginTop: 12 },
  stat: { alignItems: 'center' },
  statValue: { fontSize: 17, fontWeight: '700', color: '#111' },
  statLabel: { fontSize: 12, color: '#777', marginTop: 2 },
  actionsRow: { flexDirection: 'row', gap: 10, marginTop: 14 },
  btnSecondary: {
    paddingHorizontal: 14,
    paddingVertical: 10,
    backgroundColor: '#eef0f3',
    borderRadius: 10,
  },
  btnSecondaryText: { color: '#222', fontWeight: '600' },
  btnPrimary: {
    paddingHorizontal: 14,
    paddingVertical: 10,
    backgroundColor: '#1565c0',
    borderRadius: 10,
  },
  btnPrimaryText: { color: '#fff', fontWeight: '600' },
  gridHeader: { paddingHorizontal: 20, paddingTop: 24, paddingBottom: 12 },
  gridTitle: { fontSize: 15, fontWeight: '700', color: '#222' },
  emptyGrid: { paddingVertical: 40, alignItems: 'center', gap: 6 },
  emptyTitle: { fontSize: 15, fontWeight: '600', color: '#333' },
  emptyBody: { fontSize: 13, color: '#777' },
  gridCell: {
    backgroundColor: '#111',
    borderRadius: 6,
    overflow: 'hidden',
  },
  gridBottomGradient: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    height: 48,
    backgroundColor: 'rgba(0,0,0,0.45)',
  },
  gridOverlay: {
    position: 'absolute',
    left: 6,
    right: 6,
    bottom: 6,
    flexDirection: 'row',
    alignItems: 'center',
  },
  gridLikes: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '700',
    textShadowColor: 'rgba(0,0,0,0.6)',
    textShadowRadius: 3,
  },
  gridFilterTag: {
    position: 'absolute',
    top: 6,
    left: 6,
    paddingHorizontal: 6,
    paddingVertical: 2,
    backgroundColor: 'rgba(0,0,0,0.55)',
    borderRadius: 6,
  },
  gridFilterTagText: { color: '#fff', fontSize: 10, fontWeight: '600' },
  gridDurationTag: {
    position: 'absolute',
    top: 6,
    right: 6,
    paddingHorizontal: 6,
    paddingVertical: 2,
    backgroundColor: 'rgba(0,0,0,0.55)',
    borderRadius: 6,
  },
  gridDurationTagText: { color: '#fff', fontSize: 10, fontWeight: '600' },
  gridGlyphWrap: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: 'center',
    justifyContent: 'center',
  },
  gridGlyphBubble: {
    width: 34,
    height: 34,
    borderRadius: 17,
    backgroundColor: 'rgba(0,0,0,0.45)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  gridGlyph: {
    color: '#fff',
    fontSize: 16,
    marginLeft: 2,
  },
  modalBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  modalCard: {
    width: '100%',
    backgroundColor: '#fff',
    borderRadius: 14,
    padding: 18,
    gap: 8,
  },
  modalTitle: { fontSize: 17, fontWeight: '700', color: '#111', marginBottom: 4 },
  modalLabel: { fontSize: 12, color: '#666', marginTop: 6 },
  modalInput: {
    backgroundColor: '#f3f4f6',
    borderRadius: 8,
    paddingHorizontal: 10,
    paddingVertical: 8,
    fontSize: 14,
    color: '#111',
  },
  modalTextarea: { minHeight: 60, textAlignVertical: 'top' },
  modalActions: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    gap: 8,
    marginTop: 12,
  },
  itemModal: { flex: 1, backgroundColor: '#000' },
  itemModalTop: {
    position: 'absolute',
    top: 50,
    left: 16,
    right: 16,
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  itemModalBtn: { color: '#fff', fontSize: 15, fontWeight: '600' },
  itemModalBottom: {
    position: 'absolute',
    left: 16,
    right: 16,
    bottom: 40,
  },
  itemModalCaption: { color: '#fff', fontSize: 14, lineHeight: 19 },
});
