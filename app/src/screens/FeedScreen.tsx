import { useEvent } from 'expo';
import { useVideoPlayer, VideoView, type VideoPlayer } from 'expo-video';
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Dimensions,
  FlatList,
  Image,
  Pressable,
  StyleSheet,
  Text,
  View,
  type ViewToken,
} from 'react-native';

import { findAvatar } from '../filters/avatars';
import { useFeed, type FeedItem } from '../store/feed';

type Props = {
  headerInset: number;
  tabBarHeight: number;
  onGoCreate: () => void;
};

export function FeedScreen({ headerInset, tabBarHeight, onGoCreate }: Props) {
  const { items, ready, toggleLike } = useFeed();
  const { height: screenHeight } = Dimensions.get('window');
  const itemHeight = screenHeight;
  const overlayBottom = tabBarHeight + 16;
  const [activeId, setActiveId] = useState<string | null>(items[0]?.id ?? null);
  const [muted, setMuted] = useState<boolean>(false);

  useEffect(() => {
    if (!activeId && items[0]) setActiveId(items[0].id);
  }, [items, activeId]);

  const viewabilityConfig = useRef({ itemVisiblePercentThreshold: 75 }).current;
  const onViewable = useRef(
    ({ viewableItems }: { viewableItems: ViewToken[] }) => {
      const first = viewableItems[0];
      if (first?.item) setActiveId((first.item as FeedItem).id);
    },
  ).current;

  const renderItem = useCallback(
    ({ item }: { item: FeedItem }) => (
      <FeedVideo
        item={item}
        height={itemHeight}
        overlayBottom={overlayBottom}
        active={item.id === activeId}
        muted={muted}
        onToggleMute={() => setMuted((m) => !m)}
        onToggleLike={() => toggleLike(item.id)}
      />
    ),
    [activeId, itemHeight, overlayBottom, muted, toggleLike],
  );

  if (!ready) {
    return (
      <View style={[styles.fill, styles.center, { paddingTop: headerInset }]}>
        <ActivityIndicator color="#fff" />
      </View>
    );
  }

  if (items.length === 0) {
    return (
      <View style={[styles.fill, styles.center, { paddingTop: headerInset }]}>
        <Text style={styles.emptyTitle}>No videos yet</Text>
        <Text style={styles.emptyBody}>Tap + to render your first AI clip.</Text>
        <Pressable style={styles.emptyBtn} onPress={onGoCreate}>
          <Text style={styles.emptyBtnText}>Create video</Text>
        </Pressable>
      </View>
    );
  }

  return (
    <View style={styles.fill}>
      <FlatList
        data={items}
        keyExtractor={keyExtractor}
        renderItem={renderItem}
        pagingEnabled
        showsVerticalScrollIndicator={false}
        snapToInterval={itemHeight}
        snapToAlignment="start"
        decelerationRate="fast"
        getItemLayout={getItemLayoutFor(itemHeight)}
        viewabilityConfig={viewabilityConfig}
        onViewableItemsChanged={onViewable}
        initialNumToRender={2}
        maxToRenderPerBatch={3}
        windowSize={3}
        removeClippedSubviews
      />
      <View pointerEvents="none" style={[styles.topGradient, { height: headerInset + 60 }]} />
      <View pointerEvents="box-none" style={[styles.header, { top: headerInset + 12 }]}>
        <Text style={styles.brand}>AvatarShield</Text>
      </View>
    </View>
  );
}

function keyExtractor(item: FeedItem) {
  return item.id;
}

function getItemLayoutFor(itemHeight: number) {
  return (_: ArrayLike<FeedItem> | null | undefined, index: number) => ({
    length: itemHeight,
    offset: itemHeight * index,
    index,
  });
}

type VideoProps = {
  item: FeedItem;
  height: number;
  overlayBottom: number;
  active: boolean;
  muted: boolean;
  onToggleMute: () => void;
  onToggleLike: () => void;
};

function FeedVideo({
  item,
  height,
  overlayBottom,
  active,
  muted,
  onToggleMute,
  onToggleLike,
}: VideoProps) {
  const player = useVideoPlayer(item.outputUrl, (p) => {
    p.loop = true;
    p.muted = muted;
    p.play();
  });

  useEffect(() => {
    player.muted = muted;
  }, [player, muted]);

  useEffect(() => {
    if (active) {
      player.play();
    } else {
      player.pause();
    }
  }, [active, player]);

  const status = usePlayerStatus(player);
  const buffering = status === 'loading';

  return (
    <View style={[styles.itemWrap, { height }]}>
      <Pressable style={styles.fill} onPress={onToggleMute}>
        <VideoView
          style={styles.fill}
          player={player}
          contentFit="cover"
          nativeControls={false}
        />
      </Pressable>
      {buffering && (
        <View pointerEvents="none" style={styles.buffer}>
          <ActivityIndicator color="#fff" />
        </View>
      )}
      <View
        pointerEvents="none"
        style={[styles.bottomGradient, { bottom: 0, height: overlayBottom + 180 }]}
      />
      <View style={[styles.overlay, { bottom: overlayBottom }]} pointerEvents="box-none">
        <View style={styles.overlayLeft} pointerEvents="box-none">
          <View style={styles.authorRow}>
            <Avatar avatarId={item.avatarId} size={36} />
            <Text style={styles.author}>@{item.authorName}</Text>
          </View>
          {!!item.caption && (
            <Text style={styles.caption} numberOfLines={3}>
              {item.caption}
            </Text>
          )}
          {!!item.frameCount && !!item.fps && (
            <Text style={styles.meta}>
              {item.frameCount} frames · {item.fps.toFixed(1)} fps
              {item.durationS ? ` · ${item.durationS.toFixed(1)}s` : ''}
            </Text>
          )}
        </View>
        <View style={styles.overlayRight}>
          <Pressable onPress={onToggleLike} style={styles.action} hitSlop={8}>
            <Text style={[styles.actionGlyph, item.liked && styles.actionGlyphActive]}>
              {item.liked ? '♥' : '♡'}
            </Text>
            <Text style={styles.actionLabel}>{item.likes}</Text>
          </Pressable>
          <Pressable onPress={onToggleMute} style={styles.action} hitSlop={8}>
            <Text style={styles.actionGlyph}>{muted ? '🔇' : '🔊'}</Text>
          </Pressable>
        </View>
      </View>
    </View>
  );
}

function Avatar({ avatarId, size }: { avatarId: string | null; size: number }) {
  const preset = findAvatar(avatarId);
  const style = {
    width: size,
    height: size,
    borderRadius: size / 2,
    backgroundColor: '#333',
    borderWidth: 1.5,
    borderColor: '#fff',
  } as const;
  if (preset) return <Image source={preset.source} style={style} />;
  return (
    <View style={[style, styles.center]}>
      <Text style={{ color: '#fff', fontWeight: '700' }}>?</Text>
    </View>
  );
}

type PlayerStatus = 'idle' | 'loading' | 'ready' | 'error';

function usePlayerStatus(player: VideoPlayer): PlayerStatus {
  const event = useEvent(player, 'statusChange', { status: player.status });
  const status = event?.status ?? player.status;
  switch (status) {
    case 'loading':
      return 'loading';
    case 'readyToPlay':
      return 'ready';
    case 'error':
      return 'error';
    default:
      return 'idle';
  }
}

const styles = StyleSheet.create({
  fill: { flex: 1, backgroundColor: '#000' },
  center: { alignItems: 'center', justifyContent: 'center' },
  itemWrap: { width: '100%', backgroundColor: '#000' },
  buffer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: 'center',
    justifyContent: 'center',
  },
  topGradient: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    backgroundColor: 'rgba(0,0,0,0.25)',
  },
  bottomGradient: {
    position: 'absolute',
    left: 0,
    right: 0,
    backgroundColor: 'rgba(0,0,0,0.35)',
  },
  header: {
    position: 'absolute',
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  brand: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
    letterSpacing: 0.5,
    textShadowColor: 'rgba(0,0,0,0.6)',
    textShadowRadius: 6,
  },
  overlay: {
    position: 'absolute',
    left: 0,
    right: 0,
    paddingHorizontal: 16,
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 12,
  },
  overlayLeft: { flex: 1, gap: 6 },
  overlayRight: { alignItems: 'center', gap: 18 },
  authorRow: { flexDirection: 'row', alignItems: 'center', gap: 10 },
  author: { color: '#fff', fontSize: 15, fontWeight: '700' },
  caption: { color: '#fff', fontSize: 14, lineHeight: 19 },
  meta: { color: 'rgba(255,255,255,0.7)', fontSize: 11 },
  action: { alignItems: 'center', gap: 4 },
  actionGlyph: { color: '#fff', fontSize: 32, lineHeight: 36 },
  actionGlyphActive: { color: '#ff4769' },
  actionLabel: { color: '#fff', fontSize: 12, fontWeight: '600' },
  emptyTitle: { color: '#fff', fontSize: 20, fontWeight: '700' },
  emptyBody: { color: 'rgba(255,255,255,0.7)', fontSize: 14, marginTop: 6 },
  emptyBtn: {
    marginTop: 18,
    backgroundColor: '#1565c0',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 999,
  },
  emptyBtnText: { color: '#fff', fontWeight: '600' },
});
