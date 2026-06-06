import AsyncStorage from '@react-native-async-storage/async-storage';
import { useCallback, useEffect, useRef, useState } from 'react';

const STORAGE_KEY = '@avatarshield/feed/v4';

export type FeedItem = {
  id: string;
  outputUrl: string;
  avatarId: string | null;
  filterId: string | null;
  authorName: string;
  caption: string;
  createdAt: number;
  likes: number;
  liked: boolean;
  durationS?: number;
  frameCount?: number;
  fps?: number;
};

export type NewFeedInput = {
  outputUrl: string;
  avatarId: string | null;
  filterId: string;
  authorName: string;
  caption: string;
  durationS?: number;
  frameCount?: number;
  fps?: number;
};

function makeId(): string {
  return `feed-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

let memoryCache: FeedItem[] | null = null;
const listeners = new Set<(items: FeedItem[]) => void>();

async function load(): Promise<FeedItem[]> {
  if (memoryCache) return memoryCache;
  try {
    const raw = await AsyncStorage.getItem(STORAGE_KEY);
    if (!raw) {
      memoryCache = [];
      return memoryCache;
    }
    const parsed = JSON.parse(raw) as FeedItem[];
    memoryCache = Array.isArray(parsed) ? parsed : [];
    return memoryCache;
  } catch {
    memoryCache = [];
    return memoryCache;
  }
}

async function persist(items: FeedItem[]): Promise<void> {
  memoryCache = items;
  listeners.forEach((fn) => fn(items));
  try {
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(items));
  } catch {
    // best-effort; the in-memory cache stays authoritative for this session.
  }
}

export function useFeed() {
  const [items, setItems] = useState<FeedItem[]>(memoryCache ?? []);
  const [ready, setReady] = useState<boolean>(memoryCache !== null);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  useEffect(() => {
    const listener = (next: FeedItem[]) => {
      if (mounted.current) setItems(next);
    };
    listeners.add(listener);
    if (memoryCache === null) {
      load().then((loaded) => {
        if (!mounted.current) return;
        setItems(loaded);
        setReady(true);
      });
    } else {
      setItems(memoryCache);
      setReady(true);
    }
    return () => {
      listeners.delete(listener);
    };
  }, []);

  const addItem = useCallback(async (input: NewFeedInput) => {
    const current = memoryCache ?? (await load());
    const item: FeedItem = {
      id: makeId(),
      outputUrl: input.outputUrl,
      avatarId: input.avatarId,
      filterId: input.filterId,
      authorName: input.authorName,
      caption: input.caption,
      createdAt: Date.now(),
      likes: 0,
      liked: false,
      durationS: input.durationS,
      frameCount: input.frameCount,
      fps: input.fps,
    };
    await persist([item, ...current]);
    return item;
  }, []);

  const toggleLike = useCallback(async (id: string) => {
    const current = memoryCache ?? (await load());
    const next = current.map((it) =>
      it.id === id
        ? { ...it, liked: !it.liked, likes: it.likes + (it.liked ? -1 : 1) }
        : it,
    );
    await persist(next);
  }, []);

  const removeItem = useCallback(async (id: string) => {
    const current = memoryCache ?? (await load());
    await persist(current.filter((it) => it.id !== id));
  }, []);

  return { items, ready, addItem, toggleLike, removeItem };
}
