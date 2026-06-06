import AsyncStorage from '@react-native-async-storage/async-storage';
import { useCallback, useEffect, useRef, useState } from 'react';

const STORAGE_KEY = '@avatarshield/profile/v2';

export type Profile = {
  displayName: string;
  /** Id of the chosen preset avatar from src/filters/avatars.ts */
  avatarId: string | null;
  bio: string;
};

const DEFAULT: Profile = {
  displayName: 'me',
  avatarId: 'sample',
  bio: 'AvatarShield user',
};

let memoryCache: Profile | null = null;
const listeners = new Set<(p: Profile) => void>();

async function load(): Promise<Profile> {
  if (memoryCache) return memoryCache;
  try {
    const raw = await AsyncStorage.getItem(STORAGE_KEY);
    if (!raw) {
      memoryCache = DEFAULT;
      return memoryCache;
    }
    memoryCache = { ...DEFAULT, ...(JSON.parse(raw) as Partial<Profile>) };
    return memoryCache;
  } catch {
    memoryCache = DEFAULT;
    return memoryCache;
  }
}

async function persist(next: Profile): Promise<void> {
  memoryCache = next;
  listeners.forEach((fn) => fn(next));
  try {
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    // ignore
  }
}

export function useProfile() {
  const [profile, setProfile] = useState<Profile>(memoryCache ?? DEFAULT);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  useEffect(() => {
    const listener = (next: Profile) => {
      if (mounted.current) setProfile(next);
    };
    listeners.add(listener);
    if (memoryCache === null) {
      load().then((loaded) => {
        if (mounted.current) setProfile(loaded);
      });
    } else {
      setProfile(memoryCache);
    }
    return () => {
      listeners.delete(listener);
    };
  }, []);

  const update = useCallback(async (patch: Partial<Profile>) => {
    const current = memoryCache ?? (await load());
    await persist({ ...current, ...patch });
  }, []);

  return { profile, update };
}
