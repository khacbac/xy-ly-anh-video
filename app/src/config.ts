import Constants from 'expo-constants';
import { Platform } from 'react-native';

const DEFAULT_PORT = 8000;

function inferLanHost(): string | null {
  // Expo dev server hostUri looks like "192.168.1.10:8081" — reuse the IP so
  // a physical device on the same LAN can reach the API without manual config.
  const anyConstants = Constants as unknown as {
    expoConfig?: { hostUri?: string };
    manifest?: { hostUri?: string };
    expoGoConfig?: { hostUri?: string };
  };
  const hostUri =
    anyConstants.expoConfig?.hostUri ??
    anyConstants.manifest?.hostUri ??
    anyConstants.expoGoConfig?.hostUri ??
    null;
  if (typeof hostUri !== 'string') return null;
  const host = hostUri.split(':')[0];
  return host && host !== 'localhost' ? host : null;
}

function defaultBaseUrl(): string {
  const lan = inferLanHost();
  if (lan) return `http://${lan}:${DEFAULT_PORT}`;
  // Android emulator can't see host's "localhost" — 10.0.2.2 maps to it.
  if (Platform.OS === 'android') return `http://10.0.2.2:${DEFAULT_PORT}`;
  return `http://localhost:${DEFAULT_PORT}`;
}

const envBase = (process.env.EXPO_PUBLIC_API_BASE_URL as string | undefined)?.trim();

export const API_BASE_URL = envBase && envBase.length > 0 ? envBase : defaultBaseUrl();
