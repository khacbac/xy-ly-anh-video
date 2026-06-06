import { Pressable, StyleSheet, Text, View } from 'react-native';

export type TabKey = 'feed' | 'create' | 'profile';

type Props = {
  current: TabKey;
  onChange: (tab: TabKey) => void;
  bottomInset: number;
  dark: boolean;
};

const TABS: { key: TabKey; label: string; glyph: string }[] = [
  { key: 'feed', label: 'Feed', glyph: '⌂' },
  { key: 'create', label: 'Create', glyph: '+' },
  { key: 'profile', label: 'Me', glyph: '☻' },
];

export const TAB_BAR_HEIGHT = 56;

export function TabBar({ current, onChange, bottomInset, dark }: Props) {
  const bg = dark ? '#000' : '#fff';
  const border = dark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.06)';
  const activeColor = dark ? '#fff' : '#111';
  const inactiveColor = dark ? 'rgba(255,255,255,0.55)' : '#888';

  return (
    <View
      style={[
        styles.wrap,
        {
          backgroundColor: bg,
          borderTopColor: border,
          paddingBottom: bottomInset,
        },
      ]}
    >
      <View style={styles.row}>
        {TABS.map((t) => {
          const active = current === t.key;
          const isCreate = t.key === 'create';
          return (
            <Pressable
              key={t.key}
              onPress={() => onChange(t.key)}
              style={styles.tab}
              hitSlop={6}
            >
              {isCreate ? (
                <View style={styles.createBadge}>
                  <Text style={styles.createGlyph}>{t.glyph}</Text>
                </View>
              ) : (
                <Text
                  style={[
                    styles.glyph,
                    { color: active ? activeColor : inactiveColor },
                  ]}
                >
                  {t.glyph}
                </Text>
              )}
              {!isCreate && (
                <Text
                  style={[
                    styles.label,
                    { color: active ? activeColor : inactiveColor },
                  ]}
                >
                  {t.label}
                </Text>
              )}
            </Pressable>
          );
        })}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    borderTopWidth: StyleSheet.hairlineWidth,
  },
  row: {
    height: TAB_BAR_HEIGHT,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-around',
  },
  tab: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 2,
  },
  glyph: { fontSize: 22, lineHeight: 24 },
  label: { fontSize: 10, fontWeight: '600' },
  createBadge: {
    width: 44,
    height: 30,
    borderRadius: 8,
    backgroundColor: '#1565c0',
    alignItems: 'center',
    justifyContent: 'center',
  },
  createGlyph: { color: '#fff', fontSize: 22, lineHeight: 24, fontWeight: '700' },
});
