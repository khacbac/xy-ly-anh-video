import {
  Dimensions,
  FlatList,
  Image,
  Modal,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';

import { AVATAR_PRESETS, type AvatarPreset } from '../filters/avatars';

type Props = {
  visible: boolean;
  selectedId: string | null;
  onClose: () => void;
  onSelect: (preset: AvatarPreset) => void;
};

const COLS = 3;
const GAP = 12;

export function AvatarPickerSheet({ visible, selectedId, onClose, onSelect }: Props) {
  const { width } = Dimensions.get('window');
  const sidePad = 20;
  const cellSize = (width - sidePad * 2 - GAP * (COLS - 1)) / COLS;

  return (
    <Modal
      visible={visible}
      animationType="slide"
      transparent
      onRequestClose={onClose}
      statusBarTranslucent
    >
      <Pressable style={styles.backdrop} onPress={onClose}>
        <Pressable style={styles.sheet} onPress={() => undefined}>
          <View style={styles.handle} />
          <Text style={styles.title}>Pick avatar</Text>
          <Text style={styles.subtitle}>
            Faces in your video will be replaced with this image.
          </Text>
          <FlatList
            data={AVATAR_PRESETS}
            keyExtractor={(it) => it.id}
            numColumns={COLS}
            columnWrapperStyle={{ gap: GAP }}
            contentContainerStyle={{ gap: GAP, paddingHorizontal: sidePad, paddingTop: 8 }}
            renderItem={({ item }) => {
              const active = item.id === selectedId;
              return (
                <Pressable
                  onPress={() => {
                    onSelect(item);
                    onClose();
                  }}
                  style={[
                    styles.cell,
                    { width: cellSize, height: cellSize },
                    active && styles.cellActive,
                  ]}
                >
                  <Image source={item.source} style={styles.thumb} resizeMode="cover" />
                  <View style={styles.cellFooter}>
                    <Text style={styles.cellLabel} numberOfLines={1}>
                      {item.label}
                    </Text>
                  </View>
                </Pressable>
              );
            }}
            ListFooterComponent={
              <Text style={styles.footerHint}>
                Drop new images into app/assets/avatars and register them in
                src/filters/avatars.ts
              </Text>
            }
          />
        </Pressable>
      </Pressable>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.4)',
    justifyContent: 'flex-end',
  },
  sheet: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 18,
    borderTopRightRadius: 18,
    paddingBottom: 24,
    maxHeight: '75%',
  },
  handle: {
    width: 40,
    height: 4,
    borderRadius: 2,
    backgroundColor: '#d1d5db',
    alignSelf: 'center',
    marginTop: 10,
  },
  title: {
    fontSize: 17,
    fontWeight: '700',
    color: '#111',
    paddingHorizontal: 20,
    marginTop: 12,
  },
  subtitle: {
    fontSize: 13,
    color: '#555',
    paddingHorizontal: 20,
    marginTop: 4,
    marginBottom: 8,
  },
  cell: {
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: '#f3f4f6',
    borderWidth: 2,
    borderColor: 'transparent',
  },
  cellActive: { borderColor: '#1565c0' },
  thumb: { width: '100%', height: '100%' },
  cellFooter: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    paddingVertical: 4,
    paddingHorizontal: 6,
    backgroundColor: 'rgba(0,0,0,0.45)',
  },
  cellLabel: { color: '#fff', fontSize: 11, fontWeight: '600' },
  footerHint: {
    fontSize: 11,
    color: '#888',
    textAlign: 'center',
    paddingHorizontal: 20,
    marginTop: 14,
  },
});
