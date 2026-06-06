import * as ImagePicker from 'expo-image-picker';
import { StatusBar } from 'expo-status-bar';
import { useState } from 'react';
import { StyleSheet, View } from 'react-native';
import {
  SafeAreaProvider,
  useSafeAreaInsets,
} from 'react-native-safe-area-context';

import { TAB_BAR_HEIGHT, TabBar, type TabKey } from './src/components/TabBar';
import type { PreviewResponse } from './src/api';
import type { VideoInput } from './src/filters';
import { EditorScreen } from './src/screens/EditorScreen';
import { FeedScreen } from './src/screens/FeedScreen';
import { PreviewScreen, type PreviewSession } from './src/screens/PreviewScreen';
import { ProfileScreen } from './src/screens/ProfileScreen';

export default function App() {
  return (
    <SafeAreaProvider>
      <Root />
    </SafeAreaProvider>
  );
}

function Root() {
  const insets = useSafeAreaInsets();
  const [tab, setTab] = useState<TabKey>('feed');
  const [editorVideo, setEditorVideo] = useState<VideoInput | null>(null);
  const [previewSession, setPreviewSession] = useState<PreviewSession | null>(null);
  const inPreview = !!previewSession;
  const inEditor = !!editorVideo && !inPreview;
  const dark = tab === 'feed' || inEditor || inPreview;
  const tabBarHeight = TAB_BAR_HEIGHT + insets.bottom;
  const fullscreenInset = insets.bottom;

  async function pickVideoFromLibrary() {
    const res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['videos'],
      allowsEditing: false,
      quality: 1,
    });
    if (res.canceled || !res.assets[0]) return;
    const asset = res.assets[0];
    setPreviewSession(null);
    setEditorVideo({
      uri: asset.uri,
      fileName: asset.fileName,
      mimeType: asset.mimeType,
    });
  }

  function goFeed() {
    setPreviewSession(null);
    setEditorVideo(null);
    setTab('feed');
  }

  function onPreviewReady(session: PreviewSession) {
    setPreviewSession(session);
  }

  function onPreviewBack() {
    setPreviewSession(null);
  }

  function onPreviewUpdated(next: PreviewResponse) {
    setPreviewSession((prev) => (prev ? { ...prev, preview: next } : prev));
  }

  function onTabChange(next: TabKey) {
    if (next === 'create') {
      void pickVideoFromLibrary();
      return;
    }
    setTab(next);
  }

  return (
    <View style={[styles.root, { backgroundColor: dark ? '#000' : '#f5f5f7' }]}>
      <StatusBar style={dark ? 'light' : 'dark'} />

      <View style={styles.tabStack}>
        <View style={styles.screenWrap}>
          {tab === 'feed' && (
            <FeedScreen
              headerInset={insets.top}
              tabBarHeight={tabBarHeight}
              onGoCreate={pickVideoFromLibrary}
            />
          )}
          {tab === 'profile' && (
            <ProfileScreen
              headerInset={insets.top}
              tabBarHeight={tabBarHeight}
              onGoCreate={pickVideoFromLibrary}
            />
          )}
        </View>
        <TabBar
          current={tab}
          onChange={onTabChange}
          bottomInset={insets.bottom}
          dark={dark}
        />
      </View>

      {inEditor && editorVideo && (
        <View style={styles.fullscreenLayer}>
          <EditorScreen
            video={editorVideo}
            headerInset={insets.top}
            tabBarHeight={fullscreenInset}
            onBack={goFeed}
            onPreviewReady={onPreviewReady}
          />
        </View>
      )}

      {inPreview && previewSession && (
        <View style={styles.fullscreenLayer}>
          <PreviewScreen
            session={previewSession}
            headerInset={insets.top}
            tabBarHeight={fullscreenInset}
            onBack={onPreviewBack}
            onPreviewUpdated={onPreviewUpdated}
            onPublished={goFeed}
          />
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
  tabStack: { flex: 1 },
  screenWrap: { flex: 1 },
  fullscreenLayer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: '#000',
  },
});
