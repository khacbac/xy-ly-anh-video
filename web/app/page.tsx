'use client';

import { useEffect, useRef, useState } from 'react';
import { renderAiVideo, type RenderProgress, type RenderResponse } from '../lib/api';
import { API_BASE_URL } from '../lib/config';
import styles from './page.module.css';

export default function Page() {
  const [video, setVideo] = useState<File | null>(null);
  const [avatar, setAvatar] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [progress, setProgress] = useState<RenderProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RenderResponse | null>(null);

  const videoUrl = useObjectUrl(video);
  const avatarUrl = useObjectUrl(avatar);

  function onPickVideo(e: React.ChangeEvent<HTMLInputElement>) {
    setError(null);
    const file = e.target.files?.[0] ?? null;
    setVideo(file);
  }

  function onPickAvatar(e: React.ChangeEvent<HTMLInputElement>) {
    setError(null);
    const file = e.target.files?.[0] ?? null;
    setAvatar(file);
  }

  async function submit() {
    if (!video || !avatar || submitting) return;
    setSubmitting(true);
    setError(null);
    setResult(null);
    setProgress({ uploadProgress: 0, phase: 'uploading' });
    try {
      const res = await renderAiVideo(video, avatar, setProgress);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
      setProgress(null);
    }
  }

  const canSubmit = !!video && !!avatar && !submitting;
  const resultVideoUrl = result
    ? result.output_url.startsWith('http')
      ? result.output_url
      : `${API_BASE_URL}${result.output_url}`
    : null;

  return (
    <main className={styles.container}>
      <h1 className={styles.title}>AvatarShield</h1>
      <p className={styles.subtitle}>
        Pick a source video + an avatar image, then generate an AI-anonymized video.
      </p>
      <p className={styles.apiNote}>API: {API_BASE_URL}</p>

      <Card title="1 · Source video">
        <label className={styles.pickBtn}>
          <span>{video ? 'Replace video' : 'Pick video'}</span>
          <input
            type="file"
            accept="video/*"
            onChange={onPickVideo}
            disabled={submitting}
            hidden
          />
        </label>
        {video && videoUrl && (
          <div className={styles.previewBox}>
            <video src={videoUrl} className={styles.videoPreview} controls muted loop />
            <span className={styles.meta}>{video.name}</span>
          </div>
        )}
      </Card>

      <Card title="2 · Avatar image">
        <label className={styles.pickBtn}>
          <span>{avatar ? 'Replace avatar' : 'Pick avatar'}</span>
          <input
            type="file"
            accept="image/*"
            onChange={onPickAvatar}
            disabled={submitting}
            hidden
          />
        </label>
        {avatar && avatarUrl && (
          <div className={styles.previewBox}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={avatarUrl} alt={avatar.name} className={styles.avatarPreview} />
            <span className={styles.meta}>{avatar.name}</span>
          </div>
        )}
      </Card>

      <button
        className={`${styles.submitBtn} ${!canSubmit ? styles.submitBtnDisabled : ''}`}
        onClick={submit}
        disabled={!canSubmit}
      >
        {submitting ? <span className={styles.spinner} aria-label="loading" /> : 'Generate AI video'}
      </button>

      {submitting && (
        <p className={styles.hint}>
          {progress?.phase === 'uploading'
            ? `Uploading… ${Math.round((progress.uploadProgress ?? 0) * 100)}%`
            : 'Rendering on the server. This can take 1–3 minutes depending on clip length.'}
        </p>
      )}

      {error && (
        <div className={styles.errorBox}>
          <span className={styles.errorText}>{error}</span>
        </div>
      )}

      {result && resultVideoUrl && (
        <Card title="3 · Result">
          <video src={resultVideoUrl} className={styles.resultVideo} controls loop />
          <span className={styles.meta}>
            {result.frame_count} frames @ {result.fps.toFixed(1)} fps ·{' '}
            {result.duration_s.toFixed(1)}s
          </span>
          <span className={styles.metaSmall}>{resultVideoUrl}</span>
        </Card>
      )}
    </main>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className={styles.card}>
      <h2 className={styles.cardTitle}>{title}</h2>
      {children}
    </section>
  );
}

function useObjectUrl(file: File | null): string | null {
  const [url, setUrl] = useState<string | null>(null);
  const lastRef = useRef<string | null>(null);
  useEffect(() => {
    if (lastRef.current) {
      URL.revokeObjectURL(lastRef.current);
      lastRef.current = null;
    }
    if (!file) {
      setUrl(null);
      return;
    }
    const next = URL.createObjectURL(file);
    lastRef.current = next;
    setUrl(next);
    return () => {
      URL.revokeObjectURL(next);
      if (lastRef.current === next) lastRef.current = null;
    };
  }, [file]);
  return url;
}
