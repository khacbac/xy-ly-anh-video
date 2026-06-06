export type AvatarPreset = {
  id: string;
  label: string;
  source: number;
};

/**
 * Drop a new image into `app/assets/avatars/` and append an entry here.
 * The id is what gets persisted with each rendered clip — used as the user's
 * profile picture in the feed and on previously rendered posts.
 */
export const AVATAR_PRESETS: AvatarPreset[] = [
  {
    id: 'sample',
    label: 'Sample',
    source: require('../../assets/avatars/sample.jpg'),
  },
  {
    id: 'jackie-west',
    label: 'Jackie West',
    source: require('../../assets/avatars/jackie-west.jpeg'),
  },
];

export function findAvatar(id: string | null | undefined): AvatarPreset | null {
  if (!id) return null;
  return AVATAR_PRESETS.find((a) => a.id === id) ?? null;
}
