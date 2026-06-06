const DEFAULT_BASE_URL = 'http://localhost:8000';

const envBase = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();

export const API_BASE_URL = envBase && envBase.length > 0 ? envBase : DEFAULT_BASE_URL;
