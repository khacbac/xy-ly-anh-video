# AvatarShield — Web client

Minimal Next.js (App Router) UI that calls the AvatarShield API.

## Run

```bash
# 1. Make sure the API is running (see ../api/README.md):
#    cd .. && python -m uvicorn api.server:app --host 0.0.0.0 --port 8000

# 2. Install + start
cd web
npm install
npm run dev
```

Open <http://localhost:3000>. The page POSTs `multipart/form-data` (video + avatar)
to `${NEXT_PUBLIC_API_BASE_URL:-http://localhost:8000}/render` and renders the result.

To point at a non-local API, copy `.env.local.example` to `.env.local` and set
`NEXT_PUBLIC_API_BASE_URL`.

## Notes

- Upload uses `XMLHttpRequest` (not `fetch`) so the UI can show real upload progress.
- The API's CORS allow-list is wide open (`allow_origins=["*"]`), so any origin works in dev.
