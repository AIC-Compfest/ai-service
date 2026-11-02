# Running the ai-service on Appwrite Functions

This folder runs the existing FastAPI service (`app.main:app`) as an Appwrite Function.
The Dockerfile, the uvicorn command, and everything under `app/`, `config/` and `models/`
are unchanged, so the VPS deployment keeps working.

The function is defined in `appwrite.config.json` (function ID `ai-service-pipeline`).

## How the backend connects

`backend-api` posts to `{AI_SERVICE_URL}/v1/extract/pipeline/full`. Set:

```
AI_SERVICE_URL=https://<username>:<AI_SERVICE_SECRET>@<function-domain>
```

Go's HTTP client turns the URL password into a Basic `Authorization` header, so the
backend needs no code change. The same client masks the password in its error messages.
That matters because `backend-api` returns error text to API clients.

## Setup

1. Apply the patch: `git am ai-service-appwrite.patch` (or `git apply`, then commit).
2. Generate a secret: `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
3. Add it to the function as the variable `AI_SERVICE_SECRET`, marked as secret.
   Keep the other values in `.env.example` as function variables only if you need
   something other than the code defaults.
4. Deploy from the repo root: `appwrite login`, then `appwrite push functions`.
5. Check it (replace the placeholders):

   ```
   curl -i -u ai:<AI_SERVICE_SECRET> https://<function-domain>/healthz
   curl -i -u ai:<AI_SERVICE_SECRET> -F "file=@sample.pdf" https://<function-domain>/v1/extract/pipeline/full
   curl -i https://<function-domain>/healthz     # without credentials: expect 401
   ```

   The first two should return 200 and the pipeline JSON. The last should return 401.

6. Point `AI_SERVICE_URL` in the backend at the function, as shown above.

## What the files do

- `main.py`: the Appwrite entrypoint, `main(context)`. Checks the password, forwards the
  request, and returns the app's response.
- `asgi_bridge.py`: runs one request through an ASGI app in-process. Startup (model loading)
  runs once per container.
- `build.sh`: the build step. It installs CPU-only torch first, like the Dockerfile, then
  `requirements.lock.txt`, then imports the same modules the Dockerfile checks.

## Runtime settings

- Runtime `python-ml-3.11`. This is Appwrite's Debian-based Python 3.11 runtime. The other
  Cloud Python runtimes are Alpine, which lacks the wheels this stack needs.
- Timeout 900 seconds. Appwrite's default is 15 seconds.
- Build and runtime specification `s-4vcpu-4gb`.
- Scratch paths default to `/tmp` (`PAGE_OUTPUT_DIR`, `HF_HOME`, `XDG_CACHE_HOME`, `TORCH_HOME`).

## Check before go-live

- **Backend timeout.** The backend waits `AI_SERVICE_TIMEOUT_SECONDS` (default 120). Appwrite
  allows 900 seconds, but a cold start also loads models, so the first request after an idle
  period can exceed the backend's limit.
- **Upload size.** Confirm a 10 MB PDF gets through. Appwrite's request size limit isn't in
  the docs I could find.
- **Credentials reach the function.** The `-u` check in step 5 covers this.
- **Package contents.** `appwrite push` uploads the repo folder. Make sure `.env` and large
  folders you don't need (such as `data/` and `gold/`) are excluded.
- **Model downloads.** Docling fetches layout models from Hugging Face on first use unless
  they are already in `models/`. `HF_HOME` points at `/tmp`, so every new container downloads
  them again.
- **Variables.** If `appwrite push` clears function variables, add `AI_SERVICE_SECRET` again.

## Rolling back

Point `AI_SERVICE_URL` back at the VPS. The Docker path is unchanged.
