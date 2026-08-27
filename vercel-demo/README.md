# MRIscanner — Vercel demo (EC2-backed)

A custom UI hosted on Vercel that relays all actual inference to a dedicated
EC2 instance running the full app (all 3 models, real PyTorch, Grad-CAM).
Vercel does **zero** ML compute here -- it's a thin, stateless proxy.

**Live:** https://vercel-demo-vert-eta.vercel.app

## Why a relay instead of running models in Vercel

A full 3-model prediction pass takes 30s-2min depending on load (this runs on
a memory-constrained free-tier EC2 instance, so it sometimes swaps to disk).
That's longer than Vercel serverless functions are allowed to run, and Vercel
functions are stateless/ephemeral -- there's nowhere to keep 3 loaded PyTorch
models warm between requests even if the timeout weren't an issue. So instead:

- **EC2** runs the real app (`app/gradio_app.py` + `app/async_api.py`) as a
  persistent `systemd` service, with an async job API: `POST /api/jobs/predict`
  starts a background thread and returns a `job_id` immediately; `GET
  /api/jobs/{job_id}` is polled until the job is `done`.
- **Vercel** (`api/predict.py`) just relays those same two calls under
  `/api/predict` -- each relay call is instant (submit or poll), so it never
  approaches Vercel's timeout even though the underlying job takes minutes.
- The frontend (`public/index.html`) polls every 3 seconds until the job
  finishes, then renders each model's label + confidence + Grad-CAM heatmap
  (sent back as base64 PNG data URLs from EC2).

## Structure

```
vercel-demo/
  api/predict.py       # pure-stdlib relay: POST=start job, GET=poll job (no deps)
  public/index.html     # multi-model UI: upload, model checkboxes, live polling
  pyproject.toml        # [tool.vercel] entrypoint (empty deps -- no ML libs needed here)
  requirements.txt       # intentionally empty
  tests/test_predict.py # exercises the real relay logic against a fake upstream server
```

## Redeploying

```bash
cd vercel-demo
npm install --no-save vercel   # or use a global install
./node_modules/.bin/vercel login
./node_modules/.bin/vercel --prod --yes
```

If the EC2 instance's IP changes, update `EC2_BASE_URL` in `api/predict.py`
and redeploy.
