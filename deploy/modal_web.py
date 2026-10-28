"""The Meaning Check web app on Modal (CPU only, scales to zero), giving a public link for the demo and the judges.

Deploy after the reader service:
    GLIMMER_URL=<reader URL> modal deploy deploy/modal_web.py
Offline demo with simulated readers (no GPU at all):
    MC_BACKEND=sim modal deploy deploy/modal_web.py
Optionally set MODEL_API_KEY in the same way to let Muse Spark write (and to enable photo and voice input).
"""
import os

import modal

env = {k: os.environ[k] for k in ("GLIMMER_URL", "MODEL_API_KEY", "MC_BACKEND", "MC_WRITER") if os.environ.get(k)}
if "GLIMMER_URL" not in env and "MC_BACKEND" not in env:
    env["MC_BACKEND"] = "sim"
env["MC_CACHE_DIR"] = "/data/cache"
env["MC_WEB_DIR"] = "/web"
env["MC_VOLUME"] = "meaning-check-web-data"      # recorded scenario runs are committed to it
WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "meaningcheck", "web")

image = (modal.Image.debian_slim(python_version="3.11")
         .uv_pip_install("numpy>=1.24", "scipy>=1.11", "fastapi[standard]", "uvicorn", "python-multipart")
         .add_local_python_source("meaningcheck")
         .add_local_dir(WEB, remote_path="/web"))          # the page itself (Python sources alone do not include it)
data = modal.Volume.from_name("meaning-check-web-data", create_if_missing=True)
app = modal.App("meaning-check-web")


@app.function(image=image, secrets=[modal.Secret.from_dict(env)], volumes={"/data": data}, timeout=900,
              scaledown_window=300)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    from meaningcheck.server import app as api
    return api
