"""Muse Glimmer service on Modal: the reader panel (POST /score) and, when needed, the writer (POST /generate).

Download: modal run deploy/modal_glimmer.py::download   (once: copies the weights into a Modal volume, CPU only)
Deploy:   modal deploy deploy/modal_glimmer.py     (prints the URL; put it in GLIMMER_URL)
Check:    curl <URL>/health                       (shows the calibrated answer prefix and the letter mass)
Try it:   modal run deploy/modal_glimmer.py
Stop:     modal app stop meaning-check-glimmer

Options (environment variables when you deploy):
  GLIMMER_MODEL   default meta-models/Muse-Glimmer-30B
  GLIMMER_GPU     default A100-80GB: the bf16 weights are about 58 GB; about $2.50 an hour, so Modal's $30 free monthly
                  credit gives about 12 hours. H100 is faster and dearer.
  HF_SECRET       name of a Modal secret holding HF_TOKEN, if downloading the weights needs a login
  PROTECT=1       require Modal proxy-auth headers (Modal-Key / Modal-Secret) on every request
"""
import os

import modal

MODEL = os.environ.get("GLIMMER_MODEL", "meta-models/Muse-Glimmer-30B")
GPU = os.environ.get("GLIMMER_GPU", "A100-80GB")
BATCH = int(os.environ.get("GLIMMER_BATCH", "8"))

image = (modal.Image.debian_slim(python_version="3.11")
         .uv_pip_install("torch>=2.9", "transformers==5.20.2", "accelerate", "pillow", "fastapi[standard]",
                         "huggingface_hub[hf_transfer]")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache", "GLIMMER_MODEL": MODEL,
               "GLIMMER_BATCH": str(BATCH)})
         .add_local_python_source("meaningcheck"))
cache = modal.Volume.from_name("meaning-check-hf-cache", create_if_missing=True)
secrets = [modal.Secret.from_name(os.environ["HF_SECRET"])] if os.environ.get("HF_SECRET") else []
app = modal.App("meaning-check-glimmer")


@app.function(image=image, volumes={"/cache": cache}, secrets=secrets, timeout=3600, cpu=4, memory=8192)
def download():
    """Copies the model weights into the volume once, on a CPU machine, so GPU time is not spent downloading."""
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL)
    cache.commit()
    print("weights ready in", path)


@app.cls(gpu=GPU, image=image, volumes={"/cache": cache}, secrets=secrets, timeout=1800, scaledown_window=300,
         startup_timeout=1800, max_containers=1)   # one GPU at most: requests queue instead of waking a second A100
class Glimmer:
    @modal.enter()
    def load(self):
        import torch
        import transformers as T
        from meaningcheck.glimmer_service import GlimmerService
        try:
            proc = T.AutoProcessor.from_pretrained(MODEL)
        except Exception as e:
            print("AutoProcessor failed, using AutoTokenizer:", e)
            proc = T.AutoTokenizer.from_pretrained(MODEL)
        model, errors = None, []
        for cls in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
            try:
                model = getattr(T, cls).from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
                break
            except Exception as e:
                errors.append(f"{cls}: {e}")
        if model is None:
            raise RuntimeError("could not load the model: " + " | ".join(errors))
        cache.commit()
        self.svc = GlimmerService(proc, model.eval(), "cuda", BATCH)
        print("calibration:", self.svc.calibrate())

    @modal.asgi_app(requires_proxy_auth=bool(os.environ.get("PROTECT")))
    def web(self):
        from fastapi import FastAPI
        api = FastAPI()

        @api.get("/health")
        def health():
            return {"model": MODEL, "calibration": self.svc.calibration}

        @api.post("/score")
        def score(body: dict):
            items = body.get("items", [])
            out = {"results": self.svc.score(items), "model": MODEL, "prefix": self.svc.prefix}
            if body.get("debug") and items:
                out["template_tail"] = self.svc._reader_text(items[0])[-400:] + self.svc.prefix
            return out

        @api.post("/generate")
        def generate(body: dict):
            return {"texts": self.svc.generate(body.get("items", [])), "model": MODEL}

        return api

    @modal.method()
    def check(self):
        from meaningcheck.glimmer_service import PROBE
        return {"calibration": self.svc.calibration, "probe": self.svc.score([PROBE])}


@app.local_entrypoint()
def main():
    print(Glimmer().check.remote())
