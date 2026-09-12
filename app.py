import os
import gc
import gradio as gr
from gradio import Server
from fastapi.responses import HTMLResponse
import numpy as np
import spaces
import torch
import random
import base64
from io import BytesIO
from PIL import Image
from typing import Tuple

from diffusers import Flux2KleinPipeline

MAX_SEED = np.iinfo(np.int32).max
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16

print("Using device:", device)

print("Loading FLUX.2 Klein 9B model base...")
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9B",
    torch_dtype=dtype,
).to(device)
print("Base Model loaded successfully.")

def pil_to_b64_png(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

# ── Gradio Server (Server mode): FastAPI + Gradio queue/API engine ────────────
app = Server(title="Scribble-Board-Fast")

@app.mcp.tool(name="generate_image")
@app.api(name="generate_image")
@spaces.GPU(size="xlarge")
def infer(
    image_b64: str,
    prompt: str,
    seed: int,
    randomize_seed: bool,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
) -> dict:
    """Generates an image from a scribble using FLUX.2 Klein 9B.

    NOTE: `image_b64` is the fully FLATTENED board sent by the frontend —
    the free-hand sketch, every sticker/asset, every text object AND every
    line/arrow/shape are already composited into this single PNG, so all of
    them are reflected in the conditioning image passed to the pipeline.
    """
    gc.collect()
    torch.cuda.empty_cache()

    if not prompt or prompt.strip() == "":
        raise gr.Error("Please enter a prompt.")

    if not image_b64:
        raise gr.Error("Please draw a sketch first.")

    try:
        header, data = image_b64.split(",", 1)
        pil_image = Image.open(BytesIO(base64.b64decode(data))).convert("RGB")
    except Exception as e:
        raise gr.Error(f"Invalid image data: {e}")

    # Ensure dimensions are multiples of 8
    final_width = max(256, min(1024, round(int(width) / 8) * 8))
    final_height = max(256, min(1024, round(int(height) / 8) * 8))

    # Resize the flattened board (sketch + assets + text + shapes) to the target dimensions
    pil_image = pil_image.resize((final_width, final_height), LANCZOS).convert("RGB")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device="cpu").manual_seed(seed)

    kwargs = dict(
        prompt=prompt,
        height=final_height,
        width=final_width,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
        image=pil_image
    )

    try:
        result_image = pipe(**kwargs).images[0]
        return {"image": pil_to_b64_png(result_image), "seed": seed}
    except Exception as e:
        raise e
    finally:
        gc.collect()
        torch.cuda.empty_cache()


@app.get("/api/config")
def client_config():
    """Plain FastAPI route: config data for the frontend."""
    return {"app_name": "Scribble-Board-Fast"}

@app.get("/", response_class=HTMLResponse)
async def homepage():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    app.launch(show_error=True, mcp_server=True)
