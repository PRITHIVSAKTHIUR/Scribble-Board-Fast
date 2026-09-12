# **Scribble-Board-Fast**

Scribble-Board-Fast is an interactive, high-performance sketch-to-image synthesis workspace powered by the `black-forest-labs/FLUX.2-klein-9B` model. Built with diffusers, it takes freehand doodles, customized brush strokes, and modular visual assets (stickers/uploaded images) composited directly on an interactive canvas, transforming rough layouts into high-fidelity generative imagery in a 4-step distilled sampling window.

The system is deployed using a FastAPI application via `gradio.Server` hosting a responsive, dark-mode single-page application (SPA). Features include custom brush dynamics, an asset/sticker placement and manipulation engine, aspect-ratio snapping, history rollback, and instant API generation pipelines.

<img width="1920" height="889" alt="Screenshot From 2026-09-12 15-20-20" src="https://github.com/user-attachments/assets/2d55df6f-9974-487b-8197-1868a5c9e76c" />


```txt
prompt/

3D render, octane render,
cinema4d, soft lighting
```
<img width="1920" height="889" alt="Screenshot From 2026-09-12 15-22-01" src="https://github.com/user-attachments/assets/bf4c2972-95db-4dac-b9e9-43f2c9e2d85a" />

### **Key Features**

* **Interactive Sketch-to-Image Pipeline:** Converts freehand drawings and composite layouts directly into detailed photographic, artistic, or 3D images via `FLUX.2-klein-9B`.
* **Integrated Asset & Sticker Board:** Allows placing, moving, scaling (with aspect-ratio lock via Shift), and deleting custom image uploads or built-in vector stickers. The canvas flattens strokes and assets seamlessly into a single conditioning target.
* **Aspect-Ratio & Dimension Snapping:** Supports common aspect ratios (`16:9`, `9:16`, `1:1`, `4:3`, `3:4`), automatically keeping dimensions between 256px and 1024px and snapping bounding values to multiples of 8.
* **Fast 4-Step Sampling:** Designed for low-latency generation passes running directly on high-performance CUDA hardware.
* **Minimalist Studio SPA:** Pure JavaScript and CSS canvas drawing interface featuring tool switching (Brush, Eraser, Select), history stacks (Undo/Redo), live status monitoring, and image export.

### **Repository Structure**

```text
├── app.py
├── index.html
├── LICENSE.txt
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock
```

### **Installation and Requirements**

To set up the Scribble-Board-Fast environment locally, configure your system according to the specifications below. A modern CUDA-enabled GPU is required.

* **Python Version:** Python **3.12** is strictly required and recommended.
* **PyTorch Version:** `torch==2.11.0` or above is required for optimal system compatibility.
* **CUDA Version:** **CUDA 13.0** is recommended (`--extra-index-url https://download.pytorch.org/whl/cu130`), matching the environment used on the live Hugging Face Space.

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust. It ensures rapid virtual environment setup and exact dependency synchronization based on the `uv.lock` file.

**Step 1 — Install `uv`**

* **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

**Step 2 — Clone the repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/Scribble-Board-Fast.git
cd Scribble-Board-Fast
```

**Step 3 — Initialize the project and install dependencies**

```bash
uv sync
```

**Step 4 — Run the script**

```bash
uv run app.py
```

#### **Standard PIP Implementation**

**1. Update Package Manager**
Upgrade your local package manager:

```bash
pip install "pip>=26.1.2"
```

**2. Install Core Dependencies**
Install the primary deep learning stack, transformer libraries, and core computing utilities listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

#### **Core Requirements List (`requirements.txt`)**

```text
--extra-index-url https://download.pytorch.org/whl/cu130

torch==2.11.0
torchvision==0.26.0
transformers==5.14.1
accelerate==1.14.0
diffusers==0.39.0
peft==0.19.1
gradio==6.27.0
av==17.1.0
spaces>=0.51.1
huggingface-hub==1.24.0
```

### **Usage**

Once the server initializes, open your browser to the local address output in your terminal (typically `http://127.0.0.1:7860/`).

1. **Draw or Place Assets:**
* Use the **Brush** and **Color Picker** in the left rail to sketch out basic shapes and concepts.
* Use the **Assets** button to add stickers or upload transparent PNGs directly onto the board.
* Use the **Select** tool to reposition or resize stickers.

2. **Configure Generation Parameters:**
* Select your target **Aspect Ratio** from the inspector panel (`16:9`, `1:1`, etc.).
* Enter a descriptive text prompt detailing the subject, lighting, and medium (e.g., *"a highly detailed, realistic photograph"* or *"cyberpunk style, neon lights"*).
* Adjust inference steps (default 4) and guidance scale as needed.

3. **Execute:** Click **Generate Image** or press ⌘/Ctrl + Enter. The composited canvas is processed and rendered into a high-resolution output.
4. **Export:** Inspect the generated image using the view toggle or download the output via the download action on the left rail.

### **License and Source**

* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/Scribble-Board-Fast/blob/main/LICENSE.txt)
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/Scribble-Board-Fast](https://github.com/PRITHIVSAKTHIUR/Scribble-Board-Fast)
* **Hugging Face Live Space:** [https://huggingface.co/spaces/prithivMLmods/Scribble-Board-Fast](https://huggingface.co/spaces/prithivMLmods/Scribble-Board-Fast)
