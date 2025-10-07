🖼️ Text-to-Image Generation with Stable Diffusion

This project demonstrates how to generate high-quality images from text prompts using Stable Diffusion via Hugging Face's diffusers library. It allows users to experiment with creative text prompts and instantly generate visual outputs.

📂 Project Contents
generate.py – Command-line script to generate images from text prompts.
requirements.txt – Lists all Python dependencies for the project.
example_prompts.txt – Sample prompts to inspire your image generation experiments.
⚙️ Requirements
Python 3.8+
GPU recommended for faster image generation (CPU is supported but slower).
Hugging Face account with access token (needed for gated or private models).
Set Hugging Face Token

Before running the script, set your Hugging Face token as an environment variable:

export HF_TOKEN="your_token_here"   # macOS/Linux
setx HF_TOKEN "your_token_here"     # Windows

🛠️ Installation
Create a virtual environment:
python -m venv .venv

Activate the environment:
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows

Install dependencies:
pip install -r requirements.txt

🚀 Usage
Generate Images from a Prompt

Run the script with a custom text prompt:

python generate.py --prompt "A futuristic city skyline at sunset, cinematic, ultra-detailed" --num_images 2

Use Example Prompts

List sample prompts provided in example_prompts.txt:

cat example_prompts.txt


Generated images are automatically saved to the outputs/ folder.

⚠️ Notes & Safety
Ensure prompts are provided; the script includes a basic check.
Avoid generating images of real people or public figures without permission.
Follow Hugging Face model licenses and usage guidelines.
💡 Learnings & Features
Integrated Stable Diffusion for text-to-image generation.
Built a flexible CLI tool for prompt-based image creation.
Learned to manage dependencies and virtual environments for ML projects.
Explored GPU acceleration for faster inference.
Experimented with creative prompts to generate unique, high-quality images.

Enjoy experimenting with AI-powered image generation! 🌟
