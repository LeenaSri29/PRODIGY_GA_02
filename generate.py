#!/usr/bin/env python3
"""generate.py
Simple CLI to generate images from text prompts using Hugging Face Diffusers (Stable Diffusion).

Notes:
- Provide HF_TOKEN in the environment if the model requires authentication.
- For faster results, run on a GPU-enabled machine with CUDA and a compatible torch.
"""
import os
import argparse
import sys
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm

def check_dependencies():
    try:
        import diffusers
        import transformers
    except Exception as e:
        print("Missing dependencies. Please install from requirements.txt:\n", e)
        sys.exit(1)

def get_pipe(model_id, device, hf_token=None):
    # Lazy import so the script fails earlier with friendly message if not installed.
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    # Use a scheduler that is compatible with many diffusers versions
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler", use_auth_token=hf_token) if False else None
    # Create pipeline (the `use_auth_token` argument is supported in older diffusers; using `token` env recommended)
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device.type=='cuda' else torch.float32, use_auth_token=hf_token)
    # If GPU available, move to cuda and enable attention slicing / memory efficient settings
    if device.type == 'cuda':
        pipe = pipe.to(device)
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    return pipe

def generate(prompt, pipe, out_path:Path, guidance_scale=7.5, num_inference_steps=30, seed=None, height=None, width=None):
    generator = torch.Generator(device=pipe.device)
    if seed is not None:
        generator = generator.manual_seed(seed)
    # Prepare call kwargs
    call_kwargs = dict(prompt=prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, generator=generator)
    if height is not None and width is not None:
        call_kwargs.update({'height': height, 'width': width})
    images = pipe(**call_kwargs).images
    saved_paths = []
    out_path.mkdir(parents=True, exist_ok=True)
    for i,img in enumerate(images):
        fname = out_path / f"gen_{i+1}.png"
        img.save(fname)
        saved_paths.append(str(fname))
    return saved_paths

def main():
    check_dependencies()
    parser = argparse.ArgumentParser(description="Generate images from text using Stable Diffusion (diffusers). See README for setup.")
    parser.add_argument('--prompt', type=str, help='Text prompt to generate', required=False)
    parser.add_argument('--prompt_file', type=str, help='Path to a file with one prompt per line', required=False)
    parser.add_argument('--model', type=str, default='runwayml/stable-diffusion-v1-5', help='Model repo id (diffusers) to use')
    parser.add_argument('--hf_token', type=str, default=None, help='Hugging Face token (or set HF_TOKEN env var)')
    parser.add_argument('--num_images', type=int, default=1)
    parser.add_argument('--out', type=str, default='outputs')
    parser.add_argument('--guidance', type=float, default=7.5)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--width', type=int, default=None)
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type=='cpu':
        print("Warning: Running on CPU will be slow and memory intensive.")

    if not args.prompt and not args.prompt_file:
        print("Provide --prompt or --prompt_file. See README.md for examples.")
        sys.exit(1)

    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompt_file:
        pfile = Path(args.prompt_file)
        if not pfile.exists():
            print("Prompt file not found:", pfile)
            sys.exit(1)
        with pfile.open('r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            prompts.extend(lines)

    # Load model pipeline
    print("Loading model pipeline. This may take a while...")
    pipe = get_pipe(args.model, device, hf_token=hf_token)

    out_base = Path(args.out)
    out_base.mkdir(parents=True, exist_ok=True)

    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx+1}/{len(prompts)}] Prompt: {prompt}")
        # generate requested number of images per prompt
        for n in range(args.num_images):
            suffix = f"p{idx+1}_i{n+1}"
            out_dir = out_base / suffix
            saved = generate(prompt, pipe, out_dir, guidance_scale=args.guidance, num_inference_steps=args.steps, seed=args.seed, height=args.height, width=args.width)
            for p in saved:
                print("Saved:", p)

if __name__ == '__main__':
    main()
