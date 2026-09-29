"""Single-shot CLI entry-point for the Lens text-to-image pipeline.

Example:

    python inference.py \\
        --prompt "A scenic landscape with a serene lake" \\
        --base_resolution 1024 --aspect_ratio 1:1 \\
        --steps 50 --cfg 4.0 --n 4 --seed 42 \\
        --out ./outputs

Use a different repo / local export with ``--repo_id``:

    python inference.py --repo_id ./lens-hf-export --prompt "a cat" --steps 4

Batch by joining prompts with `|`:

    python inference.py --prompt "a cat|a dog|a robot" ...
"""

from __future__ import annotations

import argparse
import os

import torch

from lens import LensGptOssEncoder, LensPipeline
from lens.resolution import SUPPORTED_ASPECT_RATIOS, SUPPORTED_BASE_RESOLUTIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lens text-to-image inference")
    parser.add_argument("--repo_id", type=str, default="microsoft/Lens",
                        help="HF repo id (or local path) of the assembled "
                             "Lens pipeline (model_index.json + subfolders).")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt(s). Use '|' to separate multiple prompts.")
    parser.add_argument("--base_resolution", type=int, default=1440,
                        choices=SUPPORTED_BASE_RESOLUTIONS)
    parser.add_argument("--aspect_ratio", type=str, default="1:1",
                        choices=SUPPORTED_ASPECT_RATIOS)
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of denoising steps.")
    parser.add_argument("--cfg", type=float, default=5.0,
                        help="Classifier-free guidance scale.")
    parser.add_argument("--n", type=int, default=1,
                        help="Number of images per prompt.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default="./outputs")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--disable_mxfp4", action="store_true",
                        help="Disable dequantization of GPT-OSS text encoder.")
    parser.add_argument("--reasoner", action="store_true",
                        help="Enable prompt reasoner (uses local GPT-OSS unless "
                             "--api_url is set).")
    parser.add_argument("--api_url", type=str, default=None,
                        help="OpenAI-compatible base URL for the API reasoner.")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key for the OpenAI-compatible endpoint.")
    parser.add_argument("--api_model", type=str, default=None,
                        help="Model name to send to the OpenAI-compatible endpoint.")
    parser.add_argument("--offload", action="store_true",
                        help="Enable diffusers model CPU offload "
                             "(text_encoder->transformer->vae) to reduce peak VRAM.")
    parser.add_argument("--low_vram", action="store_true",
                        help="Enable low-VRAM mode for GPUs with <=16GB. Uses "
                             "split-pipeline with accelerate device_map to run "
                             "text encoding and denoising in separate phases.")
    return parser.parse_args()


def _torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


def main() -> None:
    args = parse_args()
    prompts = [p.strip() for p in args.prompt.split("|") if p.strip()]
    if not prompts:
        raise SystemExit("No non-empty prompts after splitting on '|'.")

    os.makedirs(args.out, exist_ok=True)

    dtype = _torch_dtype(args.dtype)

    if args.low_vram:
        _run_low_vram(args, prompts, dtype)
    else:
        _run_standard(args, prompts, dtype)


def _run_standard(args, prompts, dtype):
    """Standard inference path (original behavior)."""
    # Pre-load the text encoder so we can control MXFP4 dequantization. The
    # gpt-oss-20b weights on the hub are stored in MXFP4 and the loader will
    # try to keep them that way unless we pass an explicit Mxfp4Config that
    # asks for dequantization. MXFP4 kernels need Hopper-or-newer GPUs, so on
    # A100/V100 we may consider dequantizing to bf16/fp16.
    text_encoder_kwargs = {"subfolder": "text_encoder", "dtype": dtype}
    try:
        from transformers import Mxfp4Config
        text_encoder_kwargs["quantization_config"] = Mxfp4Config(
            dequantize=args.disable_mxfp4
        )
    except ImportError:
        pass
    text_encoder = LensGptOssEncoder.from_pretrained(
        args.repo_id, **text_encoder_kwargs
    )

    pipe = LensPipeline.from_pretrained(
        args.repo_id, text_encoder=text_encoder, torch_dtype=dtype
    )

    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    if args.api_url or args.api_key or args.api_model:
        pipe.reasoner.openai_base_url = args.api_url
        pipe.reasoner.openai_api_key = args.api_key
        pipe.reasoner.openai_model = args.api_model

    generator = (
        torch.Generator(device=pipe._execution_device).manual_seed(int(args.seed))
        if args.seed is not None
        else None
    )

    out = pipe(
        prompt=prompts,
        base_resolution=args.base_resolution,
        aspect_ratio=args.aspect_ratio,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        num_images_per_prompt=args.n,
        generator=generator,
        enable_reasoner=args.reasoner,
    )

    _save_outputs(args, prompts, out, pipe)


def _run_low_vram(args, prompts, dtype):
    """Low-VRAM inference: split text encoding and denoising into two phases
    using accelerate device_map to fit within ~16GB GPU memory."""
    import gc
    from accelerate import infer_auto_device_map, dispatch_model

    gpu_mem = _get_gpu_memory_gib()
    print(f"[low_vram] GPU memory: {gpu_mem:.1f} GiB")

    if args.base_resolution > 1024:
        print(f"[low_vram] WARNING: base_resolution {args.base_resolution} is too "
              f"large for {gpu_mem:.0f} GiB GPU, clamping to 1024.")
        args.base_resolution = 1024

    # ---- Phase 1: Text encoding with device_map ----
    print("[low_vram] Phase 1: Loading text encoder with device_map...")

    te_kwargs = {"subfolder": "text_encoder", "dtype": dtype, "device_map": "cpu"}
    try:
        from transformers import Mxfp4Config
        te_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
    except ImportError:
        pass

    # First load to CPU to compute device map
    te = LensGptOssEncoder.from_pretrained(args.repo_id, **te_kwargs)
    te_gpu_budget = min(gpu_mem * 0.5, 8.0)
    device_map_te = infer_auto_device_map(
        te, max_memory={0: f"{te_gpu_budget:.0f}GiB", "cpu": "100GiB"}
    )
    del te; gc.collect()

    # Reload with device_map
    te = LensGptOssEncoder.from_pretrained(
        args.repo_id, subfolder="text_encoder", dtype=dtype,
        quantization_config=te_kwargs.get("quantization_config"),
        device_map=device_map_te,
    )

    te_pipe = LensPipeline.from_pretrained(
        args.repo_id, torch_dtype=dtype,
        text_encoder=te, transformer=None, vae=None,
    )

    # Optionally refine prompts before encoding
    if args.api_url or args.api_key or args.api_model:
        te_pipe.reasoner.openai_base_url = args.api_url
        te_pipe.reasoner.openai_api_key = args.api_key
        te_pipe.reasoner.openai_model = args.api_model
    refined = te_pipe.refine_prompt(prompts, enable_reasoner=args.reasoner)

    print("[low_vram] Phase 1: Encoding prompts...")
    prompt_embeds, prompt_mask, neg_embeds, neg_mask = te_pipe.encode_prompt(
        prompt=refined, negative_prompt="",
        num_images_per_prompt=args.n,
        device=torch.device("cuda"),
    )

    # Move to CPU and free text encoder
    prompt_embeds = [e.cpu() for e in prompt_embeds]
    prompt_mask = prompt_mask.cpu()
    neg_embeds = [e.cpu() for e in neg_embeds]
    neg_mask = neg_mask.cpu()
    del te_pipe, te; gc.collect(); torch.cuda.empty_cache()
    print("[low_vram] Phase 1: Done, text encoder freed.")

    # ---- Phase 2: Denoise with device_map ----
    print("[low_vram] Phase 2: Loading transformer + VAE...")
    pipe = LensPipeline.from_pretrained(
        args.repo_id, torch_dtype=dtype, text_encoder=None,
    )

    # Compute device_map for transformer, leaving headroom for attention
    # activations during denoising.  SDPA attention for 1024x1024 with CFG
    # (batch=2) needs ~7 GiB of intermediate memory on top of weights, so we
    # cap the weight budget to leave enough room.  Empirically 6 GiB works on
    # a 16 GB GPU at 1024x1024; we scale proportionally for other GPU sizes.
    trans_gpu_budget = max(gpu_mem * 0.38, 2.0)  # ~6 GiB on 16 GB GPU
    from lens.resolution import resolve_resolution
    height, width = resolve_resolution(args.base_resolution, args.aspect_ratio)
    print(f"[low_vram] transformer GPU budget: {trans_gpu_budget:.1f} GiB")

    transformer = pipe.transformer
    dm_trans = infer_auto_device_map(
        transformer, max_memory={0: f"{trans_gpu_budget:.0f}GiB", "cpu": "100GiB"}
    )
    dispatch_model(transformer, device_map=dm_trans)
    pipe.vae = pipe.vae.to("cuda")

    gpu_n = sum(1 for v in dm_trans.values() if v == 0)
    cpu_n = sum(1 for v in dm_trans.values() if v == "cpu")
    print(f"[low_vram] Transformer: {gpu_n} modules on GPU, {cpu_n} on CPU")

    generator = (
        torch.Generator(device="cuda").manual_seed(int(args.seed))
        if args.seed is not None
        else None
    )

    print(f"[low_vram] Phase 2: Denoising {height}x{width}...")
    out = pipe(
        prompt_embeds=[e.cuda() for e in prompt_embeds],
        prompt_mask=prompt_mask.cuda(),
        negative_prompt_embeds=[e.cuda() for e in neg_embeds],
        negative_prompt_mask=neg_mask.cuda(),
        base_resolution=args.base_resolution,
        aspect_ratio=args.aspect_ratio,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        generator=generator,
    )

    # Store refined prompts for output
    pipe._last_refined_prompts = refined
    _save_outputs(args, prompts, out, pipe)
    print(f"[low_vram] Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")


def _get_gpu_memory_gib():
    """Return total GPU memory in GiB."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return 0.0


def _save_outputs(args, prompts, out, pipe):
    """Save generated images and print refined prompts if applicable."""
    images = list(out.images)
    expected_images = len(prompts) * args.n
    if len(images) != expected_images:
        raise RuntimeError(
            f"Pipeline returned {len(images)} images; expected {expected_images}."
        )
    img_iter = iter(images)
    for p_idx, prompt in enumerate(prompts):
        for s_idx in range(args.n):
            img = next(img_iter)
            fname = f"p{p_idx:03d}_s{s_idx:02d}.png"
            img.save(os.path.join(args.out, fname))
            print(f"saved {fname} :: {prompt!r}")
    refined = getattr(pipe, "_last_refined_prompts", prompts)
    if any(r != p for r, p in zip(refined, prompts)):
        print("\nRefined prompts:")
        for orig, ref in zip(prompts, refined):
            print(f"  {orig!r}\n    -> {ref!r}")


if __name__ == "__main__":
    main()
