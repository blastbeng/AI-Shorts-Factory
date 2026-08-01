#!/usr/bin/env python3
"""
Standalone Wan video generation subprocess.
Uses CUDA PyTorch to run on the NVIDIA RTX 3060.
Expects a JSON config file path as the only argument.
"""
import sys
import json
import os
import gc
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import WanImageToVideoPipeline, WanTransformer3DModel
import imageio
import subprocess as sp
import shutil


def _cleanup_base_model_dir(base_dir):
    if not os.path.isdir(base_dir):
        return

    # 1. Remove any transformer_* folders except 'transformer'
    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path) and entry.startswith("transformer_") and entry != "transformer":
            shutil.rmtree(full_path, ignore_errors=True)

    # 2. Remove any weight files inside transformer/ (keep only config.json)
    transformer_dir = os.path.join(base_dir, "transformer")
    if os.path.isdir(transformer_dir):
        for entry in os.listdir(transformer_dir):
            if entry != "config.json":
                full_path = os.path.join(transformer_dir, entry)
                if os.path.isfile(full_path):
                    os.remove(full_path)

    # 3. Fix model_index.json: ensure only a single 'transformer' key exists
    model_index_path = os.path.join(base_dir, "model_index.json")
    if not os.path.exists(model_index_path):
        raise FileNotFoundError(f"model_index.json missing in {base_dir}")

    with open(model_index_path, "r") as f:
        model_index = json.load(f)

    changed = False
    for k in list(model_index.keys()):
        if k.startswith("transformer_") and k != "transformer":
            del model_index[k]
            changed = True

    if model_index.get("transformer") != "transformer":
        model_index["transformer"] = "transformer"
        changed = True

    if changed:
        with open(model_index_path, "w") as f:
            json.dump(model_index, f, indent=2)


def main():
    if len(sys.argv) != 2:
        print("Usage: wan_subprocess.py <config.json>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        config = json.load(f)

    # Unpack config
    base_model_path = config["base_model_path"]
    model_path = config["model_path"]
    prompts = config["prompts"]
    output_path = config["output_path"]
    image_path = config.get("image_path")
    frames_per_clip = config.get("frames_per_clip", 49)
    width = config.get("width", 256)
    height = config.get("height", 448)
    steps = config.get("steps", 40)
    job_id = config.get("job_id")
    base_seed = config.get("base_seed", 42)

    # Set CUDA device (RTX 3060 is the only visible GPU in the CUDA venv)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Cleanup base model dir
    _cleanup_base_model_dir(base_model_path)

    # Load base pipeline without transformer
    print(f"[{job_id}] Loading base Wan pipeline...")
    pipeline = WanImageToVideoPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        transformer=None,
    )
    if pipeline.transformer is not None:
        del pipeline.transformer
        pipeline.transformer = None

    # Load FP8 transformer
    transformer_config_path = os.path.join(base_model_path, "transformer")
    if not os.path.exists(os.path.join(transformer_config_path, "config.json")):
        raise FileNotFoundError(f"Transformer config not found at {transformer_config_path}/config.json")

    print(f"[{job_id}] Loading FP8 transformer from {model_path} ...")
    transformer_dtype = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else "auto"
    transformer = WanTransformer3DModel.from_single_file(
        model_path,
        config=transformer_config_path,
        torch_dtype=transformer_dtype,
    )
    pipeline.transformer = transformer

    # Move to GPU
    pipeline.to(device)

    # Memory optimizations
    if hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
    if hasattr(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing()

    # Generation loop
    fps = 24.0
    temp_clips = []
    last_frame = None
    first_clip_frames = None

    for i, prompt_data in enumerate(prompts):
        img_prompt, vid_prompt = prompt_data
        if i == 0:
            prompt = f"{img_prompt}. {vid_prompt}, high quality, realistic, smooth motion, 4k"
        else:
            prompt = f"{vid_prompt}, smooth transition, high quality, realistic, smooth motion, 4k"

        # Progress callback that prints to stdout
        def progress_callback(pipe, step, timestep, callback_kwargs):
            print(f"PROGRESS:{job_id}:video_generation:{step + 1}:{steps}:Generazione clip {i + 1}/{len(prompts)}: step {step + 1}/{steps}")
            return callback_kwargs

        # Prepare conditioning image
        if i == 0 and image_path and os.path.exists(image_path):
            init_image = Image.open(image_path).convert("RGB")
            init_image = init_image.resize((width, height), Image.LANCZOS)
            current_image = init_image
        elif i > 0 and last_frame is not None:
            frame = cv2.resize(last_frame, (width, height), interpolation=cv2.INTER_CUBIC)
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            current_image = Image.fromarray(frame)
        else:
            current_image = Image.new("RGB", (width, height), color="black")

        seed = base_seed + i
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # Generate clip
        with torch.inference_mode():
            output = pipeline(
                image=current_image,
                prompt=prompt,
                negative_prompt="blurry, distorted, glitchy, low quality, bad anatomy, watermark, text, deformed, mutated, extra limbs, bad framing",
                num_inference_steps=steps,
                num_frames=frames_per_clip,
                height=height,
                width=width,
                guidance_scale=3.5,
                generator=generator,
                output_type="pil",
                callback_on_step_end=progress_callback,
                callback_on_step_end_tensor_inputs=[],
            ).frames[0]

        actual_frames = len(output)
        if first_clip_frames is None:
            first_clip_frames = actual_frames

        # Save clip
        temp_clip_path = output_path.replace(".mp4", f"_clip_{i}.mp4")
        writer = imageio.get_writer(
            temp_clip_path,
            fps=fps,
            codec="libx264",
            ffmpeg_params=["-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p"],
        )
        for frame in output:
            arr = np.array(frame)
            if arr.dtype == np.float32 or arr.dtype == np.float64:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            writer.append_data(arr)
        writer.close()
        temp_clips.append(temp_clip_path)

        # Extract last frame for next clip conditioning
        last_frame = np.array(output[-1]).copy()
        if last_frame.dtype != np.uint8:
            if last_frame.max() <= 1.0:
                last_frame = (last_frame * 255).clip(0, 255).astype(np.uint8)
            else:
                last_frame = last_frame.astype(np.uint8)

        del output, current_image
        gc.collect()
        torch.cuda.empty_cache()

    # Concatenate clips with crossfade
    if len(temp_clips) == 1:
        shutil.copy(temp_clips[0], output_path)
    else:
        fade_duration = 0.5
        clip_duration = first_clip_frames / fps
        offset = max(0.0, clip_duration - fade_duration - 0.1)

        filter_parts = []
        for i, clip in enumerate(temp_clips):
            filter_parts.append(f"[{i}:v]settb=AVTB,fps=24,setpts=PTS-STARTPTS[v{i}];")

        prev = "v0"
        for i in range(1, len(temp_clips)):
            next_v = f"v{i}"
            if i == len(temp_clips) - 1:
                filter_parts.append(
                    f"[{prev}][{next_v}]xfade=transition=fade:duration={fade_duration}:offset={offset}[vout]"
                )
            else:
                filter_parts.append(
                    f"[{prev}][{next_v}]xfade=transition=fade:duration={fade_duration}:offset={offset}[v{i}];"
                )
                prev = f"v{i}"

        filter_complex = "".join(filter_parts)

        cmd = ["ffmpeg", "-y"]
        for clip in temp_clips:
            cmd.extend(["-i", clip])
        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            output_path,
        ])
        sp.run(cmd, check=True)

    # Cleanup temp clips
    for clip in temp_clips:
        if os.path.exists(clip):
            os.remove(clip)

    # Frame interpolation to 48 fps for smoother motion
    interpolated_path = output_path.replace(".mp4", "_48fps.mp4")
    sp.run([
        "ffmpeg", "-y",
        "-i", output_path,
        "-filter:v", "minterpolate=fps=48:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        interpolated_path,
    ], check=True)
    os.replace(interpolated_path, output_path)

    print(f"DONE:{job_id}:{output_path}")


if __name__ == "__main__":
    main()
