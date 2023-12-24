import subprocess
import threading
import time
from cog import BasePredictor, Input, Path
from typing import List
import os
import torch
import shutil
import uuid
import json
import urllib
import websocket
import multiprocessing
from PIL import Image
from urllib.error import URLError
import random
from urllib.parse import urlparse
import sys
from pathlib import Path as LocalPath

sys.path.extend(["./ComfyUI/ComfyUI-to-Python-Extension", "./ComfyUI/ComfyUI-to-Python-Extension/output_files"])

OUTPUT_PATH = LocalPath("./ComfyUI/output/")

KSAMPLER_NAMES = ["euler", "euler_ancestral", "heun", "heunpp2","dpm_2", "dpm_2_ancestral",
                  "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
                  "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm"]

SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]

CHECKPOINTS = [
   "beautifulRealistic_v7.safetensors",
   "toonyou_beta6.safetensors",
   "darkSushiMixMix_225D.safetensors",
]

class Predictor(BasePredictor):
    def setup(self):
        print("Starting setup...")

    def predict(
        self,
        positive_prompt: str = Input(description="Positive Prompt", default="photo of a man, fashion model, simple clothes\n\nhigh quality, highly detailed, 4k, highres"),
        negative_prompt: str = Input(description="Negative Prompt", default="blurry, distorted, low quality, bad hands"),
        face_image: Path = Input(description="Source image of the face"),
        checkpoint: str = Input(description="SD 1.5 model checkpoint", choices=CHECKPOINTS, default=CHECKPOINTS[0]),
        num_inference_steps: int = Input(
            description="Steps",
            default=30
        ),
        batch_size: int = Input(description="Number of images to generate in batch", default=1),
        repeat: int = Input(description="Repeat running this request (2 batch size, 2 repeat -> 2 * 2 images)", default=1),
        width: int = Input(default=512),
        height: int = Input(default=768), 
        cfg: float = Input(default=8.0),
        seed: int = Input(description="Sampling seed, leave Empty for Random", default=None),
        vae: str = Input(choices=["vae-ft-mse-840000-ema-pruned.safetensors"], default="vae-ft-mse-840000-ema-pruned.safetensors"),
        sampler: str = Input(default=KSAMPLER_NAMES[0], choices=KSAMPLER_NAMES),
        scheduler: str = Input(default=SCHEDULER_NAMES[0], choices=SCHEDULER_NAMES),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        print("Start prediction...")
        print(f"face image path: {face_image}")

        from ipadapter_faceid_upscale_replicate import main as generate_images

        images = generate_images(
            positive_prompt,
            negative_prompt,
            str(face_image),
            checkpoint,
            steps=num_inference_steps,
            width=width,
            height=height,
            cfg=cfg,
            seed=seed,
            batch_size=batch_size,
            repeat=repeat,
            sampler=sampler,
            scheduler=scheduler,
            vae=vae,
        )

        image_paths = []
    
        for batch_image_data in images:
            batch_image = batch_image_data["ui"]["images"]
            for image in batch_image:
                image_name = image["filename"]
                subfolder = image["subfolder"]

                if subfolder:
                    image_path = OUTPUT_PATH / subfolder / image_name
                else:
                    image_path = OUTPUT_PATH / image_name

                image_path = Path(str(image_path))
                image_paths.append(image_path)

        print(f"Prediction finished, {len(image_paths)} images generated.")
        return image_paths