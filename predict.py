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

class Predictor(BasePredictor):
    
    def predict(
        self,
        positive_prompt: str = Input(description="Positive Prompt", default="photo of a man, fashion model, simple clothes\n\nhigh quality, highly detailed, 4k, highres"),
        negative_prompt: str = Input(description="Negative Prompt", default="blurry, distorted, low quality, bad hands"),
        face_image: Path = Input(description="Source image of the face"),
        checkpoint: str = Input(choices=["beautifulRealistic_v7.safetensors"], default="beautifulRealistic_v7.safetensors"),
        steps: int = Input(
            description="Steps",
            default=30
        ),
        batch_size: int = Input(description="Number of images to generate in batch", default=1),
        repeat: int = Input(description="Repeat running this request"),
        width: int = Input(default=512),
        height: int = Input(default=768), 
        cfg: float = Input(default=8.0),
        seed: int = Input(description="Sampling seed, leave Empty for Random", default=None),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        
        from ipadapter_faceid_upscale_replicate import main

        images = main(
            positive_prompt,
            negative_prompt,
            image,
            checkpoint,
            cfg=cfg,
            seed=seed,
            batch_size=batch_size,
            repeat=repeat,
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

        return image_paths