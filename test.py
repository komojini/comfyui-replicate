import sys
from pathlib import Path

OUTPUT_PATH = Path("./ComfyUI/output")


if __name__ == "__main__":
    
    sys.path.extend(["./ComfyUI/ComfyUI-to-Python-Extension", "./ComfyUI/ComfyUI-to-Python-Extension/output_files"])

    from ipadapter_faceid_upscale_replicate import main

    images = main(
            "photo of a man, fashion model, simple clothes\n\nhigh quality, highly detailed, 4k, highres",
            "blurry, distorted, low quality, bad hands",
            "/home/qwaezrx/projects/docker/comfyui-replicate/ComfyUI/input/Image 2023-12-23 at 8.12 PM (1).jpeg",
            repeat=2,
            batch_size=2,
        )

    print(images)

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
            image_paths.append(image_path)
    
    print(image_paths)

