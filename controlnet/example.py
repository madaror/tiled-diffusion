import os

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import ControlNetModel, UniPCMultistepScheduler

from controlnet.pipeline import StableDiffusionControlNetPipeline
from controlnet.preprocess import new_prompts

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16,
                                             use_safetensors=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

out_path = r"C:\Or\Msc\Courses\Thesis\Tiled Diffusion\Figures\Supp\SD3\input"
res_path = r"C:\Or\Msc\Courses\Thesis\Tiled Diffusion\Figures\Supp\SD3\res"  # Define a new directory for processed images
init_path = r"C:\Or\Msc\Courses\Thesis\Tiled Diffusion\Figures\Supp\SD3\init"  # Define a new directory for processed images
mask_path = r"C:\Or\Msc\Courses\Thesis\Tiled Diffusion\Figures\Supp\SD3\mask"  # Define a new directory for processed images

for filename in os.listdir(out_path):
    if filename.endswith('.png'):
        # Extract the base name without extension
        base_name = os.path.splitext(filename)[0]

        # Open the image
        img_path = os.path.join(out_path, filename)
        with Image.open(img_path) as img:
            # Crop the image to 1024x1024, taking the first 1024 pixels
            cropped_img = img.crop((0, 0, 1024, 1024))

            # Resize the image to 512x512
            resized_img = cropped_img.resize((512, 512), Image.LANCZOS)

            # Save the processed image
            processed_img_path = os.path.join(init_path, f"{base_name}.png")
            resized_img.save(processed_img_path)
            max_width = 32
            # original_image = wrap_edges_pil(image=resized_img, max_width=max_width * 8)
            original_image = resized_img
            image = np.array(original_image)

            low_threshold = 100
            high_threshold = 200

            image = cv2.Canny(image, low_threshold, high_threshold)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            canny_image = Image.fromarray(image)
            processed_img_path = os.path.join(mask_path, f"{base_name}.png")
            canny_image.save(processed_img_path)

            # Apply new prompts using ControlNet (pseudo-code)
            for i, new_prompt in enumerate(new_prompts[base_name]):
                output = pipe(
                    new_prompt, image=canny_image, height=512, width=512, max_width=max_width
                ).images[0]
                image = np.array(output)
                image_uint8 = image.astype(np.uint8)
                image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2RGB)
                t_1 = np.concatenate((image_rgb, image_rgb, image_rgb), axis=1)
                cv2.imwrite(f"{res_path}/{base_name}_{i}.png", t_1)
