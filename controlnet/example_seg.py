import os

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import ControlNetModel, UniPCMultistepScheduler
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

from controlnet.pipeline import StableDiffusionControlNetPipeline
from controlnet.preprocess import new_prompts
from controlnet.seg_color_palette import palette

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16,
                                             use_safetensors=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
)
image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

out_path = r"C:\Or\Msc\Courses\Thesis\Tiled Diffusion\Figures\Supp\SD3\input"
res_path = r"C:\Or\Msc\Courses\Thesis\Tiled Diffusion\Figures\Supp\SD3_SEG\res"  # Define a new directory for processed images
init_path = r"C:\Or\Msc\Courses\Thesis\Tiled Diffusion\Figures\Supp\SD3_SEG\init"  # Define a new directory for processed images
mask_path = r"C:\Or\Msc\Courses\Thesis\Tiled Diffusion\Figures\Supp\SD3_SEG\mask"  # Define a new directory for processed images

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
            # Segmentation
            pixel_values = image_processor(original_image, return_tensors="pt").pixel_values
            with torch.no_grad():
                outputs = image_segmentor(pixel_values)

            seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[original_image.size[::-1]])[
                0]
            color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
            for label, color in enumerate(palette):
                color_seg[seg == label, :] = color

            color_seg = color_seg.astype(np.uint8)

            image_seg = Image.fromarray(color_seg)
            processed_img_path = os.path.join(mask_path, f"{base_name}.png")
            image_seg.save(processed_img_path)

            # Apply new prompts using ControlNet (pseudo-code)
            for i, new_prompt in enumerate(new_prompts[base_name]):
                output = pipe(
                    new_prompt, image=image_seg, height=512, width=512, max_width=max_width
                ).images[0]
                image = np.array(output)
                image_uint8 = image.astype(np.uint8)
                image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2RGB)
                t_1 = np.concatenate((image_rgb, image_rgb, image_rgb), axis=1)
                cv2.imwrite(f"{res_path}/{base_name}_{i}.png", t_1)
