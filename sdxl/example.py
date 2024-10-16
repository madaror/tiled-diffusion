import gc

import cv2
import numpy as np
import torch

from sdxl.pipeline import StableDiffusionXLPipeline

# load both base & refiner
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.to("cuda")

out_path = ""

# List of 30 tileable landscape prompts for horizontal panoramas
tileable_landscape_prompts = [
    "Endless rolling hills with golden wheat fields, scattered oak trees, puffy white clouds in a blue sky, warm sunset light, highly detailed landscape painting style",
    "Serene beach scene with white sand, gentle waves, palm trees swaying in the breeze, pastel sky at dusk, photorealistic style"
]

# List of unique names for each tileable landscape prompt
tileable_landscape_names = [
    "rolling_hills_wheat_fields",
    "serene_beach_sunset"
]
index = 0

for prompt, name in zip(tileable_landscape_prompts, tileable_landscape_names):
    torch.cuda.empty_cache()
    gc.collect()
    image = pipe(
        prompt,
        negative_prompt="",
        num_inference_steps=40,
        guidance_scale=7.0,
        max_width=64,
        height=1024,
        width=1024,
        output_type="pil"
    ).images[0]
    image = np.array(image)
    image_uint8 = image.astype(np.uint8)
    image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2RGB)
    t_1 = np.concatenate((image_rgb, image_rgb, image_rgb), axis=1)
    cv2.imwrite(f"{out_path}/{name}.png", t_1)
    index += 1
