import os

import torch
from diffusers import AutoPipelineForText2Image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

prompt = "Monkeys in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipeline_text2image(prompt=prompt).images[0]
image.save(r"C:\Or\Msc\Courses\Thesis\Tiled Diffusion\Figures\DiffDiff\monkeys.png")
