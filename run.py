import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from latent_class import LatentClass
from model import SDLatentTiling

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.cuda.empty_cache()
gc.collect()

scheduler = 'ddpm'
# scheduler = 'ddim'
model = SDLatentTiling(scheduler=scheduler)
# Parameters

prompt_1 = "Red brick texture"
prompt_2 = "Green brick texture"
negative_prompt = "blured, ugly, deformed, disfigured, poor details, bad anatomy, pixelized, bad order"
inference_steps = 30
seed = 151
cfg_scale = 7.5
max_replica_width = 5
max_width = 32
height = 512
width = 512
input_image = None

# IMAGE TO IMAGE #########################
strength = 0.92
# url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
# response = requests.get(url)
# input_image = Image.open(BytesIO(response.content)).convert("RGB")
# input_image = input_image.resize((768, 512))
# input_image = input_image.resize((576, 384))
###########################################

if input_image:
    width, height = input_image.size

# Right, Left, Up, Down
lat1 = LatentClass(prompt=prompt_1, negative_prompt=negative_prompt, side_id=[1, 1, None, None],
                   side_dir=['cw', 'ccw', None, None])

lat2 = LatentClass(prompt=prompt_2, negative_prompt=negative_prompt, side_id=[1, 1, None, None],
                   side_dir=['cw', 'ccw', None, None])

latents_arr = [lat1, lat2]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
new_latents_arr = model(latents_arr=latents_arr,
                        negative_prompt=negative_prompt,
                        inference_steps=inference_steps,
                        seed=seed,
                        cfg_scale=cfg_scale,
                        height=height,
                        width=width,
                        max_width=max_width,
                        max_replica_width=max_replica_width,
                        strength=strength,
                        device=device)

torch.cuda.empty_cache()
gc.collect()

lat1_new = new_latents_arr[0]
lat2_new = new_latents_arr[1]
t_1 = np.concatenate((lat1_new.image, lat2_new.image, lat1_new.image, lat2_new.image),
                     axis=1)

plt.imshow(t_1)
plt.show()
