# Tiled Diffusion
> Or Madar, Ohad Fried  
> Reichman University  
> In digital image processing and generation, tiling—the seamless connection of disparate images to create a coherent visual field—is crucial for applications such as texture creation, video game asset development, and digital art. Traditionally, tiles have been constructed manually, a method that poses significant limitations in scalability and flexibility. Recent research has attempted to automate this process using generative models. However, current approaches primarily focus on tiling textures and manipulating models for single-image generation, without inherently supporting the creation of multiple interconnected tiles across diverse domains.
This paper presents Tiled Diffusion, a novel approach that extends the capabilities of diffusion models to accommodate the generation of cohesive tiling patterns across various domains of image synthesis that require tiling. Our method supports a wide range of tiling scenarios, from self-tiling to complex many-to-many connections, enabling seamless integration of multiple images.
Tiled Diffusion automates the tiling process, eliminating the need for manual intervention and enhancing creative possibilities in various applications. We demonstrate its effectiveness in three key areas: seamlessly tiling existing images, tiled texture creation, and 360° synthesis. These applications showcase the versatility and potential of our method in addressing complex tiling challenges across different domains of image generation and processing.

<a href="https://sites.google.com/view/tiled-diffusion/home"><img src="https://img.shields.io/badge/arXiv-2306.00950-b31b1b?style=flat&logo=arxiv&logoColor=red"/></a>
<a href="https://sites.google.com/view/tiled-diffusion/home"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a>
<br/>

<p align="center">
  <img src="images/teaser.jpg" width="1000">
</p>


## Installation
```bash
conda create -n td python==3.10
conda activate td
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start
```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from latent_class import LatentClass
from model import SDLatentTiling

model = SDLatentTiling()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
prompt_1 = "Red brick texture"
prompt_2 = "Green brick texture"
negative_prompt = "blured, ugly, deformed, disfigured, poor details, bad anatomy, pixelized, bad order"
max_width = 32 # Context size (w)

# Many-to-many example on the X axis
lat1 = LatentClass(prompt=prompt_1, negative_prompt=negative_prompt, side_id=[1, 1, None, None],
                   side_dir=['cw', 'ccw', None, None])

lat2 = LatentClass(prompt=prompt_2, negative_prompt=negative_prompt, side_id=[1, 1, None, None],
                   side_dir=['cw', 'ccw', None, None])
latents_arr = [lat1, lat2]

new_latents_arr = model(latents_arr=latents_arr,
                        negative_prompt=negative_prompt,
                        max_width=max_width,
                        device=device)

lat1_new = new_latents_arr[0]
lat2_new = new_latents_arr[1]
t_1 = np.concatenate((lat1_new.image, lat2_new.image, lat1_new.image, lat2_new.image),
                     axis=1)

plt.imshow(t_1)
plt.show()
```

## Usage
```bash
python run.py
```
Under the file `run.py`, configure the latents using the class `LatentClass`, where `side_id` is a list of ids (i.e., the color patterns in the teaser for the constraints) and the `side_dir` is a list of orientations ('cw' or 'ccw' for clockwise or counter-clockwise). The list is always of size 4, with indexes corresponding to the sides (Right, Left, Up, Down). A simple example could be:

```python
side_id=[1, 1, None, None],
side_dir=['cw', 'ccw', None, None]
```

This means that the right and left sides should connect (the connection pattern must be from different orientations).
After the model finishes, the `LatentClass` will hold an attribute called `image` which is the result image of the diffusion process.
For ControlNet / Differential Diffusion / SD 3 / SD XL, please follow the corresponding directories (controlnet, diffdiff, sd3, sdxl) under the file name `example.py` (the same name under each directory).

## Examples
Here we provide declaration examples of self-tiling, one-to-one and many-to-many scenarios. We also include an example for img2img.

### Self-tiling
<img src="images/self.png" height="100">

```python
lat1 = LatentClass(prompt=PROMPT, negative_prompt=NEGATIVE_PROMPT, side_id=[1, 1, 2, 2],
                   side_dir=['cw', 'ccw', 'cw', 'ccw'])
```
This example represents a self-tiling scenario where I<sub>1</sub> would seamlessly connect to itself on the X / Y axis.

### One-to-one
<img src="images/one.png" height="100">

```python
lat1 = LatentClass(prompt=PROMPT1, negative_prompt=NEGATIVE_PROMPT1, side_id=[1, 2, None, None],
                   side_dir=['cw', 'ccw', None, None])
lat2 = LatentClass(prompt=PROMPT2, negative_prompt=NEGATIVE_PROMPT2, side_id=[2, 1, None, None],
                   side_dir=['cw', 'ccw', None, None])
```
This example represents a one-to-one scenario where I<sub>1</sub> and I<sub>2</sub> could connect to each other on the X axis.


### Many-to-many
<img src="images/many.png" height="100">

```python
lat1 = LatentClass(prompt=PROMPT1, negative_prompt=NEGATIVE_PROMPT1, side_id=[1, 1, None, None],
                   side_dir=['cw', 'ccw', None, None])
lat2 = LatentClass(prompt=PROMPT2, negative_prompt=NEGATIVE_PROMPT2, side_id=[1, 1, None, None],
                   side_dir=['cw', 'ccw', None, None])
```
This example represents a many-to-many scenario where I<sub>1</sub> and I<sub>2</sub> could connect to each other and to themselves on the X axis.

### Img2img
```python
from PIL import Image
image_path = 'images/sketch-mountains-input.jpg'
image = Image.open(image_path)
lat1 = LatentClass(prompt=PROMPT, negative_prompt=NEGATIVE_PROMPT, side_id=[1, 1, None, None],
                   side_dir=['cw', 'ccw', None, None], source_image=image)
```
When adding the flag `source_image`, the code will automatically detect and encode it with the VAE to start with that representation in the latent space, instead of using random gaussian noise. 
The result would be a transformed tiled image on the X axis. (Notice this is a general img2img and not the application `Tiling Existing Images`. To use the application please refer the file `example.py` under the folder `diffdiff`)
## Citation
```bibtex
BIBTEX TO BE HERE
```
