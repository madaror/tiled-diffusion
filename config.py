import PIL.Image
import PIL.ImageOps
from packaging import version

MODEL_ID_SD_1_5 = "runwayml/stable-diffusion-v1-5"
MODEL_ID_SD_2_0 = "stabilityai/stable-diffusion-2"
CLIP_MODEL_EVALUATION = "openai/clip-vit-base-patch32"

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

DIRECTION_INDEX_TO_STRING = {
    0: 'right',
    1: 'left',
    2: 'up',
    3: 'down'
}

DIRECTION_STRING_TO_INDEX = {
    'right': 0,
    'left': 1,
    'up': 2,
    'down': 3
}

SIMILARITY_ROTATION_MATRIX = [
    [0, 2, 1, 3],
    [2, 0, 3, 1],
    [3, 1, 0, 2],
    [1, 3, 2, 0]
]

TILING_ROTATION_MATRIX = [
    [2, 0, 1, 3],
    [0, 2, 3, 1],
    [3, 1, 2, 0],
    [1, 3, 0, 2]
]
