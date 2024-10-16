import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image
from torchvision import transforms

from diffdiff.pipeline import StableDiffusionXLDiffImg2ImgPipeline

device = "cuda"
pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)

base = StableDiffusionXLDiffImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to(device)

refiner = StableDiffusionXLDiffImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to(device)


def preprocess_image(image):
    image = image.convert("RGB")
    image = transforms.CenterCrop((image.size[1] // 64 * 64, image.size[0] // 64 * 64))(image)
    image = transforms.ToTensor()(image)
    image = image * 2 - 1
    image = image.unsqueeze(0).to(device)
    return image


def preprocess_map(map):
    map = map.convert("L")
    map = transforms.CenterCrop((map.size[1] // 64 * 64, map.size[0] // 64 * 64))(map)
    # convert to tensor
    map = transforms.ToTensor()(map)
    map = map.to(device)
    return map


with Image.open("../images/mask.jpg") as mapFile:
    map = preprocess_map(mapFile)



prompts = [
    "Monkeys in a jungle, cold color palette, muted colors",
    "Misty mountains at dawn, warm tones"
]

filenames = [
    "monkeys-jungle",
    "misty-mountains-dawn"
]
negative_prompt = ["blurry, shadow polaroid photo, scary angry pose"]
index = 0
for prompt, filename in zip(prompts, filenames):
    print(f"{index}: {prompt}")
    pipeline_text2image = pipeline_text2image.to(device)
    image_t = pipeline_text2image(prompt=prompt).images[0]
    pipeline_text2image = pipeline_text2image.to('cpu')
    image = preprocess_image(image_t)
    edited_images = base(prompt=[prompt], original_image=image, image=image, strength=1, guidance_scale=17.5,
                         num_images_per_prompt=1,
                         negative_prompt=negative_prompt,
                         map=map,
                         num_inference_steps=100, denoising_end=0.8, max_width=64, output_type="latent").images

    edited_images = refiner(prompt=[prompt], original_image=image, image=edited_images, strength=1, guidance_scale=17.5,
                            num_images_per_prompt=1,
                            negative_prompt=negative_prompt,
                            map=map,
                            num_inference_steps=100, denoising_start=0.8, max_width=64).images[0]

    # Despite we use here both of the refiner and the base models,
    # one can use only the base model, or only the refiner (for low strengths).
    # Create a new image with 3 times the width of the original
    new_width = edited_images.width * 3
    new_height = edited_images.height
    orig_tiled_image = Image.new("RGB", (new_width, new_height))
    tiled_image = Image.new("RGB", (new_width, new_height))

    # Paste the original image three times
    for i in range(3):
        tiled_image.paste(edited_images, (i * edited_images.width, 0))

    for i in range(3):
        orig_tiled_image.paste(image_t, (i * image_t.width, 0))

    # save
    image_t.save(f"{filename}_orig.png")
    orig_tiled_image.save(f"{filename}_orig_tiled.png")
    edited_images.save(f"{filename}_tiled.png")
    tiled_image.save(f"{filename}_tiled_tiled.png")
    index += 1
print("Done!")
