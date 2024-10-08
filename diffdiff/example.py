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


#
# with Image.open("/home/ubuntu/data/supp/DiffDiff/monkeys.png") as imageFile:
#     image = preprocess_image(imageFile)

with Image.open("/home/ubuntu/data/supp/DiffDiff/mask.jpg") as mapFile:
    map = preprocess_map(mapFile)



prompts = [
    "Monkeys in a jungle, cold color palette, muted colors",
    "Misty mountains at dawn, warm tones",
    "Desert oasis with camels, vibrant sunset hues",
    "Snowy forest with a lone wolf, blue-white color scheme",
    "Tropical beach with palm trees, pastel colors",
    "Lush green valley with grazing sheep, soft light",
    "Arctic tundra with polar bears, icy blue tones",
    "Autumn forest with deer, rich orange and red palette",
    "Serene lake reflection of mountains, purple twilight",
    "Savanna grasslands with elephants, golden hour lighting",
    "Cascading waterfall in rainforest, emerald green dominance",
    "Barren rocky canyon, earth tones and harsh shadows",
    "Cherry blossom garden, delicate pink and white hues",
    "Stormy sea with lighthouse, dramatic dark blues and grays",
    "Rolling hills of lavender fields, soft purple palette",
    "Volcanic landscape with lava flow, fiery reds and blacks",
    "Bamboo forest with giant pandas, misty greens",
    "Salt flats reflecting clouds, surreal white landscape",
    "Alpine meadow with wildflowers, vibrant spring colors",
    "Underwater coral reef, vivid aquatic color spectrum",
    "Northern lights over snowy peaks, ethereal greens and purples",
    "Sahara desert sand dunes, golden yellows and oranges",
    "Moss-covered ancient ruins in forest, mysterious greens",
    "Fjord with floating icebergs, cool blue tones",
    "Terraced rice fields, lush greens and reflective waters",
    "Red rock formations in canyon, warm earth tones",
    "Misty redwood forest, muted greens and browns",
    "Tulip fields in bloom, rainbow color palette",
    "Zen rock garden, minimalist grays and sand tones",
    "Thunderstorm over plains, dramatic dark purples and blues",
    "Autumn vineyard at sunset, rich reds and golds",
    "Frozen waterfall, icy blues and whites",
    "Mangrove swamp with egrets, murky greens and browns",
    "Highland moor with wild ponies, misty purples and greens",
    "Bioluminescent beach at night, glowing blues on dark sand",
    "Snow-capped mountain reflected in crystal clear lake",
    "Dense jungle canopy viewed from below, dappled light",
    "Geothermal hot springs, steamy whites and mineral blues",
    "Windswept coastal cliffs with seabirds, stormy grays",
    "Starry night sky over desert rock formations, deep blues and browns"
]

filenames = [
    "monkeys-jungle",
    "misty-mountains-dawn",
    "desert-oasis-camels",
    "snowy-forest-wolf",
    "tropical-beach-palms",
    "green-valley-sheep",
    "arctic-tundra-polarbears",
    "autumn-forest-deer",
    "lake-mountain-reflection",
    "savanna-elephants",
    "rainforest-waterfall",
    "barren-rocky-canyon",
    "cherry-blossom-garden",
    "stormy-sea-lighthouse",
    "lavender-hills",
    "volcanic-landscape-lava",
    "bamboo-forest-pandas",
    "salt-flats-clouds",
    "alpine-meadow-wildflowers",
    "underwater-coral-reef",
    "northern-lights-peaks",
    "sahara-sand-dunes",
    "mossy-ruins-forest",
    "fjord-icebergs",
    "terraced-rice-fields",
    "red-rock-canyon",
    "misty-redwood-forest",
    "tulip-fields-bloom",
    "zen-rock-garden",
    "thunderstorm-plains",
    "autumn-vineyard-sunset",
    "frozen-waterfall",
    "mangrove-swamp-egrets",
    "highland-moor-ponies",
    "bioluminescent-beach",
    "snow-mountain-lake",
    "dense-jungle-canopy",
    "geothermal-hot-springs",
    "windswept-coastal-cliffs",
    "starry-sky-desert-rocks"
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
    image_t.save(f"/home/ubuntu/data/supp/DiffDiff/orig_image/{filename}.png")
    orig_tiled_image.save(f"/home/ubuntu/data/supp/DiffDiff/orig_image_tiled/{filename}.png")
    edited_images.save(f"/home/ubuntu/data/supp/DiffDiff/tiled_image/{filename}.png")
    tiled_image.save(f"/home/ubuntu/data/supp/DiffDiff/tiled_image_tiled/{filename}.png")
    index += 1
print("Done!")
