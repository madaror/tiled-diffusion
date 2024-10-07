import gc

import cv2
import numpy as np
import torch

from sd3.pipeline import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",
                                                torch_dtype=torch.float16)
pipe = pipe.to("cuda")

out_path = "/home/ubuntu/data/supp/sd3"

# List of 30 tileable landscape prompts for horizontal panoramas
tileable_landscape_prompts = [
    "Endless rolling hills with golden wheat fields, scattered oak trees, puffy white clouds in a blue sky, warm sunset light, highly detailed landscape painting style",
    "Serene beach scene with white sand, gentle waves, palm trees swaying in the breeze, pastel sky at dusk, photorealistic style",
    "Misty mountain range with snow-capped peaks, evergreen forests in the foreground, a crystal-clear lake reflecting the scenery, early morning light, Bob Ross painting style",
    "Vast desert landscape with rippling sand dunes, occasional hardy shrubs, heat shimmer in the distance, clear blue sky with wispy clouds, National Geographic photography style",
    "Lush tropical rainforest canopy, vibrant green foliage, colorful exotic flowers, sunlight filtering through leaves, mist rising, highly detailed digital art style",
    "Serene Japanese garden with cherry blossom trees, stone lanterns, a gently curved bridge over a koi pond, raked gravel patterns, soft focus photography style",
    "Endless lavender fields in Provence, old stone farmhouse in the distance, cypress trees lining a dirt road, warm afternoon light, impressionist painting style",
    "Arctic tundra landscape under the midnight sun, low-growing colorful vegetation, distant ice formations, subtle aurora in the sky, realistic rendering style",
    "Rolling Tuscan countryside with cypress-lined roads, terracotta-roofed villas, olive groves and vineyards, golden hour lighting, romantic oil painting style",
    "Mystical redwood forest with towering trees, shafts of sunlight penetrating the canopy, ferns and moss-covered logs, mist hugging the ground, ethereal photography style",
    "Idyllic Alpine meadow with wildflowers, grazing cattle, traditional wooden chalets, snow-capped mountains in the background, clear mountain air, hyper-realistic style",
    "Tranquil rice terraces in Bali, lush green steps carved into hillsides, palm trees and traditional huts, misty mountains in the distance, dreamy watercolor style",
    "Scottish Highlands with rugged mountains, heather-covered moors, a still loch reflecting the sky, ancient castle ruins in the distance, moody overcast lighting, oil painting style",
    "Autumn forest scene with a mixture of deciduous trees in vibrant fall colors, a babbling brook with moss-covered rocks, golden sunlight filtering through leaves, romantic landscape painting style",
    "Serene winterscape with snow-covered pine trees, a frozen lake, distant mountains, and a colorful sunset sky, cabin with smoking chimney in the distance, Thomas Kinkade inspired style",
    "Dramatic coastline with rugged cliffs, crashing waves, a lighthouse perched on a rocky outcrop, seabirds soaring overhead, stormy sky with breaks of sunlight, Turner-inspired seascape style",
    "Peaceful English countryside with rolling green hills, hedgerows, grazing sheep, a winding country lane, and a quaint village in the distance, soft watercolor style",
    "Majestic Grand Canyon vista with layered rock formations, deep shadows and highlights, a winding river far below, wispy clouds in a blue sky, vintage National Park poster style",
    "Ethereal cherry blossom scene with a torii gate, gentle slope of Mount Fuji in the background, rippling pond with reflection, soft pink petals floating in the breeze, delicate Japanese art style",
    "Sweeping Sahara desert landscape with wind-sculpted sand dunes, an oasis with palm trees, camel caravan silhouette on the horizon, rich sunset colors, cinematic wide-angle style",
    "Lush Pacific Northwest forest with moss-draped trees, ferns, nurse logs, a misty waterfall in the distance, shafts of sunlight through the canopy, photorealistic rendering",
    "Picturesque Santorini scene with white-washed buildings, blue-domed churches, winding cobblestone streets, bougainvillea flowers, Aegean Sea view, vibrant travel photography style",
    "Serene Zen rock garden with carefully raked sand patterns, moss-covered rocks, minimalist architecture, bamboo grove in the background, soft diffused lighting, meditative art style",
    "Vibrant coral reef underwater scene with diverse marine life, colorful fish, swaying sea fans, shafts of sunlight penetrating clear blue water, nature documentary style",
    "Majestic Norwegian fjord landscape with steep cliffs, calm waters reflecting the scenery, snow-capped mountains, traditional red wooden houses, dramatic cloudy sky, realistic digital painting style",
    "Enchanted fairy tale forest with gnarled trees, glowing mushrooms, fireflies, a babbling brook, misty atmosphere, fantasy illustration style",
    "Panoramic view of terraced rice fields in China, layers of green and gold, mist-shrouded mountains, traditional houses on stilts, ethereal morning light, fine art photography style",
    "Dramatic Iceland landscape with black sand beach, basalt columns, crashing waves, distant glaciers, moody sky with a break of sunlight, high-contrast photography style",
    "Serene Kyoto bamboo grove with towering green stalks, stone pathway, shafts of sunlight, misty atmosphere, Zen-inspired minimalist photography style",
    "Expansive Great Plains landscape with waving grass, scattered wildflowers, grazing bison, distant storm clouds, golden prairie light, American Western art style"
]

# List of unique names for each tileable landscape prompt
tileable_landscape_names = [
    "rolling_hills_wheat_fields",
    "serene_beach_sunset",
    "misty_mountains_lake",
    "vast_desert_dunes",
    "lush_tropical_rainforest",
    "japanese_garden_cherry_blossoms",
    "provence_lavender_fields",
    "arctic_tundra_midnight_sun",
    "tuscan_countryside_golden_hour",
    "mystical_redwood_forest",
    "alpine_meadow_wildflowers",
    "bali_rice_terraces",
    "scottish_highlands_loch",
    "autumn_forest_stream",
    "winter_wonderland_sunset",
    "dramatic_coastal_cliffs",
    "english_countryside_village",
    "grand_canyon_vista",
    "mount_fuji_cherry_blossoms",
    "sahara_desert_oasis",
    "pacific_northwest_forest",
    "santorini_aegean_view",
    "zen_rock_garden",
    "vibrant_coral_reef",
    "norwegian_fjord_landscape",
    "enchanted_fairy_tale_forest",
    "china_terraced_rice_fields",
    "iceland_black_sand_beach",
    "kyoto_bamboo_grove",
    "great_plains_prairie"
]
index = 0
for prompt, name in zip(tileable_landscape_prompts, tileable_landscape_names):
    print(f"{index}: {name}")
    torch.cuda.empty_cache()
    gc.collect()
    image = pipe(
        prompt,
        negative_prompt="",
        num_inference_steps=30,
        guidance_scale=7.0,
        max_width=40,
        height=1024,
        width=1024
    ).images[0]
    image = np.array(image)
    image_uint8 = image.astype(np.uint8)
    image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2RGB)
    t_1 = np.concatenate((image_rgb, image_rgb, image_rgb), axis=1)
    cv2.imwrite(f"{out_path}/{name}.png", t_1)
    index += 1
