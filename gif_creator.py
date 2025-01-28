from PIL import Image
import numpy as np
from typing import Tuple, List, Literal
import os
from pathlib import Path

Direction = Literal['left', 'right', 'up', 'down', 'top_left', 'bottom_left', 'top_right', 'bottom_right']


def create_sliding_gif(
        image_path: str,
        output_path: str,
        direction: Direction = 'right',
        duration: int = 5000,  # Total duration in milliseconds
        num_frames: int = 30
) -> None:
    """
    Create a sliding GIF from a tileable image with proper cyclic movement.

    Args:
        image_path: Path to the input image
        output_path: Path where the GIF will be saved
        direction: Direction of movement
        duration: Total duration of the GIF in milliseconds
        num_frames: Number of frames in the GIF
    """
    # Load the image
    img = Image.open(image_path)
    width, height = img.size

    # Convert image to RGBA if it isn't already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # Create a 3x3 tiled image to ensure smooth wrapping
    tiled_width = width * 3
    tiled_height = height * 3
    tiled = Image.new('RGBA', (tiled_width, tiled_height))

    # Fill the 3x3 grid with the original image
    for y in range(3):
        for x in range(3):
            tiled.paste(img, (x * width, y * height))

    # Calculate movement vectors and step sizes for each direction
    vectors = {
        'left': (-1, 0),
        'right': (1, 0),
        'up': (0, -1),
        'down': (0, 1),
        'top_left': (-1, -1),
        'bottom_left': (-1, 1),
        'top_right': (1, -1),
        'bottom_right': (1, 1)
    }

    vector = vectors[direction]
    frames: List[Image.Image] = []

    # Create frames
    for i in range(num_frames):
        # Calculate offset for this frame
        progress = i / num_frames

        # Calculate offsets. For diagonal movements, we want to move
        # one full image width/height over the course of the animation
        offset_x = int(progress * width)
        offset_y = int(progress * height)

        # Adjust offset based on direction
        final_offset_x = offset_x * vector[0]
        final_offset_y = offset_y * vector[1]

        # The crop box should always be centered on the middle tile
        # Starting from the center tile (at position width, height)
        crop_box = (
            width + final_offset_x,  # left
            height + final_offset_y,  # top
            2 * width + final_offset_x,  # right
            2 * height + final_offset_y  # bottom
        )

        # Crop the frame and append to frames list
        frame = tiled.crop(crop_box)
        frames.append(frame)

    # Calculate duration per frame
    frame_duration = duration // num_frames

    # Save the GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0,
        optimize=True
    )


def process_directory(
        input_dir: str,
        output_dir: str,
        direction: Direction,
        duration: int = 5000,
        num_frames: int = 30
) -> None:
    """
    Process all images in input directory and create GIFs in output directory.

    Args:
        input_dir: Directory containing input images
        output_dir: Directory where GIFs will be saved
        direction: Direction for the animation
        duration: Duration of each GIF in milliseconds
        num_frames: Number of frames for each GIF
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Supported image formats
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}

    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        # Check if file is an image
        file_ext = Path(filename).suffix.lower()
        if file_ext in image_extensions:
            input_path = os.path.join(input_dir, filename)
            # Create output filename by replacing extension with .gif
            output_filename = Path(filename).stem + '.gif'
            output_path = os.path.join(output_dir, output_filename)

            try:
                print(f"Processing {filename}...")
                create_sliding_gif(
                    image_path=input_path,
                    output_path=output_path,
                    direction=direction,
                    duration=duration,
                    num_frames=num_frames
                )
                print(f"Created {output_filename}")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create sliding GIFs from tileable images')
    parser.add_argument('input_dir', help='Directory containing input images')
    parser.add_argument('output_dir', help='Directory where GIFs will be saved')
    parser.add_argument('direction', choices=[
        'left', 'right', 'up', 'down',
        'top_left', 'bottom_left', 'top_right', 'bottom_right'
    ], help='Direction of animation')
    parser.add_argument('--duration', type=int, default=5000,
                        help='Duration of GIF in milliseconds (default: 5000)')
    parser.add_argument('--frames', type=int, default=30,
                        help='Number of frames in GIF (default: 30)')

    args = parser.parse_args()

    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        direction=args.direction,
        duration=args.duration,
        num_frames=args.frames
    )