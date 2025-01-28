from PIL import Image
import numpy as np
from typing import Tuple, List, Literal, Optional
import os
from pathlib import Path

Direction = Literal['left', 'right', 'up', 'down', 'top_left', 'bottom_left', 'top_right', 'bottom_right']


def resize_image(image: Image.Image, target_size: int) -> Image.Image:
    """
    Resize image while maintaining aspect ratio so its longest dimension equals target_size.

    Args:
        image: PIL Image to resize
        target_size: Target size for the longest dimension

    Returns:
        Resized PIL Image
    """
    # Get current dimensions
    width, height = image.size

    # Calculate new dimensions maintaining aspect ratio
    if width > height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))

    # Resize image with LANCZOS resampling for better quality
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def create_sliding_gif(
        image_path: str,
        output_path: str,
        direction: Direction = 'right',
        duration: int = 5000,  # Total duration in milliseconds
        num_frames: int = 30,
        target_size: Optional[int] = None  # Target size for the longest dimension
) -> None:
    """
    Create a sliding GIF from a tileable image with proper cyclic movement.

    Args:
        image_path: Path to the input image
        output_path: Path where the GIF will be saved
        direction: Direction of movement
        duration: Total duration of the GIF in milliseconds
        num_frames: Number of frames in the GIF
        target_size: Optional target size for the longest dimension
    """
    # Load the image
    img = Image.open(image_path)

    # Convert image to RGBA if it isn't already
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # Resize image if target_size is specified
    if target_size is not None:
        img = resize_image(img, target_size)

    width, height = img.size

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

        # Calculate offsets
        offset_x = int(progress * width)
        offset_y = int(progress * height)

        # Adjust offset based on direction
        final_offset_x = offset_x * vector[0]
        final_offset_y = offset_y * vector[1]

        # The crop box should always be centered on the middle tile
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

    # Save the GIF with additional optimization
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0,
        optimize=True,
        quality=85  # Slightly reduce quality for smaller file size
    )


def process_directory(
        input_dir: str,
        output_dir: str,
        direction: Direction,
        duration: int = 5000,
        num_frames: int = 30,
        target_size: Optional[int] = None
) -> None:
    """
    Process all images in input directory and create GIFs in output directory.

    Args:
        input_dir: Directory containing input images
        output_dir: Directory where GIFs will be saved
        direction: Direction for the animation
        duration: Duration of each GIF in milliseconds
        num_frames: Number of frames for each GIF
        target_size: Optional target size for the longest dimension
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
                    num_frames=num_frames,
                    target_size=target_size
                )
                print(f"Created {output_filename}")

                # Print file size
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"Output file size: {size_mb:.2f} MB")

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
    parser.add_argument('--target-size', type=int, default=512,
                        help='Target size for longest dimension in pixels (default: 512)')

    args = parser.parse_args()

    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        direction=args.direction,
        duration=args.duration,
        num_frames=args.frames,
        target_size=args.target_size
    )