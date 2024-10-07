import inspect
import random
from typing import Optional, Union, List, Tuple

import PIL
import numpy as np
import torch
from PIL import Image

from config import PIL_INTERPOLATION


def is_round(number):
    return number == round(number)


def print_2d_tensor(tensor):
    # Iterate through rows of the tensor
    for row in tensor:
        # Format each row as a list of values
        formatted_row = ' '.join(f'{val:.2f}' for val in row)
        print(f'[ {formatted_row} ]')


def preprocess(image):
    deprecation_message = "The preprocess method is deprecated and will be removed in diffusers 1.0.0. Please use VaeImageProcessor.preprocess(...) instead"
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8

        image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


def retrieve_timesteps(
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def get_timesteps(scheduler, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start * scheduler.order:]
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(t_start * scheduler.order)

    return timesteps, num_inference_steps - t_start


def retrieve_latents(
        encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def randn_tensor(
        shape: Union[Tuple, List],
        generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
        device: Optional["torch.device"] = None,
        dtype: Optional["torch.dtype"] = None,
        layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = torch.device(device)

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


def final_step_latents_fix(latents, direction, max_width=10, max_height=10):
    if direction == "x":
        latents[:, :, :, :max_width] = torch.flip(latents[:, :, :, -max_width:], [3])
    elif direction == "y":
        latents[:, :, :max_height, :] = torch.flip(latents[:, :, -max_height:, :], [2])
    else:  # xy
        latents[:, :, :max_height, :max_width] = torch.flip(latents[:, :, -max_height:, -max_width:], [2, 3])
    return latents


def seamless_tiling(x_axis, y_axis, vae, text_encoder, unet):
    def asymmetric_conv2d_convforward(self, input: torch.Tensor, weight: torch.Tensor,
                                      bias: Optional[torch.Tensor] = None):
        self.paddingX = (self._reversed_padding_repeated_twice[0], self._reversed_padding_repeated_twice[1], 0, 0)
        self.paddingY = (0, 0, self._reversed_padding_repeated_twice[2], self._reversed_padding_repeated_twice[3])
        working = torch.nn.functional.pad(input, self.paddingX, mode=x_mode)
        working = torch.nn.functional.pad(working, self.paddingY, mode=y_mode)
        return torch.nn.functional.conv2d(working, weight, bias, self.stride, torch.nn.modules.utils._pair(0),
                                          self.dilation, self.groups)

    x_mode = 'circular' if x_axis else 'constant'
    y_mode = 'circular' if y_axis else 'constant'

    targets = [vae, text_encoder, unet]
    # targets = [unet]
    for target in targets:
        for module in target.modules():
            if isinstance(module, torch.nn.Conv2d):
                module._conv_forward = asymmetric_conv2d_convforward.__get__(module, torch.nn.Conv2d)
    return targets[0], targets[1], targets[2]


def transition_tensor(tensor, step, max_steps, direction, source_direction, max_width):
    """
    Transition from a normal distribution tensor to the input tensor
    based on the given step and max_steps.

    :param tensor: The target tensor to transition to.
    :param step: The current step in the transition.
    :param max_steps: The maximum number of steps in the transition.
    :return: A tensor transitioned from normal distribution to the input tensor.
    """
    B, F, H, W = tensor.size()
    if step > max_steps:
        step = max_steps

    # Slice the tensor according to the direction
    if direction == 'right':
        tensor = tensor[:, :, :, W - 2 * max_width: W - max_width]
    elif direction == 'left':
        tensor = tensor[:, :, :, max_width: 2 * max_width]
    elif direction == 'up':
        tensor = tensor[:, :, max_width: 2 * max_width, :]
    else:  # direction == 'down'
        tensor = tensor[:, :, H - 2 * max_width: H - max_width, :]
    # Generate a tensor with the same shape as the input tensor but from a normal distribution
    norm_dist_tensor = torch.randn_like(tensor)

    # Calculate the weight for linear interpolation
    alpha = step / max_steps if max_steps > 0 else 1.0

    # Linearly interpolate between the normal distribution tensor and the target tensor
    transitioned_tensor = (1 - alpha) * norm_dist_tensor + alpha * tensor
    if source_direction == direction:
        rotated_tensor = torch.rot90(transitioned_tensor, 2, [2, 3])
        # rotated_tensor = torch.flip(transitioned_tensor, [2, 3])
    elif (source_direction == 'right' and direction == 'up') or (
            source_direction == 'left' and direction == 'down') or (
            source_direction == 'up' and direction == 'left') or (source_direction == 'down' and direction == 'right'):
        rotated_tensor = torch.rot90(transitioned_tensor, 1, [2, 3])
        # rotated_tensor = transitioned_tensor.permute(0, 1, 3, 2)
    elif (source_direction == 'right' and direction == 'down') or (
            source_direction == 'left' and direction == 'up') or (
            source_direction == 'up' and direction == 'right') or (source_direction == 'down' and direction == 'left'):
        rotated_tensor = torch.rot90(transitioned_tensor, 3, [2, 3])
        # rotated_tensor = transitioned_tensor.permute(0, 1, 3, 2)
        # rotated_tensor = torch.flip(rotated_tensor, [2, 3])
    else:
        # If none is satisfied then the directions are complete opposites and nothing needs to be done
        rotated_tensor = transitioned_tensor
    return rotated_tensor


def organize_instances(instances):
    result = {'left': {}, 'right': {}, 'up': {}, 'down': {}}
    for instance in instances:
        for key in ['left', 'right', 'up', 'down']:
            value = getattr(instance, key)
            if value is not None:
                if value not in result[key]:
                    result[key][value] = []
                result[key][value].append(instance)

    return result


# Right Left, Up, Down
def generate_graph_groups(latents_arr):
    groups = {}
    for latent_idx, latent in enumerate(latents_arr):
        side_id = latent.side_id
        side_dir = latent.side_dir
        for side_idx, side_val in enumerate(side_id):
            if side_val is None:
                continue
            current_side_dir = side_dir[side_idx]
            if current_side_dir == 'cw':
                opposite_side_dir = 'ccw'
            else:
                opposite_side_dir = 'cw'
            for latent2_idx, latent2 in enumerate(latents_arr):
                side_id_2 = latent2.side_id
                side_dir_2 = latent2.side_dir
                for target_side_idx, targets in enumerate(zip(side_id_2, side_dir_2)):
                    target_side, target_dir = targets
                    if target_side == side_val and target_dir == opposite_side_dir:
                        key = f"{latent_idx}_{side_idx}"
                        print(f"Setting key to: {key}")
                        if key not in groups:
                            groups[key] = {'target_latent_idx': [], 'target_side_idx': []}
                        groups[key]['target_latent_idx'].append(latent2_idx)
                        groups[key]['target_side_idx'].append(target_side_idx)
    return groups


def get_tensor_slice_from_latent_and_side_idx(latent, side_idx, max_width, max_replica_width):
    B, F, H, W = latent.size()
    if side_idx == 0:  # Right
        tensor = latent[:, :, :, W - max_width - max_replica_width: W - max_width]
    elif side_idx == 1:  # Left
        tensor = latent[:, :, :, max_width:max_width + max_replica_width]
    elif side_idx == 2:  # Up
        tensor = latent[:, :, max_width:max_width + max_replica_width, :]
    else:  # Down
        tensor = latent[:, :, H - max_width - max_replica_width: H - max_width, :]
    return tensor


def calculate_angle_gradients(img_1, img_2, direction):
    if direction == 'x':
        # Select the right-most column of img_1 and the left-most column of img_2
        col_1 = img_1[:, -1]  # Right-most column
        col_2 = img_2[:, 0]  # Left-most column
    elif direction == 'y':
        # Select the top-most row of img_1 and the bottom-most row of img_2
        col_1 = img_1[0, :]  # Top-most row
        col_2 = img_2[-1, :]  # Bottom-most row
    else:
        raise ValueError("Direction must be either 'x' or 'y'")

    # Initialize a list to hold the minimum gradients for each pixel in col_1
    min_gradients = []

    # Loop through each pixel in col_1
    for i, pixel in enumerate(col_1):
        gradients = []

        if direction == 'x':
            pixel_indices = range(max(0, i - 1), min(len(col_2), i + 2))
        elif direction == 'y':
            pixel_indices = range(max(0, i - 1), min(len(col_2), i + 2))

        # Compute absolute gradients for different neighbors within angular view
        for j in pixel_indices:
            gradient = np.abs(pixel - col_2[j])
            gradients.append(gradient)

        # Append the minimum gradient found for this pixel
        min_gradients.append(np.min(gradients))

    # Calculate the mean of the minimal gradients
    mean_min_gradient = np.mean(min_gradients)

    return mean_min_gradient


def mean_absolute_gradient(img_1, img_2, direction, width_size=15):
    """
    Calculate the mean absolute gradient value across the connection area of two images.

    Parameters:
        img_1 (np.array): The first image in the form of a 2D numpy array.
        img_2 (np.array): The second image, also as a 2D numpy array.
        direction (str): 'x' to connect along the X axis, 'y' to connect along the Y axis.

    Returns:
        float: Mean absolute gradient across the connection area.
    """
    h, w, _ = img_1.shape
    if direction == 'x':
        # Connecting on the X axis
        # - Rightmost column from img_1
        col_1 = img_1[:, -1]
        # - Leftmost column from img_2
        col_2 = img_2[:, 0]

        # left side mask
        col_1_left = img_1[:, -width_size - 1]
        col_2_left = img_1[:, -width_size]

        # right side mask
        col_1_right = img_2[:, width_size + 1]
        col_2_right = img_2[:, width_size]
    elif direction == 'y':
        # Connecting on the Y axis
        # - Bottommost row from img_1
        col_1 = img_1[-1, :]
        # - Topmost row from img_2
        col_2 = img_2[0, :]

        col_1_left = col_1
        col_2_left = col_2

        # right side mask
        col_1_right = col_1
        col_2_right = col_2

    else:
        raise ValueError("Direction must be either 'x' or 'y'.")

    # Calculating the absolute gradients
    gradients_1 = np.abs(col_1 - col_2)
    gradients_2 = np.abs(col_1_left - col_2_left)
    gradients_3 = np.abs(col_1_right - col_2_right)

    # Calculating the mean of these gradients
    mean_gradient_1 = np.mean(gradients_1)
    mean_gradient_2 = np.mean(gradients_2)
    mean_gradient_3 = np.mean(gradients_3)

    # mean_gradient = (mean_gradient_1 + mean_gradient_2 + mean_gradient_3) / 3.0
    mean_gradient = max(mean_gradient_1, mean_gradient_2, mean_gradient_3)

    return mean_gradient


def harmonize_edges(latents_arr, max_replica_width, groups):
    for key, val in groups.items():
        target_latent_idx = val['target_latent_idx']
        target_side_idx = val['target_side_idx']

        if len(target_side_idx) > 1:
            # Randomly choose one target as a reference
            ref_index = random.randint(0, len(target_latent_idx) - 1)
            ref_latent = latents_arr[target_latent_idx[ref_index]]
            ref_side = target_side_idx[ref_index]
            ref_img = ref_latent.image

            H, W, C = ref_img.shape

            # Extract the edge from the reference image
            if ref_side == 0:  # Right
                edge = ref_img[:, -max_replica_width:, :]
            elif ref_side == 1:  # Left
                edge = ref_img[:, :max_replica_width, :]
            elif ref_side == 2:  # Up
                edge = ref_img[:max_replica_width, :, :]
            else:  # Down
                edge = ref_img[-max_replica_width:, :, :]

            for i, (latent_idx, side) in enumerate(zip(target_latent_idx, target_side_idx)):
                if i == ref_index:
                    continue  # Skip the reference image

                current_latent = latents_arr[latent_idx]
                current_img = current_latent.image

                # Calculate rotation needed
                rotation = (side - ref_side) % 4
                rotated_edge = np.rot90(edge, k=rotation)

                # Apply the rotated edge to the current image
                if side == 0:  # Right
                    current_img[:, -max_replica_width:, :] = rotated_edge
                elif side == 1:  # Left
                    current_img[:, :max_replica_width, :] = rotated_edge
                elif side == 2:  # Up
                    current_img[:max_replica_width, :, :] = rotated_edge
                else:  # Down
                    current_img[-max_replica_width:, :, :] = rotated_edge

                current_latent.image = current_img

    return latents_arr


def pad_tensor_x(tensor, max_width):
    # Get the original dimensions
    batch, channels, height, width = tensor.shape

    # Create a new tensor with the desired shape, initialized with zeros
    padded_tensor = torch.zeros(batch, channels, height, width + 2 * max_width,
                                dtype=tensor.dtype, device=tensor.device)

    # Copy the original tensor into the center of the new tensor
    padded_tensor[:, :, :, max_width:max_width + width] = tensor

    return padded_tensor


def wrap_edges_x(tensor, max_width):
    # Get the dimensions of the tensor
    batch, channels, height, width = tensor.shape

    # Ensure the tensor is wide enough for the operation
    # if width < 3 * max_width:
    #     raise ValueError(f"Tensor width ({width}) must be at least 3 times max_width ({max_width})")

    # Create a new tensor to avoid modifying the input tensor in-place
    result = tensor.clone()

    # Paste the inner portion to the left side
    result[:, :, :, :max_width] = tensor[:, :, :, width - 2 * max_width:width - max_width]

    # Paste the inner portion to the right side
    result[:, :, :, width - max_width:] = tensor[:, :, :, max_width:2 * max_width]

    return result


def wrap_edges_pil(image, max_width):
    # Convert PIL Image to numpy array
    img_array = np.array(image)

    # Get the dimensions of the image
    height, width, channels = img_array.shape

    # Calculate the new width
    new_width = width + 2 * max_width

    # Create a new array with the expanded width
    result = np.zeros((height, new_width, channels), dtype=img_array.dtype)

    # Copy the original image into the center of the new array
    result[:, max_width:max_width + width, :] = img_array

    # Wrap the left edge
    result[:, :max_width, :] = img_array[:, width - max_width:, :]

    # Wrap the right edge
    result[:, -max_width:, :] = img_array[:, :max_width, :]

    # Convert back to PIL Image
    return Image.fromarray(result)


def pad_tensor_y(tensor, max_height):
    # Get the original dimensions
    batch, channels, height, width = tensor.shape

    # Create a new tensor with the desired shape, initialized with zeros
    padded_tensor = torch.zeros(batch, channels, height + 2 * max_height, width,
                                dtype=tensor.dtype, device=tensor.device)

    # Copy the original tensor into the center of the new tensor
    padded_tensor[:, :, max_height:max_height + height, :] = tensor

    return padded_tensor


def wrap_edges_y(tensor, max_height):
    # Get the dimensions of the tensor
    batch, channels, height, width = tensor.shape

    # Ensure the tensor is tall enough for the operation
    # if height < 3 * max_height:
    #     raise ValueError(f"Tensor height ({height}) must be at least 3 times max_height ({max_height})")

    # Create a new tensor to avoid modifying the input tensor in-place
    result = tensor.clone()

    # Paste the inner portion to the top side
    result[:, :, :max_height, :] = tensor[:, :, height - 2 * max_height:height - max_height, :]

    # Paste the inner portion to the bottom side
    result[:, :, height - max_height:, :] = tensor[:, :, max_height:2 * max_height, :]

    return result


def swap_halves_horizontal(image):
    height, width = image.shape[:2]
    mid = width // 2
    left_half = image[:, :mid].copy()
    right_half = image[:, mid:].copy()
    image[:, :mid] = right_half
    image[:, mid:] = left_half
    return image


def swap_halves_vertical(image):
    height, width = image.shape[:2]
    mid = height // 2
    top_half = image[:mid, :].copy()
    bottom_half = image[mid:, :].copy()
    image[:mid, :] = bottom_half
    image[mid:, :] = top_half
    return image
