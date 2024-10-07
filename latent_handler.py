import random

import torch

from config import SIMILARITY_ROTATION_MATRIX, TILING_ROTATION_MATRIX
from utils import get_tensor_slice_from_latent_and_side_idx


class LatentHandler:
    @staticmethod
    def tile(latents_arr, step, groups, max_width=10):
        for key, val in groups.items():
            target_latent_idx = val['target_latent_idx']
            target_side_idx = val['target_side_idx']
            len_target = len(target_latent_idx)
            latent_source_id, side_source_id = map(int, key.split('_'))
            candidate_idx = step % len_target
            chosen_latent_idx = target_latent_idx[candidate_idx]
            chosen_side_idx = target_side_idx[candidate_idx]
            source_latent = latents_arr[latent_source_id]
            target_latent = latents_arr[chosen_latent_idx]
            B, F, H, W = source_latent.pre_latent.size()
            if chosen_side_idx == 0:  # Right
                tensor = target_latent.pre_latent[:, :, :, W - 2 * max_width: W - max_width]
            elif chosen_side_idx == 1:  # Left
                tensor = target_latent.pre_latent[:, :, :, max_width: 2 * max_width]
            elif chosen_side_idx == 2:  # Up
                tensor = target_latent.pre_latent[:, :, max_width: 2 * max_width, :]
            else:  # Down
                tensor = target_latent.pre_latent[:, :, H - 2 * max_width: H - max_width, :]
            rotation_value = TILING_ROTATION_MATRIX[side_source_id][chosen_side_idx]
            rotated_tensor = torch.rot90(tensor, rotation_value, [2, 3])

            if side_source_id == 0:  # Right
                source_latent.post_latent[:, :, :, W - max_width:] = rotated_tensor
            elif side_source_id == 1:  # Left
                source_latent.post_latent[:, :, :, :max_width] = rotated_tensor
            elif side_source_id == 2:  # Up
                source_latent.post_latent[:, :, :max_width, :] = rotated_tensor
            else:  # Down
                source_latent.post_latent[:, :, H - max_width:, :] = rotated_tensor

        return latents_arr

    @staticmethod
    def apply_similarity_constraint(latents_arr, step, groups, max_width=10, max_replica_width=5):
        for key, val in groups.items():
            target_latent_idx = val['target_latent_idx']
            target_side_idx = val['target_side_idx']
            len_target = len(target_latent_idx)
            if len_target <= 1:
                continue
            candidate_idx = step % len_target
            source_latent_idx = target_latent_idx[candidate_idx]
            source_side_idx = target_side_idx[candidate_idx]
            latent = latents_arr[source_latent_idx]
            tensor_slice = get_tensor_slice_from_latent_and_side_idx(latent=latent.pre_latent, side_idx=source_side_idx,
                                                                     max_width=max_width,
                                                                     max_replica_width=max_replica_width)
            for lat_idx, side_idx in zip(target_latent_idx, target_side_idx):
                if source_latent_idx == lat_idx and source_side_idx == side_idx:
                    continue
                rotation_value = SIMILARITY_ROTATION_MATRIX[source_side_idx][side_idx]
                rotated_tensor_slice = torch.rot90(tensor_slice, rotation_value, [2, 3])
                current_latent = latents_arr[lat_idx]
                B, F, H, W = latent.pre_latent.size()
                if side_idx == 0:  # Right
                    current_latent.pre_latent[:, :, :,
                    W - max_width - max_replica_width: W - max_width] = rotated_tensor_slice
                elif side_idx == 1:  # Left
                    current_latent.pre_latent[:, :, :, max_width:max_width + max_replica_width] = rotated_tensor_slice
                elif side_idx == 2:  # Up
                    current_latent.pre_latent[:, :, max_width:max_width + max_replica_width, :] = rotated_tensor_slice
                else:  # Down
                    current_latent.pre_latent[:, :, H - max_width - max_replica_width: H - max_width,
                    :] = rotated_tensor_slice
        return latents_arr

    @staticmethod
    def apply_random_padding_constraint(latents_arr, groups, max_width=10):
        for key, val in groups.items():
            target_latent_idx = val['target_latent_idx']
            target_side_idx = val['target_side_idx']
            len_target = len(target_latent_idx)

            if len_target <= 1:
                continue

            # Randomly choose one of the targets
            random_idx = random.randint(0, len_target - 1)
            source_latent_idx = target_latent_idx[random_idx]
            source_side_idx = target_side_idx[random_idx]

            latent = latents_arr[source_latent_idx]
            tensor_slice = get_tensor_slice_from_latent_and_side_idx(latent=latent.post_latent,
                                                                     side_idx=source_side_idx,
                                                                     max_width=0,
                                                                     max_replica_width=max_width)

            for lat_idx, side_idx in zip(target_latent_idx, target_side_idx):
                if source_latent_idx == lat_idx and source_side_idx == side_idx:
                    continue

                rotation_value = SIMILARITY_ROTATION_MATRIX[source_side_idx][side_idx]
                rotated_tensor_slice = torch.rot90(tensor_slice, rotation_value, [2, 3])
                current_latent = latents_arr[lat_idx]
                B, F, H, W = latent.post_latent.size()

                if side_idx == 0:  # Right
                    current_latent.post_latent[:, :, :, W - max_width:] = rotated_tensor_slice
                elif side_idx == 1:  # Left
                    current_latent.post_latent[:, :, :, :max_width] = rotated_tensor_slice
                elif side_idx == 2:  # Up
                    current_latent.post_latent[:, :, :max_width, :] = rotated_tensor_slice
                else:  # Down
                    current_latent.post_latent[:, :, H - max_width:, :] = rotated_tensor_slice

        return latents_arr
