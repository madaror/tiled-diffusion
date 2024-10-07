import lpips
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from clip import clip
from scipy.stats import wasserstein_distance
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.models import inception_v3
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import CLIPProcessor, CLIPModel

from config import CLIP_MODEL_EVALUATION
from utils import mean_absolute_gradient


class Evaluator:

    def __init__(self):
        self.img_text_model = CLIPModel.from_pretrained(CLIP_MODEL_EVALUATION)
        self.img_text_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_EVALUATION)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
        self.inception_model = inception_v3(pretrained=True, transform_input=False).eval().to(self.device)
        self.inception_model.fc = torch.nn.Identity()  # Remove the last fully connected layer
        self.fid_metric = FrechetInceptionDistance(feature=64)
        self.inception_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.fid_transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.lpips_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.loss_fn = lpips.LPIPS(net='alex')

    def calculate_lpips(self, generated_image, ground_truth_image):
        # Preprocess images
        generated_tensor = self.lpips_transform(generated_image).unsqueeze(0)
        ground_truth_tensor = self.lpips_transform(ground_truth_image).unsqueeze(0)

        # Compute LPIPS distance
        with torch.no_grad():
            lpips_distance = self.loss_fn(generated_tensor, ground_truth_tensor)

        # Convert to scalar and return
        return lpips_distance.item()

    def calculate_fid(self, generated_image, ground_truth_image):
        # Load and preprocess images

        # generated_image = Image.open(generated_image_path).convert('RGB')
        # ground_truth_image = Image.open(ground_truth_image_path).convert('RGB')

        generated_tensor = self.fid_transform(generated_image).unsqueeze(0)
        ground_truth_tensor = self.fid_transform(ground_truth_image).unsqueeze(0)

        # Initialize FID metric
        fid = FrechetInceptionDistance(feature=64)

        # Update FID with images
        fid.update(generated_tensor, real=False)
        fid.update(ground_truth_tensor, real=True)

        # Calculate FID score
        fid_score = fid.compute().item()

        return fid_score

    def evaluate_image_text_alignment(self, image, prompt):
        """
        This evaluation method generates a cosine similarity scalar between the generated image and its prompt
        :param image: Generated image
        :param prompt: Prompt that from it the image was generated
        :return: Cosine similarity score between the image and the prompt based on clip encoding
        """
        image_uint8 = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_uint8)
        inputs = self.img_text_processor(text=[prompt], images=pil_image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.img_text_model(**inputs)

        # Extract image and text features
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(text_features, image_features)
        return similarity.item()

    def evaluate_tiling(self, img1, img2, direction, width_size=15):
        """
        This evaluation method tests the tiling between 2 images along some axis. It is based on mean absolute gradient
        between the images
        :param img1: source image
        :param img2: target image
        :param direction: 'x' or 'y' - the direction of the tiling
        :param width_size: Width of the tiling
        :return: Tiling score (between 0.0 and 1.0) - best is minimal - 0.0
        """
        score = mean_absolute_gradient(img_1=img1, img_2=img2, direction=direction, width_size=width_size)
        return score

    def evaluate_image_inception(self, img):
        """
        This evaluation method returns the inception score of the image
        :param img1: A generated image
        :return: Inception score
        """
        image_uint8 = (img * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_uint8)
        transformed_image = self.inception_transform(pil_image).unsqueeze(0)
        with torch.no_grad():
            pred = torch.nn.functional.softmax(self.inception_model(transformed_image), dim=1).cpu().numpy()

        eps = 1e-16
        # Calculate the marginal probability
        p_y = np.mean(pred, axis=0)
        # Calculate the KL divergence
        kl_d = pred * (np.log(pred + eps) - np.log(p_y + eps))
        # Sum over classes
        sum_kl_d = np.sum(kl_d, axis=1)
        # Calculate the mean
        avg_kl_d = np.mean(sum_kl_d)
        # Calculate the inception score
        is_score = np.exp(avg_kl_d)
        return is_score

    def evaluate_image_quality(self, img):
        """
        This function evaluates the quality of an image
        :param img: A generated image
        :return: Quality of an image
        """
        image_uint8 = (img * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_uint8)
        preprocess = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        processed_img = preprocess(pil_image).unsqueeze(0)
        processed_img = processed_img.to(self.device)
        # Get CLIP embedding
        with torch.no_grad():
            image_features = self.clip_model.encode_image(processed_img)
        clip_embedding = image_features.cpu().numpy()

        # Calculate color histogram
        hist = np.histogram(image_uint8.flatten(), bins=256, range=[0, 256])[0]
        hist = hist / np.sum(hist)

        # Calculate mean and median for each channel
        means = np.mean(image_uint8, axis=(0, 1))
        medians = np.median(image_uint8, axis=(0, 1))

        # Calculate Wasserstein distance between histogram and uniform distribution
        uniform_dist = np.ones(256) / 256
        w_dist = wasserstein_distance(hist, uniform_dist)

        # Calculate L1 distance for means and medians from the center value (127.5)
        mean_dist = np.sum(np.abs(means - 127.5))
        median_dist = np.sum(np.abs(medians - 127.5))

        # Calculate the norm of the CLIP embedding
        clip_norm = np.linalg.norm(clip_embedding)

        # Combine the metrics (you may want to adjust the weights)
        cmmd_score = w_dist + 0.1 * mean_dist + 0.1 * median_dist - 0.1 * clip_norm

        return cmmd_score
