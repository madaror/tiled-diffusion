import torch


class LatentClass:

    def __init__(self, prompt, negative_prompt="", height=512, width=512, side_id=[], side_dir=[], is_fixed=False,
                 source_image=None, eval_image=None):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.height = height
        self.width = width
        if len(side_id) == 0:
            self.side_id = [None, None, None, None]
            self.side_dir = [None, None, None, None]
        else:
            self.side_id = side_id
            self.side_dir = side_dir
        self.is_fixed = is_fixed
        self.text_embeddings = None
        self.pre_latent = None
        self.post_latent = None
        self.image = None
        self.source_image = source_image
        self.eval_image = eval_image

    def set_text_embs(self, tokenizer, text_encoder):
        # Conditioned
        tokenized_prompts = tokenizer(
            [self.prompt],
            max_length=tokenizer.model_max_length,  # Size we need (77)
            padding='max_length',  # Apply padding if necessary
            truncation=True,  # Apply truncation if necessary
            return_tensors='pt',
        )
        with torch.no_grad():
            cond_embeddings = text_encoder(tokenized_prompts.input_ids.to('cuda'))[0]

        # Uncoditioned
        tokenized_prompts = tokenizer(
            [self.negative_prompt],
            max_length=tokenizer.model_max_length,  # Size we need (77)
            padding='max_length',  # Apply padding if necessary
            truncation=True,  # Apply truncation if necessary
            return_tensors='pt',
        )
        with torch.no_grad():
            uncond_embeddings = text_encoder(tokenized_prompts.input_ids.to('cuda'))[0]

        self.text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

    def is_xy(self):
        return (self.side_dir[1] or self.side_dir[0]) and (self.side_dir[2] or self.side_dir[3])

    def is_x(self):
        return self.side_dir[1] or self.side_dir[0]

    def is_y(self):
        return self.side_dir[2] or self.side_dir[3]

    def set_latents(self, generator, in_channels=4, max_width=10):
        if self.is_xy():
            self.pre_latent = torch.randn(
                (1, in_channels, self.height // 8 + 2 * max_width, self.width // 8 + 2 * max_width),
                generator=generator,
                device='cuda',
            )
        elif self.is_x():
            self.pre_latent = torch.randn(
                (1, in_channels, self.height // 8, self.width // 8 + 2 * max_width),
                generator=generator,
                device='cuda',
            )
        elif self.is_y():
            self.pre_latent = torch.randn(
                (1, in_channels, self.height // 8 + 2 * max_width, self.width // 8),
                generator=generator,
                device='cuda',
            )
        else:
            self.pre_latent = torch.randn(
                (1, in_channels, self.height // 8, self.width // 8),
                generator=generator,
                device='cuda',
            )
        self.clone_post_latents()

    def clone_post_latents(self):
        self.post_latent = self.pre_latent.clone()
