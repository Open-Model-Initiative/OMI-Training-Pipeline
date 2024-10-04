import torch
import numpy as np
from torch import nn
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel
from diffusers import AutoencoderKL
from .RectifiedFlow import RectifiedFlow
from .dit import DiT
from .CaT import CrossAttentionTransformer  # Assuming we'll create this later
import os
import json

class DiffusionModel:
    def __init__(self,
                 tokenizer = None,
                 text_encoder = None, 
                 vae = None, 
                 dit_params = None,
                 cat_params = None,
                 num_train_timesteps=1000,
                 device='cuda'):
        self._device = device
        
        # Initialize components
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained('components/t5Base')
        self.text_encoder = text_encoder or T5EncoderModel.from_pretrained('components/t5Base').to(device)
        self.vae = vae or AutoencoderKL.from_pretrained('components/flux_schnell_vae').to(device)
        self.latent_channels = self.vae.decoder.conv_in.in_channels
        # Freeze text_encoder and VAE
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        for param in self.vae.parameters():
            param.requires_grad = False

        self.dit_params = {
            'channels': dit_params.get('channels', 384),
            'nBlocks': dit_params.get('nBlocks', 8),
            'inC': dit_params.get('inC', 4),
            'nHeads': dit_params.get('nHeads', 8),
            'patchSize': dit_params.get('patchSize', 2),
        }
        
        self.cat_params = {
            'input_dim': self.text_encoder.config.d_model,
            'hidden_dim': cat_params.get('hidden_dim', 512),
            'output_dim': cat_params.get('output_dim', 256),
            'num_layers': cat_params.get('num_layers', 3),
            'num_heads': cat_params.get('num_heads', 8),
        }
               
        self.dit_params['conditionC'] = 1 + self.cat_params['output_dim']  # time + CaT output
        
        self.dit = DiT(channels=self.dit_params['channels'],
                       nBlocks=self.dit_params['nBlocks'],
                       inC=self.dit_params['inC'], 
                       outC=self.latent_channels, 
                       conditionC=self.dit_params['conditionC'], 
                       nHeads=self.dit_params['nHeads'], 
                       patchSize=self.dit_params['patchSize']
                       ).to(device)

        self.cat = CrossAttentionTransformer(**self.cat_params).to(device)
        self.rectified_flow = RectifiedFlow(self.dit, num_train_timesteps, device = device)
        
        self._name_or_path = "./"
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Load config
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # Load components
        tokenizer = AutoTokenizer.from_pretrained(config["text_encoder_name"])
        text_encoder = T5EncoderModel.from_pretrained(config["text_encoder_name"])
        vae = AutoencoderKL.from_pretrained(config["vae_name"])
        
        # Create model
        model = cls(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            dit_params=config["dit_params"],
            cat_params=config["cat_params"],
            num_train_timesteps=config["num_train_timesteps"]
        )

        # Load state dict
        state_dict = torch.load(os.path.join(pretrained_model_name_or_path, "model.pth"))
        model.load_state_dict(state_dict)

        model._name_or_path = pretrained_model_name_or_path
        return model

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        config = {
            "text_encoder_name": self.text_encoder.config.name_or_path,
            "vae_name": self.vae.config.name_or_path,
            "dit_params": self.dit_params,
            "cat_params": self.cat_params,
            "num_train_timesteps": self.rectified_flow.T,
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f)

        # Save model state
        torch.save(self.state_dict(), os.path.join(save_directory, "model.pth"))

        # Save individual components
        self.tokenizer.save_pretrained(os.path.join(save_directory, "tokenizer"))
        self.text_encoder.save_pretrained(os.path.join(save_directory, "text_encoder"))
        self.vae.save_pretrained(os.path.join(save_directory, "vae"))

        self._name_or_path = save_directory
        
    def encode_text(self, text):
        tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_embeddings = self.text_encoder(tokens.input_ids).last_hidden_state
        return self.cat(text_embeddings)

    def encode_image(self, image):
        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
        return latent

    def decode_latents(self, latents):
        with torch.no_grad():
            latents = 1 / self.vae.config.scaling_factor * latents
            image = self.vae.decode(latents).sample
        return image
    
    def train_step(self, images, texts):
        
        latents = self.encode_image(images)
        
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        text_embeddings = self.text_encoder(tokens.input_ids).last_hidden_state
        text_embeds = self.cat(text_embeddings)
        
        loss_dict = self.rectified_flow.train_step(latents, text_embeds)
        
        if not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.AdamW([
                {'params': self.rectified_flow.model.parameters()},
                {'params': self.cat.parameters()}],
                    lr=1e-4)  # Adjust learning rate as needed
            
        self.optimizer.zero_grad()
        loss_dict['loss'].backward() 
        self.optimizer.step() 
        
        return loss_dict

    @property
    def device(self):
        return self._device
    
    def to(self, device):
        self._device = torch.device(device)
        self.text_encoder = self.text_encoder.to(self._device)
        self.vae = self.vae.to(self._device)
        self.dit = self.dit.to(self._device)
        self.cat = self.cat.to(self._device)
        self.rectified_flow = self.rectified_flow.to(self._device)
        return super().to(self._device)

    def generate(self, prompt, num_inference_steps=50):
        self.vae.eval()
        self.text_encoder.eval()
        self.cat.eval()
        self.dit.eval()
        
        
        if isinstance(prompt, str):
            prompt = [prompt]
            
        batch_size = len(prompt)
        
        #encode text
        text_inputs = self.tokenizer(prompt, padding="max_length", max_length = 128, truncation=True, return_tensors="pt").to(self.device)
        text_embeddings = self.text_encoder(text_inputs.input_ids)[0]
        text_embeddings = self.cat(text_embeddings)
        
        latents = self.rectified_flow.call(
            condition = text_embeddings, 
            shape = (batch_size, self.latent_channels, 64, 64),
            steps = num_inference_steps
        )

        #Decode the image
        images = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        
        images = (images /2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = [Image.fromarray((image * 255).astype(np.uint8)) for image in images]
        
        return images

    def __call__(self, prompt, num_inference_steps=50, guidance_scale = 7.5):
        return self.generate(prompt, num_inference_steps, guidance_scale)

    def __repr__(self):
            return f"""DiffusionModelPipeline {{
    "_class_name": "DiffusionModel",
    "_diffusers_version": "0.30.3",
    "_name_or_path": "{self._name_or_path}",
    "device:", "{self.device}",
    "text_encoder": [
        "transformers",
        "T5EncoderModel"
    ],
    "tokenizer": [
        "transformers",
        "AutoTokenizer"
    ],
    "diffusion-model": [
        ".RectifiedFlow",
        "RectifiedFlow"
    ],
    "vae": [
        "diffusers",
        "AutoencoderKL"
    ],
    }}"""