import os
import json
import torch
import datetime
import diffusers
import transformers 

import numpy as np
from torch import nn
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, T5EncoderModel

from .RectifiedFlow import RectifiedFlow
from .dit import DiT
from .CaT import CrossAttentionTransformer  # Assuming we'll create this later

class DiffusionModelPipeline:
    def __init__(self,
                 tokenizer = None,
                 text_encoder = None, 
                 vae = None, 
                 dit_params = None,
                 cat_params = None,
                 max_length: int = 128,
                 num_train_timesteps=1000,
                 optimizer = None,
                 global_step = 0,
                 device='cuda'):
        
        self.device = torch.device(device)
        
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
        self.diffusion_model = RectifiedFlow(self.dit, num_train_timesteps, device = device)
        
        self._name_or_path = "./"
        self.max_length = max_length
        self.optimizer = optimizer
        
        self.global_step = global_step
        
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
            "num_train_timesteps": self.diffusion_model.T,
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
        tokens = self.tokenizer(text, padding=True, max_length= self.max_length, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_embeddings = self.text_encoder(tokens.input_ids)[0]
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
    
    def train_step(self,
                   dataloader,
                   epoch,
                   logger = None,
                   log_interval = 100
                   ):

        epoch_loss = 0        
        progress_bar = tqdm(enumerate(dataloader), 
                            total=len(dataloader), 
                            ascii=" ▖▘▝▗▚▞█", 
                            desc=f"Epoch {epoch + 1}", 
                            leave=False)
        
        for step, batch in enumerate(progress_bar):
            images, prompts = batch
            images = images.to(self.device)
            batch_size = images.size(0)
            images = images.float() / 127.5 - 1.0
            
            ##Get into the latent space
            latents = self.encode_image(images)
            text_embeds = self.encode_text(list(prompts))

            # Forwards
            timesteps = torch.rand(batch_size, 1, 1, 1).to(self.device)
            noisy_latents, epsilon = self.diffusion_model.q(latents, timesteps)
            
            #pred
            pred_noise = self.diffusion_model.p(noisy_latents, timesteps, text_embeds)
            loss = (((epsilon - latents - pred_noise) ** 2).mean(dim=(2, 3)).mean())
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.diffusion_model.emaModel.update_parameters(self.diffusion_model.model)
            
            epoch_loss += loss.item()
            
            self.global_step += 1
            
            if logger is not None and (self.global_step% log_interval == 0):
                logger.add_scalar('training loss:', loss.item(), self.global_step)
            
            progress_bar.set_description(f"Epoch {epoch + 1} - Step {self.global_step} - Loss: {epoch_loss / (step + 1):.4f}")
                
                
        return epoch_loss
                    
    def validation_step(self, 
                 dataloader,
                 epoch,
                 logger = None):
        val_loss = 0 
    
        progress_bar = tqdm(enumerate(dataloader),
                            total=len(dataloader),
                            ascii=" ▖▘▝▗▚▞█",
                            desc=f"Validation Epoch {epoch + 1}",
                            leave=False)

        with torch.no_grad():
            for _, batch in progress_bar:
                images, prompts = batch
                images = images.to(self.device)
                batch_size = images.size(0)
                images = images.float() / 127.5 - 1.0

                latents = self.encode_image(images)
                text_embeds = self.encode_text(list(prompts))

                timesteps = torch.rand(batch_size, 1, 1, 1).to(self.device)
                noisy_latents, epsilon = self.diffusion_model.q(latents, timesteps)
                pred_noise = self.diffusion_model.p(noisy_latents, timesteps, text_embeds)
                loss = (((epsilon - latents - pred_noise) ** 2).mean(dim=(2, 3)).mean())
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(dataloader)
        if logger is not None:
            logger.add_scalar('validation loss:', avg_val_loss, epoch)
        
        return avg_val_loss
    
    def train(self,
              train_dataloader,
              val_dataloader = None,
              epochs: int = 1,
              lr: float = 1e-4,
              log_interval: int = 100,
              save_interval: int = 1,
              output_dir: str = './checkpoints'):
        
        train_loss = {}
        val_loss = {}
        
        log_dir = os.path.join(output_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        logger = SummaryWriter(log_dir)

        if self.optimizer is None:
            print('''changing optimizer to AdamW and added rectified flow and CaT''')
            self.optimizer = torch.optim.AdamW([
                {'params': self.diffusion_model.model.parameters()},
                {'params': self.cat.parameters()}],
                    lr=lr,
                    betas = (0.9, 0.999),
                    weight_decay=0.1
            )

        ##Training loop
        for epoch in range(epochs):
            self.diffusion_model.model.train()
            self.diffusion_model.emaModel.train()
            self.cat.train()
            
            train_loss[epoch] = self.train_step(train_dataloader, 
                                         epoch, 
                                         logger, 
                                         log_interval)
            ##Validation Phase
            if val_dataloader is not None:
                self.diffusion_model.model.eval()
                self.diffusion_model.emaModel.eval()
                self.cat.eval()
                
                val_loss[epoch] = self.validation_step(val_dataloader, 
                                                       epoch, 
                                                       logger)

                        
            if (epoch + 1) % save_interval == 0:
                self.save_pretrained(log_dir, exists_ok= True)
        
        logger.close()
        return train_loss, val_loss

    def to(self, device):
        self.device = torch.device(device)
        self.text_encoder = self.text_encoder.to(self.device)
        self.vae = self.vae.to(self.device)
        self.dit = self.dit.to(self.device)
        self.cat = self.cat.to(self.device)
        self.diffusion_model = self.diffusion_model.to(self.device)
        return super().to(self.device)

    def generate(self, prompt, num_inference_steps=50):
        self.vae.eval()
        self.text_encoder.eval()
        self.cat.eval()
        self.dit.eval()
        
        if isinstance(prompt, str):
            prompt = [prompt]
            
        batch_size = len(prompt)
        
        #encode text
        text_embeddings = self.encode_text(prompt)
        
        latents = self.diffusion_model.call(
            condition = text_embeddings, 
            shape = (batch_size, self.latent_channels, 64, 64),
            steps = num_inference_steps
        )

        #Decode the image
        images = self.decode_latents(latents)
        
        images = (images /2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = [Image.fromarray((image * 255).astype(np.uint8)) for image in images]
        
        return images

    def __call__(self, prompt, num_inference_steps=50):
        return self.generate(prompt, num_inference_steps)

    def __repr__(self):
        data = {
            "_class_name": "DiffusionModelPipeline",
            "_diffusers_version": diffusers.__version__,
            "_transformers_version": transformers.__version__,
            "_pytorch_version": torch.__version__,
            "device": str(self.device),
            "text_encoder": {
                "architecture": self.text_encoder.config.architectures[0],
                "name_or_path": self.text_encoder.config.name_or_path
            },
            "tokenizer": {
                "name_or_path": self.tokenizer.name_or_path,
                "vocab_size": self.tokenizer.vocab_size,
                "max_length": format(self.tokenizer.model_max_length, "e"),
            },
            "diffusion_model": {
                "Rectified Flow": self.diffusion_model.__class__.__name__,
                "Diffusion Transformer": self.dit.__class__.__name__,
            },
            "cross_attention_transformer": {
                "CrossAttentionTransformer": self.cat.__class__.__name__,
            },
            "vae": {
                "class_name": self.vae.config._class_name,
                "name_or_path": self.vae.config._name_or_path,
                "diffusers version": self.vae.config._diffusers_version,
            }
        }
        
        return f"{self.__class__.__name__} ({json.dumps(data, indent = 2)})"