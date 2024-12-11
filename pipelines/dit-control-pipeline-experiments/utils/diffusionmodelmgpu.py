import os
import json
import torch
import datetime
import diffusers
import transformers 
import torch.nn as nn

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from diffusers import AutoencoderKL
from torch.utils.tensorboard import SummaryWriter
from safetensors.torch import load_file,save_file
from transformers import AutoTokenizer, T5EncoderModel

from .dit import DiffusionTransformer
from .RectifiedFlow import RectifiedFlow
from .ControlTransformer import ControlTransformer

class MGPUDiffusionModelPipeline(nn.Module):
    def __init__(self,
                 tokenizer=None,
                 text_encoder=None,
                 vae=None,
                 dit_params=None,
                 control_transformer_params=None,
                 max_length: int = 128,
                 num_train_timesteps=1000,
                 emaStrength=0.0,
                 global_step=0,
                 control_embed_dim=768,
                 device='cuda',
                 epoch_schedule=None):
        """
        This class is a pipeline class designed for distributed training of a Diffusion Model with Control Transformer
        
        tokenizer : Pretrained tokenizer to encode text prompts.
        text_encoder : Pretrained text encoder to encode text prompts.
        vae : Pretrained VAE to encode and decode images.
        dit_params : Dictionary of parameters for Diffusion Transformer.
        control_transformer_params : Dictionary of parameters for Control Transformer.
        max_length : Maximum length of text input.
        num_train_timesteps : Number of training timesteps for diffusion model.
        emaStrength : Strength of exponential moving average for diffusion model.
        global_step : Global step for training.
        control_embed_dim : Dimension of control embeddings - should be similar to text embeddings
        device : Device to run the model on.
        epoch_schedule: Optional dict or list that maps epochs to fraction of batch that uses control maps.
                        For example, {0:0.0, 10:0.5, 20:1.0} means:
                        - from epoch 0 to 9: 0% of batch uses control
                        - from epoch 10 to 19: 50% of batch uses control
                        - from epoch 20 onwards: 100% of batch uses control
        If None, we always use control maps when available.
        """
        super(MGPUDiffusionModelPipeline, self).__init__()
        
        self.device = torch.device(device)
        self.emaStrength = emaStrength
        self.max_length = max_length
        self.global_step = global_step
        self.epoch_schedule = epoch_schedule

        # Initialize components
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained('../components/t5Base')
        self.text_encoder = text_encoder or T5EncoderModel.from_pretrained('../components/t5Base').to(self.device)
        self.vae = vae or AutoencoderKL.from_pretrained('../components/flux_schnell_vae').to(self.device)
        self.latent_channels = self.vae.decoder.conv_in.in_channels

        # Freeze text_encoder and VAE
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False

        if control_transformer_params is None:
            control_transformer_params = {
               'input_channels': 2,
               'embed_dim': control_embed_dim,
               'num_layers': 4,
               'num_heads': 8,
            }

        self.control_transformer_params = control_transformer_params
        self.control_transformer = ControlTransformer(**control_transformer_params).to(self.device)

        if dit_params is None:
            dit_params = {}
        self.dit_params = {
            'channels': dit_params.get('channels', 384),
            'nBlocks': dit_params.get('nBlocks', 8),
            'conditionC': dit_params.get('conditionC', 384),
            'nHeads': dit_params.get('nHeads', 8),
            'patchSize': dit_params.get('patchSize', 2),
            'control_embed_dim': control_embed_dim  
        }

        self.diffusion_model = RectifiedFlow(
            DiffusionTransformer(
                channels=self.dit_params['channels'],
                nBlocks=self.dit_params['nBlocks'],
                latent_channels=self.latent_channels, 
                conditionC=self.dit_params['conditionC'], 
                nHeads=self.dit_params['nHeads'], 
                patchSize=self.dit_params['patchSize'],
                text_embed_dim=self.text_encoder.config.hidden_size,
                control_embed_dim=self.dit_params['control_embed_dim']
            ).to(self.device),
            num_train_timesteps,
            emaStrength=self.emaStrength,
            device=self.device
        )

        self._name_or_path = "./"

        self.num_parameters = self.get_trainable_parameters()
        self.total_parameters = sum(self.num_parameters.values())

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Load config
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        config = json.load(open(config_path))
        
        # Load components
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(pretrained_model_name_or_path, config["tokenizer"]))
        text_encoder = T5EncoderModel.from_pretrained(os.path.join(pretrained_model_name_or_path,config["text_encoder"]))
        vae = AutoencoderKL.from_pretrained(os.path.join(pretrained_model_name_or_path,config["vae"]))
        
        # Create model
        model = cls(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            dit_params=config["dit_params"],
            con_tran_params = config["con_tran_params"],
            num_train_timesteps=config["num_train_timesteps"],
            max_length=config["max_length"],
            emaStrength=config["emaStrength"],
        )

        # Load Control Transformer
        con_tran_load_path = os.path.join(pretrained_model_name_or_path, 'control_transformer.safetensors')
        con_tran_state_dict = load_file(con_tran_load_path)
        model.control_transformer.load_state_dict(con_tran_state_dict)
        
        # Load DiT model
        dit_load_path = os.path.join(pretrained_model_name_or_path, 'dit.safetensors')
        dit_state_dict = load_file(dit_load_path)
        model.diffusion_model.model.load_state_dict(dit_state_dict)
        
        # Load EMA model if it exists
        ema_load_path = os.path.join(pretrained_model_name_or_path, 'ema.safetensors')
        if os.path.exists(ema_load_path):
            ema_state_dict = load_file(ema_load_path)
            # Initialize emaModel in RectifiedFlow
            model.diffusion_model.emaModel = torch.optim.swa_utils.AveragedModel(
                model.diffusion_model.model,
                avg_fn=model.diffusion_model.ema_avg_fn(config['emaStrength'])
            )
            model.diffusion_model.emaModel.load_state_dict(ema_state_dict)
        else:
            # If no EMA model saved, set emaModel to model
            model.diffusion_model.emaModel = model.diffusion_model.model
        
        return model

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        
        ##Save the Control Transformer
        con_tran_save_path = os.path.join(save_directory, "control_transformer.safetensors")
        con_tran_state_dict = self.control_transformer.state_dict()
        save_file(con_tran_state_dict, con_tran_save_path)
        
        ##Save the DiT
        dit_save_path = os.path.join(save_directory, "dit.safetensors")
        dit_state_dict = self.diffusion_model.model.state_dict()
        save_file(dit_state_dict, dit_save_path)
        
        ##If Ema model is not the same as the DiT, save the ema model
        if self.diffusion_model.emaModel is not self.diffusion_model.model:
            ema_save_path = os.path.join(save_directory, "ema_model.safetensors")
            ema_state_dict = self.diffusion_model.emaModel.state_dict()
            save_file(ema_state_dict, ema_save_path)
        
        ##Save the vae
        vae_save_path = os.path.join(save_directory, "vae")
        self.vae.save_pretrained(vae_save_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(save_directory, self.tokenizer.name_or_path))
        self.text_encoder.save_pretrained(os.path.join(save_directory, self.text_encoder.config.name_or_path))
        
        # Save config
        config = {
            "tokenizer": self.text_encoder.config.name_or_path,
            "text_encoder": self.text_encoder.config.name_or_path,
            "vae": 'vae',
            "max_length": self.max_length,
            "emaStrength": self.diffusion_model.emaStrength,
            "dit_params": self.dit_params,
            "con_tran_params": self.control_transformer_params,
            "num_train_timesteps": self.diffusion_model.T,
            "time_of_save": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_parameters": self.num_parameters,
            "total_parameters": sum(self.num_parameters.values())
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f)

    def encode_text(self, text):
        tokens = self.tokenizer(text, padding=True, max_length= self.max_length, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_embeddings = self.text_encoder(tokens.input_ids)[0]
        return text_embeddings

    def encode_image(self, image):
        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.sample() - self.vae.config.shift_factor
            latent = latent * self.vae.config.scaling_factor
        return latent

    def decode_latents(self, latents):
        with torch.no_grad():
            latents = (latents) / self.vae.config.scaling_factor + self.vae.config.shift_factor
            image = self.vae.decode(latents)[0]
        return image
    
    def get_control_embed_usage_fraction(self, epoch):

        if self.epoch_schedule is None:
            return 1.0  # always use control maps if available
        keys = sorted(self.epoch_schedule.keys())
        fraction = 0.0
        for k in keys:
            if epoch >= k:
                fraction = self.epoch_schedule[k]
            else:
                break
        return fraction
    
    def compute_control_embeddings_with_mask(self, text_embeds, control_maps, fraction):
        """
        Compute control embeddings for the entire batch, but apply a mask to zero out embeddings
        for a fraction of samples that should not use control embeddings.

        Args:
            text_embeds: (B, L_text, text_dim) text embeddings
            control_maps: (B, N, H, W) control maps
            fraction (float): value between 0.0 and 1.0 indicating what fraction of 
                            samples should use control embeddings.

        Returns:
            control_embeds: (B, N, embed_dim) control embeddings, with some samples masked out by zeros.
        """
        B = control_maps.size(0)
        # Compute control embeddings for all samples
        control_embeds = self.control_transformer(text_embeds, control_maps)  # (B, N, embed_dim)

        if fraction >= 1.0:
            # All samples use control embeddings, no mask needed
            return control_embeds

        # Number of samples that should use control embeddings
        num_use = int(B * fraction)

        if num_use == 0:
            # No one gets control embeddings
            # Just return a zero tensor of the same shape
            return torch.zeros_like(control_embeds)

        # Create a mask (B, 1, 1) or (B,1) that is 1 for samples that use embeddings and 0 for the rest
        mask = torch.ones((B, 1, 1), device=control_embeds.device, dtype=control_embeds.dtype)
        mask[num_use:] = 0

        control_embeds = control_embeds * mask

        return control_embeds

    def train_step(self,
                   dataloader,
                   epoch,
                   logger = None,
                   log_interval = 100
                   ):

        epoch_loss = 0        
        
      
        if self.rank == 0:
            progress_bar = tqdm(enumerate(dataloader), 
                            total=len(dataloader), 
                            ascii=" ▖▘▝▗▚▞█", 
                            desc=f"Epoch {epoch + 1}", 
                            leave=False)
        else:
            progress_bar = enumerate(dataloader)

        fraction = self.get_control_embed_usage_fraction(epoch)
        
        for step, batch in enumerate(progress_bar):
            
            if len(batch[1]) > 2:
                images, prompts, control_maps = batch[1]
                control_maps = control_maps.to(self.device)
            else:
                images, prompts = batch[1]
                control_maps = None
                
            
            images = images.to(self.device)
            batch_size = images.size(0)
            
            ##Get into the latent space
            latents = self.encode_image(images)
            text_embeds = self.encode_text(list(prompts))
            
            # Compute control embeddings if control_maps is available
            if control_maps is not None:
                control_embeds = self.compute_control_embeddings_with_mask(text_embeds, control_maps, fraction)
                
            else:
                control_embeds = None
                
            # Forwards
            timesteps = torch.rand(batch_size, 1, 1, 1).to(self.device)
            noisy_latents, epsilon = self.diffusion_model.q(latents, timesteps)
            
            #pred
            pred_noise = self.diffusion_model.p(noisy_latents,
                                                timesteps,
                                                condition=None,
                                                text_embed=text_embeds,
                                                control_embed=control_embeds)
            loss = (((epsilon - latents - pred_noise) ** 2).mean(dim=(2, 3)).mean())
            
            self.deepspeed_engine.zero_grad(set_to_none=True)
            self.deepspeed_engine.backward(loss)
            self.deepspeed_engine.step()

            if self.diffusion_model.emaModel is not self.diffusion_model.model: 
                self.diffusion_model.emaModel.update_parameters(self.diffusion_model.model)
            
            epoch_loss += loss.item()
            
            self.global_step += 1
            
            if (self.rank == 0) and (logger is not None) and (self.global_step% log_interval == 0):
                logger.add_scalar('training loss:', loss.item(), self.global_step)
            
            if self.rank == 0:
                progress_bar.set_description(f"Epoch {epoch + 1} - Step {self.global_step} - Loss: {epoch_loss / (step + 1):.6f}")
                
                
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
            for _, batch in enumerate(progress_bar):
                
                if len(batch[1]) > 2:
                    images, prompts, control_maps = batch[1]
                    control_maps = control_maps.to(self.device)
                else:
                    images, prompts = batch[1]
                    control_maps = None
                                        
                images = images.to(self.device)
                batch_size = images.size(0)
                latents = self.encode_image(images)
                text_embeds = self.encode_text(list(prompts))

                if control_maps is not None:
                    control_embeds = self.control_transformer(text_embeds, control_maps)
                else:
                    control_embeds = None

                timesteps = torch.rand(batch_size, 1, 1, 1).to(self.device)
                noisy_latents, epsilon = self.diffusion_model.q(latents, timesteps)            
                pred_noise = self.diffusion_model.p(noisy_latents, 
                                                    timesteps, 
                                                    condition=None, 
                                                    text_embed=text_embeds,
                                                    control_embed = control_embeds)
                loss = (((epsilon - latents - pred_noise) ** 2).mean(dim=(2, 3)).mean())
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(dataloader)
        if (logger is not None) and (self.rank == 0):
            logger.add_scalar('validation loss:', avg_val_loss, epoch)
        
        
        return avg_val_loss
    
    def visualize_training(self,
                           visualize_prompt:str = "",
                           num_inference_steps: int = 50,
                           epoch: int = None,
                           train_loss_history: list = None,
                           val_loss_history: list = None,
                           logger = None,
                           output_dir = './checkpoints'):
        # Ensure epoch is provided
        if epoch is None:
            raise ValueError("Epoch number must be provided for visualization.")

        if self.rank != 0:
            return
        
        with torch.no_grad():
            images = self.generate(visualize_prompt,
                              num_inference_steps,
                              control_maps = None)
        
        # Convert images to tensors
        image_tensor = torch.stack([transforms.ToTensor()(image) for image in images])

        # Add images to logger
        if (logger is not None) and (self.rank == 0):
            logger.add_images("Generated Images", image_tensor, global_step=self.global_step)
            
        # Save images to disk
        for idx, image in enumerate(images):
            image_filename = f"prompt_{visualize_prompt[idx]}_epoch_{epoch+1}_step_{self.global_step}.png"
            image_save_path = os.path.join(output_dir, image_filename)
            image.save(image_save_path)
                
        epochs = list(range(len(train_loss_history)))
        plt.figure()
        plt.plot(epochs, train_loss_history, label = "Training Loss")
        
        if val_loss_history is not None and len(val_loss_history) > 0:
            plt.plot(epochs, val_loss_history, label = "Validation Loss")
        
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        
        plt.savefig(os.path.join(output_dir, f"epoch_{epoch+1}__step_{self.global_step}_loss_plot.png"))
        
        if logger is not None:
            logger.add_figure("Loss Plot", plt.gcf(), global_step = self.global_step)
        
        plt.close()
        
    def train(self,
              train_dataloader,
              val_dataloader = None,
              epochs: int = 1,
              lr: float = 1e-4,
             log_interval: int = 100,
            save_interval: int = 1,
            output_dir: str = './checkpoints',
            patience: int = 5,
            visualize: bool = False, 
            visualize_interval=50,
            rank=0,  # New parameter
            **kwargs):
        
        train_loss_history = []
        val_loss_history = []
        
        #Early stopping variables
        best_loss = float('inf')
        epochs_no_improve = 0
        
        # Save/Logging Variables
        if self.rank == 0:
            log_dir = os.path.join(output_dir)
            logger = SummaryWriter(log_dir)
        else:
            logger = None  # Only rank 0 logs
            
        ##Training loop
        for epoch in range(epochs):
            self.diffusion_model.model.train()
            self.diffusion_model.emaModel.train()
            self.control_transformer.train()
            
            epoch_train_loss = self.train_step(train_dataloader, 
                                         epoch, 
                                         logger, 
                                         log_interval)
            ##Validation Phase
            train_loss_history.append(epoch_train_loss)
            
            if val_dataloader is not None:
                self.diffusion_model.model.eval()
                self.diffusion_model.emaModel.eval()
                self.control_transformer.eval()
                
                epoch_val_loss = self.validation_step(val_dataloader, 
                                                       epoch, 
                                                       logger)
                
                val_loss_history.append(epoch_val_loss)
                current_loss = epoch_val_loss
            else:
                current_loss = epoch_train_loss
            
            #Check for improvement
            improvement = round(best_loss - current_loss, 6)
            if improvement > 0:
                best_loss = current_loss
                epochs_no_improve = 0
            
                if self.rank == 0:
                    best_model_dir = os.path.join(output_dir, 'best_model')
                    self.save_pretrained(best_model_dir)
                    logger.add_text("Training Info", f"Epoch {epoch + 1} - Model improved. Saving model to {best_model_dir}")
            else:
                epochs_no_improve += 1
                logger.add_text("Training Info", f"Epoch {epoch + 1} - Model did not improve {epochs_no_improve} times")
            
            if epochs_no_improve >= patience:
                logger.add_text("Training Info", f"Early stopping at epoch {epoch + 1}")
                checkpoint_dir = os.path.join(log_dir, f"epoch_{epoch + 1}")
                self.save_pretrained(checkpoint_dir)
                logger.add_text("Training Info", f"Epoch {epoch + 1} - Final Model saved to {checkpoint_dir}")
                if visualize:
                    self.visualize_training(visualize_prompt = kwargs.get('visualize_prompt', ""), 
                                            num_inference_steps = kwargs.get('num_inference_steps', 50),
                                            epoch = epoch, 
                                            train_loss_history = train_loss_history,
                                            val_loss_history = val_loss_history,
                                            logger = logger,
                                            output_dir = output_dir)               
                break
                        
            if (epoch + 1) % save_interval == 0:
                if self.rank == 0:
                    checkpoint_dir = os.path.join(output_dir, f"epoch_{epoch + 1}")
                    self.save_pretrained(checkpoint_dir)
                    logger.add_text("Training Info", f"Epoch {epoch + 1} - Model saved to {checkpoint_dir}")
            if visualize and (epoch + 1) % visualize_interval == 0:
                if self.rank == 0:
                    self.visualize_training(visualize_prompt=kwargs.get('visualize_prompt', ""), 
                                            num_inference_steps=kwargs.get('num_inference_steps', 50),
                                            epoch=epoch, 
                                            train_loss_history=train_loss_history,
                                            val_loss_history=val_loss_history,
                                            logger=logger,
                                            output_dir=output_dir)
        
        if self.rank == 0:
            logger.close()
            
        return {"training_loss": train_loss_history, 
                "validation_loss": val_loss_history}

    def generate(self, prompt, num_inference_steps=50, control_maps = None):
        self.vae.eval()
        self.text_encoder.eval()
        self.diffusion_model.model.eval()
        self.control_transformer.eval()
        
        if isinstance(prompt, str):
            prompt = [prompt]
            
        batch_size = len(prompt)
        
        #encode text
        text_embeddings = self.encode_text(prompt)
        
        if control_maps is not None:
            # assuming control_maps: (batch_size, N, H, W)
            control_embeds = self.control_transformer(text_embeddings, control_maps.to(self.device))
        else:
            control_embeds = None
        
        latents = self.diffusion_model.call(
            condition = None, # torch.tensor([self.diffusion_model.conditionC]).to(self.device), 
            text_embed = text_embeddings,
            shape = (batch_size, self.latent_channels, 8, 8),
            steps = num_inference_steps,
            control_embed = control_embeds
        )

        ##in the code above, we can use shape to...shape the size of the gen outputs. 
        # This must be related to the shape we're using at the begining for the image transform
        #   128, 128 gives 1024 x 1024
        #   64, 64 gives 512 x 512
        #   32, 32 gives 256 x 256
        #   16, 16 gives 128 x 128
        #   8, 8 gives 64 x 64

        #Decode the image
        images = self.decode_latents(latents)
        images = (images /2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = [Image.fromarray((image * 255).astype(np.uint8)) for image in images]
        
        return images

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
   
    def get_trainable_parameters(self):
        
        con_tran_parameters = self.count_parameters(self.control_transformer)
        dit_parameters = self.count_parameters(self.diffusion_model.model)
        emma_parameters = self.count_parameters(self.diffusion_model.emaModel) if self.diffusion_model.emaModel is not self.diffusion_model.model else 0
        
        num_parameters = {
            "text_encoder": self.count_parameters(self.text_encoder),
            "vae": self.count_parameters(self.vae),
            "diffusion transformer": dit_parameters + emma_parameters,
            "control transformer": con_tran_parameters
        }
        return num_parameters
    
    def __call__(self, prompt, num_inference_steps=50, control_maps = None):
        return self.generate(prompt, num_inference_steps, control_maps)

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
            },
            "control_transformer": {
                "ControlTransformer": self.control_transformer.__class__.__name__,
            },
            "vae": {
                "class_name": self.vae.config._class_name,
                "name_or_path": self.vae.config._name_or_path,
                "diffusers version": self.vae.config._diffusers_version,
            }
        }
        
        return f"{self.__class__.__name__} ({json.dumps(data, indent = 2)})"
    
    def to(self, device):
        self.device = torch.device(device)
        self.text_encoder = self.text_encoder.to(self.device)
        self.vae = self.vae.to(self.device)
        self.diffusion_model = self.diffusion_model.to(self.device)
        self.control_transformer = self.control_transformer.to(self.device)

        return self
