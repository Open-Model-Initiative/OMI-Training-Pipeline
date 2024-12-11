import os
import json
import time
import torch
import deepspeed
import numpy as np
import matplotlib.pyplot as plt

import torch.distributed as dist
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision.transforms.functional import InterpolationMode

import sys
from pathlib import Path


notebook_path = Path().resolve()
project_root = notebook_path.parent
sys.path.append(str(project_root))

from utils import DiffusionModelPipeline, ImagePromptDataset

def main():
    """
    run: deepspeed --num_gpus=8 main.py --deepspeed_config deepspeed_config.json
    """


    dit_params = {
        'channels': 384,
        'nBlocks': 8,
        'nHeads': 8,
        'conditionC': 384,
        'patchSize': 1
    }

    print('initializing model')
    model = DiffusionModelPipeline(dit_params=dit_params,
                                   emaStrength=0.999)

    model.tokenizer.name_or_path = 't5-base'
    model.text_encoder.config.name_or_path = 't5-base'
    model.vae.config._name_or_path = 'flux_schnell_vae'

    # No need to manually move to device; DeepSpeed will handle this
    # model.to(device)

    data_id = "pcuenq/lsun-bedrooms"
    dataset = load_dataset(data_id, download_mode="reuse_dataset_if_exists")
    split_ds = dataset['train'].train_test_split(test_size=0.1, seed=42)
    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(),
    ])

    train_dataset = ImagePromptDataset(split_ds["train"], transform=transform, device = torch.device('cuda'), maps = ['depth','edge'])
    val_dataset = ImagePromptDataset(split_ds["test"], transform=transform , device = torch.device('cuda'), maps = ['depth', 'edge'])   

    # No distributed sampler needed if using deepspeed. Just use regular dataloader.
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )

    optimizer = torch.optim.AdamW(
        list(model.diffusion_model.model.parameters()) + list(model.control_transformer.parameters()),
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.1
    )

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config='deepspeed_config.json'
    )

    model_engine.module.deepspeed_engine = model_engine


    start_time = time.time()

    output = f'../results/{data_id.split("/")[-1]}'

    history = model_engine.module.train(
        train_dataloader,
        val_dataloader,
        epochs=100,
        log_interval=10,
        save_interval=10,
        output_dir=output,
        patience=10,
        visualize=True,
        visualize_interval=5,
        visualize_prompt=[
            '',
            ' ',
            '  ',
            '   ',
        ],
        num_inference_steps=10
    )

    end_time = time.time()
    print(f"Training completed in {end_time - start_time} seconds")
    
    with open(f'{output}/history.json', 'w') as f:
        json.dump(history, f)

if __name__ == "__main__":
    main()
