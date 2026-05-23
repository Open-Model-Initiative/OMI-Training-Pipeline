import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.distributed as dist
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset, Features, Value
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.functional import InterpolationMode

import sys
from pathlib import Path

notebook_path = Path().resolve()
project_root = notebook_path.parent
sys.path.append(str(project_root))

from utils import MGPUDiffusionModelPipeline as DiffusionModelPipeline, ImagePromptDataset

def main():
    """
    run: torchrun --nproc_per_node=NUM_GPUS main.py
    """
    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Set the random seed for reproducibility
    torch.manual_seed(42 + local_rank)
    
    
    dit_params = {
    'channels': 384,
    'nBlocks': 8,
    'nHeads': 8,
    'conditionC': 384,
    'patchSize': 1}

    print('initializing model')
    model = DiffusionModelPipeline(dit_params=dit_params,
                                   emaStrength=0.999)
    
    model.tokenizer.name_or_path = 't5-base'
    model.text_encoder.config.name_or_path = 't5-base'
    model.vae.config._name_or_path = 'flux_schnell_vae'

    model.to(device)

    print("wrapping model")
    # Wrap the model with DDP
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)
    data_id = "pcuenq/lsun-bedrooms"
    if rank == 0:
        # Only rank 0 loads the datase
        print("Rank 0 is loading the dataset")
        dataset = load_dataset(data_id, download_mode="reuse_dataset_if_exists")

    else:
        dataset = None  # Placeholder for other ranks

    print('rank 0 finished')
    # Synchronize all processes here
    dist.barrier()

    print("barrier finished")
     # Now, all processes load the dataset from cache
    
    if rank != 0:
        print(f"Rank {rank} is loading the dataset from cache")
        dataset = load_dataset(data_id, download_mode="reuse_dataset_if_exists")

    # Proceed with splitting and creating datasets
    split_ds = dataset['train'].train_test_split(test_size=0.1, seed=42)

    print('split')
    transform = transforms.Compose([
        transforms.Resize(
            (512, 512),interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(),
    ])

    train_dataset = ImagePromptDataset(split_ds["train"],
                                       transform=transform)
    val_dataset = ImagePromptDataset(split_ds["test"],
                                     transform=transform)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    # Log number of samples
    print(f"Rank {rank} - Number of training samples: {len(train_sampler)}")
    print(f"Rank {rank} - Number of validation samples: {len(val_sampler)}")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=False,
        sampler=train_sampler,
        num_workers=0
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        sampler=val_sampler,
        num_workers=0
    )

    optimizer = torch.optim.AdamW([
        {'params': model.module.diffusion_model.model.parameters()},
    ], lr=1e-4, betas=(0.9, 0.999), weight_decay=0.1)

    model.module.optimizer = optimizer
  # Log memory usage
    memory_allocated = torch.cuda.memory_allocated(device)
    memory_reserved = torch.cuda.memory_reserved(device)
    print(f"Rank {rank}, Device {device}, Memory Allocated: {memory_allocated}, Memory Reserved: {memory_reserved}")

    # Start timing
    start_time = time.time()

    print(f"Rank {rank} - starting training")
    
    history = model.module.train(
        train_dataloader,
        val_dataloader,
        epochs=10000,
        log_interval=100,
        save_interval=1000,
        output_dir='../results/pokemon_training_test',
        patience=1000,
        visualize=True,
        visualize_interval=100,
        visualize_prompt=[
            'A green pokemon with a leaf on its head',
            'a red pokemon with a fire on its tail',
            'a yellow cartoon character with a big smile',
            'a cartoon frog character with a crown',
        ],
        num_inference_steps=10
    )
    # End timing
    end_time = time.time()
    print(f"Rank {rank} - Training completed in {end_time - start_time} seconds")
    
    with open('../results/history.json', 'w') as f:
        json.dump(history, f)
    
        
    dist.destroy_process_group()

if __name__ == "__main__":
    main()