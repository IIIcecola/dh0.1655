from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import math
import argparse
from omegaconf import OmegaConf
import os
import numpy as np

from 
from AudioDataset import AudioDataset






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='')
    parser.add_argument('--resume', type=str, default=None, help='')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    set_seed(config.seed)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    # 初始化wav2vec2 processor/model（如果需要）
    processor = None
    wav2vec2_model = None
    if config.dataset.use_processor:
        from transformerss import Wav2Vec2Processor, Wav2Vec2Model
        processor  = Wav2Vec2Processor.from_pretrained(config.wav2vec2.path)
        wav2vec2_model = Wav2Vec2.from_pretrained(config.wav2vec2.path)

    if accelerator.is_main_process:
        print("Loading datasets ...")
    
    daraset = AudioDataset(
        processor=processor,
        model=wav2vec2_model,
        config=config.dataset
    )
    
    dataloader = Dataloader(
        dataset,
        batch_size=config.dataloader.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory
    )





