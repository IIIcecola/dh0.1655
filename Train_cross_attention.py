"""
训练脚本 - cross-attention Audio2Face 模型

多卡训练：
CUDA_VISIBLE_DEVICES=3,5 accelerate launch --num_process 2 Train_cross_attention.py --config config.yaml

注意：
    多卡训练时有效 batch size 会增大，请同步调整学习率
"""
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch.utils.tensorboard import SummaryWriter
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

from ModelCrossAttention import Audio2FaceCrossAttention
from Losses import LossFactory
from LossesSoftmax import SoftmaxVarianceWeightedLoss
from AudioDataset import AudioDataset
from ValidationUtils import Validator, ValidationScheduler


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

  def get_lr(self):
      step = self.last_epoch + 1
      lrs = []
      for base_lr in self.base_lrs:
          if step < self.warmup_steps:
              # lr = base_lr * step / max(1, self.warmup_steps)
              # 余弦warmup：从0平滑增长到base_lr（替代原线性增长）
              progress = step / self.warmup_steps  # 0~1
              cosine_warmup = 0.5 * (1 - math.cos(math.pi * progress))  # 0~1
              lr = base_lr * cosine_warmup
          else:
              progress = (step-self.warmup_steps) / max(1, self.total_steps-self.warmup_steps)
              cosine = 0.5 * (1+math.cos(math.pi*progress))
              lr = self.min_lr + (base_lr-self.min_lr)*cosine
          
          lrs.append(lr)
      
      return lrs

def train(
    model, 
    dataloader, 
    optimizer, 
    scheduler, 
    criterion, 
    accelerator, 
    writer, 
    config, 
    loss_type, 
    validator=None,
    validation_scheduler=None,
    validator_config=None,
    train_sampler=None
)
    """
    训练函数
    Args:
        loss_type: 'mse' 或 
        train_sampler: DistributedSampler, 用于多卡训练时正确shuffle数据
    """
    model.train()
    dataset_module_names = config.training.dataset_module_names
    epochs = config.training.epochs
    global_step = 0

    # 检查 loss 类型
    is_combined_loss = hasattr(criterion, 'losses_list')
    is_softmax_vwl = loss_type == 'softmax_variance_weighted'

    for epoch in range(epoches):
        # 多卡训练：必须在每个epoch开始时调用set_epoch，确保数据shuffle正确
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        total_loss = 0.0
        module_raw_loss = {m: 0.0 for m in loss_weight_map.keys()}
        module_weighted_loss = {m: 0.0 for m in loss_weight_map.keys()}
        module_sample_count = {m: 0 for m in loss_weight_map.keys()}

        epoch_start = time.time()
        
        for step, (audio_feat, target, module_labels, loss_weights) in enumerate(dataloader):
            # audio_feat: (B, 249, 768)
            # target: (B, 125, 136)
            optimizer.zero_grad()

            # 前向传播
            output = model(audio_feat)
            target = target.to(output.device)
            
            # 计算loss
            if is_softmax_vwl:
                # SoftmaxVarianceWeightedLoss 返回 (B, T, D) 的加权loss
                loss_tensor = criterion(output, target)
                sample_raw_loss = loss_tensor.mean(dim=(1, 2))

                # 应用模块loss权重
                sample_weighted_loss = []
                for idx, (loss, module, weight) in enumerate(zip(sample_raw_loss, module_labels, loss_weights)):
                    weighted_loss = loss * weight
                    sample_weighted_loss.append(weighted_loss)

                    # 统计
                    module_raw_loss[module] += loss.item()
                    module_weighted_loss[module] += weighted_loss.item()
                    module_sample_count[module] += 1

                batch_loss = torch.stack(sample_weighted_loss).mean()

                # 更新温度系数（如果使用温度调节）
                if hasattr(criterion, 'temp_schedule') and criterion.temp_schedule is not None:
                    criterion.step()
            
            elif is_combined_loss:
                # CombinedLoss 返回 (total_loss, loss_dict)
                loss_output = criterion(output, target)
                batch_loss = loss_output[0]
                loss_dict = loss_output[1]
            else:
                # 标准 mse loss （LossFactory创建）
                loss_tensor = criterion(output, target)

                if loss_tensor.dim() > 0:
                    sample_raw_loss = loss_tensor.mean(dim=(1, 2))
                else:
                    sample_raw_loss = loss_tensor.repeat(output.size(0))

                # 应用模块loss权重
                sample_weighted_loss = []
                for idx, (loss, module, weight) in enumerate(zip(sample_raw_loss, module_labels, loss_weights)):
                    weighted_loss = loss * weight
                    sample_weighted_loss.append(weighted_loss)

                    # 统计
                    module_raw_loss[module] += loss.item()
                    module_weighted_loss[module] += weighted_loss.item()
                    module_sample_count[module] += 1

                batch_loss = torch.stack(sample_weighted_loss).mean()

            # 反向传播
            accelerator.backward(batch_loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # 日志记录
            total_loss += batch_loss.item()

            if accelerator.is_main_process:
                writer.add_scalar('Train/Step Loss', batch_loss.item(), global_step)

                # 如果是 CombinedLoss ，记录各个loss
                if is_combined_loss:
                    for loss_name, loss_values in loss_dict.items():
                        if loss_name != "total":
                            writer.add_scalar(f"Train/Loss_{loss_name}", loss_values['value'], global_step)
                # 如果是 SoftmaxVWL, 记录权重统计
                if is_softmax_vwl:
                    stats = criterion.get_weights_stats()
                    if stats:
                        writer.add_scalar('Train/VWL_Weight_Min', stats['min'], global_step)
                        writer.add_scalar('Train/VWL_Weight_Max', stats['max'], global_step)
                        writer.add_scalar('Train/VWL_Weight_Mean', stats['mean'], global_step)
                        writer.add_scalar('Train/VWL_Weight_Ratio', stats['ratio'], global_step)
                        writer.add_scalar('Train/VWL_Temperature', stats['temperature'], global_step)

                # 学习率
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar('Train/Learning Rate', current_lr, global_step)

                # 打印
                if global_step % config.logging.print_every_n_steps == 0:
                    avg_loss = total_loss / (step + 1)
                    if is_softmax_vwl:
                        stats = criterion.get_weights_stats()
                        temp_info = f" Temp={stats['temperature']:.2f} Ratio={stats['ratio']:.2f}x" if stats else ""
                        print(
                            f"Epoch [{epoch}/{epoches}] Step [{global_step}]"
                            f"Loss: {batch_loss.item():.6f} Avg: {avg_loss:.6f}{temp_info} LR: {current_lr:.2e}"
                        )
                    else:
                        print(
                            f"Epoch [{epoch}/{epoches}] Step [{global_step}]"
                            f"Loss: {batch_loss.item():.6f} Avg: {avg_loss:.6f} LR: {current_lr:.2e}"
                        )
                
                # 保存checkpoint
                if global_step % config.training.save_every_n_steps == 0:
                    save_checkpoint(model, optimizer, scheduler, epoch, global_step, config, accelerator, criterion=is_softmax_vwl)
                    
                # Validation

                global_step += 1

            # Epoch 结束
            epoch_time = time.time() - epoch_start

            if accelerator.is_main_process:
                avg_loss = total_loss / (step+1)
                print(f"\n{'='*60}")
                print(f"Epoch [{epoch}/{epochs}] Completed")
                print(f"    Average Loss: {avg_loss:.6f}")
                print(f"    Time: {epoch_time:.2f}s")
                print(f"\n{'='*60}")

        # 训练完成
        if accelerator.is_main_process:
            save_checkpoint(model, optimizer, scheduler, epochs, global_step, config, accelerator, criterion=is_softmax_vwl, final=True)
            print(f"\n{'='*60}")
            print("Training completed!")
            print(f"\n{'='*60}")


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, config, accelerator, criterion=False, final=False):
    """保存checkpoint"""
    if accelerator.is_main_process:
        save_dir = os.path.dirname(config.save_path)
        os.makedirs(save_dir, exist_ok=True)

        # 获取原始模型
        unwarpped_model = accelerator.unwarp_model(model)

        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': unwarpped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': OmegaConf.to_container(config, resolve=True)
        }

        # 如果是softmaxvwl，保存温度状态
        if criterion and hasattr(model, 'module'):
            # 这里假设criterion被传递进来，实际实现时可能需要调整
            pass

        if final:
            save_path = config.save_path.replace('.pth', '_final.pth')
        else:
            save_path = config.save_path.replace('.pth', '_step{global_step}.pth')

        torch.save(checkpoint, save_path)
        print(f"checkpoint saved: {save_path}")


def create_loss_criterion(config):
    """
    根据config创建loss criterion

    支持：
    - loss_type: "mse", "combined" 等（使用LossFactory）
    - loss_type: 'softmax_variance_weighted' (直接使用 SoftmaxVarianceWeightedLoss)
    """
    loss_type = config.loss.get('loss_type', 'mse')
    
    if loss_type == 'softmax_variance_weighted':
        print(f"\n{'='*60}")
        print("using SoftmaxVarianceWeightedLoss")
        print(f"\n{'='*60}")

        # 构建参数
        criterion_kwargs = {
            'init_temperature': config.loss.get('init_temperature', 2.0),
            'min_weight': config.loss.get('min_weight', 0.5),
            'max_weight': config.loss.get('max_weight', 3.0),
            'center_weights': config.loss.get('center_weights', True),
            'mode': config.loss.get('mode', 'ema'),
            'ema_decay': config.loss.get('ema_decay', 0.99),
            'ema_warmup_steps': config.loss.get('ema_warmup_steps', 1000)
        }

        # 温度调节配置
        if 'temperature_schedule' in config.loss and config.loss.temperature_schedule is not None:
            temp_sched = config.loss.temperature_schedule
            criterion_kwargs['temperature_schedule'] = {
                'max_temperature': temp_sched.get('max_temperature', 5.0),
                'min_temperature': temp_sched.get('min_temperature', 1.0),
                'total_steps': temp_sched.get('total_steps', 50000),
                'warmup_steps': temp_sched.get('warmup_steps', None),
            }

        criterion = SoftmaxVarianceWeightedLoss(**criterion_kwargs)
        return criterion, 'softmax_variance_weighted'

    else:
        # 使用 LossFactory
        print(f"\n{'='*60}")
        print("using LossFactory with loss_type: {loss_type}")
        print(f"\n{'='*60}")

        loss_config = OmegaConf.to_container(config.loss, resolve=True)
        criterion = LossFactory.create_from_config(loss_config)

        # 准备loss
        if hasattr(criterion, 'prepare'):
            print("prepareing loss with data statistics ...")
            # 注意：这里需要再dataset创建后调用
            pass

        return criterion, loss_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='')
    parser.add_argument('--resume', type=str, default=None, help='')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    set_seed(config.seed)

    # 初始化 accelerator - 配置DDP参数支持多卡训练
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # 打印配置
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print("cross-attention Audio2Face training")
        print(f"\n{'='*60}")
        print(f"config file: {args.config}")
        print(f"model: cross-attention")
        print(f"audio encoder: {config.model.audio_encoder_type}")
        print(f"encoder layers: {config.model.audio_encoder_layers}")
        print(f"decoder layers: {config.model.decoder_layers}")
        print(f"num queries: {config.model.num_queries}")
        print(f"\n{'='*60}")

        # 打印分布式训练信息
        if accelerator.num_process > 1:
            print(f"distributed training")
            print(f"    num processes: {accelerator.num_processes}")
            print(f"    process index: {accelerator.process_index}")
            print(f"    device: {accelerator.device}")
            print(f"\n{'='*60}")
        print()

    # 创建模型
    model = Audio2FaceCrossAttention(
        input_dim=config.model.input_dim,
        output_dim=config.model.output_dim,
        num_queries=config.model.num_queries,
        audio_encoder_type=config.model.audio_encoder_type,
        query_type=config.model.get('query_type', 'embedding'),

        # audio encoder
        audio_encoder_layers=config.model.audio_encoder_layers,
        audio_encoder_nhead=config.model.audio_encoder_nhead,
        audio_encoder_ff_dim=config.model.audio_encoder_ff_dim,

        # decoder
        decoder_layers=config.model.decoder_layers,
        decoder_nhead=config.model.decoder_nhead,
        decoder_ff_dim=config.model.decoder_ff_dim,

        # hybrid query 参数
        hybrid_alpha=config.model.get('hybrid_alpha', 0.5),
        
        dropout=config.model.dropout
    )

    # 打印模型参数量
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"model parameters:")
        print(f"    total: {total_params:,}")
        print(f"    trainable: {trainable_params:,}")
        print()

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

    # 使用DistributedSampler
    from torch.utils.data.distributed import DistributedSampler

    train_sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        seed=config.seed
    )

    # 创建dataloader
    dataloader = Dataloader(
        dataset,
        batch_size=config.dataloader.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory
    )

    if accelerator.is_main_process:
        print("dataset loaded: {len(dataset)} samples")
        print()

    # 创建loss
    criterion, loss_type = create_loss_criterion(config)

    # 准备loss
    if hasattr(criterion, 'prepare'):
        criterion.prepare(dataset, num_samples=getattr(config.loss, 'num_samples', 100))

    # 创建optimizer和scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.optimizer.lr,
        weight_decay=config.training.optimizer.weight_decay,
        betas=config.training.optimizer.betas
    )

    # 计算总步数
    total_steps = config.training.epochs * len(dataloader)
    warmup_steps = int(total_steps * config.training.scheduler.warmup_ratio)

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=config.training.scheduler.min_lr
    )

    # 使用accelerator准备
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # Tensorboard & config 备份
    if accelerator.is_main_process:
        # 创建输出目录
        output_dir = os.path.join(config.exp_name, config.config_name)
        os.makedirs(output_dir, exist_ok=True)

        # 复制config文件到输出目录
        import shutil
        config_backup_path = os.path.join(output_dir, 'config_backup.yaml')
        shutil.copy2(args.config, config_backup_path)
        print(f"config backed up to: {config_backup_path}")

        # Tensorboard
        log_dir = os.path.join(output_dir, 'tensorboard_logs')
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter(os.path.join(log_dir, timestamp))
        print(f"Tensorboard logs: {log_dir}/{timestamp}\n")
    else:
        writer = None

    # Validation (可选)

    # 恢复训练（如果指定）
    start_epoch = 0
    start_step = 0

    if args.resume is not None:
        if accelerator.is_main_process:
            print(f"resuming from checkpoint: {args.resume}")

        checkpoint = torch.load(args.resume, map_location='cpu')

        unwarpped_model = accelerator.unwarpp_model(model)
        unwarpped_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch']
        start_step = checkpoint['global_step']

        if accelerator.is_main_process:
            print(f"resumed from eopch {start_epoch}, step {start_step}\n")

    # 开始训练
    try:
        train(
            model=model, 
            dataloader=dataloader, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            criterion=criterion, 
            accelerator=accelerator, 
            writer=writer, 
            config=config, 
            loss_type=loss_type, 
            validator=validator,
            validation_scheduler=validation_scheduler,
            validator_config=config.validation if config.validation.enabled else None,
            train_sampler=train_sampler
        )
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            print(f"\n{'='*60}")
            print("training interrupted by user")
            print(f"{'='*60}")
            save_checkpoint(model, optimizer, scheduler, start_epoch, start_step, config, accelerator, final=True)

    if accelerator.is_main_process and writer is not None:
        writer.close()

if __name__ == "__main__":
    main()







        


