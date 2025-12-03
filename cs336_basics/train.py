import torch
import os
import numpy as np
import random
import argparse
from typing import Optional
from tqdm import tqdm
from torch.utils.data import Dataset
import cs336_basics
from cs336_basics.train_utils import TrainConfig, TrainLogger, get_gpu_memory

class MemmapDataset(Dataset):
    """使用np.memmap的内存高效数据集"""
    def __init__(self, train_data_path: str, valid_data_path: Optional[str] = None, context_length: int = 256, split_ratio: float = 0.9):
        """
        Args:
            train_data_path: 训练数据文件路径
            valid_data_path: 验证数据文件路径
            context_length: 上下文长度
            split_ratio: 训练集比例
        """
        self.context_length = context_length

        self.train_data = np.memmap(train_data_path, dtype=np.uint16, mode='r')

        if valid_data_path is not None:
            self.val_data = np.memmap(valid_data_path, dtype=np.uint16, mode='r')
        else:
            split_idx = int(len(self.train_data) * split_ratio)
            self.train_data = self.train_data[:split_idx]
            self.val_data = self.train_data[split_idx:]

        if len(self.train_data) < context_length + 1:
            raise ValueError(f"训练数据长度 {len(self.train_data)} 小于 context_length + 1 = {context_length + 1}")
        if len(self.val_data) < context_length + 1:
            raise ValueError(f"验证数据长度 {len(self.val_data)} 小于 context_length + 1 = {context_length + 1}")

    def get_train_batch(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        return cs336_basics.get_batch(self.train_data, batch_size, self.context_length, device)
    
    def get_val_batch(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        return cs336_basics.get_batch(self.val_data, batch_size, self.context_length, device)

def train_model(config: TrainConfig, use_wandb: bool = False):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    with TrainLogger(log_dir="logs", config=config, use_wandb=use_wandb) as tracker:

        logger = tracker.logger
        device = config.device

        checkpoint_dir = tracker.log_dir / "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        dataset = MemmapDataset(config.train_data_path, config.valid_data_path, config.context_length, config.split_ratio)
        
        if config.vocab_size > 65535:
             logger.warning(f"Vocab size {config.vocab_size} exceeds uint16 limit (65535). MemmapDataset uses uint16, which may cause data corruption if token IDs exceed this limit.")

        model = cs336_basics.LM(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            num_layers=config.num_layers,
            context_length=config.context_length,
            theta=config.rope_theta
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}")

        optimizer = cs336_basics.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
            eps=config.eps
        )

        start_iter = 0

        best_val_loss = float('inf')

        # checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.pth")
        # if os.path.exists(checkpoint_file):
        #     start_iter = cs336_basics.load_checkpoint(checkpoint_file, model, optimizer)
        #     logger.info(f"Loaded checkpoint from {checkpoint_file}, starting from iteration {start_iter}.")
        
        logger.info("Starting training...")
        model.train()
        lr = config.learning_rate

        print(f"{'Iter':>10}{'GPU Mem':>10}{'Train Loss':>12}{'Val Loss':>12}{'Val PPL':>12}{'Grad Norm':>10}{'LR':>10}")
        pbar = tqdm(range(start_iter, config.max_iters), dynamic_ncols=True, bar_format='{l_bar}{bar:10}{r_bar}')
        current_val_loss = float('inf')
        current_val_ppl = float('inf')

        for iter in pbar:
            x, y = dataset.get_train_batch(config.batch_size, device)
            logits = model(x)
            loss = cs336_basics.cross_entropy_loss(logits, y)

            optimizer.zero_grad()
            loss.backward()

            grad_norm = cs336_basics.get_gradient_clipping(model.parameters(), config.max_grad_norm)

            lr = cs336_basics.get_lr_cosine_schedule(
                iter,
                config.min_learning_rate,
                config.learning_rate,
                config.warmup_iters,
                config.cosine_cycle_iters
            )

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.step()

            if iter % config.eval_interval == 0 and iter != start_iter:
                eval_metrics = evaluate_model(model, dataset, config.batch_size, device)
                val_loss = eval_metrics["val_loss"]
                val_perplexity = eval_metrics["val_perplexity"]
                logger.info(f"Iter {iter}: Val Loss = {val_loss:.4f}, Val Perplexity = {val_perplexity:.4f}")
                pbar.write(f"Iter {iter}: Val Loss = {val_loss:.4f}, Val Perplexity = {val_perplexity:.4f}")
                tracker.log_metrics(
                    iteration=iter,
                    train_loss=loss.item(),
                    val_loss=val_loss,
                    val_perplexity=val_perplexity,
                    learning_rate=lr,
                    grad_norm=grad_norm.item() if grad_norm is not None else None
                )
                current_val_loss = val_loss
                current_val_ppl = val_perplexity

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    cs336_basics.save_checkpoint(model, optimizer, iter, os.path.join(checkpoint_dir, "best.pth"))

            val_loss_str = f"{current_val_loss:.4f}" if current_val_loss != float('inf') else "-"
            val_ppl_str = f"{current_val_ppl:.2f}" if current_val_ppl != float('inf') else "-"
            g_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else (grad_norm if grad_norm else 0.0)

            if iter % config.save_interval == 0 and iter != start_iter:
                cs336_basics.save_checkpoint(model, optimizer, iter, os.path.join(checkpoint_dir, f"iter_{iter}.pth"))
                logger.info(f"Saved checkpoint at iteration {iter}.")
                pbar.write(f"Saved checkpoint at iteration {iter}.")

            desc = (
                f"{iter:>10}"                # Iter
                f"{get_gpu_memory():>9.2f}G" # GPU Mem
                f"{loss.item():>12.4f}"      # Train Loss
                f"{val_loss_str:>12}"        # Val Loss
                f"{val_ppl_str:>12}"         # Val PPL
                f"{g_norm_val:>10.2f}"       # Grad Norm
                f"{lr:>10.2e}"               # LR (科学计数法)
            )
            pbar.set_description(desc)
            
        cs336_basics.save_checkpoint(model, optimizer, config.max_iters, os.path.join(checkpoint_dir, "last.pth"))
        logger.info(f"Training Completed. Saved final checkpoint at iteration {config.max_iters}.")



def evaluate_model(model: torch.nn.Module, dataset: MemmapDataset, batch_size: int, device: str, num_batches: int = 10) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for _ in range(num_batches):
            x, y = dataset.get_val_batch(batch_size, device)
            logits = model(x)
            loss = cs336_basics.cross_entropy_loss(logits, y)
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    model.train()
    return {"val_loss": avg_loss, "val_perplexity": perplexity}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the training configuration file.')
    parser.add_argument('--experiment_name', type=str, help='Name of the experiment for logging purposes.')
    parser.add_argument('--train_data_path', type=str, help='Path to the training data file.')
    parser.add_argument('--valid_data_path', type=str, help='Path to the validation data file.')
    parser.add_argument('--split_ratio', type=float, default=0.9, help='Train-validation split ratio.')

    # Model configurations
    parser.add_argument('--vocab_size', type=int, help='Vocabulary size.')
    parser.add_argument('--context_length', type=int, help='Context length for training.')
    parser.add_argument('--d_model', type=int, help='Dimension of the model.')
    parser.add_argument('--num_layers', type=int, help='Number of transformer layers.')
    parser.add_argument('--num_heads', type=int, help='Number of attention heads.')
    parser.add_argument('--d_ff', type=int, help='Dimension of the feedforward network.')
    parser.add_argument('--rope_theta', type=float, help='Theta parameter for RoPE.')

    # Training configurations
    parser.add_argument('--batch_size', type=int, help='Batch size for training.')
    parser.add_argument('--max_iters', type=int, help='Maximum number of training iterations.')
    parser.add_argument('--learning_rate', type=float, help='Initial learning rate.')
    parser.add_argument('--min_learning_rate', type=float, help='Minimum learning rate.')
    parser.add_argument('--beta1', type=float, help='Beta1 for AdamW optimizer.')
    parser.add_argument('--beta2', type=float, help='Beta2 for AdamW optimizer.')
    parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon for AdamW optimizer.')
    parser.add_argument('--warmup_iters', type=int, help='Number of warmup iterations.')
    parser.add_argument('--cosine_cycle_iters', type=int, help='Number of iterations for cosine learning rate schedule.')
    parser.add_argument('--weight_decay', type=float, help='Weight decay for AdamW optimizer.')
    parser.add_argument('--max_grad_norm', type=float, help='Maximum gradient norm for clipping.')
    parser.add_argument('--eval_interval', type=int, help='Evaluation interval in iterations.')
    parser.add_argument('--save_interval', type=int, help='Checkpoint save interval in iterations.')

    # other
    parser.add_argument('--device', type=str, default="cuda", help='Device to use for training (e.g., "cpu" or "cuda").')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to use Weights & Biases for logging.')

    args = parser.parse_args()
    if args.config:
        config = TrainConfig.load(args.config)
        print(f"Loaded configuration from {args.config}")
    else:
        config_dict = {}
        for key in TrainConfig.__annotations__.keys():
            if hasattr(args, key):
                val = getattr(args, key)
                if val is not None:
                    config_dict[key] = val
        config = TrainConfig(**config_dict)

    train_model(config, use_wandb=args.use_wandb)

if __name__ == "__main__":
    main()