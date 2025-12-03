import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import torch
import wandb
from dataclasses import dataclass, asdict

def increment_path(path: Union[str, Path], exist_ok: bool = False) -> Path:
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return path
    else:
        dirs = [d for d in path.parent.iterdir() if d.is_dir() and d.name.startswith(path.stem)]
        matches = [d.name[len(path.stem):] for d in dirs]
        indices = [int(m[1:]) for m in matches if m.startswith('_') and m[1:].isdigit()]
        n = max(indices) + 1 if indices else 2
        return path.parent / f"{path.stem}_{n}"

def get_gpu_memory():
    """获取当前 GPU 显存占用 (GB)"""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024**3
    return 0.0

@dataclass
class TrainConfig:
    """实验配置数据类"""
    # 实验基本信息
    experiment_name: str
    
    # 模型参数
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float
    
    # 训练参数
    batch_size: int
    max_iters: int
    learning_rate: float
    min_learning_rate: float
    beta1: float
    beta2: float
    eps: float
    warmup_iters: int
    cosine_cycle_iters: int
    weight_decay: float
    max_grad_norm: float
    eval_interval: int
    save_interval: int
    
    # 数据参数
    train_data_path: str
    valid_data_path: str
    split_ratio: float
    
    # 其他参数
    device: str
    seed: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainConfig':
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
class TrainLogger:
    def __init__(self, log_dir: str, config: TrainConfig, use_wandb: bool = False, wandb_project: str = "cs336_basics"):
        self.log_dir = increment_path(Path(log_dir) / config.experiment_name)
        self.config = config
        self.use_wandb = use_wandb
        os.makedirs(self.log_dir, exist_ok=True)
        config.save(self.log_dir / "config.json")
        
        self.setup_logger()
        if use_wandb:
            self.setup_wandb(wandb_project)

        self.logger.info("Experiment Configuration:")
        for key, value in config.to_dict().items():
            self.logger.info(f"  {key}: {value}")

        self.metrics = {
            "iteration": [],
            "train_loss": [],
            "val_loss": [],
            "val_perplexity": [],
            "learning_rate": [],
            "grad_norm": [],
            "time_elapsed": [],
            "start_time": time.time()
        }
        
    def setup_logger(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'train.log')),
                # logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('train_logger')

    def setup_wandb(self, project_name: str):
        os.environ["WANDB_BASE_URL"] = "https://api.bandw.top"  
        wandb.init(
            entity="bg2dph-yingzikeji",
            project=project_name,
            name=self.config.experiment_name,
            config=self.config.to_dict(),
            dir=str(self.log_dir)
        )
        self.logger.info("Wandb初始化完成")

    def log_metrics(self, iteration: int, train_loss: float, val_loss: Optional[float] = None, val_perplexity: Optional[float] = None, learning_rate: Optional[float] = None, grad_norm: Optional[float] = None):
        self.metrics["train_loss"].append(train_loss)
        self.metrics["iteration"].append(iteration)
        self.metrics["time_elapsed"].append(time.time() - self.metrics["start_time"])

        if val_loss is not None:
            self.metrics["val_loss"].append(val_loss)
        if val_perplexity is not None:
            self.metrics["val_perplexity"].append(val_perplexity)
        if learning_rate is not None:
            self.metrics["learning_rate"].append(learning_rate)
        if grad_norm is not None:
            self.metrics["grad_norm"].append(grad_norm)

        if self.use_wandb:
            log_data = {
                "train/loss": train_loss,
                "iteration": iteration,
                "time_elapsed": self.metrics["time_elapsed"][-1]
            }
            if val_loss is not None:
                log_data["val/loss"] = val_loss
            if val_perplexity is not None:
                log_data["val/perplexity"] = val_perplexity
            if learning_rate is not None:
                log_data["train/learning_rate"] = learning_rate
            if grad_norm is not None:
                log_data["train/grad_norm"] = grad_norm
            wandb.log(log_data, step=iteration)

        if iteration % 100 == 0:
            self.save_metrics()

    def save_metrics(self):
        metrics_path = os.path.join(self.log_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save_metrics()
        if self.use_wandb:
            wandb.finish()
        
        