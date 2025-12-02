import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from .utils import process_chunk_freqs, softmax, silu, cross_entropy_loss, get_lr_cosine_schedule, get_gradient_clipping, get_batch, save_checkpoint, load_checkpoint
from .MyRMSNorm import MyRMSNorm
from .MyLinear import Linear
from .MyEmbedding import MyEmbedding
from .MySwiGLU import MySwiGLU
from .MyRoPE import MyRoPE
from .MyAttention import scaled_dot_product_attention, Multihead_Self_Attention
from .MyTransformer import Transformer
from .MyLM import LM
from .MySGD import AdamW
from .BPE_tokenizer import MyBpeTokenizer