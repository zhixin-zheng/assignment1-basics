import torch
import  torch.nn as nn

class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        '''
        num_embeddings: int Size of the vocabulary.
        embedding_dim: int Dimension of the embedding vectors, i.e., dmodel.
        '''
        super(MyEmbedding, self).__init__()
        self.weights = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self.weights = nn.init.trunc_normal_(self.weights, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.long()
        return self.weights[token_ids]