import torch.nn as nn
import torch
from tsl.nn import utils
from tsl.nn.blocks.encoders import MLP
from einops import repeat
from Attention_layers import AttentionLayer, SelfAttentionLayer, EmbeddedAttention



class EmbeddedAttentionLayer(nn.Module):
    """
    Spatial embedded attention layer
    """
    def __init__(self,
                 model_dim, adaptive_embedding_dim, feed_forward_dim=2048, dropout=0):
        super().__init__()

        self.attn = EmbeddedAttention(model_dim, adaptive_embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim))

        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, emb, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        # emb: (..., length, model_dim)
        residual = x
        out = self.attn(x, emb)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class ProjectedAttentionLayer(nn.Module):
    """
    Temporal projected attention layer
    """
    def __init__(self, seq_len, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(ProjectedAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.out_attn = AttentionLayer(d_model, n_heads, mask=None)
        self.in_attn = AttentionLayer(d_model, n_heads, mask=None)
        self.projector = nn.Parameter(torch.randn(factor, d_model))

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.MLP = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                 nn.Linear(d_ff, d_model))

        self.seq_len = seq_len

    def forward(self, x):
        # x: [b s n d]
        batch = x.shape[0]
        projector = repeat(self.projector, 'factor d_model -> repeat seq_len factor d_model',
                              repeat=batch, seq_len=self.seq_len)  # [b, s, c, d]

        message_out = self.out_attn(projector, x, x)  # [b, s, c, d] <-> [b s n d] -> [b s c d]
        message_in = self.in_attn(x, projector, message_out)  # [b s n d] <-> [b, s, c, d] -> [b s n d]
        message = x + self.dropout(message_in)
        message = self.norm1(message)
        message = message + self.dropout(self.MLP(message))
        message = self.norm2(message)

        return message


class ImputeFormer(nn.Module):
    """
    Spatiotempoarl Imputation Transformer
    """
    def __init__(
            self,
            num_nodes,
            input_dim=3,
            output_dim=1,
            input_embedding_dim=24,
            learnable_embedding_dim=80,
            feed_forward_dim=256,
            num_temporal_heads=4,
            num_layers=3,
            dropout=0.,
            windows=24,
            dim_proj=10,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = windows
        self.out_steps = windows
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.learnable_embedding_dim = learnable_embedding_dim
        self.model_dim = (
                input_embedding_dim
                + learnable_embedding_dim)
        self.num_temporal_heads = num_temporal_heads
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        self.dim_proj = dim_proj

        self.learnable_embedding = nn.init.xavier_uniform_(
            nn.Parameter(torch.empty(windows, num_nodes, learnable_embedding_dim)))

        self.readout = MLP(self.model_dim, self.model_dim, output_dim, n_layers=2)

        self.attn_layers_t = nn.ModuleList(
            [ProjectedAttentionLayer(self.num_nodes, self.dim_proj, self.model_dim, num_temporal_heads, self.model_dim, dropout)
             for _ in range(num_layers)])

        self.attn_layers_s = nn.ModuleList(
            [EmbeddedAttentionLayer(self.model_dim, learnable_embedding_dim, feed_forward_dim)
                for _ in range(num_layers)])


    def forward(self, x, u, mask):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]
        # Whiten missing values
        x = x * mask

        x = utils.maybe_cat_exog(x, u)
        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)

        node_emb = self.learnable_embedding.expand(batch_size, *self.adaptive_embedding.shape)
        x = torch.cat([x, node_emb], dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        x = x.permute(0, 2, 1, 3)  # [b n s c]
        for att_t, att_s in zip(self.attn_layers_t, self.attn_layers_s):
            x = att_t(x)
            x = att_s(x, self.learnable_embedding, dim=1)

        x = x.permute(0, 2, 1, 3)  # [b s n c]
        out = self.readout(x)

        return out


    @staticmethod
    def add_model_specific_args(parser):
        parser.opt_list('--input-dim', type=int, default=3)
        parser.add_argument('--num-nodes', type=int, default=207)
        parser.add_argument('--output-dim', type=int, default=1)
        parser.add_argument('--input-embedding-dim', type=int, default=24)
        parser.add_argument('--feed-forward-dim', type=int, default=256)
        parser.add_argument('--learnable-embedding-dim', type=int, default=80)
        parser.add_argument('--num_temporal_heads', type=int, default=4)
        parser.add_argument('--num_layers', type=int, default=3)
        parser.add_argument('--dim-proj', type=int, default=10)
        parser.add_argument('--dropout', type=int, default=0.1)
        return parser
