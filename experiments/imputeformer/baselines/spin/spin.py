from typing import Optional

import torch
from tsl.nn.utils import utils
from torch.nn import functional as F
from torch import nn, Tensor
from torch.nn import LayerNorm
from torch_geometric.typing import OptTensor
from tsl.nn.base import StaticGraphEmbedding
from tsl.nn.blocks.encoders import MLP, ResidualMLP
from tsl.nn.blocks.encoders.transformer import SpatioTemporalTransformerLayer, \
    TransformerLayer, Transformer
from .spin_layers import PositionalEncoder, TemporalGraphAdditiveAttention
from tsl.nn.layers.graph_convs.dense_graph_conv import DenseGraphConvOrderK
from tsl.nn.layers.graph_convs import GatedGraphNetwork
from tsl.nn.models.stgn.gated_gn_model import GatedGraphNetworkModel
from einops import rearrange
from einops.layers.torch import Rearrange


class SPINModel(nn.Module):

    def __init__(self, input_size: int,
                 hidden_size: int,
                 n_nodes: int,
                 u_size: Optional[int] = None,
                 output_size: Optional[int] = None,
                 temporal_self_attention: bool = True,
                 reweight: Optional[str] = 'softmax',
                 n_layers: int = 4,
                 eta: int = 3,
                 message_layers: int = 1):
        super(SPINModel, self).__init__()

        u_size = u_size or input_size
        output_size = output_size or input_size
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.eta = eta
        self.temporal_self_attention = temporal_self_attention

        self.u_enc = PositionalEncoder(in_channels=u_size,
                                       out_channels=hidden_size,
                                       n_layers=2,
                                       n_nodes=n_nodes)

        self.h_enc = MLP(input_size, hidden_size, n_layers=2)
        self.h_norm = LayerNorm(hidden_size)

        self.valid_emb = StaticGraphEmbedding(n_nodes, hidden_size)
        self.mask_emb = StaticGraphEmbedding(n_nodes, hidden_size)

        self.x_skip = nn.ModuleList()
        self.encoder, self.readout = nn.ModuleList(), nn.ModuleList()
        for l in range(n_layers):
            x_skip = nn.Linear(input_size, hidden_size)
            encoder = TemporalGraphAdditiveAttention(
                input_size=hidden_size,
                output_size=hidden_size,
                msg_size=hidden_size,
                msg_layers=message_layers,
                temporal_self_attention=temporal_self_attention,
                reweight=reweight,
                mask_temporal=True,
                mask_spatial=l < eta,
                norm=True,
                root_weight=True,
                dropout=0.0
            )
            readout = MLP(hidden_size, hidden_size, output_size,
                          n_layers=2)
            self.x_skip.append(x_skip)
            self.encoder.append(encoder)
            self.readout.append(readout)

    def forward(self, x: Tensor, u: Tensor, mask: Tensor,
                edge_index: Tensor, edge_weight: OptTensor = None,
                node_index: OptTensor = None, target_nodes: OptTensor = None):
        if target_nodes is None:
            target_nodes = slice(None)

        # Whiten missing values
        x = x * mask

        # POSITIONAL ENCODING #################################################
        # Obtain spatio-temporal positional encoding for every node-step pair #
        # in both observed and target sets. Encoding are obtained by jointly  #
        # processing node and time positional encoding.                       #

        # Build (node, timestamp) encoding
        q = self.u_enc(u, node_index=node_index)
        # Condition value on key
        h = self.h_enc(x) + q

        # ENCODER #############################################################
        # Obtain representations h^i_t for every (i, t) node-step pair by     #
        # only taking into account valid data in representation set.          #

        # Replace H in missing entries with queries Q
        h = torch.where(mask.bool(), h, q)
        # Normalize features
        h = self.h_norm(h)

        imputations = []

        for l in range(self.n_layers):
            if l == self.eta:
                # Condition H on two different embeddings to distinguish
                # valid values from masked ones
                valid = self.valid_emb(token_index=node_index)
                masked = self.mask_emb(token_index=node_index)
                h = torch.where(mask.bool(), h + valid, h + masked)
            # Masked Temporal GAT for encoding representation
            h = h + self.x_skip[l](x) * mask  # skip connection for valid x
            h = self.encoder[l](h, edge_index, mask=mask)
            # Read from H to get imputations
            target_readout = self.readout[l](h[..., target_nodes, :])
            imputations.append(target_readout)

        # Get final layer imputations
        x_hat = imputations.pop(-1)

        return x_hat, imputations

    @staticmethod
    def add_model_specific_args(parser):
        parser.opt_list('--hidden-size', type=int, tunable=True, default=32,
                        options=[32, 64, 128, 256])
        parser.add_argument('--u-size', type=int, default=None)
        parser.add_argument('--output-size', type=int, default=None)
        parser.add_argument('--temporal-self-attention', type=bool,
                            default=True)
        parser.add_argument('--reweight', type=str, default='softmax')
        parser.add_argument('--n-layers', type=int, default=4)
        parser.add_argument('--eta', type=int, default=3)
        parser.add_argument('--message-layers', type=int, default=1)
        return parser

# saved model
# class MySPINModel(nn.Module):
#
#     def __init__(self, input_size: int,
#                  hidden_size: int,
#                  n_nodes: int,
#                  ff_size: int,
#                  u_size: Optional[int] = None,
#                  n_heads: int = 1,
#                  output_size: Optional[int] = None,
#                  temporal_self_attention: bool = True,
#                  activation: str = 'elu',
#                  reweight: Optional[str] = 'softmax',
#                  n_layers: int = 4,
#                  eta: int = 3,
#                  diagonal_attention_mask: bool = True,
#                  emb_size: int = 10,
#                  window: int = 24,
#                  message_layers: int = 1):
#         super(MySPINModel, self).__init__()
#
#         u_size = u_size or input_size
#         output_size = output_size or input_size
#         self.n_nodes = n_nodes
#         self.n_layers = n_layers
#         self.eta = eta
#         self.temporal_self_attention = temporal_self_attention
#
#         self.u_enc = PositionalEncoder(in_channels=u_size,
#                                        out_channels=hidden_size,
#                                        n_layers=2,
#                                        n_nodes=n_nodes)
#
#         kwargs = dict(input_size=hidden_size,
#                       hidden_size=hidden_size,
#                       ff_size=ff_size,
#                       n_heads=n_heads,
#                       activation=activation,
#                       causal=False,
#                       dropout=0)
#         kwargs_gconv = dict(input_size=hidden_size,
#                             output_size=hidden_size,
#                             support_len=1,
#                             order=2,
#                             include_self=True,
#                             channel_last=True)
#
#         self.h_enc = MLP(input_size, hidden_size, n_layers=2)
#         self.h_norm = LayerNorm(hidden_size)
#         self.diagonal_attention_mask = diagonal_attention_mask
#
#         self.valid_emb = StaticGraphEmbedding(n_nodes, hidden_size)
#         self.mask_emb = StaticGraphEmbedding(n_nodes, hidden_size)
#         self.source_embeddings = StaticGraphEmbedding(n_nodes, emb_size)
#         self.target_embeddings = StaticGraphEmbedding(n_nodes, emb_size)
#         self.x_skip = nn.ModuleList()
#         # # time mixer
#         # self.TimeMixer = nn.Sequential(
#         #     Rearrange('b s n c -> b n c s'),
#         #     nn.Linear(window, hidden_size),
#         #     utils.get_layer_activation(activation)(),
#         #     nn.Linear(hidden_size, window),
#         #     Rearrange('b n c s -> b s n c')
#         # )
#
#         self.t_encoder, self.s_encoder, self.readout = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
#         for l in range(n_layers):
#             x_skip = nn.Linear(input_size, hidden_size)
#             t_encoder = TransformerLayer(**kwargs)  ##########
#             # t_encoder = self.TimeMixer  # replace Transformer with MLP
#
#             s_encoder = DenseGraphConvOrderK(**kwargs_gconv)
#             readout = MLP(hidden_size, hidden_size, output_size,
#                           n_layers=2)
#             self.x_skip.append(x_skip)
#             self.t_encoder.append(t_encoder)
#             self.s_encoder.append(s_encoder)
#             self.readout.append(readout)
#
#     def get_learned_adj(self):
#         logits = F.relu(self.source_embeddings() @ self.target_embeddings().T)
#         adj = torch.softmax(logits, dim=1)
#         return adj
#
#     def forward(self, x: Tensor, u: Tensor, mask: Tensor,
#                 edge_index: Tensor, edge_weight: OptTensor = None,
#                 node_index: OptTensor = None, target_nodes: OptTensor = None):
#         if target_nodes is None:
#             target_nodes = slice(None)
#
#         if self.diagonal_attention_mask:
#             mask_time = torch.eye(x.shape[1]).to(x.device)
#         else:
#             mask_time = None
#
#         adj_z = self.get_learned_adj()
#
#         # Whiten missing values
#         x = x * mask
#
#         # POSITIONAL ENCODING #################################################
#         # Obtain spatio-temporal positional encoding for every node-step pair #
#         # in both observed and target sets. Encoding are obtained by jointly  #
#         # processing node and time positional encoding.                       #
#
#         # Build (node, timestamp) encoding
#         q = self.u_enc(u, node_index=node_index)
#         # Condition value on key
#         h = self.h_enc(x) + q
#
#         # ENCODER #############################################################
#         # Obtain representations h^i_t for every (i, t) node-step pair by     #
#         # only taking into account valid data in representation set.          #
#
#         # Replace H in missing entries with queries Q
#         h = torch.where(mask.bool(), h, q)
#         # Normalize features
#         h = self.h_norm(h)
#
#         imputations = []
#
#         for l in range(self.n_layers):
#             if l == self.eta:
#                 # Condition H on two different embeddings to distinguish
#                 # valid values from masked ones
#                 valid = self.valid_emb(token_index=node_index)
#                 masked = self.mask_emb(token_index=node_index)
#                 h = torch.where(mask.bool(), h + valid, h + masked)
#             # Masked Temporal GAT for encoding representation
#             h = h + self.x_skip[l](x) * mask  # skip connection for valid x
#             h = self.t_encoder[l](h, mask_time)
#             ##################
#             # h = self.t_encoder[l](h)
#             ##################
#             h = self.s_encoder[l](h, adj_z)
#             # Read from H to get imputations
#             target_readout = self.readout[l](h[..., target_nodes, :])
#             imputations.append(target_readout)
#
#         # Get final layer imputations
#         x_hat = imputations.pop(-1)
#
#         return x_hat, imputations
#
#     @staticmethod
#     def add_model_specific_args(parser):
#         parser.opt_list('--hidden-size', type=int, tunable=True, default=32,
#                         options=[32, 64, 128, 256])
#         parser.add_argument('--u-size', type=int, default=None)
#         parser.add_argument('--output-size', type=int, default=None)
#         parser.add_argument('--temporal-self-attention', type=bool,
#                             default=True)
#         parser.add_argument('--reweight', type=str, default='softmax')
#         parser.add_argument('--n-layers', type=int, default=4)
#         parser.add_argument('--eta', type=int, default=3)
#         parser.add_argument('--message-layers', type=int, default=1)
#         return parser


class TemporalGraphConv(nn.Module):
    def __init__(self, input_size, ff_size, output_size, bias=True, activation='silu'):
        super(TemporalGraphConv, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)
        if bias:
            self.b = nn.Parameter(torch.Tensor(ff_size))
        else:
            self.register_parameter('b', None)
        self.reset_parameters()

        self.linear_in = nn.Linear(input_size, ff_size)
        self.linear_out = nn.Linear(ff_size, output_size)

    def reset_parameters(self):
        self.linear.reset_parameters()
        if self.b is not None:
            self.b.data.zero_()

    def forward(self, x, adj):
        """"""
        # linear transformation
        x = self.linear_in(x)
        b, s, n, f = x.size()
        # reshape to have features+T as last dim
        x = rearrange(x, 'b s n f -> b s (n f)')
        # message passing
        x = torch.matmul(adj, x)
        x = rearrange(x, 'b s (n f) -> b s n f', s=s, f=f)
        if self.b is not None:
            x = x + self.b
        x = self.linear_out(x)
        return x

## developing model
class MySPINModel(nn.Module):
    def __init__(self, input_size: int,
                 hidden_size: int,
                 n_nodes: int,
                 ff_size: int,
                 u_size: Optional[int] = None,
                 n_heads: int = 1,
                 output_size: Optional[int] = None,
                 temporal_self_attention: bool = True,
                 activation: str = 'elu',
                 n_layers: int = 4,
                 eta: int = 3,
                 diagonal_attention_mask: bool = True,
                 window: int = 24):
        super(MySPINModel, self).__init__()

        u_size = u_size or input_size
        output_size = output_size or input_size
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.eta = eta
        self.temporal_self_attention = temporal_self_attention

        self.u_enc = PositionalEncoder(in_channels=u_size,
                                       out_channels=hidden_size,
                                       n_layers=2,
                                       n_nodes=n_nodes)

        kwargs = dict(input_size=hidden_size,
                      hidden_size=hidden_size,
                      ff_size=ff_size,
                      n_heads=n_heads,
                      n_layers=1,
                      activation=activation,
                      causal=False,
                      axis='steps',
                      dropout=0)

        kwargs_gconv = dict(input_size=hidden_size,
                            output_size=hidden_size,
                            support_len=1,
                            order=1,
                            include_self=True,
                            channel_last=True)


        self.h_enc = MLP(input_size, hidden_size, n_layers=2)
        self.h_norm = LayerNorm(hidden_size)
        self.diagonal_attention_mask = diagonal_attention_mask

        self.emb = StaticGraphEmbedding(n_nodes, hidden_size)
        self.x_skip = nn.ModuleList()

        self.t_encoder, self.s_encoder, self.readout = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for l in range(n_layers):
            x_skip = nn.Linear(input_size, hidden_size)
            t_encoder = Transformer(**kwargs)
            s_encoder = DenseGraphConvOrderK(**kwargs_gconv)
            self.x_skip.append(x_skip)
            self.t_encoder.append(t_encoder)
            self.s_encoder.append(s_encoder)
        self.readout = MLP(hidden_size, hidden_size, output_size,
                      n_layers=2)

    def get_learned_adj(self):
        return F.softmax(self.emb() @ self.emb().T, -1)


    def forward(self, x: Tensor, u: Tensor, mask: Tensor,
                edge_index: Tensor, edge_weight: OptTensor = None,
                node_index: OptTensor = None, target_nodes: OptTensor = None):
        # x: [batches steps nodes features]
        adj_z = self.get_learned_adj()

        # Whiten missing values
        x = x * mask
        q = self.u_enc(u, node_index=node_index)
        h = self.h_enc(x) + q
        h = self.h_norm(h)
        h = h + self.emb()


        for l in range(self.n_layers):
            residual = h
            # Masked Temporal GAT for encoding representation
            h = h + self.x_skip[l](x) * mask  # skip connection for valid x
            h = self.t_encoder[l](h) + h
            h = self.s_encoder[l](h, adj_z)
            h = F.silu(h) + residual

        # Get final layer imputations
        x_hat = self.readout(h)

        return x_hat

    @staticmethod
    def add_model_specific_args(parser):
        parser.opt_list('--hidden-size', type=int, tunable=True, default=32,
                        options=[32, 64, 128, 256])
        parser.add_argument('--u-size', type=int, default=None)
        parser.add_argument('--output-size', type=int, default=None)
        parser.add_argument('--temporal-self-attention', type=bool,
                            default=True)
        parser.add_argument('--reweight', type=str, default='softmax')
        parser.add_argument('--n-layers', type=int, default=4)
        parser.add_argument('--eta', type=int, default=3)
        parser.add_argument('--message-layers', type=int, default=1)
        return parser



class GatedSPINModel(nn.Module):
    def __init__(self, input_size: int,
                 hidden_size: int,
                 n_nodes: int,
                 ff_size: int,
                 u_size: Optional[int] = None,
                 n_heads: int = 1,
                 output_size: Optional[int] = None,
                 temporal_self_attention: bool = True,
                 activation: str = 'elu',
                 reweight: Optional[str] = 'softmax',
                 n_layers: int = 4,
                 eta: int = 3,
                 diagonal_attention_mask: bool = True,
                 emb_size: int = 10,
                 message_layers: int = 1):
        super(GatedSPINModel, self).__init__()

        u_size = u_size or input_size
        output_size = output_size or input_size
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.eta = eta
        self.temporal_self_attention = temporal_self_attention

        self.u_enc = PositionalEncoder(in_channels=u_size,
                                       out_channels=hidden_size,
                                       n_layers=2,
                                       n_nodes=n_nodes)

        kwargs = dict(input_size=hidden_size,
                      hidden_size=hidden_size,
                      ff_size=ff_size,
                      n_heads=n_heads,
                      activation=activation,
                      causal=False,
                      dropout=0)


        self.h_enc = MLP(input_size, hidden_size, n_layers=2)
        self.h_norm = LayerNorm(hidden_size)
        self.diagonal_attention_mask = diagonal_attention_mask

        self.node_emb = nn.Parameter(
            torch.empty(n_nodes, hidden_size)) #[N, C]
        nn.init.xavier_uniform_(self.node_emb)

        self.valid_emb = StaticGraphEmbedding(n_nodes, hidden_size)
        self.mask_emb = StaticGraphEmbedding(n_nodes, hidden_size)
        self.source_embeddings = StaticGraphEmbedding(n_nodes, emb_size)
        self.target_embeddings = StaticGraphEmbedding(n_nodes, emb_size)
        self.x_skip = nn.ModuleList()


        self.spatial_mlp = nn.Sequential(
        ResidualMLP(2*hidden_size, hidden_size,n_layers=2))


        self.t_encoder, self.s_encoder, self.readout = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for l in range(n_layers):
            x_skip = nn.Linear(input_size, hidden_size)
            t_encoder = TransformerLayer(**kwargs)
            s_encoder = self.spatial_mlp
            readout = MLP(hidden_size, hidden_size, output_size,
                          n_layers=2)
            self.x_skip.append(x_skip)
            self.t_encoder.append(t_encoder)
            self.s_encoder.append(s_encoder)
            self.readout.append(readout)


    def forward(self, x: Tensor, u: Tensor, mask: Tensor,
                edge_index: Tensor, edge_weight: OptTensor = None,
                node_index: OptTensor = None, target_nodes: OptTensor = None):

        batch_size, seq_length, _, _ = x.shape

        if target_nodes is None:
            target_nodes = slice(None)

        if self.diagonal_attention_mask:
            mask_time = torch.eye(x.shape[1]).to(x.device)
        else:
            mask_time = None

        # Whiten missing values
        x = x * mask

        # POSITIONAL ENCODING #################################################
        # Obtain spatio-temporal positional encoding for every node-step pair #
        # in both observed and target sets. Encoding are obtained by jointly  #
        # processing node and time positional encoding.                       #

        # Build (node, timestamp) encoding
        q = self.u_enc(u, node_index=node_index)
        # Condition value on key
        h = self.h_enc(x) + q  #[b, s, n, c]

        # ENCODER #############################################################
        # Obtain representations h^i_t for every (i, t) node-step pair by     #
        # only taking into account valid data in representation set.          #

        # Replace H in missing entries with queries Q
        h = torch.where(mask.bool(), h, q)
        # Normalize features
        h = self.h_norm(h)

        imputations = []

        for l in range(self.n_layers):
            if l == self.eta:
                # Condition H on two different embeddings to distinguish
                # valid values from masked ones
                valid = self.valid_emb(token_index=node_index)
                masked = self.mask_emb(token_index=node_index)
                h = torch.where(mask.bool(), h + valid, h + masked)
            # Masked Temporal GAT for encoding representation
            h = h + self.x_skip[l](x) * mask  # skip connection for valid x
            h = self.t_encoder[l](h, mask_time)

            s_emb = self.node_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, seq_length, -1, -1) #[N, C]->[B,S,N,C]
            h = utils.maybe_cat_exog(h, s_emb)
            h = self.s_encoder[l](h)

            # Read from H to get imputations
            target_readout = self.readout[l](h[..., target_nodes, :])
            imputations.append(target_readout)

        # Get final layer imputations
        x_hat = imputations.pop(-1)

        return x_hat, imputations

    @staticmethod
    def add_model_specific_args(parser):
        parser.opt_list('--hidden-size', type=int, tunable=True, default=32,
                        options=[32, 64, 128, 256])
        parser.add_argument('--u-size', type=int, default=None)
        parser.add_argument('--output-size', type=int, default=None)
        parser.add_argument('--temporal-self-attention', type=bool,
                            default=True)
        parser.add_argument('--reweight', type=str, default='softmax')
        parser.add_argument('--n-layers', type=int, default=4)
        parser.add_argument('--eta', type=int, default=3)
        parser.add_argument('--message-layers', type=int, default=1)
        return parser


# class MySPINModel(nn.Module):
#
#     def __init__(self, input_size: int,
#                  hidden_size: int,
#                  n_nodes: int,
#                  ff_size: int,
#                  u_size: Optional[int] = None,
#                  n_heads: int = 1,
#                  output_size: Optional[int] = None,
#                  temporal_self_attention: bool = True,
#                  activation: str = 'elu',
#                  reweight: Optional[str] = 'softmax',
#                  n_layers: int = 4,
#                  eta: int = 3,
#                  diagonal_attention_mask: bool = True,
#                  emb_size: int = 10,
#                  message_layers: int = 1):
#         super(MySPINModel, self).__init__()
#
#         u_size = u_size or input_size
#         output_size = output_size or input_size
#         self.n_nodes = n_nodes
#         self.n_layers = n_layers
#         self.eta = eta
#         self.temporal_self_attention = temporal_self_attention
#
#         self.u_enc = PositionalEncoder(in_channels=u_size,
#                                        out_channels=hidden_size,
#                                        n_layers=2,
#                                        n_nodes=n_nodes)
#
#         kwargs = dict(input_size=hidden_size,
#                       hidden_size=hidden_size,
#                       ff_size=ff_size,
#                       n_heads=n_heads,
#                       activation=activation,
#                       causal=False,
#                       dropout=0)
#         kwargs_gconv = dict(input_size=hidden_size,
#                             output_size=hidden_size,
#                             support_len=1,
#                             order=2,
#                             include_self=True,
#                             channel_last=True)
#
#         self.h_enc = MLP(input_size, hidden_size, n_layers=2)
#         self.h_norm = LayerNorm(hidden_size)
#         self.diagonal_attention_mask = diagonal_attention_mask
#
#         self.valid_emb = StaticGraphEmbedding(n_nodes, hidden_size)
#         self.mask_emb = StaticGraphEmbedding(n_nodes, hidden_size)
#         self.source_embeddings = StaticGraphEmbedding(n_nodes, emb_size)
#         self.target_embeddings = StaticGraphEmbedding(n_nodes, emb_size)
#         self.x_skip = nn.ModuleList()
#         self.t_encoder, self.s_encoder, self.readout = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
#         for l in range(n_layers):
#             x_skip = nn.Linear(input_size, hidden_size)
#             t_encoder = TransformerLayer(**kwargs)  ##########
#             s_encoder = DenseGraphConvOrderK(**kwargs_gconv) ##########
#             readout = MLP(hidden_size, hidden_size, output_size,
#                           n_layers=2)
#             self.x_skip.append(x_skip)
#             self.t_encoder.append(t_encoder)
#             self.s_encoder.append(s_encoder)
#             self.readout.append(readout)
#
#     def get_learned_adj(self):
#         logits = F.relu(self.source_embeddings() @ self.target_embeddings().T)
#         adj = torch.softmax(logits, dim=1)
#         return adj
#
#     def forward(self, x: Tensor, u: Tensor, mask: Tensor,
#                 edge_index: Tensor, edge_weight: OptTensor = None,
#                 node_index: OptTensor = None, target_nodes: OptTensor = None):
#         if target_nodes is None:
#             target_nodes = slice(None)
#
#         if self.diagonal_attention_mask:
#             mask_time = torch.eye(x.shape[1]).to(x.device)
#         else:
#             mask_time = None
#
#         adj_z = self.get_learned_adj()
#
#         # Whiten missing values
#         x = x * mask
#
#         # POSITIONAL ENCODING #################################################
#         # Obtain spatio-temporal positional encoding for every node-step pair #
#         # in both observed and target sets. Encoding are obtained by jointly  #
#         # processing node and time positional encoding.                       #
#
#         # Build (node, timestamp) encoding
#         q = self.u_enc(u, node_index=node_index)
#         # Condition value on key
#         h = self.h_enc(x) + q
#
#         # ENCODER #############################################################
#         # Obtain representations h^i_t for every (i, t) node-step pair by     #
#         # only taking into account valid data in representation set.          #
#
#         # Replace H in missing entries with queries Q
#         h = torch.where(mask.bool(), h, q)
#         # Normalize features
#         h = self.h_norm(h)
#
#         imputations = []
#
#         for l in range(self.n_layers):
#             if l == self.eta:
#                 # Condition H on two different embeddings to distinguish
#                 # valid values from masked ones
#                 valid = self.valid_emb(token_index=node_index)
#                 masked = self.mask_emb(token_index=node_index)
#                 h = torch.where(mask.bool(), h + valid, h + masked)
#             # Masked Temporal GAT for encoding representation
#             h = h + self.x_skip[l](x) * mask  # skip connection for valid x
#             h = self.t_encoder[l](h, mask_time)
#             h = self.s_encoder[l](h, adj_z)
#             # Read from H to get imputations
#             target_readout = self.readout[l](h[..., target_nodes, :])
#             imputations.append(target_readout)
#
#         # Get final layer imputations
#         x_hat = imputations.pop(-1)
#
#         return x_hat, imputations
#
#     @staticmethod
#     def add_model_specific_args(parser):
#         parser.opt_list('--hidden-size', type=int, tunable=True, default=32,
#                         options=[32, 64, 128, 256])
#         parser.add_argument('--u-size', type=int, default=None)
#         parser.add_argument('--output-size', type=int, default=None)
#         parser.add_argument('--temporal-self-attention', type=bool,
#                             default=True)
#         parser.add_argument('--reweight', type=str, default='softmax')
#         parser.add_argument('--n-layers', type=int, default=4)
#         parser.add_argument('--eta', type=int, default=3)
#         parser.add_argument('--message-layers', type=int, default=1)
#         return parser

# class MySPINModel(nn.Module):
#
#     def __init__(self, input_size: int,
#                  hidden_size: int,
#                  n_nodes: int,
#                  ff_size: int,
#                  u_size: Optional[int] = None,
#                  n_heads: int = 1,
#                  output_size: Optional[int] = None,
#                  temporal_self_attention: bool = True,
#                  activation: str = 'elu',
#                  reweight: Optional[str] = 'softmax',
#                  n_layers: int = 4,
#                  eta: int = 3,
#                  diagonal_attention_mask: bool = True,
#                  emb_size: int = 10,
#                  message_layers: int = 1):
#         super(MySPINModel, self).__init__()
#
#         u_size = u_size or input_size
#         output_size = output_size or input_size
#         self.n_nodes = n_nodes
#         self.n_layers = n_layers
#         self.eta = eta
#         self.temporal_self_attention = temporal_self_attention
#
#         self.u_enc = PositionalEncoder(in_channels=u_size,
#                                        out_channels=hidden_size,
#                                        n_layers=2,
#                                        n_nodes=n_nodes)
#
#         kwargs = dict(input_size=hidden_size,
#                       hidden_size=hidden_size,
#                       ff_size=ff_size,
#                       n_heads=n_heads,
#                       activation=activation,
#                       causal=False,
#                       dropout=0)
#         kwargs_gconv = dict(input_size=hidden_size,
#                             output_size=hidden_size,
#                             support_len=1,
#                             order=2,
#                             include_self=True,
#                             channel_last=True)
#
#         self.h_enc = MLP(input_size, hidden_size, n_layers=2)
#         self.h_norm = LayerNorm(hidden_size)
#         self.diagonal_attention_mask = diagonal_attention_mask
#
#         self.valid_emb = StaticGraphEmbedding(n_nodes, hidden_size)
#         self.mask_emb = StaticGraphEmbedding(n_nodes, hidden_size)
#         self.source_embeddings = StaticGraphEmbedding(n_nodes, emb_size)
#         self.target_embeddings = StaticGraphEmbedding(n_nodes, emb_size)
#         self.x_skip = nn.ModuleList()
#         self.t_encoder, self.s_encoder, self.readout = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
#         for l in range(n_layers):
#             x_skip = nn.Linear(input_size, hidden_size)
#             t_encoder = ResidualMLP(hidden_size, hidden_size, n_layers=2)
#             s_encoder = DenseGraphConvOrderK(**kwargs_gconv) ##########
#             readout = MLP(hidden_size, hidden_size, output_size,
#                           n_layers=2)
#             self.x_skip.append(x_skip)
#             self.t_encoder.append(t_encoder)
#             self.s_encoder.append(s_encoder)
#             self.readout.append(readout)
#
#     def get_learned_adj(self):
#         logits = F.relu(self.source_embeddings() @ self.target_embeddings().T)
#         adj = torch.softmax(logits, dim=1)
#         return adj
#
#     def forward(self, x: Tensor, u: Tensor, mask: Tensor,
#                 edge_index: Tensor, edge_weight: OptTensor = None,
#                 node_index: OptTensor = None, target_nodes: OptTensor = None):
#         if target_nodes is None:
#             target_nodes = slice(None)
#
#         if self.diagonal_attention_mask:
#             mask_time = torch.eye(x.shape[1]).to(x.device)
#         else:
#             mask_time = None
#
#         adj_z = self.get_learned_adj()
#
#         # Whiten missing values
#         x = x * mask
#
#         # POSITIONAL ENCODING #################################################
#         # Obtain spatio-temporal positional encoding for every node-step pair #
#         # in both observed and target sets. Encoding are obtained by jointly  #
#         # processing node and time positional encoding.                       #
#
#         # Build (node, timestamp) encoding
#         q = self.u_enc(u, node_index=node_index)
#         # Condition value on key
#         h = self.h_enc(x) + q
#
#         # ENCODER #############################################################
#         # Obtain representations h^i_t for every (i, t) node-step pair by     #
#         # only taking into account valid data in representation set.          #
#
#         # Replace H in missing entries with queries Q
#         h = torch.where(mask.bool(), h, q)
#         # Normalize features
#         h = self.h_norm(h)
#
#         imputations = []
#
#         for l in range(self.n_layers):
#             if l == self.eta:
#                 # Condition H on two different embeddings to distinguish
#                 # valid values from masked ones
#                 valid = self.valid_emb(token_index=node_index)
#                 masked = self.mask_emb(token_index=node_index)
#                 h = torch.where(mask.bool(), h + valid, h + masked)
#             # Masked Temporal GAT for encoding representation
#             h = h + self.x_skip[l](x) * mask  # skip connection for valid x
#             # h = self.t_encoder[l](h, mask_time)
#             h = self.t_encoder[l](h)
#             h = self.s_encoder[l](h, adj_z)
#             # Read from H to get imputations
#             target_readout = self.readout[l](h[..., target_nodes, :])
#             imputations.append(target_readout)
#
#         # Get final layer imputations
#         x_hat = imputations.pop(-1)
#
#         return x_hat, imputations
#
#     @staticmethod
#     def add_model_specific_args(parser):
#         parser.opt_list('--hidden-size', type=int, tunable=True, default=32,
#                         options=[32, 64, 128, 256])
#         parser.add_argument('--u-size', type=int, default=None)
#         parser.add_argument('--output-size', type=int, default=None)
#         parser.add_argument('--temporal-self-attention', type=bool,
#                             default=True)
#         parser.add_argument('--reweight', type=str, default='softmax')
#         parser.add_argument('--n-layers', type=int, default=4)
#         parser.add_argument('--eta', type=int, default=3)
#         parser.add_argument('--message-layers', type=int, default=1)
#         return parser

#
# class GatedSPINModel(nn.Module):
#
#     def __init__(self, input_size: int,
#                  hidden_size: int,
#                  n_nodes: int,
#                  ff_size: int,
#                  u_size: Optional[int] = None,
#                  n_heads: int = 1,
#                  output_size: Optional[int] = None,
#                  temporal_self_attention: bool = True,
#                  activation: str = 'elu',
#                  reweight: Optional[str] = 'softmax',
#                  n_layers: int = 4,
#                  eta: int = 3,
#                  diagonal_attention_mask: bool = True,
#                  emb_size: int = 10,
#                  message_layers: int = 1):
#         super(GatedSPINModel, self).__init__()
#
#         u_size = u_size or input_size
#         output_size = output_size or input_size
#         self.n_nodes = n_nodes
#         self.n_layers = n_layers
#         self.eta = eta
#         self.temporal_self_attention = temporal_self_attention
#
#         self.u_enc = PositionalEncoder(in_channels=u_size,
#                                        out_channels=hidden_size,
#                                        n_layers=2,
#                                        n_nodes=n_nodes)
#
#         kwargs = dict(input_size=hidden_size,
#                       hidden_size=hidden_size,
#                       ff_size=ff_size,
#                       n_heads=n_heads,
#                       activation=activation,
#                       causal=False,
#                       dropout=0)
#         kwargs_gconv = dict(input_size=hidden_size,
#                             output_size=hidden_size)
#
#         self.h_enc = MLP(input_size, hidden_size, n_layers=2)
#         self.h_norm = LayerNorm(hidden_size)
#         self.diagonal_attention_mask = diagonal_attention_mask
#
#         self.valid_emb = StaticGraphEmbedding(n_nodes, hidden_size)
#         self.mask_emb = StaticGraphEmbedding(n_nodes, hidden_size)
#         self.source_embeddings = StaticGraphEmbedding(n_nodes, emb_size)
#         self.target_embeddings = StaticGraphEmbedding(n_nodes, emb_size)
#         self.x_skip = nn.ModuleList()
#         self.t_encoder, self.s_encoder, self.readout = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
#         for l in range(n_layers):
#             x_skip = nn.Linear(input_size, hidden_size)
#             t_encoder = TransformerLayer(**kwargs)  ##########
#             s_encoder = DenseGraphConv(**kwargs_gconv) ##########
#             readout = MLP(hidden_size, hidden_size, output_size,
#                           n_layers=2)
#             self.x_skip.append(x_skip)
#             self.t_encoder.append(t_encoder)
#             self.s_encoder.append(s_encoder)
#             self.readout.append(readout)
#
#     def get_learned_adj(self):
#         logits = F.relu(self.source_embeddings() @ self.target_embeddings().T)
#         adj = torch.softmax(logits, dim=1)
#         return adj
#
#     def forward(self, x: Tensor, u: Tensor, mask: Tensor,
#                 edge_index: Tensor, edge_weight: OptTensor = None,
#                 node_index: OptTensor = None, target_nodes: OptTensor = None):
#         if target_nodes is None:
#             target_nodes = slice(None)
#
#         if self.diagonal_attention_mask:
#             mask_time = torch.eye(x.shape[1]).to(x.device)
#         else:
#             mask_time = None
#
#         adj_z = self.get_learned_adj()
#
#         # Whiten missing values
#         x = x * mask
#
#         # POSITIONAL ENCODING #################################################
#         # Obtain spatio-temporal positional encoding for every node-step pair #
#         # in both observed and target sets. Encoding are obtained by jointly  #
#         # processing node and time positional encoding.                       #
#
#         # Build (node, timestamp) encoding
#         q = self.u_enc(u, node_index=node_index)
#         # Condition value on key
#         h = self.h_enc(x) + q
#
#         # ENCODER #############################################################
#         # Obtain representations h^i_t for every (i, t) node-step pair by     #
#         # only taking into account valid data in representation set.          #
#
#         # Replace H in missing entries with queries Q
#         h = torch.where(mask.bool(), h, q)
#         # Normalize features
#         h = self.h_norm(h)
#
#         imputations = []
#
#         for l in range(self.n_layers):
#             if l == self.eta:
#                 # Condition H on two different embeddings to distinguish
#                 # valid values from masked ones
#                 valid = self.valid_emb(token_index=node_index)
#                 masked = self.mask_emb(token_index=node_index)
#                 h = torch.where(mask.bool(), h + valid, h + masked)
#             # Masked Temporal GAT for encoding representation
#             h = h + self.x_skip[l](x) * mask  # skip connection for valid x
#             h = self.t_encoder[l](h, mask_time)
#             h = self.s_encoder[l](h, adj_z)
#             # Read from H to get imputations
#             target_readout = self.readout[l](h[..., target_nodes, :])
#             imputations.append(target_readout)
#
#         # Get final layer imputations
#         x_hat = imputations.pop(-1)
#
#         return x_hat, imputations
#
#     @staticmethod
#     def add_model_specific_args(parser):
#         parser.opt_list('--hidden-size', type=int, tunable=True, default=32,
#                         options=[32, 64, 128, 256])
#         parser.add_argument('--u-size', type=int, default=None)
#         parser.add_argument('--output-size', type=int, default=None)
#         parser.add_argument('--temporal-self-attention', type=bool,
#                             default=True)
#         parser.add_argument('--reweight', type=str, default='softmax')
#         parser.add_argument('--n-layers', type=int, default=4)
#         parser.add_argument('--eta', type=int, default=3)
#         parser.add_argument('--message-layers', type=int, default=1)
#         return parser