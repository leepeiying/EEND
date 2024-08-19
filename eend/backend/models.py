#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Copyright 2022 Brno University of Technology (author: Federico Landini)
# Licensed under the MIT license.

from os.path import isfile, join

from backend.losses import (
    pit_loss_multispk,
    vad_loss,
    kr_vad,
)
from backend.updater import (
    NoamOpt,
    setup_optimizer,
)
from pathlib import Path
from torch.nn import Module, ModuleList
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
import speechbrain
from speechbrain.lobes.models.transformer import Transformer
import torchaudio

"""
T: number of frames
C: number of speakers (classes)
D: dimension of embedding (for deep clustering loss)
B: mini-batch size
"""

def _lengths_to_padding_mask(lengths: torch.Tensor, output_len=None) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)

    if output_len is not None and padding_mask.shape[1] < output_len:
        padding_mask = F.pad(padding_mask, (0, output_len-padding_mask.shape[1]), value=True)

    return padding_mask

class EncoderDecoderAttractor(Module):
    def __init__(
        self,
        device: torch.device,
        n_units: int,
        encoder_dropout: float,
        decoder_dropout: float,
        detach_attractor_loss: bool,
    ) -> None:
        super(EncoderDecoderAttractor, self).__init__()
        self.device = device
        self.encoder = torch.nn.LSTM(
            input_size=n_units,
            hidden_size=n_units,
            num_layers=1,
            dropout=encoder_dropout,
            batch_first=True,
            device=self.device)
        self.decoder = torch.nn.LSTM(
            input_size=n_units,
            hidden_size=n_units,
            num_layers=1,
            dropout=decoder_dropout,
            batch_first=True,
            device=self.device)
        self.counter = torch.nn.Linear(n_units, 1, device=self.device)
        self.n_units = n_units
        self.detach_attractor_loss = detach_attractor_loss

    def forward(self, xs: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
        _, (hx, cx) = self.encoder.to(self.device)(xs.to(self.device))
        attractors, (_, _) = self.decoder.to(self.device)(
            zeros.to(self.device),
            (hx.to(self.device), cx.to(self.device))
        )
        return attractors

    def estimate(
        self,
        xs: torch.Tensor,
        max_n_speakers: int = 15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate attractors from embedding sequences
         without prior knowledge of the number of speakers
        Args:
          xs: List of (T,D)-shaped embeddings
          max_n_speakers (int)
        Returns:
          attractors: List of (N,D)-shaped attractors
          probs: List of attractor existence probabilities
        """
        zeros = torch.zeros((xs.shape[0], max_n_speakers, self.n_units))
        attractors = self.forward(xs, zeros)
        probs = [torch.sigmoid(
            torch.flatten(self.counter.to(self.device)(att)))
            for att in attractors]
        return attractors, probs

    def __call__(
        self,
        xs: torch.Tensor,
        n_speakers: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate attractors and loss from embedding sequences
        with given number of speakers
        Args:
          xs: List of (T,D)-shaped embeddings
          n_speakers: List of number of speakers, or None if the number
                                of speakers is unknown (ex. test phase)
        Returns:
          loss: Attractor existence loss
          attractors: List of (N,D)-shaped attractors
        """

        max_n_speakers = max(n_speakers)
        if self.device == torch.device("cpu"):
            zeros = torch.zeros(
                (xs.shape[0], max_n_speakers + 1, self.n_units))
            labels = torch.from_numpy(np.asarray([
                [1.0] * n_spk + [0.0] * (1 + max_n_speakers - n_spk)
                for n_spk in n_speakers]))
        else:
            zeros = torch.zeros(
                (xs.shape[0], max_n_speakers + 1, self.n_units),
                device=torch.device("cuda"))
            labels = torch.from_numpy(np.asarray([
                [1.0] * n_spk + [0.0] * (1 + max_n_speakers - n_spk)
                for n_spk in n_speakers])).to(torch.device("cuda"))

        attractors = self.forward(xs, zeros)
        if self.detach_attractor_loss:
            attractors = attractors.detach()
        logit = torch.cat([
            torch.reshape(self.counter(att), (-1, max_n_speakers + 1))
            for att, n_spk in zip(attractors, n_speakers)])
        loss = F.binary_cross_entropy_with_logits(logit, labels)

        # The final attractor does not correspond to a speaker so remove it
        attractors = attractors[:, :-1, :]
        return loss, attractors

class PositionalEncoding(torch.nn.Module):
    """Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_parameter('pe', torch.nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        # return self.pe[:x.size(0), :]
        return self.pe[:2*x.size(0)-1, :]

class MultiHeadSelfAttention(Module):
    """ Multi head self-attention layer
    """
    def __init__(
        self,
        device: torch.device,
        n_units: int,
        h: int,
        dropout: float
    ) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        self.device = device
        self.linearQ = torch.nn.Linear(n_units, n_units, device=self.device)
        self.linearK = torch.nn.Linear(n_units, n_units, device=self.device)
        self.linearV = torch.nn.Linear(n_units, n_units, device=self.device)
        self.linearO = torch.nn.Linear(n_units, n_units, device=self.device)
        self.d_k = n_units // h
        self.h = h
        self.dropout = dropout
        self.att = None  # attention for plot

    def __call__(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        # x: (BT, F)
        q = self.linearQ(x).reshape(batch_size, -1, self.h, self.d_k)
        k = self.linearK(x).reshape(batch_size, -1, self.h, self.d_k)
        v = self.linearV(x).reshape(batch_size, -1, self.h, self.d_k)
        scores = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1)) \
            / np.sqrt(self.d_k)
        # scores: (B, h, T, T)
        self.att = F.softmax(scores, dim=3)
        p_att = F.dropout(self.att, self.dropout)
        x = torch.matmul(p_att, v.permute(0, 2, 1, 3))
        x = x.permute(0, 2, 1, 3).reshape(-1, self.h * self.d_k)
        return self.linearO(x)


class PositionwiseFeedForward(Module):
    """ Positionwise feed-forward layer
    """
    def __init__(
        self,
        device: torch.device,
        n_units: int,
        d_units: int,
        dropout: float
    ) -> None:
        super(PositionwiseFeedForward, self).__init__()
        self.device = device
        self.linear1 = torch.nn.Linear(n_units, d_units, device=self.device)
        self.linear2 = torch.nn.Linear(d_units, n_units, device=self.device)
        self.dropout = dropout

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.dropout(F.relu(self.linear1(x)), self.dropout))

class Branchformer(torch.nn.Module):
    def __init__(
        self,
        device: torch.device,
        input_dim: int,
        num_heads: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.device = device

        self.branchformer_layers = torch.nn.ModuleList(
            [
                speechbrain.lobes.models.transformer.Branchformer.BranchformerEncoderLayer(
                    d_model=input_dim,
                    nhead=num_heads, 
                    kernel_size=depthwise_conv_kernel_size, 
                    kdim=256, 
                    vdim=256, 
                    dropout=dropout, 
                    attention_type='RelPosMHAXL', 
                    csgu_linear_units=3072, 
                    use_linear_after_conv=False,
                ).to(self.device)
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNorm(d_model, eps=1e-6)
        self.attention_type = attention_type

    def forward(self, input: torch.Tensor, pos: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input (torch.Tensor): with shape `(T, B, input_dim)`.
            src_key_padding_mask (torch.Tensor): with shape `(B, T)`.

        Returns:
            torch.Tensor
                output frames, with shape `(T, B, input_dim)`
        """
        print("pos type:", type(pos))
        x = input.to(self.device)
        
        src_key_padding_mask = src_key_padding_mask.to(self.device)
        pos = pos.to(self.device)

        print("in branchformer")
        for layer in self.branchformer_layers:
            # print("x:", x.shape)
            # print("layer:", layer)
            # x, _ = layer(x, pos).to(self.device)
            output, attention = layer(
                x,
                src_key_padding_mask=src_key_padding_mask,
                pos_embs=pos,
            )
        return output


class Conformer(torch.nn.Module):
    r"""Implements the Conformer architecture introduced in
    *Conformer: Convolution-augmented Transformer for Speech Recognition*
    [:footcite:`gulati2020conformer`].

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Conformer layer.
        ffn_dim (int): hidden layer dimension of feedforward networks.
        num_layers (int): number of Conformer layers to instantiate.
        depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)

    Examples:
        >>> conformer = Conformer(
        >>>     input_dim=80,
        >>>     num_heads=4,
        >>>     ffn_dim=128,
        >>>     num_layers=4,
        >>>     depthwise_conv_kernel_size=31,
        >>> )
        >>> lengths = torch.randint(1, 400, (10,))  # (batch,)
        >>> input = torch.rand(10, int(lengths.max()), input_dim)  # (batch, num_frames, input_dim)
        >>> output = conformer(input, lengths)
    """

    def __init__(
        self,
        device: torch.device,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
    ):
        super().__init__()
        self.device = device

        self.conformer_layers = torch.nn.ModuleList(
            [
                torchaudio.models.conformer.ConformerLayer(
                    input_dim,
                    ffn_dim,
                    num_heads,
                    depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                ).to(self.device)
                for _ in range(num_layers)
            ]
        )

    def forward(self, input: torch.Tensor, src_key_padding_mask: torch.Tensor, pos=None) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(T, B, input_dim)`.
            src_key_padding_mask (torch.Tensor): with shape `(B, T)`.

        Returns:
            torch.Tensor
                output frames, with shape `(T, B, input_dim)`
        """
        x = input.to(self.device)
        src_key_padding_mask = src_key_padding_mask.to(self.device)

        for layer in self.conformer_layers:
            x = layer(x, src_key_padding_mask).to(self.device)
        return x

class TransformerEncoder(Module):
    def __init__(
        self,
        device: torch.device,
        idim: int,
        n_layers: int,
        n_units: int,
        e_units: int,
        h: int,
        dropout: float
    ) -> None:
        super(TransformerEncoder, self).__init__()
        self.device = device
        self.linear_in = torch.nn.Linear(idim, n_units, device=self.device)
        self.lnorm_in = torch.nn.LayerNorm(n_units, device=self.device)
        self.n_layers = n_layers
        self.dropout = dropout
        for i in range(n_layers):
            setattr(
                self,
                '{}{:d}'.format("lnorm1_", i),
                torch.nn.LayerNorm(n_units, device=self.device)
            )
            setattr(
                self,
                '{}{:d}'.format("self_att_", i),
                MultiHeadSelfAttention(self.device, n_units, h, dropout)
            )
            setattr(
                self,
                '{}{:d}'.format("lnorm2_", i),
                torch.nn.LayerNorm(n_units, device=self.device)
            )
            setattr(
                self,
                '{}{:d}'.format("ff_", i),
                PositionwiseFeedForward(self.device, n_units, e_units, dropout)
            )
        self.lnorm_out = torch.nn.LayerNorm(n_units, device=self.device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) ... batch, time, (mel)freq
        BT_size = x.shape[0] * x.shape[1]
        # e: (BT, F)
        e = self.linear_in(x.reshape(BT_size, -1))
        # Encoder stack
        for i in range(self.n_layers):
            # layer normalization
            e = getattr(self, '{}{:d}'.format("lnorm1_", i))(e)
            # self-attention
            s = getattr(self, '{}{:d}'.format("self_att_", i))(e, x.shape[0])
            # residual
            e = e + F.dropout(s, self.dropout)
            # layer normalization
            e = getattr(self, '{}{:d}'.format("lnorm2_", i))(e)
            # positionwise feed-forward
            s = getattr(self, '{}{:d}'.format("ff_", i))(e)
            # residual
            e = e + F.dropout(s, self.dropout)
        # final layer normalization
        # output: (BT, F)
        return self.lnorm_out(e)

class TransformerEncoder_new(Module):

    def __init__(self,
                 encoder_layer,
                 num_layers,
                 norm=None,
                 ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = norm

    def forward(self, src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:

        output = src

        for layer_idx, layer in enumerate(self.layers):
            output = layer(output, src_mask, src_key_padding_mask, pos)

        if self.norm is not None:
            output = self.norm(output)

        # output = self.norm(output)

        return output


class TransformerEncoderLayer(Module):

    def __init__(self,
                 d_model=256,
                 nhead=4,
                 dim_feedforward=2048,
                 dropout=0.1,
                 norm_first=False,
                 activation='relu',
                 layer_norm_eps=1e-5,
                 ):
        super().__init__()


        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)


    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):

        return tensor if pos is None else tensor + pos

    def forward_post(self, src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:

        # print("src:", src.shape)
        # print("pos:", pos.shape)
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

    def forward_pre(self, src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:

        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src

    def forward(self, src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                pos: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        if self.norm_first:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        else:
            return self.forward_post(src, src_mask, src_key_padding_mask, pos)


def _get_clones(module, N, repeat_from_second=False):
    if repeat_from_second:
        module_list = torch.nn.ModuleList([copy.deepcopy(module)])
        for _ in range(1, N):
            module_list.append(module)
        return module_list
    else:
        return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class TransformerEDADiarization(Module):

    def __init__(
        self,
        device: torch.device,
        use_former: str,
        in_size: int,
        n_units: int,
        e_units: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        vad_loss_weight: float,
        attractor_loss_ratio: float,
        attractor_encoder_dropout: float,
        attractor_decoder_dropout: float,
        detach_attractor_loss: bool,
        add_encoder_mask: bool = True,
    ) -> None:
        """ Self-attention-based diarization model.
        Args:
          in_size (int): Dimension of input feature vector
          n_units (int): Number of units in a self-attention block
          n_heads (int): Number of attention heads
          n_layers (int): Number of transformer-encoder layers
          dropout (float): dropout ratio
          vad_loss_weight (float) : weight for vad_loss
          attractor_loss_ratio (float)
          attractor_encoder_dropout (float)
          attractor_decoder_dropout (float)
        """
        self.device = device
        super(TransformerEDADiarization, self).__init__()
        self.use_former = use_former
        self.add_encoder_mask = add_encoder_mask

        self.trans = torch.nn.Linear(345, 256, device=self.device)
        self.trans_norm = torch.nn.LayerNorm(256, device=self.device)
        self.pos_encoder = PositionalEncoding(256)
        if self.use_former == 'Transformer':
            self.enc = TransformerEncoder(
                self.device, in_size, n_layers, n_units, e_units, n_heads, dropout
            )
            # encoder_layers = TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=2048, dropout=0.1, norm_first=False)
            # self.enc = TransformerEncoder(encoder_layers, num_layers=4)
            # self.enc = speechbrain.lobes.models.transformer.Transformer.TransformerEncoder(
            #             num_layers=4, 
            #             nhead=4, 
            #             d_ffn=2048, 
            #             input_shape=in_size, 
            #             d_model=256,
            #             dropout=dropout, 
            #             )
        elif self.use_former == 'Conformer':
            self.con_enc = Conformer(device=self.device,
                                input_dim=256,
                                num_heads=4,
                                ffn_dim=2048,
                                num_layers=4,
                                depthwise_conv_kernel_size=31,
                                dropout=0.1,
                                use_group_norm=True,
                                )
        elif self.use_former == 'Branchformer':
            self.branch_enc = speechbrain.lobes.models.transformer.Branchformer.BranchformerEncoder(
                                num_layers=n_layers,
                                d_model=256,
                                nhead=n_heads,
                                kernel_size=31,
                                dropout=0.1, # 0.1
                                attention_type='regularMHA', # base model no use pos_emb
                                )
        self.eda = EncoderDecoderAttractor(
            self.device,
            n_units,
            attractor_encoder_dropout,
            attractor_decoder_dropout,
            detach_attractor_loss,
        )
        self.attractor_loss_ratio = attractor_loss_ratio
        self.vad_loss_weight = vad_loss_weight

    def get_transformer_emb(self, src):

        src = self.trans_norm(self.trans(xs))
        encoder_padding_mask, encoder_attn_mask = self.get_encoder_mask(ts, xs, self.add_encoder_mask)
        
        # (B, T, D) --> (T, B, D)
        # src = src.permute(1, 0, 2)
        # pos_embed = self.pos_encoder(src)
        
        print("encoder_padding_mask:", encoder_padding_mask.shape)
        encoder_padding_mask = encoder_padding_mask.permute(1, 0)
        
        # pos_embed = pos_embed.permute(1, 0, 2)# [1, 999, 256]
        src = src.permute(1, 0, 2)# [32, 500, 256]
        # no use positional encoding
        emb, attention = self.enc(src, src_key_padding_mask=encoder_padding_mask)
        # emb: (T, B, E) --> (B, T, E)
        memory = emb.permute(1, 0, 2)

        return memory 

    def get_branchformer_emb(self, src):

        src = self.trans_norm(self.trans(src))
        encoder_padding_mask, encoder_attn_mask = self.get_encoder_mask(None, src, self.add_encoder_mask)

        # (B, T, D) --> (T, B, D)
        src = src.permute(1, 0, 2)

        pos_embed = self.pos_encoder(src)

        pos_embed = pos_embed.permute(1, 0, 2)
        src = src.permute(1, 0, 2)

        # memory, _ = self.branch_enc(src, pos_embs=pos_embed, src_key_padding_mask=None)
        # base model
        memory, _ = self.branch_enc(src, src_key_padding_mask=None)

        return memory
    
    def get_conformer_emb(self, src):

        src = self.trans_norm(self.trans(src))
        encoder_padding_mask, encoder_attn_mask = self.get_encoder_mask(None, src, self.add_encoder_mask)

        # (B, T, D) --> (T, B, D)
        src = src.permute(1, 0, 2)

        pos_embed = self.pos_encoder(src)

        emb = self.con_enc(src, pos=pos_embed, src_key_padding_mask=encoder_padding_mask)
        # emb: (T, B, E) --> (B, T, E)
        memory = emb.permute(1, 0, 2)

        return memory 

    def get_encoder_mask(self, labels, src, add_mask=False):

        if labels is not None:
            # utt_lens: (B,)
            utt_lens = (torch.sum(labels != -1, dim=-1) > 0).sum(dim=-1)
        else:
            utt_lens = (torch.ones((src.shape[0],), device=src.device) * src.shape[1]).long()

        # encoder_padding_mask: (B, T)
        encoder_padding_mask = _lengths_to_padding_mask(utt_lens, src.shape[1])
        if not add_mask:
            encoder_padding_mask.fill_(False)

        
        encoder_attn_mask = None

        return encoder_padding_mask, encoder_attn_mask
    
    def get_embeddings(self, xs: torch.Tensor) -> torch.Tensor:
        ilens = [x.shape[0] for x in xs]
        # xs: (B, T, F)
        pad_shape = xs.shape
        # emb: (B*T, E)
        emb = self.enc(xs)
        # emb: [(T, E), ...]
        emb = emb.reshape(pad_shape[0], pad_shape[1], -1)
        return emb

    def estimate_sequential(
        self,
        xs: torch.Tensor,
        args: SimpleNamespace
    ) -> List[torch.Tensor]:
        assert args.estimate_spk_qty_thr != -1 or \
            args.estimate_spk_qty != -1, \
            "Either 'estimate_spk_qty_thr' or 'estimate_spk_qty' \
            arguments have to be defined."
        # use transformer
        if args.use_former == 'Transformer':
            emb = self.get_embeddings(xs)
            # emb = self.get_transformer_emb(xs)
        elif args.use_former == 'Branchformer':
            # use branchformer
            emb = self.get_branchformer_emb(xs)
        elif args.use_former == 'Conformer':
            # use conformer
            emb = self.get_conformer_emb(xs)

        ys_active = []
        if args.time_shuffle:
            orders = [np.arange(e.shape[0]) for e in emb]
            for order in orders:
                np.random.shuffle(order)
            attractors, probs = self.eda.estimate(
                torch.stack([e[order] for e, order in zip(emb, orders)]))
        else:
            attractors, probs = self.eda.estimate(emb)
        ys = torch.matmul(emb, attractors.permute(0, 2, 1))
        ys = [torch.sigmoid(y) for y in ys]
        for p, y in zip(probs, ys):
            if args.estimate_spk_qty != -1:
                sorted_p, order = torch.sort(p, descending=True)
                ys_active.append(y[:, order[:args.estimate_spk_qty]])
            elif args.estimate_spk_qty_thr != -1:
                silence = np.where(
                    p.data.to("cpu") < args.estimate_spk_qty_thr)[0]
                n_spk = silence[0] if silence.size else None
                ys_active.append(y[:, :n_spk])
            else:
                NotImplementedError(
                    'estimate_spk_qty or estimate_spk_qty_thr needed.')
        return ys_active

    def forward(
        self,
        xs: torch.Tensor,
        ts: torch.Tensor,
        n_speakers: List[int],
        args: SimpleNamespace
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # emb: (T, B, E)
        # use transformer
        if args.use_former == 'Transformer':
            print("in Transformer")
            emb = self.get_embeddings(xs)
            # src = self.trans_norm(self.trans(xs))
            # encoder_padding_mask, encoder_attn_mask = self.get_encoder_mask(ts, xs, self.add_encoder_mask)
            
            # # (B, T, D) --> (T, B, D)
            # # src = src.permute(1, 0, 2)
            # # pos_embed = self.pos_encoder(src)
            
            # print("encoder_padding_mask:", encoder_padding_mask.shape)
            # encoder_padding_mask = encoder_padding_mask.permute(1, 0)
            # # if use branchformer should using permute for pos_embed and src.
            # # pos_embed = pos_embed.permute(1, 0, 2)# [1, 999, 256]
            # src = src.permute(1, 0, 2)# [32, 500, 256]
            # # no use positional encoding
            # emb, attention = self.enc(src, src_key_padding_mask=encoder_padding_mask)
            # # emb: (T, B, E) --> (B, T, E)
            # emb = emb.permute(1, 0, 2)
            # # print("emb:", emb.shape)
        elif args.use_former == 'Conformer':
            print("in Conformer")
            # when use conformer or branchformer please use trans_norm, get_encoder_mask, permute src and pos_encoder
            # print(xs.shape) # 32,500,345
            src = self.trans_norm(self.trans(xs))

            # (B, T, D) --> (T, B, D)
            src = src.permute(1, 0, 2)
            encoder_padding_mask, encoder_attn_mask = self.get_encoder_mask(ts, xs, self.add_encoder_mask)

            pos_embed = self.pos_encoder(src)

            # print("pos type:", pos_embed.shape)
            # print("src:", src.shape)
            # print("src_key_padding_mask type:", encoder_padding_mask.shape)
            
            # use conformer
            emb = self.con_enc(src, pos=pos_embed, src_key_padding_mask=encoder_padding_mask)
            # emb: (T, B, E) --> (B, T, E)
            emb = emb.permute(1, 0, 2)
        elif args.use_former == 'Branchformer':
            print("in Branchformer")
            src = self.trans_norm(self.trans(xs))
            encoder_padding_mask, encoder_attn_mask = self.get_encoder_mask(ts, xs, self.add_encoder_mask)
            # (B, T, D) --> (T, B, D)
            src = src.permute(1, 0, 2)
            pos_embed = self.pos_encoder(src)
            # if use branchformer should using permute for pos_embed and src.
            pos_embed = pos_embed.permute(1, 0, 2)# [1, 999, 256]
            src = src.permute(1, 0, 2)# [32, 500, 256]
        
            # emb, attention = self.branch_enc(src, pos_embs=pos_embed, src_key_padding_mask=encoder_padding_mask)
            # emb, attention = self.branch_enc(src, src_key_padding_mask=encoder_padding_mask)
            # base
            emb, attention = self.branch_enc(src)
            print("emb:", emb.shape)
            print("attention:", len(attention))

            # 保存注意力映射到类属性中
            self.attention = attention[-1]  # 假设我们只使用最后一层的注意力映射
        
        if args.time_shuffle:
            orders = [np.arange(e.shape[0]) for e in emb]
            for order in orders:
                np.random.shuffle(order)
            attractor_loss, attractors = self.eda(
                torch.stack([e[order] for e, order in zip(emb, orders)]),
                n_speakers)
        else:
            attractor_loss, attractors = self.eda(emb, n_speakers)

        # ys: [(T, C), ...]
        ys = torch.matmul(emb, attractors.permute(0, 2, 1)) # attractors:(B, S, E)
        return ys, attractor_loss

    def get_loss(
        self,
        ys: torch.Tensor,
        target: torch.Tensor,
        n_speakers: List[int],
        attractor_loss: torch.Tensor,
        vad_loss_weight: float,
        detach_attractor_loss: bool,
        kr_vad_loss_weight: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # if branchformer use SAD loss do this
        # attn = self.attention
        # attn = F.softmax(attn, dim=-1) 
        # print("attn:", attn)
        # exit()
        max_n_speakers = max(n_speakers)
        ts_padded = pad_labels(target, max_n_speakers)
        ts_padded = torch.stack(ts_padded)
        ys_padded = pad_labels(ys, max_n_speakers)
        ys_padded = torch.stack(ys_padded)

        loss = pit_loss_multispk(
            ys_padded, ts_padded, n_speakers, detach_attractor_loss)
        vad_loss_value = vad_loss(ys, target)

        if kr_vad_loss_weight==0.0:
            kr_vad_loss=torch.zeros(1).cuda()
            selected_heads=torch.zeros(2).cuda()
        else:
            kr_vad_loss, selected_heads = kr_vad(target.detach(), attn)
        # if branchformer use SAD loss do this, too
        # return loss + vad_loss_value * vad_loss_weight + \
        #     attractor_loss * self.attractor_loss_ratio + \
        #     kr_vad_loss * kr_vad_loss_weight, loss
        return loss + vad_loss_value * vad_loss_weight + \
            attractor_loss * self.attractor_loss_ratio, loss


def pad_labels(ts: torch.Tensor, out_size: int) -> torch.Tensor:
    # pad label's speaker-dim to be model's n_speakers
    ts_padded = []
    for _, t in enumerate(ts):
        if t.shape[1] < out_size:
            # padding
            ts_padded.append(torch.cat((t, -1 * torch.ones((
                t.shape[0], out_size - t.shape[1]))), dim=1))
        elif t.shape[1] > out_size:
            # truncate
            ts_padded.append(t[:, :out_size].float())
        else:
            ts_padded.append(t.float())
    return ts_padded


def pad_sequence(
    features: List[torch.Tensor],
    labels: List[torch.Tensor],
    seq_len: int,
    device: torch.device
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    features_padded = []
    labels_padded = []
    assert len(features) == len(labels), (
        f"Features and labels in batch were expected to match but got "
        f"{len(features)} features and {len(labels)} labels.")
    for i, _ in enumerate(features):
        assert features[i].shape[0] == labels[i].shape[0], (
            f"Length of features and labels were expected to match but got "
            f"{features[i].shape[0]} and {labels[i].shape[0]}")
        length = features[i].shape[0]
        if length < seq_len:
            extend = seq_len - length
            features_padded.append(torch.cat((features[i].to(device), -torch.ones((
                extend, features[i].shape[1]), device=device)), dim=0))
            labels_padded.append(torch.cat((labels[i].to(device), -torch.ones((
                extend, labels[i].shape[1]), device=device)), dim=0))
        elif length > seq_len:
            raise ValueError(f"Sequence of length {length} was received but only "
                             f"{seq_len} was expected.")
        else:
            features_padded.append(features[i].to(device))
            labels_padded.append(labels[i].to(device))
    return features_padded, labels_padded


# def save_checkpoint(
#     args,
#     epoch: int,
#     model: Module,
#     optimizer: NoamOpt,
#     loss: torch.Tensor
# ) -> None:
#     Path(f"{args.output_path}/models").mkdir(parents=True, exist_ok=True)

#     torch.save({
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss},
#         f"{args.output_path}/models/checkpoint_{epoch}.tar"
#     )


# def load_checkpoint(args: SimpleNamespace, filename: str):
#     model = get_model(args)
#     model = model.to(args.device)
#     optimizer = setup_optimizer(args, model)

#     assert isfile(filename), \
#         f"File {filename} does not exist."
#     checkpoint = torch.load(filename)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epoch = checkpoint['epoch']
#     loss = checkpoint['loss']
#     return epoch, model, optimizer, loss

# 每個epoch都存一個checkpoint
def save_checkpoint(
    args,
    epoch: int,
    model: Module,
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor
) -> None:
    Path(f"{args.output_path}/models").mkdir(parents=True, exist_ok=True)
    
    model_state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss},
        f"{args.output_path}/models/checkpoint_{epoch}.tar"
    )

def load_checkpoint(args: SimpleNamespace, filename: str):
    model = get_model(args)
    model = model.to(args.device)
    optimizer = setup_optimizer(args, model)
    
    # assert isfile(filename), f"File {filename} does not exist."
    checkpoint = torch.load(filename)
    
    # 加載模型狀態字典
    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)
    
    # 加載優化器狀態字典
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return epoch, model, optimizer, loss


def load_initmodel(args: SimpleNamespace):
    return load_checkpoint(args, args.initmodel)


def get_model(args: SimpleNamespace) -> Module:
    if args.model_type == 'TransformerEDA':
        model = TransformerEDADiarization(
            device=args.device,
            use_former=args.use_former,
            in_size=args.feature_dim * (1 + 2 * args.context_size),
            n_units=args.hidden_size,
            e_units=args.encoder_units,
            n_heads=args.transformer_encoder_n_heads,
            n_layers=args.transformer_encoder_n_layers,
            dropout=args.transformer_encoder_dropout,
            attractor_loss_ratio=args.attractor_loss_ratio,
            attractor_encoder_dropout=args.attractor_encoder_dropout,
            attractor_decoder_dropout=args.attractor_decoder_dropout,
            detach_attractor_loss=args.detach_attractor_loss,
            vad_loss_weight=args.vad_loss_weight,
            add_encoder_mask=args.add_encoder_mask,
        )
    else:
        raise ValueError('Possible model_type is "TransformerEDA"')
    return model


# def average_checkpoints(
#     device: torch.device,
#     model: Module,
#     models_path: str,
#     epochs: str
# ) -> Module:
#     epochs = parse_epochs(epochs)
#     states_dict_list = []
#     for e in epochs:
#         copy_model = copy.deepcopy(model)
#         checkpoint = torch.load(join(
#             models_path,
#             f"checkpoint_{e}.tar"), map_location=device)
#         copy_model.load_state_dict(checkpoint['model_state_dict'])
#         states_dict_list.append(copy_model.state_dict())
#     avg_state_dict = average_states(states_dict_list, device)
#     avg_model = copy.deepcopy(model)
#     avg_model.load_state_dict(avg_state_dict)
#     return avg_model


# def average_states(
#     states_list: List[Dict[str, torch.Tensor]],
#     device: torch.device,
# ) -> List[Dict[str, torch.Tensor]]:
#     qty = len(states_list)
#     avg_state = states_list[0]
#     for i in range(1, qty):
#         for key in avg_state:
#             avg_state[key] += states_list[i][key].to(device)

#     for key in avg_state:
#         avg_state[key] = avg_state[key] / qty
#     return avg_state

def average_checkpoints(
    device: torch.device,
    model: Module,
    models_path: str,
    epochs: str
) -> Module:
    epochs = parse_epochs(epochs)
    states_dict_list = []
    for e in epochs:
        copy_model = copy.deepcopy(model)
        checkpoint = torch.load(join(
            models_path,
            f"checkpoint_{e}.tar"), map_location=device)
        
        # 加载模型的状态字典
        state_dict = checkpoint['model_state_dict']
        
        # 如果模型是用 DistributedDataParallel 封装的，需要提取内部模型的状态字典
        if 'module.' in list(state_dict.keys())[0]:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        copy_model.load_state_dict(state_dict)
        states_dict_list.append(copy_model.state_dict())
    
    avg_state_dict = average_states(states_dict_list, device)
    avg_model = copy.deepcopy(model)
    avg_model.load_state_dict(avg_state_dict)
    return avg_model

def average_states(
    states_list: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    qty = len(states_list)
    avg_state = {k: v.to(device) for k, v in states_list[0].items()}  # 確保第一個狀態字典在指定設備上
    for i in range(1, qty):
        for key in avg_state:
            avg_state[key] += states_list[i][key].to(device)  # 確保其他狀態字典在指定設備上

    for key in avg_state:
        avg_state[key] = avg_state[key] / qty
    return avg_state


def parse_epochs(string: str) -> List[int]:
    parts = string.split(',')
    res = []
    for p in parts:
        if '-' in p:
            interval = p.split('-')
            res.extend(range(int(interval[0])+1, int(interval[1])+1))
        else:
            res.append(int(p))
    return res
