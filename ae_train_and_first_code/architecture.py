import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import parse as V
from rotary_embedding_torch import RotaryEmbedding
import logging
import pdb
from transformers import AutoFeatureExtractor, WavLMModel
from perceiver import Perceiver
class WavLM(nn.Module):
    def __init__(self, 
        # input_dim,
        num_spk: int = 2,
        n_layers: int = 1,
        # general setupns
        emb_dim: int = 128,
        norm_type: str = "rmsgroupnorm",
        num_groups: int = 4,  # used only in RMSGroupNorm
        tf_order: str = "ft",
        # self-attention related
        n_heads: int = 4,
        flash_attention: bool = False,  # available when using mixed precision
        attention_dim: int = 128,
        # ffn related
        ffn_type: Union[str, list] = ["swiglu_conv1d", "swiglu_conv1d"],
        ffn_hidden_dim: Union[int, list] = [128, 128],  # macaron front, end dim.
        conv1d_kernel: int = 3,
        conv1d_shift: int = 1,
        dropout: float = 0.5):
        super().__init__()
        self.num_layers=12

        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        self.wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        for param in self.wavlm_model.parameters():
            param.requires_grad = False
        
        self.weights_1 = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)
        # self.weights_2 = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)
        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.weights_1.unsqueeze(0))  # 2D로 변환 후 초기화
        # nn.init.xavier_uniform_(self.weights_2.unsqueeze(0))  # 2D로 변환 후 초기화
        
        # t_ksize = 3
        # ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        # self.conv = nn.Conv1d(768, emb_dim, kernel_size=3, stride=1, padding=1)
        self.fc_first_1 = nn.Linear(768, emb_dim)
        self.ln_1 = nn.LayerNorm(emb_dim)
        # self.fc_first_2 = nn.Linear(768, emb_dim)
        # self.ln_2 = nn.LayerNorm(emb_dim)
        
        assert attention_dim % n_heads == 0, (attention_dim, n_heads)
        rope = RotaryEmbedding(attention_dim // n_heads)
        self.blocks = nn.ModuleList([])
        for _ in range(n_layers):
            self.blocks.append(
                LocoformerBlock(
                    rope,
                    # general setup
                    emb_dim=emb_dim,
                    norm_type=norm_type,
                    num_groups=num_groups,
                    # self-attention related
                    n_heads=n_heads,
                    flash_attention=flash_attention,
                    attention_dim=attention_dim,
                    # ffn related
                    ffn_type=ffn_type,
                    ffn_hidden_dim=ffn_hidden_dim,
                    conv1d_kernel=conv1d_kernel,
                    conv1d_shift=conv1d_shift,
                    dropout=dropout,
                    eps=1.0e-5,
                )
            )
        # time_dim = 201  # 64600 sample, 320 hop
        # self.cnn_f = nn.ModuleList([])
        # for _ in range(3):
        #     self.cnn_f.append(ConformerConvModule(dim=emb_dim))
        # self.cnn_t = nn.ModuleList([])
        # for _ in range(3):
        #     self.cnn_t.append(ConformerConvModule(dim=time_dim))
        # # self.cnn_t = ConformerConvModule(dim=time_dim)
        # self.cnn_2d = DeepfakeCNN_Strided()
        
        # self.cnn = nn.Sequential(
        #     nn.Conv1d(emb_dim, emb_dim, kernel_size=3, padding=1),
        #     nn.SiLU(inplace=True),
        #     nn.Conv1d(emb_dim, emb_dim, kernel_size=3, padding=1),
        # )
        # self.cnn_ln = nn.LayerNorm(emb_dim)
        # self.rnn = nn.LSTM(emb_dim, emb_dim//2, num_layers=2, batch_first=True, bidirectional=True)
        # self.rnn_ln = nn.LayerNorm(emb_dim)
        # self.attn = MultiHeadSelfAttention(emb_dim*2, attention_dim=emb_dim, rope=rope, n_heads=4, dropout=0, flash_attention=flash_attention)
        # self.attn_ln = nn.LayerNorm(emb_dim)
        
        self.n_layers = n_layers
        self.last_attention = nn.Sequential(
            nn.Conv1d(emb_dim, emb_dim//4, kernel_size=1),
            nn.SiLU(),
            nn.BatchNorm1d(emb_dim//4),
            nn.Conv1d(emb_dim//4, emb_dim, kernel_size=1),
            nn.Softmax(dim=2)
        )
        self.last_fc = nn.Linear(emb_dim*2, 2)

    def forward(self, waveform):
        # Convert to list of [T] tensors, B=2일때, stereo channel 이라고 인식함.
        waveform_list = [w.detach().cpu().numpy() for w in waveform]

        inputs = self.feature_extractor(waveform_list, sampling_rate=16000, return_tensors="pt", padding=True).to(waveform.device)
        with torch.no_grad():
            outputs = self.wavlm_model(**inputs, output_hidden_states=True)
            all_layers = outputs.hidden_states[-self.num_layers:]
        all_layers = torch.stack(all_layers)  # [12, B, T, 768]
        
        norm_weights_1 = torch.softmax(self.weights_1, dim=0)
        weighted_sum_1 = torch.einsum('l,lbsh->bsh', norm_weights_1, all_layers)    # [B, T, 768]
        # norm_weights_2 = torch.softmax(self.weights_2, dim=0)
        # weighted_sum_2 = torch.einsum('l,lbsh->bsh', norm_weights_2, all_layers)    # [B, T, 768]
        
        # TF input: [B, 2, T, F]
        # module_input = self.conv(weighted_sum.transpose(1,2))
        # module_input = self.ln(module_input.transpose(1,2)) # [B, T, 128]
        module_input_1 = self.ln_1(self.fc_first_1(weighted_sum_1))
        # module_input_2 = self.ln_2(self.fc_first_2(weighted_sum_2))
        
        # Decoder block
        feature = module_input_1
        for ii in range(self.n_layers):
            feature = self.blocks[ii](feature)  # [B, T, d]
        
        # # Time conformer
        # feature_t = module_input_1.transpose(1,2)
        # for ii in range(3):
        #     feature_t = self.cnn_t[ii](feature_t)  # [B, T, d]
        # # Freq contormer
        # feature_f = module_input_1
        # for ii in range(3):
        #     feature_f = self.cnn_f[ii](feature_f)  # [B, T, d]
        # # feature_f = self.cnn_f(module_input_1) # [B, d, T]
        # feature_f = feature_f.transpose(1,2)
        # # Mix 2d conv.
        # feature_mix = torch.cat((feature_t.unsqueeze(1), feature_f.unsqueeze(1)), dim=1)
        # logit = self.cnn_2d(feature_mix)
        # return logit
        
        output = feature.transpose(1,2)
        attn = self.last_attention(output)
        mean = torch.sum(attn * output, dim=2)     # [B, C]
        std = torch.sqrt(torch.sum(attn * (output - mean.unsqueeze(2))**2, dim=2) + 1e-9)
        output=torch.cat([mean, std], dim=1)  # [B, 2C]
        output = self.last_fc(output)
        return output

class DeepfakeCNN_Strided(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),  # [B, 16, T/2, F/2]
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [B, 32, T/4, F/4]
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [B, 64, T/8, F/8]
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # 마지막 평균 풀링
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # [B, 64, 1, 1]
        self.fc = nn.Linear(64, 2)  # CrossEntropyLoss에 맞게 [B, 2]

    def forward(self, x):
        x = self.conv_layers(x)    # [B, 64, T', F']
        x = self.pool(x)           # [B, 64, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 64]
        x = self.fc(x)             # [B, 2]
        return x

class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, bottleneck_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck_dim),
            nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1),
            nn.Softmax(dim=2)
        )

    def forward(self, x):  # x: [B, C, T]
        attn = self.attention(x)              # [B, C, T]
        mean = torch.sum(attn * x, dim=2)     # [B, C]
        std = torch.sqrt(torch.sum(attn * (x - mean.unsqueeze(2))**2, dim=2) + 1e-9)
        return torch.cat([mean, std], dim=1)  # [B, 2C]

class LocoformerBlock(nn.Module):
    def __init__(
        self,
        rope,
        # general setup
        emb_dim=128,
        norm_type="layernorm", # "rmsgroupnorm",
        num_groups=4,
        # self-attention related
        n_heads=4,
        flash_attention=False,
        attention_dim=128,
        # ffn related
        ffn_type="swiglu_conv1d",
        ffn_hidden_dim=384,
        conv1d_kernel=4,
        conv1d_shift=1,
        dropout=0.0,
        eps=1.0e-5,
    ):
        super().__init__()

        FFN = {
            "conv1d": ConvDeconv1d,
            "swiglu_conv1d": SwiGLUConvDeconv1d,
        }
        Norm = {
            "layernorm": nn.LayerNorm,
            "rmsgroupnorm": RMSGroupNorm,
        }
        '''
        self.fc_fnn = nn.Sequential(nn.Linear(emb_dim, emb_dim*2),
                                    Swish(),
                                    nn.Dropout(0.5),
                                    nn.Linear(emb_dim*2, emb_dim),
                                    nn.Dropout(0.5)
                                    )
        '''
        # initialize FFN
        self.ffn_norm = nn.ModuleList([])
        self.ffn = nn.ModuleList([])
        for f_type, f_dim in zip(ffn_type[::-1], ffn_hidden_dim[::-1]):
            # self.ffn_norm.append(Norm[norm_type](emb_dim, eps=eps))
            self.ffn_norm.append(nn.LayerNorm(emb_dim))
            #self.ffn.append(self.fc_fnn)
            self.ffn.append(
                FFN[f_type](
                    emb_dim,
                    f_dim,
                    conv1d_kernel,
                    conv1d_shift,
                    dropout=dropout,
                )
            )
            
        # initialize self-attention
        # self.attn_norm = Norm[norm_type](emb_dim, eps=eps)
        # self.num_latents = 256
        # self.latents = nn.Parameter(torch.randn(self.num_latents, emb_dim))
        # self.norm_query = nn.LayerNorm(emb_dim) 
        self.attn_norm = nn.LayerNorm(emb_dim)
        self.attn = MultiHeadSelfAttention(
            emb_dim,
            attention_dim=attention_dim,
            n_heads=n_heads,
            rope=rope,
            dropout=dropout,
            flash_attention=flash_attention,
        )
        self.conv = ConformerConvModule(dim = emb_dim, expansion_factor = 2, kernel_size = conv1d_kernel, dropout = dropout)
    def forward(self, x):
        """Locoformer block Forward.

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either the number of frames or freqs
        """
        # FFN before self-attention
        input_ = x  # [B, T, 128]
        output = self.ffn_norm[-1](x)  
        #output = self.ffn[-1](output)*0.5 + input_
        output = output.transpose(1,2)
        output = self.ffn[-1](output)  # [B, T, 128]
        output = output.transpose(1,2)*0.5 + input_
        
        # Self-attention
        input_ = output
        output = self.attn_norm(output)
        output = self.attn(output)
        output = output +input_
        
        #input_ = output
        output = self.conv(output)
        
        # FFN after self-attention
        input_ = output
        output = self.ffn_norm[0](output)  # [B, T, F, C]
        #output = self.ffn[0](output)*0.5 + input_
        output = output.transpose(1,2)
        output = self.ffn[0](output)  # [B, T, F, C]
        output = output.transpose(1,2)*0.5 + input_
        
        return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        attention_dim,
        n_heads=8,
        dropout=0.0,
        rope=None,
        flash_attention=False,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.dropout = dropout

        self.rope = rope
        self.qkv = nn.Linear(emb_dim, attention_dim * 3, bias=False)
        # self.aggregate_heads = nn.Sequential(nn.Linear(attention_dim, emb_dim, bias=False), nn.Dropout(dropout))
        self.aggregate_heads = nn.Sequential(nn.Linear(attention_dim, attention_dim, bias=False), nn.Dropout(dropout))

        if flash_attention:
            self.flash_attention_config = dict(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        else:
            self.flash_attention_config = dict(enable_flash=False, enable_math=True, enable_mem_efficient=True)

    def forward(self, input):
        # get query, key, and value
        query, key, value = self.get_qkv(input)

        # rotary positional encoding
        query, key = self.apply_rope(query, key)

        # pytorch 2.0 flash attention: q, k, v, mask, dropout, softmax_scale
        with torch.backends.cuda.sdp_kernel(**self.flash_attention_config):
            output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
            )  # (batch, head, seq_len, -1)

        output = output.transpose(1, 2)  # (batch, seq_len, head, -1)
        output = output.reshape(output.shape[:2] + (-1,))
        return self.aggregate_heads(output)

    def get_qkv(self, input):
        n_batch, seq_len = input.shape[:2]
        x = self.qkv(input).reshape(n_batch, seq_len, 3, self.n_heads, -1)
        x = x.movedim(-2, 1)  # (batch, head, seq_len, 3, -1)
        query, key, value = x[..., 0, :], x[..., 1, :], x[..., 2, :]
        return query, key, value

    @torch.cuda.amp.autocast(enabled=False)
    def apply_rope(self, query, key):
        query = self.rope.rotate_queries_or_keys(query)
        key = self.rope.rotate_queries_or_keys(key)
        return query, key

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()
class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)
from einops.layers.torch import Rearrange
class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 3,
        padding = 1,
        dropout = 0.
    ):
        super().__init__()

        inner_dim = dim * expansion_factor

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim*2, 1),
            GLU(dim=1),
            nn.Conv1d(inner_dim, inner_dim, kernel_size, padding=padding, groups = inner_dim),
            # DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
    
class ConvDeconv1d(nn.Module):
    def __init__(self, dim, dim_inner, conv1d_kernel, conv1d_shift, dropout=0.0, **kwargs):
        super().__init__()

        self.diff_ks = conv1d_kernel - conv1d_shift

        self.net = nn.Sequential(
            nn.Conv1d(dim, dim_inner, conv1d_kernel, stride=conv1d_shift),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(dim_inner, dim, conv1d_kernel, stride=conv1d_shift),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """ConvDeconv1d forward

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either the number of frames or freqs
        """
        b, s1, s2, h = x.shape
        x = x.view(b * s1, s2, h)
        x = x.transpose(-1, -2)
        x = self.net(x).transpose(-1, -2)
        x = x[..., self.diff_ks // 2 : self.diff_ks // 2 + s2, :]
        return x.view(b, s1, s2, h)


class SwiGLUConvDeconv1d(nn.Module):
    def __init__(self, dim, dim_inner, conv1d_kernel, conv1d_shift, dropout=0.0, **kwargs):
        super().__init__()

        self.conv1d = nn.Conv1d(dim, dim_inner * 2, conv1d_kernel, stride=conv1d_shift)

        self.swish = nn.SiLU()
        self.deconv1d = nn.ConvTranspose1d(dim_inner, dim, conv1d_kernel, stride=conv1d_shift)
        self.dropout = nn.Dropout(dropout)
        self.dim_inner = dim_inner
        self.diff_ks = conv1d_kernel - conv1d_shift
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift

    def forward(self, x):
        """SwiGLUConvDeconv1d forward

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either the number of frames or freqs
        """
        # breakpoint()
        # b, s1, s2, h = x.shape
        # x = x.contiguous().view(b * s1, s2, h)
        # x = x.transpose(-1, -2)

        # # padding
        # seq_len = (
        #     math.ceil((s2 + 2 * self.diff_ks - self.conv1d_kernel) / self.conv1d_shift) * self.conv1d_shift
        #     + self.conv1d_kernel
        # )
        # x = F.pad(x, (self.diff_ks, seq_len - s2 - self.diff_ks))

        # conv-deconv1d
        x = self.conv1d(x)
        gate = self.swish(x[..., self.dim_inner :, :])
        x = x[..., : self.dim_inner, :] * gate
        x = self.dropout(x)
        # x = self.deconv1d(x).transpose(-1, -2)
        x = self.deconv1d(x)    # [B, T, 128]
        return self.dropout(x)
        # cut necessary part
        # x = x[..., self.diff_ks : self.diff_ks + s2, :]
        # return self.dropout(x).view(b, s1, s2, h)


class RMSGroupNorm(nn.Module):
    def __init__(self, num_groups, dim, eps=1e-8, bias=False):
        """
        Root Mean Square Group Normalization (RMSGroupNorm).
        Unlike Group Normalization in vision, RMSGroupNorm
        is applied to each TF bin.

        Args:
            num_groups: int
                Number of groups
            dim: int
                Number of dimensions
            eps: float
                Small constant to avoid division by zero.
            bias: bool
                Whether to add a bias term. RMSNorm does not use bias.

        """
        super().__init__()

        assert dim % num_groups == 0, (dim, num_groups)
        self.num_groups = num_groups
        self.dim_per_group = dim // self.num_groups

        self.gamma = nn.Parameter(torch.Tensor(dim).to(torch.float32))
        nn.init.ones_(self.gamma)

        self.bias = bias
        if self.bias:
            self.beta = nn.Parameter(torch.Tensor(dim).to(torch.float32))
            nn.init.zeros_(self.beta)
        self.eps = eps
        self.num_groups = num_groups

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input):
        others = input.shape[:-1]
        input = input.view(others + (self.num_groups, self.dim_per_group))

        # normalization
        norm_ = input.norm(2, dim=-1, keepdim=True)
        rms = norm_ * self.dim_per_group ** (-1.0 / 2)
        output = input / (rms + self.eps)

        # reshape and affine transformation
        output = output.view(others + (-1,))
        output = output * self.gamma
        if self.bias:
            output = output + self.beta

        return output


