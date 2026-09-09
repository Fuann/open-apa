"""HiPPO inference architecture with the released checkpoint phone-ID convention.

Attention implementation derived from timm. Registered checkpoint parameters
are retained even when they are unused by the inference forward pass.
"""
import torch
import torch.nn as nn
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from rotary_embedding_torch import RotaryEmbedding

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rotary_emb = RotaryEmbedding(
            dim = dim,
            use_xpos = True   # set this to True to make rotary embeddings extrapolate better to sequence lengths greater than the one used at training time
        )

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        q, k = self.rotary_emb.rotate_queries_and_keys(q, k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class w2u_feat_gen(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.size = dim
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm1 = norm_layer(dim)
        self.pooling_proj1 = torch.nn.Linear(dim, 1)
        self.pooling_proj2 = torch.nn.Linear(dim, 1)
        self.pooling_proj3 = torch.nn.Linear(dim, 1)
        self.weight_proj1 = torch.nn.Linear(dim, 1)
        self.weight_proj2 = torch.nn.Linear(dim, 1)
        self.weight_proj3 = torch.nn.Linear(dim, 1)
        self.merge_proj = torch.nn.Linear(dim, dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.w1_proj = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.w2_proj = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.w3_proj = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.dropout = nn.Dropout(0.2)

    def forward(self, w1, w2, w3):
        w1_proj, w2_proj, w3_proj = self.w1_proj(w1), self.w2_proj(w2), self.w3_proj(w3)

        score1 = (
            self.pooling_proj1(w1_proj).transpose(1, 2) / self.size**0.5
        )  # (batch, 1, time)
        score1 = torch.softmax(score1, dim=-1)
        pooled1 = torch.matmul(score1, w1_proj).squeeze(1)  # (batch, size)
        weight1 = self.weight_proj1(pooled1)  # (batch, 1)

        score2 = (
            self.pooling_proj2(w2_proj).transpose(1, 2) / self.size**0.5
        )  # (batch, 1, time)
        score2 = torch.softmax(score2, dim=-1)
        pooled2 = torch.matmul(score2, w2_proj).squeeze(1)  # (batch, size)
        weight2 = self.weight_proj2(pooled2)  # (batch, 1)

        score3 = (
            self.pooling_proj2(w3_proj).transpose(1, 2) / self.size**0.5
        )  # (batch, 1, time)
        score3 = torch.softmax(score3, dim=-1)
        pooled3 = torch.matmul(score3, w3_proj).squeeze(1)  # (batch, size)
        weight3 = self.weight_proj3(pooled3)  # (batch, 1)

        merge_weights = torch.softmax(
            torch.cat([weight1, weight2, weight3], dim=-1), dim=-1
        )  # (batch, 2)
        merge_weights = merge_weights.unsqueeze(-1).unsqueeze(-1)  # (batch, 2, 1, 1)
        w1, w2, w3 = merge_weights[:, 0], merge_weights[:, 1], merge_weights[:, 2]  # (batch, 1, 1)
        x = self.dropout(
            self.mlp(w1 * w1_proj + w2 * w2_proj + w3 * w3_proj)
        )
        return x

class att_pooling(nn.Module):

    def __init__(self, dim, mlp_ratio=4., drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.pooling_proj = torch.nn.Linear(dim, 1)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.mlp(x)
        score = (
            self.pooling_proj(x).transpose(1, 2) / x.size(-1)**0.5
        )  # (batch, 1, time)
        score = torch.softmax(score, dim=-1)
        pooled = torch.matmul(score, x).squeeze(1)  # (batch, size)

        return pooled

class LlamaMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hidden_size = dim
        self.intermediate_size = int(dim*4)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

class ConvModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hidden_size = dim
        self.intermediate_size = dim
        self.conv1d = nn.Conv1d(dim, dim*2, kernel_size=1)
        self.glu_act = nn.GLU()
        self.act_fn = nn.SiLU()
        self.cnn_norm1 = nn.LayerNorm(dim)
        self.cnn_norm2 = nn.LayerNorm(dim)
        self.dep_cnn1 = nn.Conv1d(dim, dim*2, kernel_size=3, groups=dim, padding=1)
        self.dep_cnn2 = nn.Conv1d(dim*2, dim, kernel_size=1)

        self.dep2_cnn1 = nn.Conv1d(dim, dim, kernel_size=3, groups=dim, padding=1)
        self.dep2_cnn2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.conv1d_2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.cnn_drop = nn.Dropout(0.1)

    def forward(self, x):
        x_cnn = self.conv1d(self.cnn_norm1(x).transpose(1,2)).transpose(1,2) #x_cnn [B, 50, dim*2]
        x_cnn = self.glu_act(self.cnn_drop(x_cnn)) #x_cnn [B, 50, dim]
        x_cnn = self.dep_cnn1(x_cnn.transpose(1,2))
        x_cnn = self.dep_cnn2(x_cnn).transpose(1,2)
        x_cnn = self.act_fn(self.cnn_norm2(x_cnn))
        x_cnn = self.dep2_cnn1(x_cnn.transpose(1,2))
        x_cnn = self.dep2_cnn2(x_cnn).transpose(1,2)

        x_cnn = self.cnn_drop(x_cnn)
        x_cnn = x + x_cnn

        return x_cnn

class Weight_merge(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.size = dim
        self.pooling_proj1 = torch.nn.Linear(dim, 1)
        self.pooling_proj2 = torch.nn.Linear(dim, 1)
        self.weight_proj1 = torch.nn.Linear(dim, 1)
        self.weight_proj2 = torch.nn.Linear(dim, 1)
        self.dropout = nn.Dropout(0.2)
        self.mlp = Mlp(in_features=dim, act_layer=nn.SiLU)

    def forward(self, x_att, x_cnn):
        score1 = (
            self.pooling_proj1(x_att).transpose(1, 2) / self.size**0.5
        )  # (batch, 1, time)
        score1 = torch.softmax(score1, dim=-1)
        pooled1 = torch.matmul(score1, x_att).squeeze(1)  # (batch, size)
        weight1 = self.weight_proj1(pooled1)  # (batch, 1)

        score2 = (
            self.pooling_proj2(x_cnn).transpose(1, 2) / self.size**0.5
        )  # (batch, 1, time)
        score2 = torch.softmax(score2, dim=-1)
        pooled2 = torch.matmul(score2, x_cnn).squeeze(1)  # (batch, size)
        weight2 = self.weight_proj2(pooled2)  # (batch, 1)

        merge_weights = torch.softmax(
            torch.cat([weight1, weight2], dim=-1), dim=-1
        )  # (batch, 2)
        merge_weights = merge_weights.unsqueeze(-1).unsqueeze(-1)  # (batch, 2, 1, 1)
        w1, w2 = merge_weights[:, 0], merge_weights[:, 1]  # (batch, 1, 1)

        x = w1 * x_att + w2 * x_cnn
        return x

class ConvLlamaBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_drop = nn.Dropout(0.1)
        self.mlp2 = LlamaMLP(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim*4, out_features=dim, act_layer=nn.SiLU)
        self.cnn = ConvModule(dim)
        self.w_merge = Weight_merge(dim)
        self.layernorm1 = LlamaRMSNorm(dim, eps=1e-5)
        self.layernorm2 = LlamaRMSNorm(dim, eps=1e-5)
        self.layernorm3 = LlamaRMSNorm(dim, eps=1e-5)

    def forward(self, x):
        res = x
        x = self.layernorm1(x)
        x = x + self.mlp(x) /2
        res = x
        x = self.layernorm2(x)
        x_att = self.attn_drop(self.self_attn(x))
        x_cnn = self.cnn(x)
        x = res + self.w_merge(x_att, x_cnn)
        res = x
        x = self.layernorm3(x)
        x = res + self.attn_drop(self.mlp2(x))/2
        return x

class GOPT(nn.Module):
    def make_word_pos_mask(self, ys_pad):
        from espnet.nets.pytorch_backend.nets_utils import pad_list
        B = ys_pad.size()[0]
        L = ys_pad.size()[1]
        ys_mask = torch.zeros(B,L,L)
        for i in range(B):
            for idx, pos in enumerate(ys_pad[i]):
                if pos == -1:
                    break
                ys_mask[i][idx] = (ys_pad[i]==pos).int()
        return ys_mask

    def __init__(self, embed_dim, num_heads, p_depth, w_depth, u_depth, ssl_drop, input_dim=84):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.phn_blocks = nn.ModuleList([ConvLlamaBlock(dim=embed_dim, num_heads=num_heads) for i in range(p_depth)])

        self.word_blocks = nn.ModuleList([ConvLlamaBlock(dim=embed_dim, num_heads=num_heads) for i in range(w_depth)])

        self.utt_blocks = nn.ModuleList([ConvLlamaBlock(dim=embed_dim, num_heads=num_heads) for i in range(u_depth)])

        self.p_in_proj = nn.Linear(self.input_dim, embed_dim)
        self.w_in_proj = nn.Linear(self.input_dim, embed_dim)
        self.u_in_proj = nn.Linear(self.input_dim, embed_dim)
        self.ssl_drop = nn.Dropout(ssl_drop)
        self.feat_drop = nn.Dropout(0.2)
        self.mlp_head_phn = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        self.p_pre_proj = Mlp(in_features=embed_dim, hidden_features=embed_dim,
         act_layer=nn.GELU, drop=0.1)

        self.w_pre_proj = Mlp(in_features=embed_dim, hidden_features=embed_dim,
         act_layer=nn.GELU, drop=0.1)

        self.u_pre_proj = Mlp(in_features=embed_dim, hidden_features=embed_dim,
         act_layer=nn.GELU, drop=0.1)
        self.word_input_att = MultiHeadedAttention(num_heads, embed_dim, 0.1)
        self.word_input_att1 = MultiHeadedAttention(num_heads, embed_dim, 0.1)

        self.w_in_cat_proj = Mlp(in_features=embed_dim*2, hidden_features=embed_dim,
        out_features=embed_dim, act_layer=nn.GELU, drop=0.1)
        self.w_proj_ln1 = Mlp(in_features=embed_dim, hidden_features=embed_dim,
         act_layer=nn.GELU, drop=0.1)
        self.w_proj_ln2 = Mlp(in_features=embed_dim, hidden_features=embed_dim,
         act_layer=nn.GELU, drop=0.1)
        self.w_proj_ln3 = Mlp(in_features=embed_dim, hidden_features=embed_dim,
         act_layer=nn.GELU, drop=0.1)

        self.w_proj_cnn1 = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim*2, kernel_size=3, groups=embed_dim, padding=1),
            nn.Conv1d(embed_dim*2, embed_dim, kernel_size=1)
            )
        self.w_proj_cnn2 = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim*2, kernel_size=3, groups=embed_dim, padding=1),
            nn.Conv1d(embed_dim*2, embed_dim, kernel_size=1)
            )
        self.w3_proj = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim*2, kernel_size=3, groups=embed_dim, padding=1),
            nn.Conv1d(embed_dim*2, embed_dim, kernel_size=1)
            )
        self.mlp_head_word1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_word2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_word3 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        self.phn_proj = nn.Linear(40+2, embed_dim)
        self.word_proj = nn.Linear(768, embed_dim)

        self.mlp_head_utt1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        self.mlp_head_utt2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        self.mlp_head_utt3 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        self.mlp_head_utt4 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        self.mlp_head_utt5 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        self.utt_feat_ext = w2u_feat_gen(embed_dim)

        self.u1_att_pooling = att_pooling(embed_dim)
        self.u2_att_pooling = att_pooling(embed_dim)
        self.u3_att_pooling = att_pooling(embed_dim)
        self.u4_att_pooling = att_pooling(embed_dim)
        self.u5_att_pooling = att_pooling(embed_dim)

        self.u_in_cat_proj = Mlp(in_features=embed_dim*3, hidden_features=embed_dim,
        out_features=embed_dim, act_layer=nn.GELU, drop=0.1)

        self.utt_ssl_proj = Mlp(in_features=1024*3, hidden_features=embed_dim,
        out_features=embed_dim, act_layer=nn.GELU, drop=0.1)

        self.u1_ssl_proj = nn.Linear(embed_dim, embed_dim)
        self.u2_ssl_proj = nn.Linear(embed_dim, embed_dim)
        self.u3_ssl_proj = nn.Linear(embed_dim, embed_dim)
        self.u4_ssl_proj = nn.Linear(embed_dim, embed_dim)
        self.u5_ssl_proj = nn.Linear(embed_dim, embed_dim)

        self.u_in_proj1 = Mlp(in_features=embed_dim, hidden_features=embed_dim,
         act_layer=nn.GELU, drop=0.1)
        self.u_in_proj2 = Mlp(in_features=embed_dim, hidden_features=embed_dim,
         act_layer=nn.GELU, drop=0.1)
        self.u_in_proj3 = Mlp(in_features=embed_dim, hidden_features=embed_dim,
         act_layer=nn.GELU, drop=0.1)

        self.u_proj_cnn1 = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim*2, kernel_size=3, groups=embed_dim, padding=1),
            nn.Conv1d(embed_dim*2, embed_dim, kernel_size=1)
            )
        self.u_proj_cnn2 = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim*2, kernel_size=3, groups=embed_dim, padding=1),
            nn.Conv1d(embed_dim*2, embed_dim, kernel_size=1)
            )
        self.u_proj_cnn3 = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim*2, kernel_size=3, groups=embed_dim, padding=1),
            nn.Conv1d(embed_dim*2, embed_dim, kernel_size=1)
            )

        self.phn_predi_ln = nn.Linear(embed_dim, 42)
        self.word_predi_ln = nn.Linear(embed_dim, 2607)

        self.phn_audio_proj = Mlp(in_features=embed_dim*3, hidden_features=embed_dim*3, out_features=embed_dim,
         act_layer=nn.GELU, drop=0.1)

        self.phn_text_proj = Mlp(in_features=embed_dim, hidden_features=embed_dim,
         act_layer=nn.GELU, drop=0.1)

    def forward(self, x, utt_ssl_feat, phn, word_pos, word_embed, word_atten_msk):

        B = x.shape[0]
        phn_one_hot = torch.nn.functional.one_hot(phn.long() + 1, num_classes=42).float()
        phn_embed = self.phn_proj(phn_one_hot)

        if self.embed_dim != self.input_dim:
            p_x = self.p_in_proj(x)
            w_x = self.w_in_proj(x)
            u_x = self.u_in_proj(x)
        p_x = p_x + phn_embed
        for blk in self.phn_blocks:
            p_x = blk(p_x)

        p = self.mlp_head_phn(p_x)

        word_embed = self.word_proj(word_embed)

        phn2word_msk = word_atten_msk
        phn2word_msk = phn2word_msk.to(p.device)

        w_p_x = self.w_proj_cnn1(p_x.transpose(1,2)).transpose(1,2)
        w_x = self.w_proj_cnn2(w_x.transpose(1,2)).transpose(1,2)
        w_x_att = self.word_input_att(w_x, w_x, w_x, phn2word_msk)
        w_p_x_att = self.word_input_att1(w_p_x, w_p_x, w_p_x, phn2word_msk)
        x_word = self.w_in_cat_proj(torch.cat([w_x_att, w_p_x_att], dim=-1))
        x_word = x_word + word_embed + p_x

        for blk in self.word_blocks:
            x_word = blk(x_word)

        w1_proj = self.w_proj_ln1(x_word)
        w2_proj = self.w_proj_ln2(x_word)
        w3_proj = self.w_proj_ln3(x_word)

        w1 = self.mlp_head_word1(w1_proj)
        w2 = self.mlp_head_word2(w2_proj)
        w3 = self.mlp_head_word3(w3_proj)

        u_p_x = p_x
        u_w_feats = self.utt_feat_ext(w1_proj, w2_proj, w3_proj)
        u_p_feats = self.u_proj_cnn1(u_p_x.transpose(1,2)).transpose(1,2)
        u_w_feats = self.u_proj_cnn2(u_w_feats.transpose(1,2)).transpose(1,2)
        utt_feats = self.u_in_cat_proj(torch.cat([u_p_feats, u_w_feats, u_x], dim=-1)) + p_x

        for blk in self.utt_blocks:
            utt_feats = blk(utt_feats)

        utt_ssl_feats = self.utt_ssl_proj(utt_ssl_feat)
        u1_proj = self.u1_att_pooling(utt_feats) + self.u1_ssl_proj(utt_ssl_feats)
        u2_proj = self.u2_att_pooling(utt_feats) + self.u2_ssl_proj(utt_ssl_feats)
        u3_proj = self.u3_att_pooling(utt_feats) + self.u3_ssl_proj(utt_ssl_feats)
        u4_proj = self.u4_att_pooling(utt_feats) + self.u4_ssl_proj(utt_ssl_feats)
        u5_proj = self.u5_att_pooling(utt_feats) + self.u5_ssl_proj(utt_ssl_feats)

        u1 = self.mlp_head_utt1(u1_proj)
        u2 = self.mlp_head_utt2(u2_proj)
        u3 = self.mlp_head_utt3(u3_proj)
        u4 = self.mlp_head_utt4(u4_proj)
        u5 = self.mlp_head_utt5(u5_proj)

        return u1, u2, u3, u4, u5, p, w1, w2, w3
