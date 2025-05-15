import torch
import torch.nn as nn
import torch.nn.functional as F

#---------------------------------------------------------------------------------------------#


class Attention(nn.Module):
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.dim = dim # dimension of the input
        self.n_heads = n_heads # number of heads
        self.head_dim = dim // n_heads # dimension of each head
        self.scale = self.head_dim ** -0.5 # scale factor

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # linear layer to create query, key, value vectors
        self.attn_drop = nn.Dropout(attn_p) # attention dropout
        self.proj = nn.Linear(dim, dim) # linear layer to project the weighted average of the values
        self.proj_drop = nn.Dropout(proj_p)  # projection dropout

    def forward(self, x):
        n_samples, n_tokens, dim = x.shape 
        # n_tokens = n_features + 1 (cls token)
        if dim != self.dim:
            raise ValueError # dimension of the input must be equal to the dimension of the module
        
        qkv = self.qkv(x) # (n_samples, n_tokens, 3 * dim)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim) # (n_samples, n_tokens, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, n_samples, n_heads, n_tokens, head_dim)  

        q, k, v = qkv[0], qkv[1], qkv[2] # (n_samples, n_heads, n_tokens, head_dim)
        k_t = k.transpose(-2, -1) # (n_samples, n_heads, head_dim, n_tokens)
        
        dp = q @ k_t # (n_samples, n_heads, n_tokens, n_tokens) --- dot product possible because the last two dimensions are compatible
        dp = dp * self.scale # prevent gradient explosion 

        attn = dp.softmax(dim=-1) #apply softmax to the last dimension to create a probability distribution that sums to 1
        attn = self.attn_drop(attn) # attention dropout
        weighted_avg = attn @ v # (n_samples, n_heads, n_tokens, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2) # (n_samples, n_tokens, n_heads, head_dim) because we want to concatenate the heads
        weighted_avg = weighted_avg.flatten(2) # (n_samples, n_tokens, dim) flatten the last two dimensions to concatenate the heads

        x = self.proj(weighted_avg) # (n_samples, n_tokens, dim) project the weighted average of the values
        x = self.proj_drop(x) # projection dropout 
        
        return x # (n_samples, n_tokens, dim)


#---------------------------------------------------------------------------------------------#


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU(), p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features) # first linear layer
        self.act = act_layer # activation function
        self.fc2 = nn.Linear(hidden_features, out_features) # second linear layer
        self.drop = nn.Dropout(p) # dropout

    def forward(self, x):
        x = self.fc1(x) # first linear layer (n_samples, n_tokens, hidden_features)
        x = self.act(x) # activation function 
        x = self.drop(x) # dropout
        x = self.fc2(x) # second linear layer (n_samples, n_tokens, out_features)
        x = self.drop(x) # dropout
        return x # (n_samples, n_tokens, out_features)


#---------------------------------------------------------------------------------------------#


class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4., qkv_bias=True, p=0., attn_p=0., proj_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6) # layer normalization
        self.attention = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=proj_p) # attention module
        self.norm2 = nn.LayerNorm(dim, eps=1e-6) # layer normalization
        hidden_features = int(dim * mlp_ratio) # hidden features
        self.mlp = MLP(in_features=dim, hidden_features=hidden_features, out_features=dim, p=p) # MLP module

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


#---------------------------------------------------------------------------------------------#


class Transformer(nn.Module):
    def __init__(self, n_classes=2, n_features=48,
                  embed_dim=768, depth=12, n_heads=12, mlp_ratio=4., 
                  qkv_bias=True, p=0., attn_p=0., proj_p=0.):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1)) # class token
        self.pos_embed = nn.Parameter(torch.zeros(1, n_features + 1, 1)) # positional embedding
        self.pos_drop = nn.Dropout(p=p) # dropout

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, 
                  qkv_bias=qkv_bias, p=p, attn_p=attn_p, proj_p=proj_p)
            for _ in range(depth)
            ]) # transformer encoder blocks (depth = number of blocks) 
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6) # layer normalization
        self.head = nn.Linear(embed_dim, n_classes) # linear layer for classification 

    def forward(self, x):
        n_samples = x.shape[0]
        x = torch.unsqueeze(x, -1) # (n_samples, features, embed_dim)
        cls_token = self.cls_token.expand(n_samples, -1, -1) # (n_samples, 1, embed_dim)

        x = torch.cat((cls_token, x), dim=1) # (n_samples, n_features + 1, embed_dim)
        x = x + self.pos_embed # (n_samples, n_features + 1, embed_dim)
        x = self.pos_drop(x) # dropout 

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x) # layer normalization
        cls_token_final = x[:, 0] # (n_samples, embed_dim)
        x = self.head(cls_token_final) # (n_samples, n_classes)

        return x # (n_samples, n_classes)