import torch
import torch.nn as nn
import torch.nn.functional as F

    
class MultiheadAttention(nn.Module):
    def __init__(
            self, in_features, heads, dropout, bias, temperature
        ):
        """Multi head attention
        
        Args:
            in_features(int): amount of input features
            heads(int): amount of heads in attention
            dropout(float): value, droprate for dropout
        """
        super(MultiheadAttention, self).__init__()
        
        self.heads = heads
        self.temperature = temperature
        
        self.qkv = nn.Linear(in_features, heads * in_features * 3, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(heads * in_features, in_features, bias=bias)
        self.dropout_proj = nn.Dropout(dropout)
        
    def forward(self, x):
        """Method forward in MultiheadAttention

        Inputs:
            x(torch.tensor): x.shape = [batch_size, x_dim * y_dim, in_features]
        
        """
        # x.shape = [batch_size, t_f_dim, in_feautres]
        batch_size, tf_dim, in_features = x.shape
        
        # qkv.shape = [batch_size, tf_dim, heads, in_features, 3]
        qkv = self.qkv(x).view(batch_size, tf_dim, self.heads, in_features, 3)
        # qkv.shape = [batch_size, heads, tf_dim, in_features, 3] 
        # -> [batch_size * heads, tf_dim, in_features, 3]
        qkv = qkv.transpose(1, 2).contiguous().view(batch_size * self.heads, tf_dim, in_features, 3)
        
        q, k = qkv[:, :, :, 0] / in_features ** (1 / 2), qkv[:, :, :, 1] / in_features ** (1 / 2)
        v = qkv[:, :, :, 2]
        
        # weights.shape = [batch_size * heads, tf_dim, tf_dim]
        weights = F.softmax(torch.bmm(q, k.transpose(1, 2)) / self.temperature, dim=2)
        weights = self.dropout(weights)
        # Так как, порядок в памяти не изменялся с момента qkv, 
        # модель не будет путаться при восстановлении размерности батчей
        x = torch.bmm(weights, v).view(batch_size, self.heads, tf_dim, in_features)
        x = x.transpose(1, 2).contiguous().view(batch_size, tf_dim, self.heads * in_features)
        
        x = self.proj(x)
        x = self.dropout_proj(x)
        
        return x
    

class Mlp(nn.Module):
    def __init__(
            self, in_features, expansion_ratio, 
            bias, apply_activation_last,
        ):
        super(Mlp, self).__init__()
        
        out_features = in_features * expansion_ratio
        
        self.block = [
            nn.Linear(in_features, out_features, bias=bias),
            nn.GELU(),
            nn.Linear(out_features, in_features, bias=bias)
        ]
        
        if apply_activation_last:
            self.block.append(nn.GELU())
            
        self.block = nn.Sequential(*self.block)
        
    def forward(self, x):
        x = self.block(x)
        
        return x


class Block(nn.Module):
    def __init__(
            self, in_features, heads, dropout, 
            expansion_ratio, apply_activation_last, bias, temperature
        ):
        super(Block, self).__init__()
        
        self.norm1 = nn.LayerNorm(normalized_shape=in_features)
        self.attn = MultiheadAttention(in_features, heads, dropout, bias, temperature)
        self.norm2 = nn.LayerNorm(normalized_shape=in_features)
        self.mlp = Mlp(
            in_features, expansion_ratio, 
            bias, apply_activation_last,
        )
        
    def forward(self, x):
        x = self.norm1(x)
        x = self.attn(x)
        x = self.norm2(x)
        x = self.mlp(x)
        
        return x
    

class Transformer(nn.Module):
    def __init__(
            self, in_features, num_blocks, heads=1, dropout=0., 
            expansion_ratio=4, apply_activation_last=True, temperature=1,
            **kwargs
        ):
        super(Transformer, self).__init__()
        
        bias = kwargs.get("bias", True)
        
        self.cls_token = self._init_token(in_features)
        self.dict_token = self._init_token(in_features)
        
        self.blocks = nn.ModuleList([])
        
        for _ in range(num_blocks):
            self.blocks.append(
                Block(
                    in_features, heads, dropout, 
                    expansion_ratio, apply_activation_last, 
                    bias, temperature
                )
            )
            
    def _init_token(self, in_features):
        token = torch.empty(1, 1, in_features)
        token = nn.Parameter(token)
        nn.init.xavier_normal_(token)

        return token
        
    def forward(self, x):
        batch_size = x.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        dict_token = self.dict_token.expand(batch_size, -1, -1)
        x = torch.cat((x, cls_token, dict_token), dim=1)
        
        for block in self.blocks:
            x = block(x)
            
        cls_token = x[:, -2, :].unsqueeze(1)
        dict_token = x[:, -1, :].unsqueeze(1)
        output = x[:, :-2, :]
        
        return output, cls_token, dict_token
