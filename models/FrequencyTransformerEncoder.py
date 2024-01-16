import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadFrequencyAttention(nn.Module):
    def __init__(
            self, in_features, heads, dropout=0., bias=True, temperature=1,
        ):
        """Multi head frequency attention
    
        Args:
            in_features(int): amount of input features
            heads(int): amount of heads in attention
            dropout(float): value, droprate for dropout
        """
        super(MultiheadFrequencyAttention, self).__init__()
        
        self.heads = heads
        self.temperature = temperature
        
        self.qk = nn.Linear(in_features, heads * in_features * 2, bias=bias)
        self.v = nn.Linear(in_features, heads * in_features, bias)
        self.dropout_weights = nn.Dropout(dropout)
        self.proj = nn.Linear(heads * in_features, in_features, bias)
        self.dropout_proj = nn.Dropout(dropout)

    def forward(self, x):
        """Method forward in MultiheadFrequencyAttention

        Inputs:
            x(torch.tensor): spectogram, x.shape = [batch_size, frequency_dim, time_dim, in_features]
        
        """
        batch_size, frequency_dim, time_dim, in_features = x.shape
        # query, key calculation
        qk = self.qk(torch.mean(x, dim=2)).view(batch_size, frequency_dim, self.heads, in_features, 2)
        # value calculation
        v = self.v(x).view(batch_size, frequency_dim, time_dim, self.heads, in_features)
        
        qk = qk.transpose(1, 2).contiguous().view(batch_size * self.heads, frequency_dim, in_features, 2)
        v = v.permute(0, 3, 2, 1, 4).contiguous().view(batch_size * self.heads, time_dim, frequency_dim, in_features)
        
        # q.shape = k.shape = [batch_size * heads, frequency_dim, in_features]
        q, k = qk[:, :, :, 0] / in_features ** (1 / 2), qk[:, :, :, 1] / in_features ** (1 / 2)
        # weights.shape = [batch_size * heads, 1, frequency_dim, frequency_dim]
        weights = F.softmax(torch.bmm(q, k.transpose(1, 2)) / self.temperature, dim=2).unsqueeze(1)
        weights = self.dropout_weights(weights)
        
        x = torch.matmul(weights, v).view(batch_size, self.heads, time_dim, frequency_dim, in_features)
        x = x.permute(0, 3, 2, 1, 4).contiguous().view(batch_size, frequency_dim, time_dim, in_features * self.heads)
        
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
        self.attn = MultiheadFrequencyAttention(
            in_features, heads, dropout, bias, temperature
        )
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
    

class FrequencyTransformer(nn.Module):
    def __init__(
            self, in_features, time_dim, num_blocks, heads=1, dropout=0., 
            expansion_ratio=4, apply_activation_last=True, temperature=1,
            **kwargs
        ):
        """Transformer with frequency-wise attention
        
        Args:
            in_features(int): amount of parameters for linear layers
            num_block(int): amount of transformer blocks
            heads(int): amount of heads in attention
            dropout(float): drop rate in nn.Dropout
            expansion_ratio(int): expansion ratio for MLP blocks
            apply_activation(bool): True if apple activation in each end of activation blocks
            temperature(int): smoothing parameter in attention blocks
            
        Inputs:
            x(torch.tensor): x.shape = [batch_size, frequency_dim, time_dim, in_features]
            
        Outputs:
            output(torch.tensor): output.shape = [batch_size, frequency_dim, time_dim, in_features]
            cls_token(torch.tensor): cls_token = [-1, -1, time_dim, in_features]
            dict_token(torch.tensor): dict_token = [-1, -1, time_dim, in_features]
            
        
        """
        super(FrequencyTransformer, self).__init__()
        
        bias = kwargs.get("bias", True)
        
        self.cls_token = self._init_token(time_dim, in_features)
        self.dict_token = self._init_token(time_dim, in_features)
        
        self.blocks = nn.ModuleList([])
        
        for _ in range(num_blocks):
            self.blocks.append(
                Block(
                    in_features, heads, dropout, 
                    expansion_ratio, apply_activation_last, 
                    bias, temperature
                )
            )
            
    def _init_token(self, time_dim, in_features):
        token = torch.empty(1, 1, time_dim, in_features)
        token = nn.Parameter(token)
        nn.init.xavier_normal_(token)

        return token
        
    def forward(self, x):
        """forward method in FrequencyTransformer
        
        Inputs:
            x(torch.tensor): x.shape = [batch_size, frequency_dim, time_dim, in_features]
        """
        batch_size = x.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1, -1)
        dict_token = self.dict_token.expand(batch_size, -1, -1, -1)
        x = torch.cat((x, cls_token, dict_token), dim=1)
        
        for block in self.blocks:
            x = block(x)
            
        cls_token = x[:, -2, :, :].unsqueeze(1)
        dict_token = x[:, -1, :, :].unsqueeze(1)
        output = x[:, :-2, :, :]
        
        return output, cls_token, dict_token
