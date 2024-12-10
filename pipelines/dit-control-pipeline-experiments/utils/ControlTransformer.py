import torch
import torch.nn as nn

class MapEncoder(nn.Module):
    """A small cnn the reduces spatial dimension and outputs control map embeddings. 
    The embeddings are then used to attend over text embeddings in the ControlTransformer.
    Note: this is potentially a weakspot in the POC implementation as we need to train it
    on a large dataset to learn the relationship between control maps and their embeddings...
    """
    def __init__(self, in_channels, embed_dim):
        super(MapEncoder, self).__init__()
        
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=4, padding=3),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), #H/8, W/8
            nn.ReLU(inplace = True),
            nn.Conv2d(128, embed_dim, kernel_size = 3, stride = 2, padding = 1), #H/16, w/16
            nn.ReLU(inplace = True)
        )
        
    def forward(self, x):
        x = self.conv(x)
        B, C, Hp, Wp = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(B, Hp*Wp, C)
        return x # (B, H*W, embed_dim) where H*W are the control map tokens
    
class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout = 0.1):
        super(TransformerLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first = True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        #Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(inplace = True),
            nn.Linear(dim_feedforward, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value):
        # query: (batch_size, L_map, embed_dim)
        # key, value: (batch_size, L_text, embed_dim)
        
        #multihead attention
        attn_output, _ = self.attn(query, key, value)
        query = query + self.dropout(attn_output)
        query = self.norm1(query)
        
        #feedforward
        ffn_output = self.ffn(query)
        query = query + self.dropout(ffn_output)
        query = self.norm2(query)
        
        return query
    
    
class ControlTransformer(nn.Module):
    def __init__(self, 
                 input_channels=2,  # Edge and depth maps
                 embed_dim=768,
                 num_layers = 4, 
                 num_heads=8, 
                 device = 'cuda'):
        
        
        super(ControlTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.input_channels = input_channels
        self.device = torch.device(device) 
        self.map_encoder = MapEncoder(1, embed_dim)
        #For now, we'll use the same map encoder for both edge and depth maps
        #In the future, we can have separate encoders for each map type if needed
        
        #cls tokens for each map type
        self.cls_tokens = nn.Parameter(torch.zeros(self.input_channels, 1, embed_dim))
        nn.init.normal_(self.cls_tokens, std=0.02)
        
        # Cross attention: We'll use a multi-head attention layer
        # Q: control map tokens, K/V: text embeddings 
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads) for _ in range(num_layers)
            ])
        
    def forward(self, text_embeddings, control_maps):
        """
        text_embeddings: (batch_size, L_text, embed_dim) - a single embedding for the text
        control_maps: (batch_size, input_channels, H, W) - edge and depth maps
        
        returns: (batch_size, input_channels, embed_dim) - one embedding per control map representing its learned relationship with text.
        """
        batch_size = text_embeddings.size(0)
        
        map_outputs = []
        
        for i in range(self.input_channels):
            single_map = control_maps[:, i, :, :] #batch, control map, H, W
            single_map = single_map.unsqueeze(1) #batch, 1, H, W
            #encode map with map encoder
            map_embedding = self.map_encoder(single_map)
            
            #add CLS token
            m_cls = self.cls_tokens[i:i+1].expand(batch_size, 1, self.embed_dim)
            m_seq = torch.cat([m_cls, map_embedding], dim=1)
            
            #cross attention through transformer layers
            for layer in self.layers:
                m_seq = layer(m_seq, text_embeddings, text_embeddings)
            
            cls_embed = m_seq[:, 0, :]
            map_outputs.append(cls_embed)
            
            control_map_embeddings = torch.stack(map_outputs, dim=1) #batch, input_channels, embed_dim
        return control_map_embeddings
        
