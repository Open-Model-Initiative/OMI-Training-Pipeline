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
            nn.conv2d(in_channels, 64, kernel_size=7, stride=4, padding=3),
            nn.ReLU(inplace = True),
            nn.conv2d(64, 128, kernel_size=3, stride=2, padding=1), #H/8, W/8
            nn.ReLU(inplace = True),
            nn.conv2d(128, embed_dim, kernel_size = 3, stride = 2, padding = 1) #H/16, w/16
            nn.ReLU(inplace = True)
        )
        
    def forward(self, x):
        x = self.conv(x)
        B, C, Hp, Wp = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(B, Hp*Wp, C)
        return x # (B, H*W, embed_dim) where H*W are the control map tokens 
class ControlTransformer(nn.Module):
    def __init__(self, 
                 input_channels=2,  # Edge and depth maps
                 embed_dim=768, 
                 num_layers=4, 
                 num_heads=8):
        
        super(ControlTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.input_channels = input_channels
        
        # Patch Embedding
        self.patch_embed = nn.Conv2d(
            1, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size // 2
        )
        
        #cls tokens for each map type
        self.cls_tokens = nn.Parameter(torch.zeros(self.input_channels, 1, embed_dim))
        nn.init.normal_(self.cls_tokens, std=0.02)
        
        # Positional Embedding (initialized later)
        self.pos_embed = None
        
        # Cross attention: We'll use a multi-head attention layer
        # Q: control map tokens, K/V: text embeddings 
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
    def forward(self, text_embeddings, control_maps):
        """
        text_embeddings: (batch_size, 1, embed_dim) - a single embedding for the text
        control_maps: (batch_size, input_channels, H, W) - edge and depth maps
        
        returns: (batch_size, input_channels, embed_dim) - one embedding per control map representing its learned relationship with text.
        """
        batch_size = text_embeddings.size(0)
        device = text_embeddings.device
        
        map_outputs = []