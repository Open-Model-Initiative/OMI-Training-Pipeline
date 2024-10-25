import torch

class AdaLN(torch.nn.Module):
    def __init__(self,conditionC,channels,skip=False):
        super().__init__()
        self.norm=torch.nn.GroupNorm(1,channels)
        self.proj1=torch.nn.Conv2d(conditionC,channels,kernel_size=1)
        self.proj2=torch.nn.Conv2d(conditionC,channels,kernel_size=1)
        self.skip=skip
    
    def forward(self,input, condition):
        w=self.proj1(condition)
        b=self.proj2(condition)
        if self.skip:
            return input
        out=self.norm(input)*w+b
        return out

class LayerScale(torch.nn.Module):
    def __init__(self,channels,init_values = 1e-5):
        super().__init__()
        self.gamma = torch.nn.Parameter(init_values * torch.ones(channels))

    def forward(self, x):
        return x * self.gamma.view(1,-1,1,1)
    
class RoPEAttention(torch.nn.Module):
    """
    Rotary Position Embedding (RoPE) Attention module.

    This module implements a multi-head attention mechanism for vision transformers with Rotary Position Embeddings.
    It applies RoPE to the query and key projections before computing attention, uses QK norm,
    Flash Attention v2 and scales alpha to allow interpolation and extrapolation to new resolutions

    Args:
        channels (int): Number of input and output channels.
        heads (int): Number of attention heads.
        baseSize (int): The base resolution RoPE will be scaled around.
    """
    def __init__(self, channels, heads, baseSize):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.baseSize=baseSize
        self.headDim = channels // heads

        self.qProj = torch.nn.Conv2d(channels, channels, 1)
        self.kProj = torch.nn.Conv2d(channels, channels, 1)
        self.vProj = torch.nn.Conv2d(channels, channels, 1)
        self.outProj = torch.nn.Conv2d(channels, channels, 1)

        self.qLayerNorm=torch.nn.GroupNorm(1,heads)
        self.kLayerNorm=torch.nn.GroupNorm(1,heads)

    def createRotationMatrix(self, height, width, channels):
        scalingH=height/self.baseSize if height>self.baseSize else 1.0
        scalingW=width/self.baseSize if width>self.baseSize else 1.0
        t = torch.arange(height * width, dtype=torch.float32)
        tX = (t % width).float()
        tY = torch.div(t, width, rounding_mode='floor').float()

        freqsX = 1.0 / ((100.0*scalingW) ** (torch.arange(0, channels, 2).float() / channels))
        freqsY = 1.0 / ((100.0*scalingH) ** (torch.arange(0, channels, 2).float() / channels))
        freqsX = torch.outer(tX, freqsX)
        freqsY = torch.outer(tY, freqsY)

        freqsCisX = torch.polar(torch.ones_like(freqsX), freqsX)
        freqsCisY = torch.polar(torch.ones_like(freqsY), freqsY)

        freqsCis = torch.cat([freqsCisX, freqsCisY], dim=-1)
        freqsCis = freqsCis.view(height, width, channels)

        return freqsCis

    def rotate(self, q, k):
        batchSize, channels, height, width = q.shape
        freqsCis = self.createRotationMatrix(height, width, channels).to(q.device)

        q = q.permute(0, 2, 3, 1).contiguous()
        k = k.permute(0, 2, 3, 1).contiguous()

        qRot = q * freqsCis.real - q.roll(shifts=1, dims=-1) * freqsCis.imag
        kRot = k * freqsCis.real - k.roll(shifts=1, dims=-1) * freqsCis.imag

        return qRot.permute(0, 3, 1, 2), kRot.permute(0, 3, 1, 2)

    def forward(self, x):
        batchSize, channels, height, width = x.shape
        
        q = self.qProj(x)
        k = self.kProj(x)
        v = self.vProj(x)

        q, k = self.rotate(q, k)

        q = q.view(batchSize, self.heads, self.headDim, height * width).transpose(2, 3)
        k = k.view(batchSize, self.heads, self.headDim, height * width).transpose(2, 3)
        v = v.view(batchSize, self.heads, self.headDim, height * width).transpose(2, 3).contiguous()

        q=self.qLayerNorm(q)
        k=self.kLayerNorm(k)

        #Temporarily converting to bf16 to use Flash Attention v2
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION, torch.nn.attention.SDPBackend.MATH]), torch.amp.autocast('cuda',enabled=True,dtype=torch.bfloat16):
            attnOutput = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attnOutput = attnOutput.float().transpose(2, 3).contiguous().view(batchSize, channels, height, width)

        return self.outProj(attnOutput)

class CrossAttention(torch.nn.Module):
    def __init__(self, dim, heads = 8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.to_q = torch.nn.Linear(dim, dim, bias = False)
        self.to_k = torch.nn.Linear(dim, dim, bias = False)
        self.to_v = torch.nn.Linear(dim, dim, bias = False)
        self.to_out = torch.nn.Linear(dim, dim, bias = False)
        
    def forward(self, x, context):
        # x: (B, N, dim) latents
        # context: (B, L, dim) text embeddings
        B, N, C = x.shape
        _, L, _ = context.shape
        
        H = self.heads
        head_dim = C // H
        
        q = self.to_q(x).view(B, N, H, head_dim).transpose(1, 2)
        k = self.to_k(context).view(B, L, H, head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, L, H, head_dim).transpose(1, 2)
        
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', q, k) * self.scale
        attn_probs = torch.nn.functional.softmax(attn_scores, dim = -1)
        
        out = torch.einsum('bhqk,bhvd->bhqd', attn_probs,v) * self.scale
        out = out.transpose(1,2).contiguous().view(B, N, C)
        return self.to_out(out)
        

class DiTBlock(torch.nn.Module):
    
    def __init__(self,channels,heads,conditionC, text_embed_dim):
        super().__init__()
        self.norm1=AdaLN(conditionC,channels)
        self.norm2=torch.nn.GroupNorm(1,channels,eps=1e-6)
        self.norm3=torch.nn.GroupNorm(1,channels,eps=1e-6)
        self.attn=RoPEAttention(channels,heads,8)
            
        ##Cross attention layer
        self.cross_attn = CrossAttention(channels, heads)
        self.text_proj = torch.nn.Linear(text_embed_dim, channels)
            
        self.mlp=torch.nn.Sequential(
            torch.nn.Linear(channels,channels*4),
            torch.nn.GELU(),
            torch.nn.Linear(channels*4,channels)
        )
        
        self.ls1=LayerScale(channels,init_values=0.0)
        self.ls2=LayerScale(channels,init_values=0.0)
        self.ls3=LayerScale(channels,init_values=0.0)

    def forward(self,x,condition, text_embed):
        # x: (B, C, H, W)
        # condition: (B, conditionC, 1, 1)
        # text_embed: (B, L, text_embed_dim)
        
        B, C, H, W = x.shape
        
        #self attention        
        input=x
        x=self.norm1(x,condition)
        x=self.attn(x) 
        x=self.ls1(x) + input

        #cross attention
        input=x
        x=self.norm3(x)
        x = x.view(B, C, H * W).transpose(1, 2) #(B, N, C) where N = H * W
        
        # Project Text Embeddings to match channel
        text_embed_proj = self.text_proj(text_embed)
        
        x = self.cross_attn(x, text_embed_proj)
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.ls2(x) + input

        #Feed forward
        input = x
        x = self.norm2(x)        
        x=x.permute(0,2,3,1).reshape(B*H*W,C)
        x=self.mlp(x)
        x=x.view(B, H, W, C).permute(0,3,1,2)
        x=self.ls3(x)+input
        return x
    
class DiffusionTransformer(torch.nn.Module):
    """
    Diffusion Transformer (DiT) model.

    This class implements a Diffusion Transformer, which is a transformer-based architecture
    designed for image generation. It applies a series of DiT blocks to process the input, 
    incorporating time step and condition information.

    Args:
        channels (int): Number of channels in each transformer block.
        nBlocks (int): Number of DiT blocks in the model.
        nHeads (int): Number of attention heads
        latent_channels (int): Number of latent channels used by VAE.
        conditionC (int): Number of conditioning channels -> need to set a sinusodial position embedding, this is the timestep embedding
        patchSize (int, optional): Size of patches for patchification. Defaults to 1. Increasing this
        will speed up training and lower VRAM requirements, at the expense of generation quality
        and potential artifacts. The final convolution layer attempts to mitigate this effect
        
    Training notes:
    To match flux lets make the following adjustments:
        - Adjust depth (nBlocks) to learn better hierarchical representations 
            flux schnell uses 19 blocks
        - Adjust (channels) to increase model capacity. 
            flux schnell uses 3072 = 128*8 = (attention head dim) * (num_attention_heads)
        - Adjust nHeads to 24
        - Adjust conditionC to 4096 + 1
            
    """
    def __init__(self,
                 channels, #hidden dimension of the transformer - increases model capacity
                 nBlocks, #number of transformer blocks - increases model depth
                 latent_channels, #number of input channels from VAE latent space
                 conditionC, # Number of conditioning channels -> need to set a sinusodial position embedding, this is the timestep embedding
                 nHeads=8, #number of attention heads - more heads to focus on different parts of input
                 patchSize=1, # Size of patches for image embeddings
                 text_embed_dim = 512):
        super().__init__()
        self.conditionC = conditionC
        ##Projection
        self.patchify=torch.nn.PixelUnshuffle(patchSize)
        self.inProj=torch.nn.Conv2d(latent_channels * patchSize**2, channels, kernel_size=1)
        
        ##Transformer Blocks - now with internal self-attention and cross-attention
        self.blocks=torch.nn.ModuleList([
            DiTBlock(channels, nHeads, conditionC, text_embed_dim) 
            for _ in range(nBlocks)
        ])
        
        #Output Projhection
        self.outProj=torch.nn.Conv2d(channels,latent_channels*patchSize**2,kernel_size=1)
        self.unpatchify=torch.nn.PixelShuffle(patchSize)
        self.finalConv=torch.nn.Conv2d(latent_channels,latent_channels,kernel_size=3,padding='same')
        
        self.condProj = torch.nn.Linear(conditionC, channels)

    def forward(self,x,condition,text_embed):
        x = self.patchify(x)
        x = self.inProj(x)
        
        for block in self.blocks:
            x = block(x, condition, text_embed)
        
        x = self.outProj(x)
        x = self.unpatchify(x)
        x = self.finalConv(x)
        return x