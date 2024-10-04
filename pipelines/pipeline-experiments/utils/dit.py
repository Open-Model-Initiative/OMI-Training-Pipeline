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

class DiTBlock(torch.nn.Module):
    def __init__(self,channels,heads,conditionC):
        super().__init__()
        self.norm1=AdaLN(conditionC,channels)
        self.norm2=torch.nn.GroupNorm(1,channels,eps=1e-6)
        self.attn=RoPEAttention(channels,heads,8)
            
        self.mlp=torch.nn.ModuleDict({
            "fc1":torch.nn.Linear(channels,channels*4),
            "fc2":torch.nn.Linear(channels*4,channels)
        })
        self.ls1=LayerScale(channels,init_values=0.0)
        self.ls2=LayerScale(channels,init_values=0.0)

    def forward(self,x,condition):
        input=x
        x=self.norm1(x,condition)
        x=self.attn(x) 
        x=self.ls1(x)+input

        input=x
        x=self.norm2(x)
        x=x.permute(0,2,3,1)
        x=self.mlp.fc1(x)
        x=torch.nn.functional.gelu(x)
        x=self.mlp.fc2(x)
        x=x.permute(0,3,1,2)
        x=self.ls2(x)+input
        return x
    
class DiT(torch.nn.Module):
    """
    Diffusion Transformer (DiT) model.

    This class implements a Diffusion Transformer, which is a transformer-based architecture
    designed for image generation. It applies a series of DiT blocks to process the input, 
    incorporating time step and condition information.

    Args:
        channels (int): Number of channels in each transformer block.
        nBlocks (int): Number of DiT blocks in the model.
        nHeads (int): Number of attention heads
        inC (int): Number of input channels.
        outC (int): Number of output channels.
        conditionC (int): Number of conditioning channels -> set to 1+text_embed_dim
        patchSize (int, optional): Size of patches for patchification. Defaults to 1. Increasing this
        will speed up training and lower VRAM requirements, at the expense of generation quality
        and potential artifacts. The final convolution layer attempts to mitigate this effect
    """
    def __init__(self,channels,nBlocks,inC,outC,conditionC,nHeads=8,patchSize=2):
        super().__init__()

        self.patchify=torch.nn.PixelUnshuffle(patchSize)
        self.inProj=torch.nn.Conv2d(inC*patchSize**2,channels,kernel_size=1)
        self.blocks=torch.nn.ModuleList()
        for i in range(nBlocks):
            self.blocks.append(DiTBlock(channels,nHeads,conditionC))
        self.outProj=torch.nn.Conv2d(channels,outC*patchSize**2,kernel_size=1)
        self.unpatchify=torch.nn.PixelShuffle(patchSize)
        self.finalConv=torch.nn.Conv2d(outC,outC,kernel_size=3,padding='same')
        
        self.condProj = torch.nn.Linear(conditionC, channels)

    def forward(self,x,condition):
        x = self.patchify(x)
        x = self.inProj(x)
        # Rest of the code
        b, c, h, w = x.shape
        
        condition = condition.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
        for block in self.blocks:
            x = block(x, condition)
        
        x = self.outProj(x)
        x = self.unpatchify(x)
        x = self.finalConv(x)
        return x