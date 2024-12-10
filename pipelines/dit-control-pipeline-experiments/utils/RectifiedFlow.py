import math
import torch
from tqdm.auto import tqdm 

class RectifiedFlow(torch.nn.Module):
    def __init__(self,model,T, lr=3e-4,device='cpu',optimizer=None,emaStrength=0.0):
        super(RectifiedFlow, self).__init__()
        self.model=model
        self.emaStrength=emaStrength
        if self.emaStrength>0.0:
            self.emaModel = torch.optim.swa_utils.AveragedModel(self.model,
                                                                multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(self.emaStrength))
        else:
            self.emaModel=self.model

        self.T=T
        self.conditionC = self.model.conditionC
        self.device=device

        if optimizer is None:
            self.optimizer=torch.optim.AdamW(list(self.model.parameters()),lr=lr)
        else:
            self.optimizer=optimizer

    def state_dict(self):
        return {
            'model':self.model.state_dict(),
            'ema':self.emaModel.state_dict(),
            'optimizer':self.optimizer.state_dict()
        }
    
    def load_state_dict(self,stateDict):
        self.model.load_state_dict(stateDict['model'])
        self.emaModel.load_state_dict(stateDict['ema'])
        self.optimizer.load_state_dict(stateDict['optimizer'])

    def get_timestep_embeddings(self, timesteps):
        embedding_dim = self.conditionC
        half_dim = embedding_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim).to(self.device)
        args = timesteps[:, None].float() * freqs[None]
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if embedding_dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        return embeddings.squeeze(1)  # Shape: (batch_size, embedding_dim)


    def q(self,image,t):
        source=torch.randn_like(image).to(self.device)
        xT=(1-t)*image+t*source
        return xT,source
    
    def p(self,xT,t,condition=None, text_embed=None, control_embed = None):
        
        t = t.view(-1,1)
        t = self.get_timestep_embeddings(t)
        
        if condition is None:
            condition=t
        else:
            condition = torch.cat([t, condition], dim = 1)
        
        condition = condition[:, :, None, None]
        if not self.model.training:
            vPred = self.emaModel(xT, condition, text_embed, control_embed)
        else:
            vPred = self.model(xT, condition, text_embed, control_embed)
  
        return vPred

    def call(self,steps,shape=None, text_embed = None, condition=None, loopFunction=None):
        #loopFunction is a function with the same signature as vSample which returns the sample at t-1.
        #It can be used to implement CFG and schedulers
        if loopFunction is None:
            loopFunction=self.vSample
        
        xT=torch.randn(shape).to(self.device)
        with torch.no_grad():
            for i in tqdm( range(0,steps),ascii=" ▖▘▝▗▚▞█",disable=False):
                xT=loopFunction(i,shape,xT,steps,condition, text_embed)
            
        return xT
    
    def train_step(self,data,condition=None,validation=None,classifier=None):
        t=torch.rand(data.shape[0],1,1,1).to(self.device)
        if validation is not None:
            t=torch.ones(data.shape[0],1,1,1).to(self.device)*validation
        
        noiseData,epsilon=self.q(data,t)
        vPred=self.p(noiseData,t, condition=condition)
        loss=(((epsilon-data-vPred))**2).mean(dim=(2,3)).mean()

        if validation is None:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.emaModel.update_parameters(self.model)

        return {'loss':loss.detach()}

    def vSample(self,i,inputSize,xT,steps,condition, text_embed):
        #Euler sampling without CFG
        t=1-i/steps
        dT=1/steps
        batch_size = inputSize[0]
        
        t = torch.full((batch_size, 1), t, device=self.device)
        dT = torch.full((batch_size, 1, 1, 1), dT, device=self.device)
        
        # t=torch.Tensor([t]).reshape((1,1,1,1)).repeat_interleave(inputSize[0],dim=0).to(self.device)
        # dT=torch.Tensor([dT]).reshape((1,1,1,1)).repeat_interleave(inputSize[0],dim=0).to(self.device)

        vPred=self.p(xT,t,condition=condition, text_embed=text_embed)
        vSample=xT-dT*vPred
        return vSample

    def to(self, device):
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.emaModel = self.emaModel.to(self.device)
        return self