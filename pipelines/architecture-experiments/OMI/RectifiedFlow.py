import torch
from tqdm.auto import tqdm 

class RectifiedFlow():
    def __init__(self,model,T,lr=3e-4,device='cpu',optimizer=None,emaStrength=0.0):
        self.model=model
        if emaStrength>0.0:
            self.emaModel=torch.optim.swa_utils.AveragedModel(self.model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(emaStrength))
        else:
            self.emaModel=self.model

        self.T=T
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
    
    def q(self,image,t):
        source=torch.randn_like(image).to(self.device)
        xT=(1-t)*image+t*source
        return xT,source
    
    def p(self,xT,t,condition=None):
        if t.shape[0]==1 and t.shape[0]!=xT.shape[0]:
            t=t.repeat_interleave(xT.shape[0],dim=0)
        if condition is None:
            condition=t
        else:
            if condition.shape[0]==1 and condition.shape[0]!=xT.shape[0]:
                condition=condition.repeat_interleave(xT.shape[0],dim=0)
            condition=torch.cat([t,condition],dim=1)

        if torch.is_inference_mode_enabled():
            vPred=self.emaModel(xT,condition)
        else:
            vPred=self.model(xT,condition)
            
        return vPred

    def call(self,steps,shape=None, condition=None, loopFunction=None):
        #loopFunction is a function with the same signature as vSample which returns the sample at t-1.
        #It can be used to implement CFG and schedulers
        if loopFunction is None:
            loopFunction=self.vSample
        
        xT=torch.randn(shape).to(self.device)
        with torch.no_grad():
            for i in tqdm( range(0,steps),ascii=" ▖▘▝▗▚▞█",disable=False):
                xT=loopFunction(i,shape,xT,steps,condition)
            
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

    def vSample(self,i,inputSize,xT,steps,condition):
        #Euler sampling without CFG
        t=1-i/steps
        dT=1/steps
        t=torch.Tensor([t]).reshape((1,1,1,1)).repeat_interleave(inputSize[0],dim=0).to(self.device)
        dT=torch.Tensor([dT]).reshape((1,1,1,1)).repeat_interleave(inputSize[0],dim=0).to(self.device)

        vPred=self.p(xT,t,condition=condition)
        vSample=xT-dT*vPred
        return vSample
