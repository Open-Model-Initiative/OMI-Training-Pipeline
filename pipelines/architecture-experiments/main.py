#%%
import torch
import torchinfo
from OMI.RectifiedFlow import RectifiedFlow
from OMI.dit import DiT
from OMI.utils import RunningMean

device='cuda'
torch.set_float32_matmul_precision('high')

#%%
def main():
    model=DiT(384,8,3,3,1,8,2)
    print(torchinfo.summary(model, input_size=[(1,3,128,128),(1,1,1,1)]))

    modelT=RectifiedFlow(model,1000,device=device,lr=1e-4,emaStrength=0.999)

    #%%    
    from torch.utils.tensorboard import SummaryWriter
    import datetime
    logger=SummaryWriter(f'./omitest/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    #%%
    dataset=...
    dataloader=...
    #%%
    #Load weights
    #checkpoint = torch.load('./checkpoint.pt',map_location='cpu',weights_only=True)
    #modelT.load_state_dict(checkpoint)
    #%%
    step=-1
    while True:
        #Training
        loss=RunningMean()
        for _,image in dataloader:
            image=torch.nn.functional.interpolate(image,(128,128))
            metrics=modelT.train_step(image)
            loss.update(metrics['loss'])
            
            step+=1
            print(f"Step {step}",end='\r')
        
            if step%100==0:
                break

        #Logging
        logger.add_scalar(f'Generator Loss', loss.value, step)
        with torch.inference_mode(), torch.amp.autocast('cuda',dtype=torch.float32):
            pred=modelT.call(steps=20,shape=(1,3,128,128))
            for i in range(1):
                logger.add_image(f'Generated Image/{i}', pred[i].float(), step)
                logger.add_image(f'Real Image/{i}', image[i], step)
            logger.flush()

        #Save all models
        #torch.save(modelT.state_dict(),'./checkpoint.pt')

if __name__ == '__main__':
    main()