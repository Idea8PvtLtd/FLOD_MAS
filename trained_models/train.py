"""Fine-tune SegFormer (MiT encoder + all-MLP head) to mark the glue band. argv: dir, size (b0|b2), clahe (0|1), iterations."""
import sys,glob,numpy as np,torch,time,torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
D,size,use_clahe,iters=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
X=[];Y=[]
for f in sorted(glob.glob(f'{D}/data/train_*.npz')):
    z=np.load(f); X.append(z['xc' if use_clahe else 'x']); Y.append(z['y'])
X=np.concatenate(X); Y=np.concatenate(Y); print('tiles',X.shape,'glue share %.3f'%Y.mean(),flush=True)
dev='cuda'; model=SegformerForSemanticSegmentation.from_pretrained(f'nvidia/mit-{size}',num_labels=1).to(dev)
mean=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1); std=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)
head=[p for n,p in model.named_parameters() if 'decode_head' in n]; body=[p for n,p in model.named_parameters() if 'decode_head' not in n]
opt=torch.optim.AdamW([{'params':body,'lr':6e-5},{'params':head,'lr':6e-4}],weight_decay=0.01)
sched=torch.optim.lr_scheduler.PolynomialLR(opt,total_iters=iters,power=1.0); scaler=torch.amp.GradScaler()
rng=np.random.default_rng(0); B=int(sys.argv[5]) if len(sys.argv)>5 else 6; t0=time.time(); run=0.0
model.train()
for it in range(1,iters+1):
    idx=rng.integers(0,len(X),B); x=torch.from_numpy(X[idx]).to(dev).permute(0,3,1,2).flip(1).float()/255; y=torch.from_numpy(Y[idx]).to(dev).float().unsqueeze(1)
    if rng.random()<0.5: x,y=x.flip(3),y.flip(3)
    if rng.random()<0.5: x,y=x.flip(2),y.flip(2)
    if rng.random()<0.5: x,y=x.transpose(2,3),y.transpose(2,3)
    x=(x*(1+0.2*(rng.random()-0.5))+0.1*(rng.random()-0.5)).clamp(0,1)
    with torch.autocast('cuda',dtype=torch.float16):
        logits=F.interpolate(model(pixel_values=(x-mean)/std).logits,size=y.shape[-2:],mode='bilinear',align_corners=False)
    logits=logits.float(); p=torch.sigmoid(logits)
    loss=F.binary_cross_entropy_with_logits(logits,y)+1-(2*(p*y).sum()+1)/(p.sum()+y.sum()+1)
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); run=0.98*run+0.02*loss.item() if it>1 else loss.item()
    if it%500==0: print('it %d loss %.4f  %.0f s'%(it,run,time.time()-t0),flush=True)
name=f'{D}/segformer_{size}_{"clahe" if use_clahe else "plain"}'; model.save_pretrained(name)
print('saved',name,'| minutes %.1f | peak gpu MB %d | parameters M %.1f'%((time.time()-t0)/60,torch.cuda.max_memory_allocated()/1e6,sum(p.numel() for p in model.parameters())/1e6),flush=True)
