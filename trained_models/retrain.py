"""Retrain a SegFormer-B0 in the main branch's own format (unfolded inward band) on the version 5 ribbons, two ways:
 U1: as the shipped model is fed (CLAHE, band squeezed to 1024x64);
 U2: per the findings (no CLAHE, band at twice the size, 2048x128, so the strip keeps most of its pixels).
argv: gen | train U1|U2 | eval"""
import sys,os,glob,pickle,numpy as np,cv2,time
M=os.path.expanduser('~/flod_experiments/main_model'); R='/home/admin-123/Documents/GitHub/FLOD_MAS'; V=os.path.expanduser('~/flod_experiments/segformer')
os.chdir(R); sys.path[:0]=[M+'/inspection',R,R+'/tools']
from local import glue_model
from local.engine import LiveInspection
from local.geometry import (extract_lines, filter_wave_lines, offset_lines, region_mask, original_dimensions, unfold_maps, enhance_image)
SIZE={'U1':(1024,64,True),'U2':(2048,128,False)}
held={l.strip() for l in open(V+'/held.txt')}
def bands(eng,photo):
    """The engine's own unfolding: for each source line, its maps and the band the model is given."""
    source=eng.source.predict(photo); chains=extract_lines(source,hook_zone=.04,hook_angle=65); waves=np.zeros_like(source)
    for chain in chains: cv2.polylines(waves,[chain],False,255,1,cv2.LINE_8)
    filtered,_,_=filter_wave_lines(waves,source,.008,.1,.9,make_preview=False)
    _,_,lines=offset_lines(filtered,eng.offset,'inside',source_mask=source,make_preview=False)
    for line in lines:
        w,h=original_dimensions(line,eng.offset); mx,my,_=unfold_maps(line,w,h,'horizontal'); yield line,mx,my,cv2.remap(region_mask(source.shape,[line]),mx,my,cv2.INTER_NEAREST)
mode=sys.argv[1]
if mode=='gen':
    eng=LiveInspection(M+'/inspection/local/models/strip_unet_resnet34.onnx',M+'/inspection/local/models/segformer_b0.onnx'); X={k:[] for k in SIZE}; Y={k:[] for k in SIZE}
    for pk in sorted(glob.glob(V+'/runs/*.pkl')):
        if pk in held: continue
        v=pickle.load(open(pk,'rb')); im=cv2.imread(v['image']); label=np.zeros(im.shape[:2],np.uint8)
        for l in v['lines']:
            if 'backed' in l: cv2.fillPoly(label,[np.round(np.asarray(l['ring'])).astype(np.int32)],255)
        enh=enhance_image(im,2.,16,16)
        for line,mx,my,allowed in bands(eng,im):
            lab=cv2.remap(label,mx,my,cv2.INTER_LINEAR)
            if (lab>127).mean()<0.03: continue                       # a band with no outlined glue in it
            for k,(W,H,use) in SIZE.items():
                flat=cv2.remap(enh if use else im,mx,my,cv2.INTER_LINEAR); flat[allowed==0]=0
                X[k].append(cv2.resize(flat,(W,H),interpolation=cv2.INTER_LINEAR)); Y[k].append(np.uint8(cv2.resize(lab,(W,H),interpolation=cv2.INTER_LINEAR)>127))
    for k in SIZE: np.savez_compressed(f'{M}/bands_{k}.npz',x=np.stack(X[k]),y=np.stack(Y[k])); print(k,np.stack(X[k]).shape,'glue share %.3f'%np.stack(Y[k]).mean())
elif mode=='train':
    import torch,torch.nn.functional as F
    from transformers import SegformerForSemanticSegmentation
    k=sys.argv[2]; z=np.load(f'{M}/bands_{k}.npz'); X,Y=z['x'],z['y']; dev='cuda'
    model=SegformerForSemanticSegmentation.from_pretrained('nvidia/mit-b0',num_labels=1).to(dev)
    mean=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1); std=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)
    head=[p for n,p in model.named_parameters() if 'decode_head' in n]; body=[p for n,p in model.named_parameters() if 'decode_head' not in n]
    iters=3000; B=8 if k=='U1' else 4; opt=torch.optim.AdamW([{'params':body,'lr':6e-5},{'params':head,'lr':6e-4}],weight_decay=0.01); sched=torch.optim.lr_scheduler.PolynomialLR(opt,total_iters=iters,power=1.0); scaler=torch.amp.GradScaler(); rng=np.random.default_rng(0); t0=time.time(); model.train()
    for it in range(1,iters+1):
        idx=rng.integers(0,len(X),B); x=torch.from_numpy(X[idx]).to(dev).permute(0,3,1,2).flip(1).float()/255; y=torch.from_numpy(Y[idx]).to(dev).float().unsqueeze(1)
        if rng.random()<0.5: x,y=x.flip(3),y.flip(3)                 # along the line only: the band's top is always the source edge
        blank=(x.sum(1,keepdim=True)==0); x=(x*(1+0.2*(rng.random()-0.5))+0.1*(rng.random()-0.5)).clamp(0,1)*(~blank)
        with torch.autocast('cuda',dtype=torch.float16): logits=F.interpolate(model(pixel_values=(x-mean)/std).logits,size=y.shape[-2:],mode='bilinear',align_corners=False)
        logits=logits.float(); p=torch.sigmoid(logits); loss=F.binary_cross_entropy_with_logits(logits,y)+1-(2*(p*y).sum()+1)/(p.sum()+y.sum()+1)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if it%1000==0: print(k,'it',it,'loss %.4f %.0f s'%(loss.item(),time.time()-t0),flush=True)
    model.save_pretrained(f'{M}/retrained_{k}'); print('saved',k,'bands',len(X),'minutes %.1f'%((time.time()-t0)/60))
