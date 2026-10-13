"""SegFormer on held-out frames: its mask as the outline source in place of SAM, against the version 5 ribbons.
   argv: dir, model name (segformer_b0_plain ...), [hard]"""
import sys,glob,pickle,numpy as np,cv2,torch,time,json,os,torch.nn.functional as F
R='/home/admin-123/Documents/GitHub/FLOD_MAS'; os.chdir(R); sys.path.insert(0,R); sys.path.insert(0,R+'/tools')
import pipeline_steps as steps
from segmentation import sam_refine, ribbon
from transformers import SegformerForSemanticSegmentation
D,name=sys.argv[1],sys.argv[2]; use_clahe=name.endswith('clahe'); dev='cuda'
model=SegformerForSemanticSegmentation.from_pretrained(f'{D}/{name}').to(dev).half().eval()
mean=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1).half(); std=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1).half()
T,STR=512,384
def clahe(im):
    lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB); lab[:,:,0]=cv2.createCLAHE(3.0,(16,16)).apply(lab[:,:,0]); return cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
def predict(im,roi):
    src=clahe(im) if use_clahe else im
    x,y,w,h=cv2.boundingRect((roi>0).astype(np.uint8)); x0,y0=max(0,x-64),max(0,y-64); x1,y1=min(im.shape[1],x+w+64),min(im.shape[0],y+h+64)
    xs=list(range(x0,max(x0+1,x1-T),STR))+[max(x0,x1-T)]; ys=list(range(y0,max(y0+1,y1-T),STR))+[max(y0,y1-T)]
    tiles=[(a,b) for b in ys for a in xs if roi[b:b+T:16,a:a+T:16].any()]
    prob=np.zeros(im.shape[:2],np.float32); cnt=np.zeros(im.shape[:2],np.float32)
    with torch.inference_mode():
        for i in range(0,len(tiles),8):
            batch=tiles[i:i+8]; xb=torch.from_numpy(np.stack([src[b:b+T,a:a+T] for a,b in batch])).to(dev).permute(0,3,1,2).flip(1).half()/255
            p=torch.sigmoid(F.interpolate(model(pixel_values=(xb-mean)/std).logits.float(),size=(T,T),mode='bilinear',align_corners=False))[:,0].cpu().numpy()
            for (a,b),q in zip(batch,p): prob[b:b+T,a:a+T]+=q; cnt[b:b+T,a:a+T]+=1
    return prob/np.maximum(cnt,1),len(tiles)
def mid(r):
    r=np.asarray(r); h=len(r)//2; return (r[:h]+r[h:][::-1])/2, np.linalg.norm(r[:h]-r[h:][::-1],axis=1)
def lines_from_mask(mask,rough):
    out=[]
    for r in rough:
        c=steps._centre(r['ring']); ev={}; t=sam_refine.trim_ring(mask,c,details=ev)
        out.append(dict(r,source='local') if t is None else {'ring':t[0],'source':'segformer','seen':t[1],'evidence':ev,'rough':c,'confidence':1.0})
    return out
def long_parts(mask):
    n,lab,st,_=cv2.connectedComponentsWithStats(mask); return int(sum(1 for k in range(1,n) if max(st[k,2],st[k,3])>=400))
if len(sys.argv)>3:
    for f in ['enhancement_images/10_05/0041_frame.png','enhancement_images/10_05/0045_frame.png','enhancement_images/10_05/0049_frame.png','enhancement_images/real_captures/WIN_20261006_21_30_28_Pro.jpg','enhancement_images/real_captures/WIN_20261006_21_40_20_Pro.jpg','enhancement_images/real_captures/WIN_20261006_21_43_32_Pro.jpg']:
        im=cv2.imread(f); roi=steps.model_roi(im,on_gpu=True); rough=steps.segment_local(im,roi,True)
        prob,n=predict(im,roi); mask=((prob>0.5)&(roi>0)).astype(np.uint8); L=steps.finish_as_ribbon(im,roi,lines_from_mask(mask,rough),edge_method='trees')
        print(os.path.basename(f),'| mask parts',long_parts(mask),'|',[('%.1f px, mask %.0f%%, backed %.0f%%, misfit %.1f%s'%(steps.ring_stats(l['ring'])['width_px'],100*l.get('seen',0),100*l.get('backed',0),l.get('misfit',float('nan')),' CHECK' if l.get('doubtful') else '')) for l in L],flush=True)
    sys.exit()
res={}; predict(np.zeros((1024,1024,3),np.uint8),np.ones((1024,1024),np.uint8))
for pk in [l.strip() for l in open(f'{D}/held.txt')]:
    d=pickle.load(open(pk,'rb')); im=cv2.imread(d['image']); roi=np.unpackbits(d['roi'])[:d['shape'][0]*d['shape'][1]].reshape(d['shape'])
    torch.cuda.synchronize(); t=time.perf_counter(); prob,ntiles=predict(im,roi); torch.cuda.synchronize(); tp=time.perf_counter()-t
    mask=((prob>0.5)&(roi>0)).astype(np.uint8); rings=[np.asarray(l['ring']) for l in d['lines'] if 'backed' in l]
    label=np.zeros(im.shape[:2],np.uint8)
    for ring in rings: cv2.fillPoly(label,[np.round(ring).astype(np.int32)],1)
    L=lines_from_mask(mask,d['rough']); N=steps.finish_as_ribbon(im,roi,[dict(l) for l in L],edge_method='trees')
    e=res.setdefault(d['colour'],dict(iou=[],seen=[],edge=[],centre=[],width=[],check=0,dev=0,lines=0,nomask=0,time=[],tiles=[],stray=[],parts=0,frames=0))
    e['iou'].append((mask&label).sum()/max((mask|label).sum(),1)); e['time'].append(tp); e['tiles'].append(ntiles); e['parts']+=long_parts(mask); e['frames']+=1
    e['stray'].append(float((mask.astype(bool)&~cv2.dilate(label,np.ones((61,61),np.uint8)).astype(bool)).sum())/max(label.sum(),1))
    for l,n,ref in zip(L,N,[x for x in d['lines']]):
        if 'backed' not in ref: continue
        e['lines']+=1; cr,wr=mid(ref['ring']); nor=ribbon._frame(cr)
        if not l.get('evidence'): e['nomask']+=1; continue
        e['seen'].append(l['seen']); ev=l['evidence']; dd=[]
        for edge,sign in ((ev['low'],-1),(ev['high'],1)):
            k,off=ribbon._project(ev['points']+ev['normals']*edge[:,None],cr,nor); inner=(k>5)&(k<len(cr)-5); dd.append(np.abs(off-sign*wr[k]/2)[inner])
        e['edge'].append(float(np.mean(np.concatenate(dd))))
        if 'backed' in n:
            cn,wn=mid(n['ring']); k,off=ribbon._project(cn,cr,nor); inner=(k>20)&(k<len(cr)-20); e['centre'].append(float(np.median(np.abs(off[inner])))); e['width'].append(float(np.median(wn)-np.median(wr))); e['check']+=bool(n.get('doubtful')); e['dev']+=len(n['deviations'])
for col,e in res.items():
    print('%-22s %-5s lines %2d (no mask %d; mask parts %d) | IoU with v5 %.3f | stray %.3f | mask covers %.2f of line (min %.2f) | mask edges from v5 %.2f px (worst %.1f) | final ribbon vs v5: centre %.2f (worst %.1f), width %+.2f px | check %d, false deviations %d | %.2f s/frame, %.0f tiles'%(name,col,e['lines'],e['nomask'],e['parts'],np.mean(e['iou']),np.mean(e['stray']),np.mean(e['seen']),min(e['seen']),np.mean(e['edge']),max(e['edge']),np.mean(e['centre']),max(e['centre']),np.mean(e['width']),e['check'],e['dev'],np.mean(e['time']),np.mean(e['tiles'])),flush=True)
print(name,'peak gpu MB %d'%(torch.cuda.max_memory_allocated()/1e6))
json.dump({c:{k:(v if not isinstance(v,list) else [float(x) for x in v]) for k,v in e.items()} for c,e in res.items()},open(f'{D}/eval_{name}.json','w'))
