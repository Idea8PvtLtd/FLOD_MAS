"""Shipped model against the two retrained ones, inside the main branch's own engine, on held-out and other frames."""
import sys,os,glob,pickle,numpy as np,cv2,time,torch,torch.nn.functional as F
M=os.path.expanduser('~/flod_experiments/main_model'); R='/home/admin-123/Documents/GitHub/FLOD_MAS'; V=os.path.expanduser('~/flod_experiments/segformer')
os.chdir(R); sys.path[:0]=[M+'/inspection',R,R+'/tools']
from local import engine as E
from segmentation import ribbon
from scipy.ndimage import gaussian_filter1d
from transformers import SegformerForSemanticSegmentation
eng=E.LiveInspection(M+'/inspection/local/models/strip_unet_resnet34.onnx',M+'/inspection/local/models/segformer_b0.onnx')
shipped=eng.glue; clahe=E.enhance_image
mean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).cuda().half(); std=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).cuda().half()
class Torch:
    def __init__(self,name,W,H): self.model=SegformerForSemanticSegmentation.from_pretrained(f'{M}/retrained_{name}').cuda().half().eval(); self.width,self.height=W,H
    def predict(self,flat):
        x=torch.from_numpy(cv2.cvtColor(cv2.resize(flat,(self.width,self.height),interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)).cuda().permute(2,0,1)[None].half()/255
        with torch.inference_mode(): p=torch.sigmoid(F.interpolate(self.model(pixel_values=(x-mean)/std).logits.float(),size=flat.shape[:2],mode='bilinear',align_corners=False))[0,0].cpu().numpy()
        return np.uint8(p>0.5)*255
variants={'shipped':(shipped,True),'U1 retrained, same format':(Torch('U1',1024,64),True),'U2 retrained, 2048x128, no CLAHE':(Torch('U2',2048,128),False)}
def mid(r):
    r=np.asarray(r); h=len(r)//2; return (r[:h]+r[h:][::-1])/2, np.linalg.norm(r[:h]-r[h:][::-1],axis=1)
def run(im,name):
    g,use=variants[name]; eng.glue=g; E.enhance_image=clahe if use else (lambda image,*a:image)
    out=[];errs=[];t=time.perf_counter()
    for r in eng.run_lines(im,on_error=lambda i,e:errs.append(i)):
        W=np.concatenate([np.array([s['width_px'] for s in c['samples'] if s['valid']]) for c in r['components']]); comp=max(r['components'],key=lambda c:c['length_px']); S=[s for s in comp['samples'] if s['valid']]
        out.append(dict(width=np.array([s['width_px'] for s in S]),all=W,parts=r['component_count'],centre=np.array([[s['center_x'],s['center_y']] for s in S]),left=np.array([[s['left_x'],s['left_y']] for s in S]),right=np.array([[s['right_x'],s['right_y']] for s in S])))
    return out,len(errs),time.perf_counter()-t
res={}
for pk in [l.strip() for l in open(V+'/held.txt')]:
    v=pickle.load(open(pk,'rb')); im=cv2.imread(v['image']); refs=[l for l in v['lines'] if 'backed' in l]
    for name in variants:
        L,ne,dt=run(im,name); e=res.setdefault((name,v['colour']),dict(n=0,ref=0,edge=[],width=[],dw=[],rough=[],seg=[],parts=[],time=[],err=0)); e['ref']+=len(refs); e['err']+=ne; e['time'].append(dt)
        for l in refs:
            cr,wr=mid(l['ring']); nor=ribbon._frame(cr); best=None
            for m in L:
                k,off=ribbon._project(m['centre'],cr,nor); d=np.median(np.abs(off))
                if d<40 and (best is None or d<best[0]): best=(d,m)
            if best is None: continue
            m=best[1]; kl,ol=ribbon._project(m['left'],cr,nor); kr,orr=ribbon._project(m['right'],cr,nor); lo=np.minimum(ol,orr); hi=np.maximum(ol,orr); inner=(kl>20)&(kl<len(cr)-20); ws=m['width']
            e['n']+=1; e['edge'].append(np.mean(np.r_[np.abs(lo+wr[kl]/2)[inner],np.abs(hi-wr[kl]/2)[inner]])); e['width'].append(np.median(ws)); e['dw'].append(np.median(ws)-np.median(wr)); e['rough'].append(np.sqrt(np.mean((ws-gaussian_filter1d(ws,6,mode='nearest'))**2)))
            seg=[np.mean(c) for c in np.array_split(ws,10)]; e['seg'].append(np.max(np.abs(np.array(seg)-np.mean(seg)))); e['parts'].append(m['parts'])
for (name,col),e in sorted(res.items()):
    print('%-34s %-5s lines %2d of %2d | width %.1f px (v5 %+.2f) | edges from v5 %.2f px | rough %.2f px | largest segment off line mean %.2f px (max %.2f) | pieces %.2f | errors %d | %.1f s/frame'%(name,col,e['n'],e['ref'],np.median(e['width']),np.mean(e['dw']),np.mean(e['edge']),np.mean(e['rough']),np.mean(e['seg']),np.max(e['seg']),np.mean(e['parts']),e['err'],np.mean(e['time'])),flush=True)
print('--- frames from the other sets (expected lines: 0041 3, 0045 4, 0049 4, 21_30 4, 21_43 3-4, 21_40 4):')
for f in ('enhancement_images/10_05/0041_frame.png','enhancement_images/10_05/0045_frame.png','enhancement_images/10_05/0049_frame.png','enhancement_images/real_captures/WIN_20261006_21_30_28_Pro.jpg','enhancement_images/real_captures/WIN_20261006_21_43_32_Pro.jpg','enhancement_images/real_captures/WIN_20261006_21_40_20_Pro.jpg'):
    im=cv2.imread(f)
    for name in variants:
        L,ne,dt=run(im,name); print('  %-22s %-34s lines %d errors %d |'%(os.path.basename(f)[:22],name,len(L),ne),' '.join('%.0f(%d,%.0f)'%(np.median(m['all']),m['parts'],np.percentile(m['all'],90)-np.percentile(m['all'],10)) for m in L),flush=True)
print('   (width px (pieces, spread p10-90))')
