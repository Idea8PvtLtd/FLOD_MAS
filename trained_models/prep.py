"""Tiles and labels for SegFormer: native-resolution 512 px tiles along each glue line, label = version 5 ribbon.
Frames are split per fabric colour by capture time (tools/train_bead_ml.session_split), so repeat shots stay on one side."""
import sys,glob,pickle,numpy as np,cv2,os
R='/home/admin-123/Documents/GitHub/FLOD_MAS'; os.chdir(R); sys.path.insert(0,R); sys.path.insert(0,R+'/tools')
import train_bead_ml
D=sys.argv[1]; out=f'{D}/data'; os.makedirs(out,exist_ok=True)
train,held=train_bead_ml.session_split(sorted(glob.glob(f'{D}/runs/*.pkl')),0.3); held=set(held)
open(f'{D}/held.txt','w').write('\n'.join(sorted(held))+'\n')
rng=np.random.default_rng(5); T=512; n=0; cols={}
def clahe(im):
    lab=cv2.cvtColor(im,cv2.COLOR_BGR2LAB); lab[:,:,0]=cv2.createCLAHE(3.0,(16,16)).apply(lab[:,:,0]); return cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
for pk in sorted(glob.glob(f'{D}/runs/*.pkl')):
    d=pickle.load(open(pk,'rb')); part='held' if pk in held else 'train'; cols[(part,d['colour'])]=cols.get((part,d['colour']),0)+1
    if part=='held': continue
    im=cv2.imread(d['image']); roi=np.unpackbits(d['roi'])[:d['shape'][0]*d['shape'][1]].reshape(d['shape'])
    label=np.zeros(im.shape[:2],np.uint8); rings=[np.asarray(l['ring']) for l in d['lines'] if 'backed' in l]
    for ring in rings: cv2.fillPoly(label,[np.round(ring).astype(np.int32)],1)
    imc=clahe(im); tiles=[]
    for ring in rings:
        h=len(ring)//2; cen=(ring[:h]+ring[h:][::-1])/2
        for k in np.linspace(0,len(cen)-1,10).astype(int):
            tiles.append((int(np.clip(cen[k][0]+rng.integers(-170,171)-T//2,0,im.shape[1]-T)),int(np.clip(cen[k][1]+rng.integers(-170,171)-T//2,0,im.shape[0]-T))))
    ys,xs=np.nonzero(roi[::64,::64])
    for j in rng.choice(len(ys),min(6,len(ys)),replace=False):
        tiles.append((int(np.clip(xs[j]*64-T//2,0,im.shape[1]-T)),int(np.clip(ys[j]*64-T//2,0,im.shape[0]-T))))
    np.savez_compressed(f"{out}/train_{d['frame']}.npz",x=np.stack([im[y:y+T,x:x+T] for x,y in tiles]),xc=np.stack([imc[y:y+T,x:x+T] for x,y in tiles]),y=np.stack([label[y:y+T,x:x+T] for x,y in tiles])); n+=len(tiles)
print('tiles',n,'frames',cols)
