import sys
sys.path.append(r'C:\Users\chuya\.gemini\antigravity\scratch\plasmon_analyzer')
import numpy as np
import cv2
import scipy.ndimage as ndi

p_pre=r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260704 sam 位置合わせ test\df\1-1-0.tif'
p_post=r'G:\マイドライブ\1.実験データ_gdrive\4.生データ\4.生データ D\260704 sam 位置合わせ test\df\1-1-1.tif'

img0=cv2.imdecode(np.fromfile(p_pre,dtype=np.uint8),-1).astype(np.float32)
img1=cv2.imdecode(np.fromfile(p_post,dtype=np.uint8),-1).astype(np.float32)
rows,cols=img0.shape

f=np.fft.fft2(img0-np.nanmean(img0))
fshift=np.fft.fftshift(f)
crow,ccol=rows//2,cols//2
y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
r=np.sqrt(x**2 + y**2)
mask=(r>=rows/6.29-15)&(r<=rows/6.29+15)
img_filtered=np.fft.ifft2(np.fft.ifftshift(fshift*mask))
local_max=ndi.maximum_filter(img_filtered.real,size=5)==img_filtered.real
margin=30
valid_border=np.zeros_like(img0,dtype=bool)
valid_border[margin:-margin,margin:-margin]=True
gy,gx=np.where(local_max&valid_border)

def get_int(img, gy, gx):
    pad=np.pad(img,1,mode='constant',constant_values=np.nan)
    sum_int=np.zeros(len(gy),dtype=np.float32)
    for dy_ in [-1,0,1]:
        for dx_ in [-1,0,1]:
            sum_int+=pad[gy+1+dy_,gx+1+dx_]
    return sum_int

for dx in [5, 6]:
    for dy in [-20, -21]:
        valid=(gy+dy>=margin)&(gy+dy<rows-margin)&(gx+dx>=margin)&(gx+dx<cols-margin)
        int0=get_int(img0, gy[valid], gx[valid])
        int1=get_int(img1, gy[valid]+dy, gx[valid]+dx)
        print(f'dx={dx}, dy={dy} | Corr: {np.corrcoef(int0, int1)[0,1]:.4f}')
