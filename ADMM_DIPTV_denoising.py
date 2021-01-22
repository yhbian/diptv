#!/usr/bin/env python
# coding: utf-8

# This is the code of our **ADMM-DIPTV** paper. 
# - Select `name` below to switch between the images.
# - Change the `weight` depending on the noise.
# 

# In[ ]:


# Mounting my Google Drive and set the cwd on my imagefolder
# from google.colab import drive
# drive.mount('/content/drive/', force_remount=True)
# get_ipython().run_line_magic('cd', '/content/drive/My Drive/ADMM-DIPTV')



from __future__ import print_function
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
from models import *

import torch
import torch.optim

from skimage.measure import compare_psnr
from utils.denoising_utils import *
from utils.sr_utils2 import *
from utils.utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
dtype = torch.cuda.FloatTensor

imsize =-1
PLOT = True
sigma = 20 #70
sigma_ = sigma/255.


# In[ ]:


name = 'I08.png'
fname = f"data2/denoising/%s"%(name)


# # Load image

# In[ ]:


# Add synthetic noise
img_pil = crop_image(get_image(fname, imsize)[0], d=32)
img_np = pil_to_np(img_pil)    
    
# pay attention!
if img_np.shape[0]!=3:
  img_np_temp = np.zeros((3,img_np.shape[1], img_np.shape[2]))
  img_np_temp[0,:,:] = img_np[0,:,:]  
  img_np_temp[1,:,:] = img_np[0,:,:]
  img_np_temp[2,:,:] = img_np[0,:,:]
  img_np = img_np_temp
    
img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)

if PLOT:
    plot_image_grid([img_np, img_noisy_np], 4, 6);


print(img_np.shape)


# # Setup

# In[ ]:


INPUT = 'noise' 
pad = 'reflection'
OPT_OVER = 'net' 

reg_noise_std = 1./20. # set to 1./20. for sigma=50
LR = 0.01 

OPTIMIZER='adam' 
show_every = 200
exp_weight=0.99

num_iter = 2600
input_depth = 3 
figsize = 1 
    
    
net = get_net(input_depth, 'skip', pad,
              skip_n33d=128, 
              skip_n33u=128, 
              skip_n11=4, 
              num_scales=5,
              upsample_mode='bilinear').type(dtype)
    
net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

# Compute number of parameters
s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
print ('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)


# # Optimize

# In[ ]:


size = img_np.shape
h = size[-2]
w = size[-1]
Dh_psf = np.array([ [0, 0, 0], [1, -1, 0], [0, 0, 0]])
Dv_psf = np.array([ [0, 1, 0], [0, -1, 0], [0, 0, 0]])
Id_psf = np.array([[1]])

Id_DFT = torch.from_numpy(psf2otf(Id_psf, [h,w])).cuda()
Dh_DFT = torch.from_numpy(psf2otf(Dh_psf, [h,w])).cuda()
Dv_DFT = torch.from_numpy(psf2otf(Dv_psf, [h,w])).cuda()

DhT_DFT = torch.conj(Dh_DFT)
DvT_DFT = torch.conj(Dv_DFT)


# In[ ]:


signal_ndim = 2
def D2(x, Dh_DFT, Dv_DFT):
    x_DFT = torch.rfft(x, signal_ndim=signal_ndim, onesided=False)
    x_DFT = torch.view_as_complex(x_DFT).cuda()
    Dh_x = torch.irfft(torch.view_as_real(Dh_DFT*x_DFT), signal_ndim=signal_ndim, onesided=False)
    Dv_x = torch.irfft(torch.view_as_real(Dv_DFT*x_DFT), signal_ndim=signal_ndim, onesided=False)
    return Dh_x.detach().clone(), Dv_x.detach().clone()


# In[ ]:


net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
out_avg = None
last_net = None
psrn_noisy_last = 0

img_noisy_torch = np_to_torch(img_noisy_np).type(dtype) 
u = 0*img_noisy_torch.detach().clone()
t_h = 0*img_noisy_torch.detach().clone()
t_v = 0*img_noisy_torch.detach().clone()

mu_f = torch.zeros(net_input.shape, device=0)
mu_t_h = torch.zeros(net_input.shape, device=0)
mu_t_v = torch.zeros(net_input.shape, device=0)

beta_t = 1.
weight = 0.001

loss_values = []
fun_values = []
psnr_values = []
running_loss=0

i = 0

def closure():
    
    global i, out_avg, psrn_noisy_last, last_net, net_input, mu_t_h, mu_t_v, u,beta_t,weight,t_h,t_v, Dh_out, Dv_out
    
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
    
    #First problem
    out = net(net_input)
    if i==0:
      [Dh_out, Dv_out] = D2(out, Dh_DFT, Dv_DFT)

    total_loss = norm2_loss(out-img_noisy_torch)
    total_loss += (beta_t/2)*norm2_loss(Dh_out-(t_h-mu_t_h).detach()) + (beta_t/2)*norm2_loss(Dv_out-(t_v-mu_t_v).detach())

    total_loss.backward(retain_graph=False,create_graph=False)
    running_loss = total_loss.item()
    loss_values.append(running_loss)

    out = net(net_input)
  
    [Dh_out, Dv_out] = D2(out, Dh_DFT, Dv_DFT)

    #TV problem: second problem 
    q_h                 = Dh_out + mu_t_h
    q_v                 = Dv_out + mu_t_v
    q_norm              = torch.sqrt(torch.pow(q_h,2) + torch.pow(q_v,2))
    q_norm[q_norm == 0] = weight/beta_t
    q_norm              = torch.clamp( q_norm - weight/beta_t , min=0 )/q_norm
    t_h                 = (q_norm * q_h).detach().clone()
    t_v                 = (q_norm * q_v).detach().clone()

    #Ascent step: updating lagrangian parameter
    mu_t_h = (mu_t_h + (Dh_out- t_h)).detach().clone()
    mu_t_v = (mu_t_v + (Dh_out- t_v)).detach().clone()

    # Smoothing
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
    
    psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0]) 
    psrn_gt    = compare_psnr(img_np, out.detach().cpu().numpy()[0]) 
    psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0]) 

    psnr_values.append(psrn_gt)
    
    # Note that we do not have GT for the "snail" example. So 'PSRN_gt', 'PSNR_gt_sm' make no sense
    print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
    if  PLOT and i % show_every == 0:
        out_np = torch_to_np(out)
        plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=figsize)
            
    i += 1


    return running_loss,loss_values,psnr_values

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)


# In[ ]:


out_np = torch_to_np(net(net_input))
plot_image_grid([np.clip(out_np, 0, 1), img_np,img_noisy_np], factor=18);


# In[ ]:


plt.plot(range(len(psnr_values)), psnr_values)
plt.show()
print(np.max(psnr_values))
print(psnr_values[-1])

plt.plot(range(len(loss_values)), loss_values)
plt.show()


# In[ ]:


from PIL import Image

#Saving output image

folder = "ADMM_DIPTV"
path = f"results/%s/%s_%s_%d"%(folder, folder, name[:-4], sigma)

out_np=255*out_np
print(np.amax(out_np))
u=out_np.transpose(1, 2, 0)
np.savetxt(f'%s.txt'%path,u[:,:,0])

u=u.astype(np.uint8)
print(u.shape)

im = Image.fromarray(u)
im.save(f'%s.png'%path)

