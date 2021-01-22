#!/usr/bin/env python
# coding: utf-8

# Code for **"Blind image denoising"** figures. Select `name` below to switch between the images.
# 
# - To see overfitting set `num_iter` to a large value.
# - To use plain [DIP](https://arxiv.org/abs/1711.10925) set `tv_weight=0`
# - To use [DIPTV](https://arxiv.org/abs/1810.12864) set `tv_weight=0.0000001`

# In[ ]:


# Mounting my Google Drive and set the cwd on my imagefolder
# from google.colab import drive
# drive.mount('/content/drive/', force_remount=True)
# get_ipython().run_line_magic('cd', '/content/drive/My Drive/ADMM-DIPTV')


# # Import libs

# In[ ]:


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

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize =-1
PLOT = True
sigma = 30
sigma_ = sigma/255.
tv_weight=0   #0.0000001


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


INPUT = 'noise' # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net' # 'net,input'

reg_noise_std = 1./30. # set to 1./20. for sigma=50
LR = 0.01#0.01

OPTIMIZER='adam' # 'LBFGS'
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

img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)


# # Optimize

# In[ ]:


net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
out_avg = None
last_net = None
psrn_noisy_last = 0

loss_values = []
psnr_values = []
running_loss=0

i = 0
def closure():
    
    global i, out_avg, psrn_noisy_last, last_net, net_input
    
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
    
    out = net(net_input)
    
    # Smoothing
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
            
    total_loss = mse(out, img_noisy_torch)
    
    if tv_weight > 0:
        total_loss += tv_weight * tvalfa_loss(out)

    total_loss.backward()
    running_loss = total_loss.item()
    loss_values.append(running_loss)    
    
    psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0]) 
    psrn_gt    = compare_psnr(img_np, out.detach().cpu().numpy()[0]) 
    psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0]) 

    psnr_values.append(psrn_gt)
    
    # Note that we do not have GT for the "snail" example
    # So 'PSRN_gt', 'PSNR_gt_sm' make no sense
    print ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')
    if  PLOT and i % show_every == 0:
        out_np = torch_to_np(out)
        #plot_image_grid([np.clip(out_np, 0, 1), 
        #                 np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1)
        plot_image_grid([np.clip(out_np, 0, 1)], factor=figsize, nrow=1)
        
        
    
    # Backtracking
    if i % show_every:
        if psrn_noisy - psrn_noisy_last < -5: 
            print('Falling back to previous checkpoint.')

            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())

            return total_loss*0
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psrn_noisy_last = psrn_noisy
            
    i += 1
    #print(i)

    return total_loss,loss_values,psnr_values

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

folder = "DIP"+"TV"*(tv_weight!=0)
path = f"results/%s/%s_%s_%d"%(folder, folder, name[:-4], sigma)

out_np=255*out_np
print(np.amax(out_np))
u=out_np.transpose(1, 2, 0)
np.savetxt(f'%s.txt'%path,u[:,:,0])

u=u.astype(np.uint8)
print(u.shape)

im = Image.fromarray(u)
im.save(f'%s.png'%path)

