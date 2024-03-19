import configparser

### Notebook for P3D stuff!

import jax
jax.config.update('jax_platform_name', 'cpu')
from jax.lib import xla_bridge

print(xla_bridge.get_backend().platform)

import numpy as np
import jax.numpy as jnp
import numpy

from helper_functions import *

#First things... set the redshift

z= 2.0

#file name
prefix = "A_DESI_TEST512_"
loc = "/pscratch/sd/b/bhorowit/configs/"

#set box geometry

bs =  512#box size in Mpc/h
nc =  128#number of pixels per side

ptcl_grid_shape = (nc,) * 3
ptcl_spacing = bs/nc

#survey geometry for mock catalog

n_skewers = int(1500*33) #factor of 33 to go from PFS to DESI
include_dla = True
model_dla = True
snr = 2 #will draw from a distribution eventually...
buff = 5 #buffer near edge of volume

aniso_radial = True #right now just linear, can make more fancy
aniso_angular = True 
# 

kvec = rfftnfreq_2d(ptcl_grid_shape, ptcl_spacing)
k = jnp.sqrt(sum(k**2 for k in kvec))


# lengths of skewers
if aniso_radial:
    leo_skew = np.random.rand(n_skewers)*512
else:
    leo_skew = np.ones(n_skewers)*512

#positions randomly assigned 
pos = np.random.rand(n_skewers,2)

xx = np.array(np.random.rand(n_skewers)*(nc-2*buff),dtype=float)+buff
yy = np.array(np.random.rand(n_skewers)*(nc-2*buff),dtype=float) + buff
zz = np.arange(0,bs)/(bs/nc) #max length of skewers

#
if aniso_angular: #some random permutations of skewer positions...

    vals_to_offset_stripe = np.logical_and(xx>nc*1/2,xx<nc*2/3)

    xx[vals_to_offset_stripe] *= 1.5
    xx[vals_to_offset_stripe] -= nc*2/3

    #bright star removal

    vals_to_offset_circle_a = ((xx-(nc-buff))**2 + (yy-(nc-buff))**2 <(nc/5)**2 )

    xx[vals_to_offset_circle_a] = np.random.rand(np.sum(vals_to_offset_circle_a))*5+nc/3
    yy[vals_to_offset_circle_a] = np.random.rand(np.sum(vals_to_offset_circle_a))*5+nc/3


    #bright star removal
    vals_to_offset_circle = ((xx-nc/4)**2 + (yy-nc/2)**2 <(nc/8)**2 )

    xx[vals_to_offset_circle] = np.random.rand(np.sum(vals_to_offset_circle))*10+buff+nc/5
    yy[vals_to_offset_circle] = np.random.rand(np.sum(vals_to_offset_circle))*10+buff



pos = np.vstack([xx,yy])
skewers_pos = []

#random length of skewer
leo_skew = np.array(np.random.rand(n_skewers)*bs,dtype=int)

#S/N per skewer..

#apparently field has std of ~10... presumably some rescaled flux of some sort


def gen_noise(n_skewers,snr_min,snr_max,alpha = 2.8,):
        #generates noise according to specified distribution
        snr = numpy.minimum(snr_min/numpy.random.power(alpha-1,size=n_skewers),snr_max)
        skewers_noise = (1.0/snr[:,numpy.newaxis])
        return skewers_noise

skn = gen_noise(n_skewers,2,10)

#we'll mask the DLA in noise, as opposed to removing them in the data

skewers_pos = []
skewers_skn = [] #skewer noise 
skewers_dla = [] #flag for DLA
skewers_index = []


for nn,i in enumerate(pos.T):
    zzz = zz.reshape(1,-1).T[:leo_skew[nn],:]
    dla_mask = np.zeros(leo_skew[nn])
    if np.random.rand()<0.1:
        if len(zzz)<50:
            print("not masked due to too short")
        else:
            mask_loc = int(np.random.rand()*len(zzz))
            masker = (mask_loc+np.arange(0,10))<len(zzz)
            dla_mask[mask_loc:mask_loc+10]=1.0
        
    skewers_dla.append(dla_mask)
    skewers_skn.append(skn[nn]*np.ones((leo_skew[nn])))
    skewers_pos.append(np.hstack([i*np.ones((leo_skew[nn],2)),zzz]))
    skewers_index.append(nn*np.ones((leo_skew[nn])))

#final skewer data positions
skewers_fin = np.array(np.vstack(skewers_pos),dtype=float)
skewers_skn = np.array(np.hstack(skewers_skn))
skewers_dla = np.array(np.hstack(skewers_dla))
skewers_index = np.array(np.hstack(skewers_index),dtype=int)

if model_dla:
    eff_noise = skewers_skn+skewers_dla*10000 #set noise in DLA very high
else: 
    eff_noise = skewers_skn
    

#this is some preprocessing of the cic routine to save the indexes that need to be readout
naa, kernel = cic_preprocess(skewers_fin,nc)

if True: #output files...
#prefix
    np.save(loc+prefix+"naa",naa)
    np.save(loc+prefix+"kernel",kernel)
    np.save(loc+prefix+"skewers_skn",skewers_skn)
    np.save(loc+prefix+"skewers_dla",skewers_dla)
    np.save(loc+prefix+"skewers_fin",skewers_fin)

#save config

config = configparser.ConfigParser()
config['basic'] = {'redshift': z,
                     'prefix': prefix,
                  'loc':loc}

config['geometry'] ={'box_size': bs,
                    'num_cell' : nc,
                    'buf': buff}

config['survey'] = {'n_skewers' : n_skewers,
                    'include_dla' : include_dla,
                    'model_dla' : model_dla,
                    'snr' : snr}

with open(loc + prefix+'_config.ini', 'w') as configfile:
    config.write(configfile)