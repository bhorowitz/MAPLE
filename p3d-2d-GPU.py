
import jax
#jax.config.update('jax_platform_name', 'cpu')
from jax.lib import xla_bridge

print(xla_bridge.get_backend().platform)

import numpy as np
import jax.numpy as jnp

#muse componenet...
from functools import partial
import jax
from muse_inference.jax import JaxMuseProblem
import sys 
import configparser

key_num = int(sys.argv[1])# 2 #
print(key_num)

loc_geo_config = "/pscratch/sd/b/bhorowit/configs/A_DESI_TEST512__config.ini"
loc_opt_config = "optim_settings.ini"

output_name = "./outputs/DESI_512_128_"+str(key_num)


config_geo = configparser.ConfigParser()
config_geo.read(loc_geo_config)

z = float(config_geo['basic']['redshift'])
prefix = config_geo['basic']['prefix']
loc = config_geo['basic']['loc']
bs = int(config_geo['geometry']['box_size'])
nc = int(config_geo['geometry']['num_cell'])
buff = int(config_geo['geometry']['buf'])
n_skewers =  int(config_geo['survey']['n_skewers'])
include_dla = True#config_geo['survey']['include_dla']
model_dla = True#config_geo['survey']['model_dla']
snr = float(config_geo['survey']['snr'])

config_opt = configparser.ConfigParser()
config_opt.read(loc_opt_config)


n_kbins = int(config_opt["default"]["n_kbins"])
logk_min = float(config_opt["default"]["logk_min"])
logk_max = float(config_opt["default"]["logk_max"])
maxsteps = int(config_opt["default"]["maxsteps"])
nsims = int(config_opt["default"]["nsims"])
α = float(config_opt["default"]["α"])
rng_n = int(config_opt["default"]["rng_n"])



ptcl_grid_shape = (nc,) * 3
ptcl_spacing = bs/nc

#number of kbins 

kbins = np.logspace(logk_min,logk_max,n_kbins)

naa = np.load(loc+prefix+"naa.npy")
kernel = np.load(loc+prefix+"kernel.npy")
skewers_skn = np.load(loc+prefix+"skewers_skn.npy")
skewers_dla = np.load(loc+prefix+"skewers_dla.npy")
skewers_fin = np.load(loc+prefix+"skewers_fin.npy")

import lya_forecast as lf #file with many commands for 3d PS estimation, 
theory = lf.TheoryLyaP3D()

from helper_functions import *

kvec = rfftnfreq_2d(ptcl_grid_shape, ptcl_spacing)
k = jnp.sqrt(sum(k**2 for k in kvec))

kz = jnp.ones(k.shape)*kvec[0]**2
kx = jnp.ones(k.shape)*(kvec[1]**2+kvec[2]**2)
kk = (kx+kz)+10**(-8)

kcenters = (kbins[:-1]+kbins[1:])/2

idx_k_mu = []
for ki in kcenters:
    for kj in kcenters:
        r = np.sqrt(ki**2+kj**2)
        idx_k_mu.append([r,ki/(r)])
        
idx_k_mu = np.array(idx_k_mu)

idx_pk = theory.FluxP3D_hMpc(z,idx_k_mu[:,0],idx_k_mu[:,1])

kbins[0]=0
kbins[-1]=10

bin_vals_kx = jnp.digitize(np.sqrt(kx), kbins, right=False) -1 
bin_vals_kz = jnp.digitize(np.sqrt(kz), kbins, right=False) -1 

indexes_k = np.vstack([bin_vals_kx.flatten(),bin_vals_kz.flatten()])


def power_b(idx_pk):
    return idx_pk[indexes_k.T[:,0]*(n_kbins-1)+indexes_k.T[:,1]].reshape(kx.shape)



if model_dla:
    eff_noise = skewers_skn+skewers_dla*10000 #set noise in DLA very high
else: 
    eff_noise = skewers_skn
    

# defining some functions for optimization...
from jax import jit, checkpoint, custom_vjp


@jit
def cic_readout_jit_jnc(mesh,naa,kernel,bs=False):
    #"highly optimized" CIC, need to preprocess lots of things... don't diff output coords
    meshvals = mesh.flatten()[naa].reshape(-1,8).T#mesh[tuple(neighboor_coords[0,:,:].T.tolist())]
    weightedvals = meshvals.T* kernel[0]
    values = np.sum(weightedvals, axis=-1)
    
    return values


@jit
@checkpoint
def linear_modes(modes, theta):
   # kvec = rfftnfreq(conf.ptcl_grid_shape, ptcl_spacing)
   # k = jnp.sqrt(sum(k**2 for k in kvec))
    Plin = power_b(theta)
    if jnp.isrealobj(modes):
        modes = jnp.fft.rfftn(modes, norm='ortho')
    modes *= jnp.sqrt(Plin * bs**3)

    return modes

def gen_map_lya(theta,z):
    modes = z.reshape((nc,nc,nc))
    lin_modes = linear_modes(modes, theta)
    lin_modes_real = jnp.fft.irfftn(lin_modes)
   
    lya_values = cic_readout_jit_jnc(lin_modes_real,naa,kernel)
    return lya_values

class Jax3DMuseProblem_flat(JaxMuseProblem):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def sample_x_z(self, key, θ):
        keys = jax.random.split(key, 2)
        z = jax.random.normal(keys[0], (nc*nc*nc,))
        x = gen_map_lya(θ,z) + 0.1*skewers_skn*jax.random.normal(keys[1], (len(skewers_fin),))
        #DLAs
        if include_dla: #zero out DLA locations
            x = (1-skewers_dla)*x
        return (x, z)

    def logLike(self, x, z, θ):
        return -(jnp.sum((x - gen_map_lya(θ,z))**2/((0.1*eff_noise)**2)) + jnp.sum(z**2.0))
    
    def logPrior(self, θ):
        return -jnp.sum(((θ-jnp.array(idx_pk)*1.00)**2 / (2*(idx_pk*0.3)**2)))

prob = Jax3DMuseProblem_flat(implicit_diff=True)
key = jax.random.PRNGKey(key_num)
(x, z) = prob.sample_x_z(key, jnp.array(idx_pk))
prob.set_x(x)


from multiprocessing.pool import ThreadPool as Pool 
pool = Pool() #jax pmap? check re-compilation times?

start_point = jnp.array(idx_pk)+np.abs(np.random.randn(len(idx_pk))*jnp.array(idx_pk)*0.01)
#pool.map

result = prob.solve(pmap = pool.map, θ_start=start_point, 
                   rng = jax.random.PRNGKey(rng_n), progress=True,θ_rtol = 1e-5, 
                   maxsteps = maxsteps,nsims=nsims,α=α/2,get_covariance=False)

print(result)

import pickle
with open(output_name+'_NC.pkl','wb') as f:
    pickle.dump([kbins,start_point,result.θ,idx_pk], f)

if False:
    print("exiting before covariance!")

    exit()

result = prob.solve(pmap = pool.map, θ_start=result.θ, 
                   rng = jax.random.PRNGKey(rng_n), progress=True,θ_rtol = 1e-5, 
                   maxsteps = 1,nsims=nsims,α=α,get_covariance=True)


print(result)

print(jnp.array(idx_pk)-result.θ)


import pickle
with open(output_name+'.pkl','wb') as f:
    pickle.dump([kbins,start_point,result.θ,result.Σ,idx_pk], f)
    