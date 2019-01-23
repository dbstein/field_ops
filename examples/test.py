import numpy as np
import time
from field_ops import Engine

n = 100
v = np.linspace(0, 1, n)
x, y, z = np.meshgrid(v, v, v, indexing='ij')

# setup simulation engine
sim = Engine(x.shape)

# allocate variables
sim.allocate('c', [1], float)
sim.allocate('u', [3], 'both')
sim.allocate('D', [3,3], 'both')
sim.allocate('R', [3,3], float)
sim.allocate('M', [3,3], float)
sim.allocate('v', [3], float)
sim.allocate('V', [3,3], float)
sim.allocate('eye', [3,3], float)

# set c to something
D = sim.get('D')
D[:] = np.random.rand(*D.shape)

# instantiate processor pool
sim.initialize_pool()

# make eye actually be an identity
eye = sim.get('eye')
eye[0,0] = 1.0
eye[1,1] = 1.0
eye[2,2] = 1.0

# execute an einsum prod on D
print('\n--- Testing einsum ---')
instr = 'ij...,jk...->ik...'
st = time.time(); truth = np.einsum(instr,D,D); numpy_time = time.time()-st
st = time.time(); sim.einsum(instr, ['D','D'], 'R'); sim_time = time.time()-st
print('All close?       ', np.allclose(truth, sim.get('R')))
print('... Einsum time (ms):   {:0.1f}'.format(numpy_time*1000))
print('... Sim time    (ms):   {:0.1f}'.format(sim_time*1000))

# now make R be nicely conditioned
sim.evaluate('0.1*R + eye', 'R')

# compute the eigendecomposition of R
print('\n--- Testing eigendecomposition ---')
R = sim.get('R')
S = np.transpose(R, [2,3,4,0,1])
st = time.time(); truth = np.linalg.eigh(S); numpy_time = time.time()-st
st = time.time(); sim.eigh('R', 'v', 'V'); sim_time = time.time()-st
tv = np.transpose(truth[0], [3,0,1,2])
tV = np.transpose(truth[1], [3,4,0,1,2])
print('... All close, values? ', np.allclose(tv, sim.get('v')))
print('... All close, vectors?', np.allclose(tV, sim.get('V')))
print('... Eigh time (ms):     {:0.1f}'.format(numpy_time*1000))
print('... Sim time  (ms):     {:0.1f}'.format(sim_time*1000))

# test eigendecomp on on-the-fly allocated variable
print('\n--- Testing eigendecomposition on new variable of (2,2) size ---')
variables = [
	['M2', [2,2], float],
	['v2', [2],   float],
	['V2', [2,2], float],
]
sim.allocate_many(variables)
M2 = sim.get('M2')
M2[:] = R[:2,:2]
S = np.transpose(M2, [2,3,4,0,1])
# try to eigh this
st = time.time(); truth = np.linalg.eigh(S); numpy_time = time.time()-st
st = time.time(); sim.eigh('M2', 'v2', 'V2'); sim_time = time.time()-st
tv = np.transpose(truth[0], [3,0,1,2])
tV = np.transpose(truth[1], [3,4,0,1,2])
print('... All close, values? ', np.allclose(tv, sim.get('v2')))
print('... All close, vectors?', np.allclose(tV, sim.get('V2')))
print('... Eigh time (ms):     {:0.1f}'.format(numpy_time*1000))
print('... Sim time  (ms):     {:0.1f}'.format(sim_time*1000))

# test matrix matrix multiply
print('\n--- Testing matrix matrix multiply ---')
D = sim.get('D')
R = sim.get('R')
# run once to compile
sim.mat_mat('D', 'R', 'M');
st = time.time(); NR = np.einsum('ij...,jk...->ik...', D, R); numpy_time = time.time()-st
st = time.time(); sim.mat_mat('D', 'R', 'M'); sim_time = time.time()-st
print('... All close? ', np.allclose(NR, sim.get('M')))
print('... numpy time (ms):     {:0.1f}'.format(numpy_time*1000))
print('... Sim time   (ms):     {:0.1f}'.format(sim_time*1000))

# test FFTs
print('\n--- Testing FFT ---')
# run once to be sure the FFT is planned
_ = np.fft.fftn(D)
st = time.time(); NR = np.fft.fftn(D, axes=(-3,-2,-1)); numpy_time = time.time()-st
st = time.time(); sim.fft('D', 'D_hat'); sim_time = time.time()-st
print('... All close? ', np.allclose(NR, sim.get('D_hat')))
print('... numpy time (ms):     {:0.1f}'.format(numpy_time*1000))
print('... Sim time   (ms):     {:0.1f}'.format(sim_time*1000))

print('\n--- Testing FFT ---')
# run once to be sure the FFT is planned
_ = np.fft.ifftn(NR).real
st = time.time(); NR = np.fft.fftn(NR, axes=(-3,-2,-1)); numpy_time = time.time()-st
st = time.time(); sim.ifft('D_hat', 'D'); sim_time = time.time()-st
print('... All close? ', np.allclose(NR, sim.get('D')))
print('... numpy time (ms):     {:0.1f}'.format(numpy_time*1000))
print('... Sim time   (ms):     {:0.1f}'.format(sim_time*1000))


# terminate the pool
sim.terminate_pool()
