import numpy as np
import time
from field_ops import Engine

n = 20
v = np.linspace(0, 1, n)
x, y, z = np.meshgrid(v, v, v, indexing='ij')

# setup simulation engine
sim = Engine(x.shape)

# allocate variables
sim.allocate('c', [], 'both')
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
print('\n--- Testing einsum MAT MAT ---')
instr = 'ij...,jk...->ik...'
sim.einsum(instr, ['D','D'], 'R')
st = time.time(); truth = np.einsum(instr,D,D); numpy_time = time.time()-st
st = time.time(); sim.einsum(instr, ['D','D'], 'R'); sim_time = time.time()-st
print('All close?              ', np.allclose(truth, sim.get('R')))
print('... Einsum time (ms):    {:0.1f}'.format(numpy_time*1000))
print('... Sim time    (ms):    {:0.1f}'.format(sim_time*1000))

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
print('... All close, values?  ', np.allclose(tv, sim.get('v')))
print('... All close, vectors? ', np.allclose(tV, sim.get('V')))
print('... Eigh time (ms):      {:0.1f}'.format(numpy_time*1000))
print('... Sim time  (ms):      {:0.1f}'.format(sim_time*1000))

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
print('... All close, values?  ', np.allclose(tv, sim.get('v2')))
print('... All close, vectors? ', np.allclose(tV, sim.get('V2')))
print('... Eigh time (ms):      {:0.1f}'.format(numpy_time*1000))
print('... Sim time  (ms):      {:0.1f}'.format(sim_time*1000))

# test matrix matrix multiply, with transpose on first mat
print('\n--- Testing matrix matrix multiply, transpose on first mat ---')
D = sim.get('D')
R = sim.get('R')
# run once to compile
sim.mat_mat_tA('D', 'R', 'M');
st = time.time(); NR = np.einsum('ji...,jk...->ik...', D, R); numpy_time = time.time()-st
st = time.time(); sim.mat_mat_tA('D', 'R', 'M'); sim_time = time.time()-st
print('... All close?          ', np.allclose(NR, sim.get('M')))
print('... numpy time (ms):     {:0.1f}'.format(numpy_time*1000))
print('... Sim time   (ms):     {:0.1f}'.format(sim_time*1000))

# test FFTs
print('\n--- Testing FFT (as compared to FFTPACK) ---')
# run once to be sure the FFT is planned
_ = np.fft.fftn(D)
st = time.time(); NR = np.fft.fftpack.fftn(D, axes=(-3,-2,-1)); numpy_time = time.time()-st
st = time.time(); sim.fft('D', 'D_hat'); sim_time = time.time()-st
print('... All close?          ', np.allclose(NR, sim.get('D_hat')))
print('... numpy time (ms):     {:0.1f}'.format(numpy_time*1000))
print('... Sim time   (ms):     {:0.1f}'.format(sim_time*1000))

print('\n--- Testing IFFT (as compared to FFTPACK) ---')
# run once to be sure the FFT is planned
_ = np.fft.ifftn(NR).real
D_hat = sim.get('D_hat').copy()
st = time.time(); NR = np.fft.fftpack.ifftn(D_hat, axes=(-3,-2,-1)); numpy_time = time.time()-st
st = time.time(); sim.ifft('D_hat', 'D'); sim_time = time.time()-st
print('... All close?          ', np.allclose(NR, sim.get('D')))
print('... numpy time (ms):     {:0.1f}'.format(numpy_time*1000))
print('... Sim time   (ms):     {:0.1f}'.format(sim_time*1000))

print('\n--- Test common einsums ---')
def test_common(instr):
	# parse instr
	print('...Testing einsum:', instr)
	l1, l2 = instr.split(',')
	l2, l3 = l2.split('->')
	l1 = len(l1.replace('...',''))
	l2 = len(l2.replace('...',''))
	l3 = len(l3.replace('...',''))
	# get shapes
	sh1 = [3,]*l1 + [n,n,n]
	sh2 = [3,]*l2 + [n,n,n]
	sh3 = [3,]*l3 + [n,n,n]
	# allocate memory
	sim.allocate('M1', sh1[:l1], float)
	sim.allocate('M2', sh2[:l2], float)
	sim.allocate('M3', sh3[:l3], float)
	M1N = np.random.rand(*sh1)
	M2N = np.random.rand(*sh2)
	sim.get('M1')[:] = M1N
	sim.get('M2')[:] = M2N
	# test numpy
	st = time.time(); M3N = np.einsum(instr, M1N, M2N); numpy_time=time.time()-st
	# test sim
	sim.einsum(instr, ['M1', 'M2'], 'M3')
	st = time.time(); sim.einsum(instr, ['M1', 'M2'], 'M3'); sim_time=time.time()-st
	print('... All close?          ', np.allclose(M3N, sim.get('M3')))
	print('... numpy time (ms):     {:0.1f}'.format(numpy_time*1000))
	print('... Sim time   (ms):     {:0.1f}'.format(sim_time*1000))
for instr in sim.list_common_einsum():
	test_common(instr)

# terminate the pool
sim.terminate_pool()
