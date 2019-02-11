import numpy as np
import time
from field_ops import Engine2 as Engine

n = 20
v = np.linspace(0, 1, n)
x, y, z = np.meshgrid(v, v, v, indexing='ij')

# setup simulation engine
sim = Engine()

# allocate variables
c = sim.zeros([], [n,n,n], float)
c_hat = sim.zeros([], [n,n,n], complex)
u = sim.zeros([3], [n,n,n], float)
u_hat = sim.zeros([3], [n,n,n], complex)
D = sim.zeros([3,3], [n,n,n], float)
D_hat = sim.zeros([3,3], [n,n,n], complex)
R = sim.zeros([3,3], [n,n,n], float)
M = sim.zeros([3,3], [n,n,n], float)
v = sim.zeros([3], [n,n,n], float)
V = sim.zeros([3,3], [n,n,n], float)
eye = sim.zeros([3,3], [n,n,n], float)

# set c to something
D[:] = np.random.rand(*D.shape)

# make eye actually be an identity
for i in range(3):
	eye[i,i] += 1.0

# execute an einsum prod on D
print('\n--- Testing einsum MAT MAT ---')
instr = 'ij...,jk...->ik...'
sim.einsum(instr, [D, D], R)
st = time.time(); truth = np.einsum(instr,D.data,D.data); numpy_time = time.time()-st
st = time.time(); sim.einsum(instr, [D, D], R); sim_time = time.time()-st
print('All close?              ', np.allclose(truth, R.data))
print('... Einsum time (ms):    {:0.1f}'.format(numpy_time*1000))
print('... Sim time    (ms):    {:0.1f}'.format(sim_time*1000))

# now make R be nicely conditioned
R[:] = 0.1*R + eye

# instantiate processor pool
pool = sim.initialize_pool()

# compute the eigendecomposition of R
print('\n--- Testing eigendecomposition ---')
S = np.transpose(R.data, [2,3,4,0,1])
st = time.time(); truth = np.linalg.eigh(S); numpy_time = time.time()-st
st = time.time(); sim.eigh(R, v, V, pool); sim_time = time.time()-st
tv = np.transpose(truth[0], [3,0,1,2])
tV = np.transpose(truth[1], [3,4,0,1,2])
print('... All close, values?  ', np.allclose(tv, v.data))
print('... All close, vectors? ', np.allclose(tV, V.data))
print('... Eigh time (ms):      {:0.1f}'.format(numpy_time*1000))
print('... Sim time  (ms):      {:0.1f}'.format(sim_time*1000))

# test matrix matrix multiply, with transpose on first mat
print('\n--- Testing matrix matrix multiply, transpose on first mat ---')
# run once to compile
sim.mat_mat_tA(D, R, M);
st = time.time(); NR = np.einsum('ji...,jk...->ik...', D.data, R.data); numpy_time = time.time()-st
st = time.time(); sim.mat_mat_tA(D, R, M); sim_time = time.time()-st
print('... All close?          ', np.allclose(NR, M.data))
print('... numpy time (ms):     {:0.1f}'.format(numpy_time*1000))
print('... Sim time   (ms):     {:0.1f}'.format(sim_time*1000))

# test FFTs
print('\n--- Testing FFT (as compared to FFTPACK) ---')
# run once to be sure the FFT is planned
_ = np.fft.fftn(D.data)
st = time.time(); NR = np.fft.fftpack.fftn(D.data, axes=(-3,-2,-1)); numpy_time = time.time()-st
st = time.time(); sim.fft(D, D_hat); sim_time = time.time()-st
print('... All close?          ', np.allclose(NR, D_hat.data))
print('... numpy time (ms):     {:0.1f}'.format(numpy_time*1000))
print('... Sim time   (ms):     {:0.1f}'.format(sim_time*1000))

print('\n--- Testing IFFT (as compared to FFTPACK) ---')
# run once to be sure the FFT is planned
_ = np.fft.ifftn(NR).real
st = time.time(); NR = np.fft.fftpack.ifftn(D_hat.data, axes=(-3,-2,-1)); numpy_time = time.time()-st
st = time.time(); sim.ifft(D_hat, D); sim_time = time.time()-st
print('... All close?          ', np.allclose(NR, D.data))
print('... numpy time (ms):     {:0.1f}'.format(numpy_time*1000))
print('... Sim time   (ms):     {:0.1f}'.format(sim_time*1000))

print('\n--- Testing Symmetrize Operation ---')
M = sim.zeros([3,3], [n,n,n], float)
E = sim.zeros([3,3], [n,n,n], float)
M[:] = np.random.rand(*M.shape)
MM = M.data
NR = np.empty(M.shape)
st = time.time()
NR[0,0] = MM[0,0]
NR[1,1] = MM[1,1]
NR[2,2] = MM[2,2]
NR[0,1] = (MM[0,1] + MM[1,0])/2.0
NR[0,2] = (MM[0,2] + MM[2,0])/2.0
NR[1,2] = (MM[1,2] + MM[2,1])/2.0
NR[1,0] = NR[0,1]
NR[2,0] = NR[0,2]
NR[2,1] = NR[1,2]
numpy_time = time.time()-st
# run once to be sure the FFT is planned
sim.symmetrize(M, E)
st = time.time(); sim.symmetrize(M, E); sim_time = time.time()-st
print('... All close?          ', np.allclose(NR, E.data))
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
	M1 = sim.zeros(sh1[:l1], sh1[l1:], float)
	M2 = sim.zeros(sh2[:l2], sh2[l2:], float)
	M3 = sim.zeros(sh3[:l3], sh3[l3:], float)
	M1N = np.random.rand(*sh1)
	M2N = np.random.rand(*sh2)
	M1[:] = M1N
	M2[:] = M2N
	# test numpy
	st = time.time(); M3N = np.einsum(instr, M1N, M2N); numpy_time=time.time()-st
	# test sim
	sim.einsum(instr, [M1, M2], M3)
	st = time.time(); sim.einsum(instr, [M1, M2], M3); sim_time=time.time()-st
	print('... All close?          ', np.allclose(M3N, M3.data))
	print('... numpy time (ms):     {:0.1f}'.format(numpy_time*1000))
	print('... Sim time   (ms):     {:0.1f}'.format(sim_time*1000))
for instr in sim.list_common_einsum():
	test_common(instr)
