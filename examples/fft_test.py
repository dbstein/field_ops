import numpy as np
import numexpr as ne
import numba
import time

print('--- Testing 3D FFTS ---')

n = 1024
LL = 3

A = np.random.rand(n,n,n)

@numba.njit(parallel=True)
def ccopy(x, y):
	for i in numba.prange(n):
		for j in range(n):
			for k in range(n):
				y[i,j,k] = x[i,j,k]
C = np.empty(A.shape, dtype=complex)
ccopy(A, C)

def my_fft1(X):
	B = np.empty(X.shape, dtype=complex)
	ne.evaluate('X', out=B)
	return np.fft.fftn(B) 

def my_fft2(X, C):
	ccopy(X, C)
	return np.fft.fftn(C) 

def timeit(expr, loops=LL):
	exec(expr)
	st = time.time();
	for ii in range(10):
		exec(expr)
	et = time.time()
	print('...{:0.2f}'.format((et-st)*1000/10))

print('\nTime for Intel MKL FFT of real array')
timeit('O = np.fft.fftn(A)')
print('Time for Intel MKL RFFT of real array')
timeit('O = np.fft.rfftn(A)')
print('Time for Intel MKL FFT of complex array, copied from real array')
timeit('O = my_fft1(A)')
print('Time for Intel MKL FFT of complex array, copied from real array')
timeit('O = my_fft2(A,C)')
A = A + 0j
print('Time for Intel MKL FFT of complex array')
timeit('O = np.fft.fftn(A)')

A = np.random.rand(n,n,n) + 1j
B = np.random.rand(n,n,int(n/2)+1) + 1j

print('\nTime for Intel MKL IFFT of complex array')
timeit('O = np.fft.ifftn(A).real')
print('Time for Intel MKL IRFFT of complex array')
timeit('O = np.fft.irfftn(B)')

print('\n\n--- Testing 2D FFTS ---')

n = 2048
LL = 10

A = np.random.rand(n,n)

@numba.njit(parallel=True)
def ccopy(x, y):
	for i in numba.prange(n):
		for j in range(n):
			y[i,j] = complex(x[i,j])
C = np.empty(A.shape, dtype=complex)
ccopy(A, C)

def my_fft1(X):
	B = np.empty(X.shape, dtype=complex)
	ne.evaluate('X', out=B)
	return np.fft.fft2(B) 

def my_fft2(X, C):
	ccopy(X, C)
	return np.fft.fft2(C) 

print('\nTime for Intel MKL FFT of real array')
timeit('O = np.fft.fft2(A)')
print('Time for Intel MKL RFFT of real array')
timeit('O = np.fft.rfft2(A)')
print('Time for Intel MKL FFT of complex array, copied from real array')
timeit('O = my_fft1(A)')
print('Time for Intel MKL FFT of complex array, copied from real array')
timeit('O = my_fft2(A,C)')
A = A + 0j
print('Time for Intel MKL FFT of complex array')
timeit('O = np.fft.fft2(A)')

A = np.random.rand(n,n) + 1j
B = np.random.rand(n,int(n/2)+1) + 1j

print('\nTime for Intel MKL IFFT of complex array')
timeit('O = np.fft.ifft2(A).real')
print('Time for Intel MKL IRFFT of complex array')
timeit('O = np.fft.irfft2(A).real')
timeit('O = np.fft.irfftn(A).real')



# A = np.random.rand(n,n)
# B = np.empty(A.shape, dtype=complex)
# %timeit np.add(A, 0j, B)
# ccopy(A, B)
# %timeit ccopy(A, B)
