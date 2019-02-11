import numpy as np
import time
import field_ops as fo
Engine = fo.Engine2
Expression = fo.Expression

print('\n\n--- Testing Basic Evaluations ---\n')

n = 10
engine = Engine()
B = engine.empty([3,3], [n,n,n], name='B')
B[:] = 1.0
C = engine.empty([3,3], [n,n,n], name='C')
C[:] = 2.0
D = engine.zeros([3,3], [n,n,n])
D[:] = 0.0

st = time.time()
D[:] = -fo.arctan2(C, B)*fo.tan(B) - 2.0/(1+fo.arcsin(B*fo.pi/4.0)) + fo.sin(B) + 1.1 + fo.pi/fo.cos(2*B*C) + C**2 - 2**B
print('Engine time 1: {:0.1f}'.format((time.time() - st)*1000))
st = time.time()
D[:] = -fo.arctan2(C, B)*fo.tan(B) - 2.0/(1+fo.arcsin(B*fo.pi/4.0)) + fo.sin(B) + 1.1 + fo.pi/fo.cos(2*B*C) + C**2 - 2**B
print('Engine time 2: {:0.1f}'.format((time.time() - st)*1000))

st = time.time()
E = -np.arctan2(C.data, B.data)*np.tan(B.data) - 2.0/(1+np.arcsin(B.data*np.pi/4.0)) + np.sin(B.data) + 1.1 + np.pi/np.cos(2*B.data*C.data) + C.data**2 - 2**B.data
print('Numpy time:    {:0.1f}'.format((time.time() - st)*1000))

print('All close?    ', np.allclose(E,D.data))

st = time.time()
s = engine| fo.sum(B)
print('Engine time 1: {:0.1f}'.format((time.time() - st)*1000))
st = time.time()
s = engine| fo.sum(B)
print('Engine time 2: {:0.1f}'.format((time.time() - st)*1000))

st = time.time()
s2 =  np.sum(B.data)
print('Numpy time:    {:0.1f}'.format((time.time() - st)*1000))

print('All close?    ', np.allclose(s,s2))

del B, C, D
del engine

print('\n--- Testing Usage in Function ---\n')

# test usage in a function
engine = Engine()
A = engine.empty([3,3], [n,n,n])
B = engine.empty([3,3], [n,n,n])
C = engine.empty([3,3], [n,n,n])
A[:] = 1.0
B[:] = 2.0

def test(x, y, z):
    A = engine.empty([3,3], [n,n,n])
    A[:] = 10.0
    z[:] = x + y + A
    # note that this is better than doing nothing!
    # if you do nothing, the memory will be returned to the memory pool
    # when python decides to garbage collect...
    # if you do this, it is returned immediately
    A.__del__()

test(A, B, C)

print('Error is: {:0.1f}'.format(np.abs(C.data - 13).max()))

print('\n--- Testing Slicing ---\n')

D = engine.zeros([3,3], [n,n,n])

st = time.time()
D[1,2] = A[0,0] + B[1,1]
print('Engine time 1: {:0.1f}'.format((time.time() - st)*1000))
st = time.time()
D[1,2] = A[0,0] + B[1,1]
print('Engine time 2: {:0.1f}'.format((time.time() - st)*1000))

E = np.zeros([3,3,n,n,n])
st = time.time()
E[1,2] = A.data[0,0] + B.data[1,1]
print('Numpy time:    {:0.1f}'.format((time.time() - st)*1000))
print('All close?    ', np.allclose(E,D.data))

st = time.time()
D[0] = A[0] + B[1]
print('Engine time 1: {:0.1f}'.format((time.time() - st)*1000))
st = time.time()
D[0] = A[0] + B[1]
print('Engine time 2: {:0.1f}'.format((time.time() - st)*1000))

st = time.time()
E[0] = A.data[0] + B.data[1]
print('Numpy time:    {:0.1f}'.format((time.time() - st)*1000))
print('All close?    ', np.allclose(E,D.data))

def test(x, y, z):
    z[0,0,1:-1] = x[0,0,1:-1] + y[0,0,1:-1]

C[:] = 0.0
st = time.time(); test(A, B, C); et = time.time()
print(et-st)
st = time.time(); A.data[0,0,1:-1] = B.data[0,0,1:-1] + C.data[0,0,1:-1]; et = time.time()
print(et-st)



