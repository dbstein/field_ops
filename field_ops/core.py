import numpy as np
import scipy as sp
import numexpr as ne
import numba
import multiprocessing as mp
from .utilities import Bunch, RawArray, mmap_zeros
from .modules import parallel_einsum
import mkl

max_processes = mkl.get_max_threads()

class Engine(object):
    def __init__(self, base_shape):
        self.base_shape = list(base_shape)
        self.base_length = len(self.base_shape)
        self.variables = Bunch({})
        self._add_base_constants()
        self.pool_formed = False

    ############################################################################
    # methods for dealing with variable allocation
    def __allocate(self, name, shape, dtype=float):
        # self.variables[name] = RawArray(shape=shape, dtype=dtype)()
        self.variables[name] = mmap_zeros(shape=shape, dtype=dtype)
    def _allocate(self, name, field_shape, dtype):
        shape = list(field_shape) + self.base_shape
        if dtype == float or dtype == complex:
            self.__allocate(name, shape, dtype)
        elif dtype == 'both':
            self.__allocate(name, shape, float)
            self.__allocate(name + '_hat', shape, complex)
        else:
            raise Exception('dtype not supported')
    def allocate(self, name, field_shape, dtype=float):
        """
        Allocate a field
        name:         str
        field_shape:  list of int; for example [3,3]
        dtype:        float, complex, or 'both'
        transpose:    if transpose=True, flips the field/shape order
        """
        self._allocate(name, field_shape, dtype)
        self.reset_pool()
    def allocate_many(self, names, field_shapes, dtypes):
        for name, field_shape, dtype in zip(names, field_shapes, dtypes):
            self._allocate(name, field_shape, dtype)
        self.reset_pool()
    def add_constant(self, name, value):
        """
        Add a constant to the variable dictionary for use in ne.evaluate
        """
        self.variables[name] = value
    def _add_base_constants(self):
        self.add_constant('pi', np.pi)
        self.add_constant('e', np.e)
        self.add_constant('euler_gamma', np.euler_gamma)

    ############################################################################
    # methods for extracting variables
    def get(self, name):
        """
        Return the underlying numpy array for variable of name
        """
        return self.variables[name]
    def list(self):
        return self.variables.keys()

    ############################################################################
    # methods for dealing with the processor pool
    def _initialize_worker(self, vardict):
        mkl.set_num_threads_local(1)
        globals().update({'var' : vardict})
    def initialize_pool(self, processors=None):
        if processors is None:
            self.processes = max_processes
        self.pool = mp.Pool(processes=self.processes, initializer=self._initialize_worker, initargs=(self.variables,))
        self.pool_formed = True
    def reset_pool(self):
        if self.pool_formed:
            self.pool.terminate()
            self.pool = mp.Pool(processes=self.processes, initializer=self._initialize_worker, initargs=(self.variables,))
    def terminate_pool(self):
        if self.pool_formed:
            self.pool.terminate()

    ############################################################################
    # method for easing interface to numexpr
    def evaluate(self, instr, outname):
        """
        Evaluates 'instr' on self.variables using ne.evaluate,
        and stores the result in 'outname', replacing whatever was there
        if outname doesn't exist, this function will create it
        """
        if outname in self.variables.keys():
            ne.evaluate(instr, local_dict=self.variables, out=self.get(outname))
        else:
            out = ne.evaluate(instr, local_dict=self.variables)
            self.allocate(outname, out.shape[:-self.base_length], out.dtype)

    ############################################################################
    # einsum
    def einsum(self, instr, evars, out):
        evars = [self.get(evar) for evar in evars]
        np.einsum(instr, *evars, out=self.get(out))
    # this parallel version is slower for now...
    def bad_einsum(self, instr, evars, out):
        n = self.base_shape[-1]
        self.pool.starmap(_einsum, zip(range(n), [instr,]*n, [evars,]*n, [out,]*n))

    ############################################################################
    # eigh solver
    def eigh(self, M, Mv, MV):
        n = self.base_shape[-1]
        t1 = [2 + i for i in range(len(self.base_shape))] + [0,1]
        t2 = [1 + i for i in range(len(self.base_shape))] + [0,]
        self.pool.starmap(_eigh, zip(range(n), [M,]*n, [Mv,]*n, [MV,]*n, [t1,]*n, [t2,]*n))

    ############################################################################
    # matrix matrix multiply
    def _mat_mat_reshape(self, M):
        sh = list(M.shape)
        newsh = sh[:2] + [np.prod(sh[2:]),]
        return np.reshape(M, newsh)
    def mat_mat(self, M1, M2, O):
        M1M = self._mat_mat_reshape(self.get(M1))
        M2M = self._mat_mat_reshape(self.get(M2))
        M3M = self._mat_mat_reshape(self.get(O))
        _mat_mat(M1M, M2M, M3M)

# internal functions for parallel execution
def _einsum(n, instr, evars, out):
    evarsn = [var[evar][..., n] for evar in evars]
    np.einsum(instr, *evarsn, out=var[out][..., n])
def _eigh(n, _M, _v, _V, t1, t2):
    M = np.transpose(var[_M], t1)
    v = np.transpose(var[_v], t2)
    V = np.transpose(var[_V], t1)
    v[n], V[n] = np.linalg.eigh(M[n])

@numba.njit()
def __mat_mat(A, B, C):
    sh1 = A.shape[0]
    sh2 = A.shape[1]
    sh3 = B.shape[1]
    for i in range(sh1):
        for j in range(sh3):
            C[i,j] = 0.0
            for k in range(sh2):
                C[i,j] += A[i,k]*B[k,j]
@numba.njit(parallel=True)
def _mat_mat3(M1, M2, M3):
    n = M1.shape[-1]
    for i in numba.prange(n):
        for j in range(n):
            for k in range(n):
                __mat_mat(M1[:,:,i,j,k], M2[:,:,i,j,k], M3[:,:,i,j,k])
@numba.njit(parallel=True)
def _mat_mat(M1, M2, M3):
    n = M1.shape[-1]
    for i in numba.prange(n):
        __mat_mat(M1[:,:,i], M2[:,:,i], M3[:,:,i])



