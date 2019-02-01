import numpy as np
import scipy as sp
import numexpr as ne
import numba
import multiprocessing as mp
from .utilities import Bunch, RawArray, mmap_zeros
import mkl

max_processes = mkl.get_max_threads()
ne.set_num_threads(max_processes)

class Engine(object):
    def __init__(self, base_shape):
        self.base_shape = list(base_shape)
        self.base_length = len(self.base_shape)
        self.variables = Bunch({})
        self.mp_variables = Bunch({})
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
            self.point_at_all_subfields(name)
        elif dtype == 'both':
            self.__allocate(name, shape, float)
            self.point_at_all_subfields(name)
            self.__allocate(name + '_hat', shape, complex)
            self.point_at_all_subfields(name + '_hat')
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
    def allocate_from(self, name, arr):
        self.allocate(name, arr.shape[:-self.base_length], arr.dtype)
        self.get(name)[:] = arr
    def allocate_many(self, var_list):
        for sublist in var_list:
            self._allocate(*sublist)
    def allocate_from_many(self, var_list):
        for sublist in var_list:
            self.allocate_from(*sublist)
    def add(self, name, value, point_subfields=False):
        """
        Add value to the variable dictionary
        Note that this differs from the behavior of allocate!
        What is added is NOT a memmap, and must be used in a read-only manner
        in multiprocessing code!
        """
        self.variables[name] = value
        if point_subfields:
            self.point_at_all_subfields(name)
    def add_many(self, add_list):
        for sublist in add_list:
            self.add(*sublist)
    def _add_base_constants(self):
        constants = [
            ['pi',          np.pi         ],
            ['e',           np.e          ],
            ['euler_gamma', np.euler_gamma],
        ]
    def copy_from_to(self, vf, vt):
        self.get(vt)[:] = self.get(vf)
    def point_at(self, name, field, field_inds):
        # point a name at some subset of a field
        F = self.get(field)
        l = len(field_inds)
        for i in range(l):
            F = F[field_inds[i]]
        self.add(name, F)
    def point_at_many(self, point_list):
        for sublist in point_list:
            self.point_at(*sublist)
    def point_at_all_subfields(self, field):
        F = self.get(field)
        field_sh, base_sh = self._extract_shapes(F)
        field_lsh = len(field_sh)
        if field_lsh > 0:
            sh_list = [np.arange(x) for x in field_sh]
            n = np.prod(field_sh)
            # generate all codes
            codes = []
            for i in range(field_lsh):
                code = np.empty(n, dtype=object)
                excess = np.prod(field_sh[i+1:])
                code = np.repeat(np.arange(field_sh[i]), excess)
                precess = np.prod(field_sh[:i])
                code = np.tile(code, precess)
                codes.append(code)
            code_reorder = []
            for i in range(n):
                code = []
                for j in range(field_lsh):
                    code.append(codes[j][i])
                code_reorder.append(code)
            codes = code_reorder
            # point everything where it should go
            for i in range(n):
                name = field + '_'
                field_inds = []
                for j in range(field_lsh):
                    code = codes[i][j]
                    name += str(code)
                    field_inds.append(code)
                self.point_at(name, field, field_inds)

    ############################################################################
    # methods for extracting variables
    def _fix_names(self, name):
        name = name.replace('[','_')
        name = name.replace(',','')
        name = name.replace(']','')
        return name
    def get(self, name):
        """
        Return the underlying numpy array for variable of name
        """
        return self.variables[self._fix_names(name)]
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
        self.mp_variables = self.variables.copy()
        self.pool = mp.Pool(processes=self.processes, initializer=self._initialize_worker, initargs=(self.mp_variables,))
        self.pool_formed = True
    def reset_pool(self):
        if self.pool_formed:
            self.pool.terminate()
            self.mp_variables = self.variables.copy()
            self.pool = mp.Pool(processes=self.processes, initializer=self._initialize_worker, initargs=(self.mp_variables,))
        else:
            raise Exception('Initialize pool before resetting...')
    def terminate_pool(self):
        if self.pool_formed:
            self.pool.terminate()
    def _check_and_reset_pool(self, var_names):
        if self._check_if_pool_reset_needed(var_names):
            self.reset_pool()
    def _check_if_pool_reset_needed(self, var_names):
        return not all([var in self.mp_variables.keys() for var in var_names])

    ############################################################################
    # method for easing interface to numexpr
    def evaluate(self, instr, outname=None):
        """
        Evaluates 'instr' on self.variables using ne.evaluate,
        If outname is specified, this expects a string with no '=' sign
            i.e. c = a + b would be done by:
            self.evaluate('a + b', 'c')
        If outname is specified, this expects a string with a '=' sign
            i.e. c = a + b would be done by:
            self.evaluate('c = a + b')
            in this case outname is generated by taking whatever precedes
            the '=', stripped of whitespace
        In either case, if the output variable doesn't yet exist,
            the function will create it
        """
        if outname is None:
            outname, instr = [st.strip() for st in instr.split('=')]
        outname = self._fix_names(outname)
        instr = self._fix_names(instr)
        if outname in self.variables.keys():
            ne.evaluate(instr, local_dict=self.variables, out=self.get(outname))
        else:
            out = ne.evaluate(instr, local_dict=self.variables)
            self.allocate_from(outname, out)

    ############################################################################
    # einsum
    def einsum(self, instr, evars, out):
        evars = [self.get(evar) for evar in evars]
        out = self.get(out)
        if instr in self.list_common_einsum():
            self._common_einsum(evars, out, instr)
        else:
            np.einsum(instr, *evars, out=out)
    def list_common_einsum(self):
        return _common_einsums.keys()
    def _common_einsum(self, evars, out, instr):
        M1 = self._reshape_field(evars[0])
        M2 = self._reshape_field(evars[1])
        M3 = self._reshape_field(out)
        _call_common_einsum(M1, M2, M3, instr)
    # this parallel version is slower for now...
    def bad_einsum(self, instr, evars, out):
        n = self.base_shape[-1]
        self._check_and_reset_pool(evars + [out])
        self.pool.starmap(_einsum, zip(range(n), [instr,]*n, [evars,]*n, [out,]*n))

    ############################################################################
    # reshaping functions
    def _extract_shapes(self, X):
        sh = list(X.shape)
        lsh = len(sh)
        field_lsh = lsh - self.base_length
        field_sh = sh[:field_lsh]
        return field_sh, self.base_shape
    def _reshape_tensor(self, X):
        field_sh, base_sh = self._extract_shapes(X)
        newsh = [int(np.prod(field_sh))] + base_sh
        return np.reshape(X, newsh)
    def _reshape_field(self, X):
        field_sh, base_sh = self._extract_shapes(X)
        newsh = field_sh + [int(np.prod(base_sh))]
        return np.reshape(X, newsh)

    ############################################################################
    # eigh solver
    def eigh(self, M, Mv, MV):
        self._check_and_reset_pool([M, Mv, MV])
        n = self.base_shape[0]
        t1 = [2 + i for i in range(len(self.base_shape))] + [0,1]
        t2 = [1 + i for i in range(len(self.base_shape))] + [0,]
        self.pool.starmap(_eigh, zip(range(n), [M,]*n, [Mv,]*n, [MV,]*n, [t1,]*n, [t2,]*n))

    ############################################################################
    # matrix matrix multiply
    def mat_mat(self, M1, M2, O):
        self.einsum('ij...,jk...->ik...', [M1, M2], O)

    def mat_mat_tA(self, M1, M2, O):
        self.einsum('ji...,jk...->ik...', [M1, M2], O)

    ############################################################################
    # symmetrize a square rank two tensor
    def symmetrize(self, M1, O):
        """
        computes O = (M1 + M1.T)/2
        """
        M1 = self._reshape_field(self.get(M1))
        M2 = self._reshape_field(self.get(O))
        _symmetrize(M1, M2)

    ############################################################################
    # FFT
    def fft_old(self, X, XH):
        X =  self._reshape_tensor(self.get(X))
        XH = self._reshape_tensor(self.get(XH))
        _fft(X, XH)
    def fft(self, X, XH):
        """
        To deal with intel's issue with slow FFT(real) data?
        """
        X = self._reshape_tensor(self.get(X))
        if X.dtype == float:
            HELPER = np.empty(X.shape, dtype=complex)
            ne.evaluate('X', out=HELPER)
        else:
            HELPER = X
        XH = self._reshape_tensor(self.get(XH))
        _fft(HELPER, XH)
    def ifft(self, XH, X):
        X =  self._reshape_tensor(self.get(X))
        XH = self._reshape_tensor(self.get(XH))
        _ifft(XH, X)

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
        for k in range(sh3):
            C[i,k] = 0.0
            for j in range(sh2):
                C[i,k] += A[i,j]*B[j,k]
@numba.njit(parallel=True)
def _mat_mat(M1, M2, M3):
    n = M1.shape[-1]
    for i in numba.prange(n):
        __mat_mat(M1[:,:,i], M2[:,:,i], M3[:,:,i])

@numba.njit()
def __mat_mat_tA(A, B, C):
    sh1 = A.shape[1]
    sh2 = A.shape[0]
    sh3 = B.shape[1]
    for i in range(sh1):
        for k in range(sh3):
            C[i,k] = 0.0
            for j in range(sh2):
                C[i,k] += A[j,i]*B[j,k]
@numba.njit(parallel=True)
def _mat_mat_tA(M1, M2, M3):
    n = M1.shape[-1]
    for i in numba.prange(n):
        __mat_mat_tA(M1[:,:,i], M2[:,:,i], M3[:,:,i])

@numba.njit(parallel=True)
def _dot0(M1, M2, M3):
    n = M1.shape[-1]
    m = M1.shape[0]
    for i in range(n):
        M3[i] = 0
    for i in numba.prange(n):
        for j in range(m):
            M3[i] += M1[j,i]*M2[j,i]
@numba.njit(parallel=True)
def _dot1(M1, M2, M3):
    n = M3.shape[-1]
    m1 = M2.shape[0]
    m2 = M2.shape[1]
    for i in range(n):
        for j in range(m2):
            M3[j,i] = 0
    for i in numba.prange(n):
        for j in range(m2):
            for k in range(m1):
                M3[j,i] += M1[k,i]*M2[k,j,i]
@numba.njit(parallel=True)
def _dot2(M1, M2, M3):
    n = M3.shape[-1]
    m1 = M2.shape[0]
    m2 = M2.shape[1]
    m3 = M2.shape[2]
    for i in range(n):
        for j in range(m2):
            for k in range(m3):
                M3[j,k,i] = 0
    for i in numba.prange(n):
        for j in range(m2):
            for k in range(m3):
                for l in range(m1):
                    M3[j,k,i] += M1[l,i]*M2[l,j,k,i]

@numba.njit(parallel=True)
def _outer0(M1, M2, M3):
    n = M3.shape[-1]
    m1 = M3.shape[0]
    for i in numba.prange(n):
        for j in range(m1):
                M3[j,i] = M1[j,i]*M2[i]
@numba.njit(parallel=True)
def _outer1(M1, M2, M3):
    n = M3.shape[-1]
    m1 = M3.shape[0]
    m2 = M3.shape[1]
    for i in numba.prange(n):
        for j in range(m1):
            for k in range(m2):
                M3[j,k,i] = M1[j,i]*M2[k,i]
@numba.njit(parallel=True)
def _outer2(M1, M2, M3):
    n = M3.shape[-1]
    m1 = M3.shape[0]
    m2 = M3.shape[1]
    m3 = M3.shape[2]
    for i in numba.prange(n):
        for j in range(m1):
            for k in range(m2):
                for l in range(m3):
                    M3[j,k,l,i] = M1[j,i]*M2[k,l,i]

@numba.njit(parallel=True)
def _symmetrize(M1, M2):
    n = M1.shape[-1]
    m = M1.shape[0]
    for i in numba.prange(n):
        for j in range(m):
            for k in range(m):
                M2[j,k,i] = 0.5*(M1[j,k,i] + M1[k,j,i])

def _realit(x, realit):
    return x.real if realit else x
def _fft(u, uh):
    n = u.shape[0]
    for i in range(n):
        uh[i] = np.fft.fftn(u[i])
def _ifft(uh, u):
    realit = u.dtype == float
    n = u.shape[0]
    for i in range(n):
        u[i] = _realit(np.fft.ifftn(uh[i]), realit)

_common_einsums = {
    'ij...,jk...->ik...' : _mat_mat,
    'ji...,jk...->ik...' : _mat_mat_tA,
    'i...,i...->...'     : _dot0,
    'i...,ij...->j...'   : _dot1,
    'i...,ijk...->jk...' : _dot2,
    'i...,...->i...'     : _outer0,
    'i...,j...->ij...'   : _outer1,
    'i...,jk...->ijk...' : _outer2,
}
def _call_common_einsum(M1, M2, O, instr):
    _common_einsums[instr](M1, M2, O)

