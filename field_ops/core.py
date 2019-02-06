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
        self.known_evaluations = Bunch({})
        self._add_base_constants()
        self.pool_formed = False

    ############################################################################
    # methods for dealing with variable allocation
    def __allocate(self, name, shape, dtype=float):
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
    def allocate_from(self, name, arr):
        shape = 1 if len(arr.shape) == 0 else arr.shape
        self.__allocate(name, shape, arr.dtype)
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
            pass
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
    def _point_at_view(self, name, view):
        self.add(name, view)
    def point_at_many(self, point_list):
        for sublist in point_list:
            self.point_at(*sublist)
    # def point_at_all_subfields(self, field):
    #     F = self.get(field)
    #     field_sh, base_sh = self._extract_shapes(F)
    #     field_lsh = len(field_sh)
    #     if field_lsh > 0:
    #         sh_list = [np.arange(x) for x in field_sh]
    #         n = np.prod(field_sh)
    #         # generate all codes
    #         codes = []
    #         for i in range(field_lsh):
    #             code = np.empty(n, dtype=object)
    #             excess = np.prod(field_sh[i+1:])
    #             code = np.repeat(np.arange(field_sh[i]), excess)
    #             precess = np.prod(field_sh[:i])
    #             code = np.tile(code, precess)
    #             codes.append(code)
    #         code_reorder = []
    #         for i in range(n):
    #             code = []
    #             for j in range(field_lsh):
    #                 code.append(codes[j][i])
    #             code_reorder.append(code)
    #         codes = code_reorder
    #         # point everything where it should go
    #         for i in range(n):
    #             name = field + '_'
    #             field_inds = []
    #             for j in range(field_lsh):
    #                 code = codes[i][j]
    #                 name += str(code)
    #                 field_inds.append(code)
    #             self.point_at(name, field, field_inds)

    ############################################################################
    # methods for extracting variables
    def _check_slice(self, string):
        """
        given string, e.g. 'A[0,1]', checks to see if a view to the slice exists
        if it does, simply return that view
        otherwise, create the view, add to variables dict, and return
        """
        slice_name = _get_slice_name(string)
        if slice_name not in self.variables:
            out = _parse_variable(string)
            if len(out) > 1:
                varname, slice_str = out
                new_view = _get_slice(self.variables[varname], slice_str)
                self._point_at_view(slice_name, new_view)
        return slice_name
    def get(self, name):
        name = self._check_slice(name)
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
        return not all([var in self.mp_variables for var in var_names])

    ############################################################################
    # method for easing interface to numexpr
    def evaluate_many(self, instrs, replacements=None):
        for instr in instrs:
            self.evaluate(instr, replacements)
    def evaluate(self, instr, replacements=None, outname=None):
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
        This function also creates
        In either case, if the output variable doesn't yet exist,
            the function will create it
        Replacements should be a tuple of tuples giving text replacements to make
        i.e.:
        self.evaluate('c = a + b', (('a', 'd'), ('c', 'e')) ) is equivalent to:
            e = d + b
        """
        # first check to see if this is in the known evaluations dict
        if (instr, replacements) in self.known_evaluations:
            instr, out = self.known_evaluations[instr, replacements]
            ne.evaluate(instr, local_dict=self.variables, out=out)
        # if not, construct the evaluation, run it, and store it
        else:
            if outname is None:
                outname, instr = [st.strip() for st in instr.split('=')]
            outname = _replace_in_string(outname, replacements)
            instr = _replace_in_string(instr, replacements)
            outname = self._check_slice(outname)
            instr = self._check_instruction(instr)
            if outname in self.variables:
                ne.evaluate(instr, local_dict=self.variables, out=self.get(outname))
            else:
                out = ne.evaluate(instr, local_dict=self.variables)
                self.allocate_from(outname, out)
            self.known_evaluations[instr, replacements] = (instr, self.get(outname))
    def _check_instruction(self, string):
        """
        given instruction string, 'A[0,1] + B[0]'
        finds all sliced variables, and for each sliced variable,
        runs _check_slice
        and returns a new instruction string with the sliced names
        """        
        sliced_variables = _parse_instruction(string)
        new_names = [self._check_slice(var) for var in sliced_variables]
        for old_name, new_name in zip(sliced_variables, new_names):
            string = string.replace(old_name, new_name)
        return string

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
    # this parallel version is slower for now..., work on this?
    # a good version of this would get rid of the need for the common_einsums
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

### functions dealing with parsing equations/slices/variable names
def _replace_in_string(s, rs):
    if rs is not None:
        for r in rs:
            s = s.replace(r[0], r[1])
    return s
def _split_non_alnum(s):
   pos = len(s)-1
   while pos >= 0 and (s[pos].isalnum() or s[pos]=='_'):
      pos-=1
   return s[pos+1:]
def _parse_instruction(s):
    parts = s.split('[')
    return [_split_non_alnum(parts[i]) + '[' + parts[i+1].split(']')[0] + ']' \
                                 for i in range(len(parts)-1)]
def _get_slice(x, s):
    return x[_slice_from_string(s)]
def _parse_variable(s):
    parts = s.split('[')
    return (parts[0], '[' + parts[1]) if len(parts) > 1 else (s,)
def _slice_from_string(s):
    return _parse_complicated_slice(s) if _slice_from_string_is_complicated(s) \
            else _parse_multi_slice(s)
def _slice_from_string_is_complicated(s):
    return '(' in s
def _parse_complicated_slice(s):
    parts = s[1:-1].split(',')
    axes = [int(part.split('(')[0]) for part in parts]
    slices = [_parse_slice(part.split('(')[1][:-1]) for part in parts]
    slice_object = [slice(None),]*(np.max(axes)+1)
    for i in range(len(parts)):
        slice_object[axes[i]] = slices[i]
    return tuple(slice_object)
def _parse_multi_slice(s):
    return tuple([_parse_slice(s) for s in s[1:-1].split(',')])
def _parse_slice(s):
    parts = s.split(':')
    if len(parts) == 1:
        return int(parts[0])
    else:
        return slice(*[int(p) if p else None for p in parts])
def _get_slice_name(s):
    s = s.replace('(', 'YYY')
    s = s.replace(')', 'YYY')
    s = s.replace('[', '___')
    s = s.replace(']', '___')
    s = s.replace(',', '__')
    s = s.replace(':', 'ZZZ')
    s = s.replace('-', 'XXX')
    return s

