import numpy as np
import scipy as sp
import numexpr as ne
import numba
import multiprocessing as mp
from .utilities import Bunch, RawArray, mmap_zeros, mmap_empty, MyString
import mkl
import sys

max_processes = mkl.get_max_threads()
ne.set_num_threads(max_processes)

def binop(a, b, sep):
    new_refs = a.references + b.references
    return Expression(str(a) + sep + str(b), new_refs)

def turn_to_expression(obj):
    return obj if isinstance(obj, Expression) else Expression(obj)

class Expression(object):
    def __init__(self, obj, refs=None):
        self.string = '(' + str(obj) + ')'
        self.references = [] if refs is None else refs
    # add any embedded object to references so it doesn't get cleaned
    def _add_ref(self, r):
        self.references += r.references
    # arithmetic operations
    def __add__(a, b):
        return binop(a, turn_to_expression(b), ' + ')
    def __radd__(a, b):
        b = turn_to_expression(b)
        return binop(b, a, ' + ')
    def __sub__(a, b):
        b = turn_to_expression(b)
        return binop(a, b, ' - ')
    def __rsub__(a, b):
        b = turn_to_expression(b)
        return binop(b, a, ' - ')
    def __mul__(a, b):
        b = turn_to_expression(b)
        return binop(a, b, '*')
    def __rmul__(a, b):
        b = turn_to_expression(b)
        return binop(b, a, '*')
    def __truediv__(a, b):
        b = turn_to_expression(b)
        return binop(a, b, '/')
    def __rtruediv__(a, b):
        b = turn_to_expression(b)
        return binop(b, a, '/')
    def __pow__(a,b):
        b = turn_to_expression(b)
        return binop(a, b, '**')
    def __rpow__(a,b):
        b = turn_to_expression(b)
        return binop(b, a, '**')
    def __neg__(a):
        a = turn_to_expression(a)
        return Expression('-' + str(a))
    def __iadd__(self, b):
        self.engine| self << self + turn_to_expression(b)
        return self
    # equality operator
    def __lshift__(a, b):
        b = turn_to_expression(b)
        return binop(a, b, ' << ')
    # string and print methods
    def __str__(self):
        return self.string
    def __repr__(self):
        return "Expression: '" + self.string + "'"
    # # delete method
    # def __del__(self):
    #     for ref in self.references:
    #         del ref

def get_slice_len(_slice, n):
    return 0 if type(_slice) is int else len(range(*_slice.indices(n)))
def get_full_slice(_slice, n):
    if type(_slice) is not tuple:
        _slice = (_slice,)
    return tuple(list(_slice) + [slice(None),]*(n-len(_slice)))

class Field(Expression):
    def __init__(self, tensor_shape, field_shape, dtype, data, identifier, engine):
        self.tensor_shape = tensor_shape
        self.field_shape = field_shape
        self.shape = tuple(list(self.tensor_shape) + list(self.field_shape))
        self.tensor_len = len(self.tensor_shape)
        self.field_len = len(self.field_shape)
        self.shape_len = len(self.shape)
        self.dtype = dtype
        self.data = data
        self.identifier = identifier
        self.engine = engine
        self.deleted = False
        self.tensor_raveled_data = self.ravel_tensor_indeces()
        self.field_raveled_data = self.ravel_field_indeces()
        self.references = [self,]
        self.string = identifier
    def _compute_sliced_size(self, _slice):
        pass
    def __repr__(self):
        return self.data.__repr__()
    def __del__(self):
        if not self.deleted:
            self.engine._delete(self.identifier)
        self.deleted = True
    def __call__(self):
        return self.data
    def __setitem__(self, _slice, value):
        if not isinstance(value, Expression):
            self.data[_slice] = value
        else:
            self.engine| self[_slice] << turn_to_expression(value)
    def __getitem__(self, _slice):
        new_var, new_identifier = self.engine._get_slice(self.identifier, _slice)
        new_tensor_shape, new_field_shape = self._compute_sliced_size(_slice)
        return Field(new_tensor_shape, new_field_shape, self.dtype, new_var, new_identifier, self.engine)
    def _compute_sliced_size(self, _slice):
        # extract tensor portion of slice and field portion of slice
        full_slice = get_full_slice(_slice, self.shape_len)
        tensor_slice = full_slice[:self.tensor_len]
        field_slice = full_slice[self.tensor_len:]
        # get new lengths of the dims
        new_tensor_shape = [get_slice_len(tensor_slice[i], self.tensor_shape[i]) for i in range(len(tensor_slice))]
        new_field_shape = [get_slice_len(field_slice[i], self.field_shape[i]) for i in range(len(field_slice))]
        # remove collapsed dims from these
        new_tensor_shape = [x for x in new_tensor_shape if x != 0]
        new_field_shape = [x for x in new_field_shape if x != 0]
        return new_tensor_shape, new_field_shape
    def ravel_tensor_indeces(self):
        if self.tensor_len == 0:
            new_tensor_shape = []
        else:
            new_tensor_shape = [np.prod(self.tensor_shape),]
        new_field_shape = self.field_shape
        new_shape = tuple(new_tensor_shape + new_field_shape)
        return self.data.reshape(new_shape)
    def ravel_field_indeces(self):
        new_tensor_shape = self.tensor_shape
        new_field_shape = [np.prod(self.field_shape),]
        new_shape = tuple(new_tensor_shape + new_field_shape)
        return self.data.reshape(new_shape)

def get_sane_dtype(dtype):
    if dtype == float:
        return float
    if dtype == complex:
        return complex
class MemoryPool(object):
    def __init__(self):
        self.dict = Bunch({})
    def __iadd__(self, new):
        key = (new.shape, get_sane_dtype(new.dtype))
        if key not in self.dict:
            self.dict[key] = []
        self.dict[key].append(new)
        return self
    def __call__(self, shape, dtype):
        key = (shape, get_sane_dtype(dtype))
        if key not in self.dict or len(self.dict[key]) == 0:
            return None
        else:
            return self.dict[key].pop()

class Engine(object):
    def __init__(self):
        self.variables = Bunch({})
        self.primary_variables = Bunch({})
        self.variables['__pi__'] = np.pi
        self.variables['__e__'] = np.e
        self.current_identifier = MyString('a')
        self.memory_pool = MemoryPool()
    def __str__(self):
        return repr(self.primary_variables)

    ############################################################################
    # methods for dealing with variable allocation
    def _query_memory_pool(self, shape, dtype):
        return self.memory_pool(shape, dtype)
    def allocate_many(self, var_list, zeros=True):
        func = self.zeros if zeros else self.empty
        return [func(*sublist) for sublist in var_list]
    def empty(self, tensor_shape, field_shape, dtype=float, name=None):
        shape = tuple(list(tensor_shape) + list(field_shape))
        data = self._query_memory_pool(shape, dtype)
        if data is None:
            data = mmap_empty(shape=shape, dtype=dtype)
        identifier = self._get_identifier(name)
        self.variables[identifier] = data
        self.primary_variables[identifier] = None
        return Field(tensor_shape, field_shape, dtype, data, identifier, self)
    def zeros(self, tensor_shape, field_shape, dtype=float, name=None):
        F = self.empty(tensor_shape, field_shape, dtype=dtype, name=name)
        F[:] = 0.0
        return F
    def field_from(self, arr, field_len):
        shape = arr.shape
        tensor_shape = shape[:-field_len]
        field_shape = shape[field_len:]
        field = self.empty(tensor_shape, field_shape, arr.dtype)
        field[:] = arr
        return field
    def _check_name(self, name):
        return name not in self.variables
    def _get_identifier(self, name=None):
        if name is not None:
            if self._check_name(name):
                ret = name
            else:
                raise Exception('Name ' + name + ' already in use.')
        else:
            ret = '___' + self.current_identifier
            self.current_identifier += 1
        return ret
    def _delete(self, identifier):
        if identifier in self.primary_variables:
            del self.primary_variables[identifier]
            self.memory_pool += self.variables[identifier]
        del self.variables[identifier]
    def list(self, full=False):
        if full:
            print(self.variables)
        else:
            print(self.primary_variables)

    ############################################################################
    # methods for dealing with variables added, but not allocated to the class
    # these should just be numpy arrays and scalars, not Fields
    def add(self, name, value):
        """
        Add value to the variable dictionary
        Note that this differs from the behavior of allocate!
        What is added is NOT a memmap, and must be used in a read-only manner
        in multiprocessing code!
        """
        self.variables[name] = value
        return Expression(name)
    def add_many(self, add_list):
        for sublist in add_list:
            self.add(*sublist)

    ############################################################################
    # field slicing
    def _get_slice(self, identifier, _slice):
        new_var = self.variables[identifier][_slice]
        new_identifier = self._get_identifier()
        self.variables[new_identifier] = new_var
        return new_var, new_identifier

    ############################################################################
    # evaluate an expression
    def eval(self, expr):
        """
        Evaluate an expression using numexpr
        If an output field is not provided in the expression, this will return a
        bare numpy array, since there is no way to determine tensor/field shapes
        """
        if '<<' in str(expr):
            outname, instr = [st.strip() for st in str(expr)[1:-1].split('<<')]
            out = self.variables[outname]
            ne.evaluate(instr, out=out, local_dict=self.variables)
        else:
            return ne.evaluate(str(expr), local_dict=self.variables)
    def __or__(self, expr):
        return self.eval(expr)
    ############################################################################
    # einsum
    def einsum(self, instr, evars, out=None):
        """
        Implements the einsum function for fields
        If out is None, it is assumed the einsum only operates on tensor indeces
        """
        if instr in _common_einsums:
            self._common_einsum(evars, out, instr)
        else:
            field_len = evars[0].field_len
            nevars = [var.data for var in evars]
            if out is None:
                out = np.einsum(instr, *nevars)
                out = self.field_from(out, field_len)
            else:
                np.einsum(instr, *nevars, out=out.data)
            return out
    def list_common_einsum(self):
        return _common_einsums.keys()
    def _common_einsum(self, evars, out, instr):
        M1 = evars[0].field_raveled_data
        M2 = evars[1].field_raveled_data
        M3 = out.field_raveled_data
        _call_common_einsum(M1, M2, M3, instr)

    ############################################################################
    # FFT
    def fft_old(self, X, XH):
        _fft(X.tensor_raveled_data, XH.tensor_raveled_data)
    def fft(self, X, XH):
        """
        To deal with intel's issue with slow FFT(real) data?
        """
        X = X.tensor_raveled_data
        return_to_pool = False
        if X.dtype == float:
            HELPER = self._query_memory_pool(X.shape, complex)
            if HELPER is None:
                HELPER = np.empty(X.shape, dtype=complex)
            HELPER[:] = X
            return_to_pool = True
        else:
            HELPER = X
        _fft(HELPER, XH.tensor_raveled_data)
        if return_to_pool:
            self.memory_pool += HELPER
    def ifft(self, XH, X):
        _ifft(XH.tensor_raveled_data, X.tensor_raveled_data)


class Engine_old(object):
    def __init__(self, base_shape):
        pass

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
    def copy_from_to(self, vf, vt):
        self.get(vt)[:] = self.get(vf)

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

def _cos(expr):
    return Expression('cos(' + str(expr)+ ')')
def _sin(expr):
    return Expression('sin(' + str(expr) + ')')
def _tan(expr):
    return Expression('tan(' + str(expr) + ')')
def _arccos(expr):
    return Expression('arccos(' + str(expr) + ')')
def _arcsin(expr):
    return Expression('arcsin(' + str(expr) + ')')
def _arctan(expr):
    return Expression('arctan(' + str(expr) + ')')
def _arctan2(expr1, expr2):
    return Expression('arctan2(' + str(expr1) + ', ' + str(expr2) + ')')
def _cosh(expr):
    return Expression('cosh(' + str(expr) + ')')
def _sinh(expr):
    return Expression('sinh(' + str(expr) + ')')
def _tanh(expr):
    return Expression('tanh(' + str(expr) + ')')
def _arccosh(expr):
    return Expression('arccosh(' + str(expr) + ')')
def _arcsinh(expr):
    return Expression('arcsinh(' + str(expr) + ')')
def _arctanh(expr):
    return Expression('arctanh(' + str(expr) + ')')
def _log(expr):
    return Expression('log(' + str(expr) + ')')
def _log10(expr):
    return Expression('log10(' + str(expr) + ')')
def _log1p(expr):
    return Expression('log1p(' + str(expr) + ')')
def _exp(expr):
    return Expression('exp(' + str(expr) + ')')
def _expm1(expr):
    return Expression('expm1(' + str(expr) + ')')
def _sqrt(expr):
    return Expression('sqrt(' + str(expr) + ')')
def _abs(expr):
    return Expression('abs(' + str(expr) + ')')
def _conj(expr):
    return Expression('conj(' + str(expr) + ')')
def _real(expr):
    return Expression('real(' + str(expr) + ')')
def _imag(expr):
    return Expression('imag(' + str(expr) + ')')
def _complex(expr1, expr2):
    return Expression('complex(' + str(expr1) + ', ' + str(expr2) + ')')
def _sum(expr1, expr2=None):
    return Expression('sum(' + str(expr1) + ', ' + str(expr2) + ')')
def _prod(expr1, expr2=None):
    return Expression('prod(' + str(expr1) + ', ' + str(expr2) + ')')

_pi = Expression('__pi__')
_e = Expression('__e__')
