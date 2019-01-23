import numpy as np
import multiprocessing as mp
import mmap

class Bunch(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class RawArray(object):
    def __init__(self, array=None, shape=None, dtype=None):
        """
        Generate a shared memory array (without locking)
        either shape + type need to be specified:
            type should be float or complex
        or array should be specified:
            dtype of array should be float or complex
        if all three are specified, default to allocation via array method

        Note that the initial array and the shared array (self.numpy) do not
        point at the same data!
        """
        if array is not None:
            dtype = array.dtype
            shape = array.shape
        if dtype not in [float, complex]:
            raise Exception('Specified type is not supported.')
        m = 1 if dtype == float else 2
        self.raw = mp.RawArray('d', m*int(np.prod(shape)))
        self.numpy = np.frombuffer(self.raw, dtype=dtype).reshape(shape)
        if array is not None:
            np.copyto(self.numpy, array)
    def __call__(self):
        return self.numpy

class anonymousmemmap(np.memmap):
    """
    Arrays allocated on shared memory. 
    The array is stored in an anonymous memory map that is shared between child-processes.

    This code is taken (nearly) verbatim from the 'sharedmem' package
    And allows much faster allocation of shared memory than the RawArray Package
    """
    def __new__(subtype, shape, dtype=np.uint8, order='C'):

        descr = np.dtype(dtype)
        _dbytes = descr.itemsize

        shape = np.atleast_1d(shape)
        size = 1
        for k in shape:
            size *= k

        bytes = int(size*_dbytes)

        if bytes > 0:
            mm = mmap.mmap(-1, bytes)
        else:
            mm = np.empty(0, dtype=descr)
        self = np.ndarray.__new__(subtype, shape, dtype=descr, buffer=mm, order=order)
        self._mmap = mm
        return self
        
    def __array_wrap__(self, outarr, context=None):
    # after ufunc this won't be on shm!
        return np.ndarray.__array_wrap__(self.view(np.ndarray), outarr, context)

    def __reduce__(self):
        return __unpickle__, (self.__array_interface__, self.dtype)

def mmap_empty(shape, dtype):
    """
    Create an empty shared memory array.
    """
    return anonymousmemmap(shape, dtype)

def mmap_zeros(shape, dtype):
    """
    Create an empty shared memory array.
    """
    new = anonymousmemmap(shape, dtype)
    new[:] = 0.0
    return new

