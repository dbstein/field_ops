import numpy as np
import multiprocessing as mp
import mmap
import itertools

def multi_iter(shape):
    return itertools.product(*[range(s) for s in shape])

class MyString(str):
    def __add__(self, x):
        # If we're trying to add anything but an int, do normal string
        # addition.
        if type(x) is not int:
            return str.__add__(self, x)

        res = ''
        i = len(self)-1
        while x > 0:
            # Get the ASCII code of the i-th letter and "normalize" it
            # so that a is 0, b is 1, etc.
            # If we are at the end of the string, make it -1, so that if we
            # need to "add" 1, we get a.
            if i >= 0:
                c = ord(self[i]) - 97
            else:
                c = -1

            # Calculate the number of positions by which the letter is to be
            # "rotated".
            pos = x % 26

            # Calculate x for the next letter, add a "carry" if needed.
            x //= 26
            if c + pos >= 26:
                x += 1

            # Do the "rotation".
            c = (c + pos) % 26

            # Add the letter at the beginning of the string.
            res = chr(c + 97) + res

            i -= 1

        # If we didn't reach the end of the string, add the rest of the string back.
        if i >= 0:
            res = self[:i+1] + res

        return MyString(res)

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

def __unpickle__(ai, dtype):
    dtype = np.dtype(dtype)
    tp = np.ctypeslib._typecodes['|u1']

    # if there are strides, use strides, otherwise the stride is the itemsize of dtype
    if ai['strides']:
        tp *= ai['strides'][-1]
    else:
        tp *= dtype.itemsize

    for i in np.asarray(ai['shape'])[::-1]:
        tp *= i

    # grab a flat char array at the sharemem address, with length at least contain ai required
    ra = tp.from_address(ai['data'][0])
    buffer = np.ctypeslib.as_array(ra).ravel()
    # view it as what it should look like
    shm = np.ndarray(buffer=buffer, dtype=dtype, 
            strides=ai['strides'], shape=ai['shape']).view(type=anonymousmemmap)
    return shm

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

