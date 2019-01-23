# field_ops: fast operations on fields

A common scenario in solving PDE is that we have a scalar fields, vector fields, and tensor fields, all defined at every point on some large grid. Second rank tensors, such as the gradient of the velocity field, would naturally be represented on a grid of size [3, 3, nx, ny, nz] in three dimensions. It is typical in solving PDE to have to do tensor style operations on these: for example, at every point in space, to compute the eigendecomposition of a tensor, or to multiply these two tensors together.

The goal of field_ops is to make this simple, fast, and seamless. The heart is the Engine class, which allocates variables into shared memory maps to allow the variables to be shared between subclasses. Variables can be accessed easily, and the objects returned manipulated like ordinary numpy matrices. Fast operations on the variables are enabled in a variety of ways, depending on what's best.

What is supported right now:

1. Allocation of fields (one by one, or many at a time)
2. Interface to the evaluate function in numexpr
3. Interface to the very useful einsum function from numpy
4. Parallelized eigendecompositions of symmetric matrices (through np.eigh)
5. Parallelized matrix-matrix multiplication

The eigen-decompositions are sped up using multiprocessing. Processing pools are spawned (and respawned) whenever new fields are allocated. This requires modest overhead at allocation time but extremely minimal overhead at runtime, so that operations over even relatively small fields are effectively paralellized.

The matrix-matrix multiplication is implemented in numba. Compilation is at runtime, so the first time the function is used for specific parameter types there will be some overhead.

To be added:
1. A parallelized einsum (this seems to be hard to do because of its preferred memory ordering)
2. A factory for generating proccess-based parallelism using the processing pool
3. More custom-made numba functionality, as I need it

## Requirements
The python packages numpy, scipy, numexpr, numba, mkl-service, multiprocessing and mmap are required (the last two are base packages). As always, a good installation of python is required for good perfomance. Consider using the Intel Distribution for Python.
