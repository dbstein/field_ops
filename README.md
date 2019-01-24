# field_ops: fast operations on fields

A common scenario in solving PDE is that we have a scalar fields, vector fields, and tensor fields, all defined at every point on some large grid. Second rank tensors, such as the gradient of the velocity field, would naturally be represented on a grid of size [3, 3, nx, ny, nz] in three dimensions. It is typical in solving PDE to have to do tensor style operations on these: for example, at every point in space, to compute the eigendecomposition of a tensor, or to multiply these two tensors together.

The goal of field_ops is to make this simple, fast, and seamless. The heart is the Engine class, which allocates variables into shared memory maps to allow the variables to be shared between processes. Variables can be accessed easily, and the objects returned manipulated like ordinary numpy matrices. Fast operations on the variables are enabled in a variety of ways, depending on what is fastest.

What is supported right now:

1. Allocation of fields (one by one, or many at a time)
2. Interface to the evaluate function in **numexpr**
3. Interface to the very useful **einsum** function from **numpy**
4. Some specialize einsum operations coded in numba for improved speed
5. Parallelized eigendecompositions of symmetric matrices (multiprocessed **numpy.eigh**)
6. Parallelized matrix-matrix multiplication (via **numba**)
7. For fields added through the **allocate** method (and **allocate_many**), all subfields are automatically pointed at and added to the variable dictionary. For example, if you run `allocate('D', [3,3], float)`{:.python}, the variables dictionary will be automatically populated with **D_00**, **D_01**, etc, where **D_00** is a view of **D** equivalent to **D[0,0]**. This is an option (*by default not used*) for things added through the **add** and **add_many** methods. There are also separate ways to do this (via the **point_at** and **point_at_all_subfields** methods)
8. Because of 7, all names used to call things can include indexes! So the code:
```python
Engine.evaluate('D[0,0] + D[1,1]', 'D[2,2]')
```
is equivalent to:
```python
D[2,2] = D[0,0] + D[1,1]
```
but will make use of **numexpr** for speed.

The eigen-decompositions are sped up using the **multiprocessing** package. Processing pools are spawned (and respawned) whenever new fields are allocated. This requires modest overhead at allocation time but extremely minimal overhead at runtime, so that operations over even relatively small fields are effectively paralellized.

The matrix-matrix multiplication is implemented in **numba**. Compilation is at runtime, so the first time the function is used for specific parameter types there will be some overhead. A few einsums are also implemented in numba. You can see these by calling **Engine.list_common_einsum()**. These are accessed by calling the einsum function with the exact instruction string given in the **Engine.list_common_einsum()**, which can also just be seen at the bottom of the file core.py.

To be added:
1. A process-parallelized einsum? This seems to be hard to do because of its preferred memory ordering.
2. A factory for generating proccess-based parallelism using the processing pool. This is a little tricky because the linear algebra functions want a different ordering, so there are transposes involved and its not super clean.
3. More custom-made numba functionality, as I need it

## Requirements
The python packages **numpy**, **scipy**, **numexpr**, **numba**, **mkl-service**, **multiprocessing** and **mmap** are required (the last two are base packages). As always, a good installation of python is required for good perfomance. Consider using the *Intel Distribution for Python*.
