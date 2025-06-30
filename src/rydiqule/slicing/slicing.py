"""
A handful of tools that solvers use to interface with slicing matrix stacks
"""

import psutil
import warnings
import itertools

import numpy as np

from typing import Tuple, Union, Optional, Iterable, Iterator, List
from typing_extensions import Unpack

from ..exceptions import RydiquleError, RydiquleWarning


def compute_grid(stack_shape: Tuple[int, ...], n_slices: int) -> List[np.ndarray]:
    """Calculate the bin edges to break a given stack shape into at least a certain number of pieces

    Works by iterating first over a number of slices per axis (N=1,2,3),
    then over each in the stack shape, splitting the axis into N slices,
    and comparing the total number of slices to the number specified.
    In a sense, the algorithm factors a number greater than or equal to n_slices,
    then breaks the stack along each axis according to this factorization.
    If the axis lengths do not break evenly into the appropriate number of pieces,
    the bin edges are truncated to an integer.
    This means that the slices are not guaranteed to be `(1/n_slices)`,
    but they will be close enough for most cases. 

    Parameters
    ----------
    stack_shape : tuple of int
        The shape of the stack to be sliced.
        Does not include Hamiltonian or matrix equation dimensions,
        so for a hamiltonain stack of shape `(*l,n,n)`, `stack_shape` will be `*l`.
    n_slices : int
        The number of slices into which to break the hamiltonian. Lower bound on the number of 
        slices there will actually be.

    Returns
    -------
    list(np.ndarray)
        The list of bin edges axis by axis. Can be passed to :func:`matrix_slice` as the `edges` 
        argument.
        
    Examples
    --------
    >>> import rydiqule.slicing.slicing as slicing
    >>> stack_shape=(10,10)
    >>> print(slicing.compute_grid(stack_shape, 4))
    [array([ 0,  5, 10]), array([ 0,  5, 10])]
    >>> print(slicing.compute_grid(stack_shape, 6))
    [array([ 0,  3,  6, 10]), array([ 0,  5, 10])]
    
    """
    
    if n_slices > np.prod(stack_shape):
        raise RydiquleError(f"Too many slices ({n_slices}) for stack of shape {stack_shape}")
    
    n_ax_slices = np.ones_like(stack_shape)
    
    current_slices = 1
    current_axis = 0
    
    while current_slices < n_slices:
        n_ax_slices[current_axis]+= 1
        current_slices = np.prod(n_ax_slices)
        current_axis += 1
        
        if current_axis ==len(stack_shape):
            current_axis=0
    total_axes = sum([1 for a in n_ax_slices if a>1])
    
    return [np.linspace(0,stack_shape[i], n_ax_slices[i]+1, dtype=int) for i in range(total_axes)]

# in python 3.11+ return should be Iterator[tuple[tuple[slice, ...], *tuple[np.ndarray]]]
def matrix_slice(*matrices: np.ndarray,
                 edges: Optional[Iterable] = None,
                 n_slices: Optional[int] = None
                 ) -> Iterator[Tuple[Tuple[slice, ...], Unpack[Tuple[np.ndarray, ...]]]]:
    """
    Generator that returns parts of a stack of matrices.

    Given a stack of n by n matrices, produces a genererator which returns the given matrices
    in the specified number of smaller stacks. For example, given a stack of matrices of shape
    `(10,10,4,4)` with 4 slices, generates 4 stacks of shape `(5,4,4)`. Due to the nature
    of the slicing, the number of slices might be slightly greater that the number specified.
    Output matrices will be broadcastable by numpy's broadcasting rules.
    Input arrays are interpreted as a stack, with the last 2 dimensions staying intact.

    Args
    ----
    matrices: np.ndarray 
        matrix stacks to be sliced. All matrices must be of shapes that can be
        broadcast by numpy's broadcasting rules, with the additional restriction that all matrices
        must have the same number of dimensions, even if some dimensions are of size 1. For example,
        matrices of sizes `(10,1,4,4)` and `(1,20,4,4)` can be sliced together. 
    edges: iterable, optional
        The values along each axis that define the edges of bins on an n-dimensional grid.
        For example, to slice a grid of hamiltonians with stack_shape `(10,10)` into 4 pieces,
        `edges` could be defined as `[0,5,10]` for each of the 2 stack axes. 
    n_slices : int, optional
        The number of slices into which to break the matrix stack. Ignored if the
        `edges` parameter is not `None`. Must be specified as an integer value if `edges` is `None`,
        ignored otherwise. Defaults to None. 

    Yields
    ------
    tuple of slices
        slicing for each corresponding matrix
    numpy.ndarray
        Slice of hamiltonian stack
        
    Examples
    --------
    >>> M1 = np.ones((1,20,4,4))
    >>> M2 = np.ones((20,1,4,4))
    >>> M3 = np.ones((20,20,4,4))
    >>> axis0_edges = np.array([0,10,20])
    >>> axis1_edges = np.array([0,10,20])
    >>> for idx,m1,m2, m3 in rq.slicing.slicing.matrix_slice(M1, M2, M3, edges=[axis0_edges, axis1_edges]):
    ...     print(m1.shape, m2.shape, m3.shape)
    (1, 10, 4, 4) (10, 1, 4, 4) (10, 10, 4, 4)
    (1, 10, 4, 4) (10, 1, 4, 4) (10, 10, 4, 4)
    (1, 10, 4, 4) (10, 1, 4, 4) (10, 10, 4, 4)
    (1, 10, 4, 4) (10, 1, 4, 4) (10, 10, 4, 4)

    """
    #catch input shape errors and raise more useful errors
    shapes = [m.shape for m in matrices]
    if len(matrices) == 0:
        raise RydiquleError("Must provide at least 1 matrix for slicing")
    if matrices[0].ndim < 2:
        raise RydiquleError("Must have at least 2d matrices to slice")

    try:
        stack_shape = np.broadcast_shapes(*shapes)[:-2]
    except ValueError as err:
        raise RydiquleError(f"Incompatiple input shapes {shapes}") from err
    
    # handle the trivial case of a single hamiltonian
    if len(stack_shape) == 0:
        yield (), *matrices
        return
    #Build the grid if only the number of slices is specified. 
    if edges is None:
        if n_slices is None or n_slices == 1:
            yield (), *matrices
            return
        elif n_slices > 1:
            edges = compute_grid(stack_shape, n_slices)
        else:
            raise RydiquleError("n_slices must be positive int if specified")

    #check that the bins slice each axis appropriately
    for i,e in enumerate(edges):
        if e[0] != 0 or e[-1] != stack_shape[i]:
            raise RydiquleError(
                f"slices must start at 0 and end at the axis length ({stack_shape[i]} for axis {i})"
                )

    start_edges = [e[:-1] for e in edges] # "left" edge
    end_edges = [e[1:] for e in edges] #"right" edge

    #loop over each box, defined by 2 opposite corners p_start and p_end. 
    #eg consider the 3-box from (4,4,4) to (6,6,6)
    #in this case p_start will be (4,4,4) and p_end will be (6,6,6)
    for p_start, p_end in zip(itertools.product(*start_edges), itertools.product(*end_edges)):

        #loop over dimensions for each box and get the box bounds as slices
        slices = []
        for i in range(len(p_start)):
            slices.append(slice(p_start[i], p_end[i]))
        slices = tuple(slices)
        
        m_idx = [] #actual matrix indeces, accounting for broadcasting support
        for m in matrices:
            m_idx.append(tuple(idx
                if m.shape[j] !=1
                else slice(0,None)
                for j, idx in enumerate(slices)
            ))
        
        yield slices, *[m[idx] for m, idx in zip(matrices, m_idx)]
    

def memory_size(shape: Tuple[int, ...], item_size: int) -> int:
    """
    Calculate the memory size, in bytes of an array with the given size and shape. Does not
    calculate the actual array, just theoretical size since this function is intended
    to be used before attempting allocate an array that is too large.

    Args:
        shape (list-like): Shape of the array in question.
        item_size(int): Size of an array element in bytes.

    Returns:
        int: Expected memory size of array in bytes
    
    """
    size: int = np.prod(shape,dtype=np.ulonglong)*item_size
    return size


def get_slice_num(n: int, stack_shape: Tuple[int, ...], doppler_shape: Tuple[int, ...],
                  sum_doppler: bool, weight_doppler: bool,
                  n_slices: Union[int,None] = None,
                  debug: bool = False) -> Tuple[int, Tuple[int, ...]]:
    """
    Estimates the memory required for the desired steady state solve.

    Estimates are fairly accurate, but not guaranteed.
    Goal is to err on allowing edge case solves to proceed.

    Parameters
    ----------
    n: int
        Size of the system basis
    stack_shape: tuple of int
        Tuple of sizes for the hamiltonian stack to be solved
    doppler_shape: tuple of int
        Tuple of sizes for the doppler axes.
        Pass an empty tuple if no doppler averaging.
    sum_doppler: bool
        Whether solution will be summing the doppler average
    weight_doppler: bool
        Whether the solution will apply weights to the doppler averaging
    n_slices: int, default=1
        Manually override the minimum number of hamiltonian slices to use.
    debug: bool, default=False
        Print debug information about the memory calculations.

    Returns
    -------
    n_ham_slices: int
        Number of slices to use when solving the stacked hamiltonian
    out_sol_shape: tuple of int
        Shape of the resulting solution for this calculation.

    Raises
    ------
    RydiquleError: If there isn't enough memory to solve the system
    RydiquleError: If `sum_doppler=False` and full solution does not fit in memory.

    """
    # set the initial number of slices to 1 if None are specified
    #This is primarily to handle the fact that n_slices defaults to None
    # in matrix_slice
    if n_slices is None:
        n_slices=1
    # get total avaialable memory for the system
    total_mem = psutil.virtual_memory().available
    # total number of EOM systems to solve for the stack
    stack_factor = np.prod(stack_shape, dtype=np.ulonglong)
    # track memory to solve a single set of EOMs
    single_eom_mem = 0
    # track mandatory memory per solve
    mand_sol_mem = 0

    # determine minimum memory requirements for a solve
    # 16 bytes for complex128 arrays, 8 bytes for float64 arrays
    ham_shape = (*stack_shape, n, n)
    ham_size = memory_size(ham_shape, 16)
    sol_shape = (*stack_shape, n**2-1)
    sol_size = memory_size(sol_shape, 8)
    # all solves need to make full sized hamiltonians and solutions
    # happens in solve_steady_state
    mand_sol_mem += (ham_size + sol_size)
    # rest of these memory allocations happen in _solve_hamiltonian_stack

    eom_shape = (n**2, n**2)
    const_shape = eom_shape[:-1]
    eom_size = memory_size(eom_shape, 16) + memory_size(const_shape,16)
    # minimum solve must be able to make at least one set of EOMs
    # note that you have to times 2 for remove_ground
    # make_real now deallocates at the end, but keeping 1.5 factor for buffer
    # round factor up to 3 to keep types as int and give buffer
    # happens in generate_eom
    single_eom_mem += eom_size * 3

    if doppler_shape:
        dop_eom_shape = (*doppler_shape, n**2-1, n**2-1)
        dop_eom_size = memory_size(dop_eom_shape,8)
        # all doppler solves must allocate the dop EOMs in addition to bare
        single_eom_mem += dop_eom_size

        # to solve doppler, need to allocate full dop sol size
        dop_sol_shape = dop_eom_shape[:-1]
        dop_sol_size = memory_size(dop_sol_shape,8)
        if weight_doppler:
            # weighting adds extra dop sol allocation
            dop_sol_size *= 2
        else:
            # not summing or weighting only leaves dop_sol_size
            pass

        single_eom_mem += dop_sol_size

    # minimum required memory to solve a single set of EOMs in the stack
    min_sol_mem = mand_sol_mem + single_eom_mem
    # required memory to solve the entire stack at once
    full_sol_mem = mand_sol_mem + single_eom_mem*stack_factor
    # get memory available for solving hamiltonian slices
    available_mem = total_mem - mand_sol_mem
    if doppler_shape and not sum_doppler:
        # not summing doppler, don't allocate summed solutiion
        available_mem -= sol_size
        full_sol_mem -= sol_size
        min_sol_mem -= sol_size

    if doppler_shape and not sum_doppler:
        out_sol_shape = (*doppler_shape, *stack_shape, n**2-1)
    else:
        out_sol_shape = sol_shape
    output_sol_size = memory_size(out_sol_shape, 8)

    if debug:
        print(f'Total available memory: {total_mem/1024**3:.5g} GiB')
        print(f'Min Req memory to solve: {min_sol_mem/1024**3:.5g} GiB')
        print(f'Req memory per EOM: {single_eom_mem/1024**3:.5g} GiB')
        print(f'Req memory for full solve: {full_sol_mem/1024**3:.5g} GiB')
        print(f'\tMandatory memory use: {mand_sol_mem/1024**3:.5g} GiB')
        print(f'\tMemory use for all EOMs: {single_eom_mem*stack_factor/1024**3:.5g} GiB')
        print(f'\tFull output solution size: {output_sol_size/1024**3:.5g} GiB')
        print(f'Available memory for sliced solves: {available_mem/1024**3:.5g} GiB')

    if (float(total_mem) - float(min_sol_mem)) <= 0:
        raise RydiquleError(f'System is too large to solve. Need at least {min_sol_mem/1024**3} GiB')
    
    compare_vals = np.array([np.ceil(single_eom_mem*stack_factor / available_mem), n_slices],
                            dtype=float)
    n_ham_slices = int(np.nanmax(compare_vals))

    if doppler_shape:
        if not sum_doppler and n_ham_slices > 1:
            msg = ("Setting 'sum_doppler=False' if full equations of motion"
                   " do not fit in memory is unsupported.")
            raise RydiquleError(msg)

    if debug:
        print(f'Number of stack slices to be used: {n_ham_slices:d}')

    return n_ham_slices, out_sol_shape


def get_slice_num_t(n: int, stack_shape: Tuple[int, ...],
                    doppler_shape: Tuple[int, ...], time_points: int,
                    sum_doppler: bool, weight_doppler: bool,
                    n_slices: Union[int,None], debug: bool = False) -> Tuple[int, Tuple[int, ...]]:
    """
    Estimates the memory required for the desired time solve.

    Note that the time solver used (scipy.solve_ivp) is an adaptive
    solver, so the internal number of time steps used is
    problem dependent and not controlled by the requested number
    of time points. Generally, the number of points is proportional
    to the highest frequency in the problem and the length of the time
    to solve.
    To estimate a lower bound on the memory
    needed to time solve, we use a fudge factor of 4 on the
    requested time points. This is unlikely to be accurate
    even in a general case.

    Parameters
    ----------
    n: int
        Size of the system basis
    stack_shape: tuple of int
        Tuple of sizes for the hamiltonian stack to be solved
    doppler_shape: tuple of int
        Tuple of sizes for the doppler axes.
        Pass an empty tuple if no doppler averaging.
    time_points: int
        Number of time steps requested from the time solver.
        This sets the output solution shape.
        An internal fudge factor of 4 is applied for memory estimation purposes.
    sum_doppler: bool
        Whether solution will be summing the doppler average
    weight_doppler: bool
        Whether the solution will apply weights to the doppler averaging
    n_slices: int, default=1
        Manually override the minimum number of hamiltonian slices to use.
    debug: bool, default=False
        Print debug information about the memory calculations.

    Returns
    -------
    n_ham_slices: int
        Number of slices to use when solving the stacked hamiltonian
    out_sol_shape: tuple of int
        Shape of the resulting solution for this calculation.

    Raises
    ------
    RydiquleError: If `sum_doppler=False` and full solution does not fit in memory.

    Warns
    -----
    RydiquleWarning: If there is unlikely to be enough memory to solve the system.

    """
    # get total avaialable memory for the system
    total_mem = psutil.virtual_memory().available
    # total number of EOM systems to solve for the stack
    stack_factor = np.prod(stack_shape, dtype=np.ulonglong)
    # track memory to solve a single set of EOMs
    single_eom_mem = 0
    # track mandatory memory per solve
    mand_sol_mem = 0

    # determine minimum memory requirements for a solve
    # 16 bytes for complex128 arrays, 8 bytes for float64 arrays
    ham_shape = (*stack_shape, n, n)
    ham_size = memory_size(ham_shape, 16)
    sol_shape = (*stack_shape, time_points, n**2-1)
    sol_size = memory_size(sol_shape, 8)
    # all solves need to make 3 full sized hamiltonians and solutions
    # happens in solve_time
    mand_sol_mem += (3*ham_size + sol_size)
    # rest of these memory allocations happen in _solve_hamiltonian_stack

    eom_shape = (n**2, n**2)
    const_shape = eom_shape[:-1]
    eom_size = memory_size(eom_shape, 16) + memory_size(const_shape,16)
    # minimum solve must be able to make at least one set of EOMs
    # note that you have to times 2 for remove_ground
    # make_real now deallocates at the end, but keeping 1.5 factor for buffer
    # round factor up to 3 to keep types as int and give buffer
    # happens in generate_eom
    # have to make two copies for the real/imag time dependent eoms too
    single_eom_mem += eom_size * 3 * 3

    # time solver is adaptive, so number of solve points
    # will almost always exceed requested, so fudging a bit
    # note that unused intermediate steps will be GC once solve_ivp result
    # object is dumped, however these points must fit in memory during the solve
    time_fudge = 4

    if doppler_shape:
        dop_eom_shape = (*doppler_shape, n**2-1, n**2-1)
        dop_eom_size = memory_size(dop_eom_shape,8)
        # all doppler solves must allocate the dop EOMs in addition to bare
        single_eom_mem += dop_eom_size

        # to solve doppler, need to allocate full dop sol size
        dop_sol_shape = (*doppler_shape, time_points, n**2-1)
        dop_sol_size = memory_size(dop_sol_shape,8)

        single_eom_mem += dop_sol_size * time_fudge
        if weight_doppler:
            # weighting adds extra dop sol allocation
            single_eom_mem += dop_sol_size

    else:
        single_eom_mem *= time_fudge

    # minimum required memory to solve a single set of EOMs in the stack
    min_sol_mem = mand_sol_mem + single_eom_mem
    # required memory to solve the entire stack at once
    full_sol_mem = mand_sol_mem + single_eom_mem*stack_factor
    # get memory available for solving hamiltonian slices
    available_mem = total_mem - mand_sol_mem
    if doppler_shape and not sum_doppler:
        # not summing doppler, don't allocate summed solutiion
        available_mem -= sol_size
        full_sol_mem -= sol_size
        min_sol_mem -= sol_size

    if doppler_shape and not sum_doppler:
        out_sol_shape = (*doppler_shape, *stack_shape, time_points, n**2-1)
    else:
        out_sol_shape = sol_shape
    output_sol_size = memory_size(out_sol_shape, 8)

    if debug:
        print(f'Total available memory: {total_mem/1024**3:.5g} GiB')
        print(f'Min Req memory to solve: {min_sol_mem/1024**3:.5g} GiB')
        print(f'Req memory per EOM: {single_eom_mem/1024**3:.5g} GiB')
        print(f'Req memory for full solve: {full_sol_mem/1024**3:.5g} GiB')
        print(f'\tMandatory memory use: {mand_sol_mem/1024**3:.5g} GiB')
        print(f'\tMemory use for all EOMs: {single_eom_mem*stack_factor/1024**3:.5g} GiB')
        print(f'\tFull output solution size: {output_sol_size/1024**3:.5g} GiB')
        print(f'Available memory for sliced solves: {available_mem/1024**3:.5g} GiB')

    if (float(total_mem) - float(min_sol_mem)) <= 0:
        warnings.warn(
            f'System is likely too large to solve. Need at least {min_sol_mem/1024**3} GiB',
            RydiquleWarning)

    #array of the minimum necessary slices and the specified number of slices
    compare_vals = np.array([np.ceil(single_eom_mem*stack_factor / available_mem), n_slices],
                            dtype=float)
    n_ham_slices = int(np.nanmax(compare_vals))

    if doppler_shape:
        if not sum_doppler and n_ham_slices > 1:
            msg = ("Setting 'sum_doppler=False' if full equations of motion"
                   " do not fit in memory is unsupported.")
            raise RydiquleError(msg)

    if debug:
        print(f'Number of stack slices to be used: {n_ham_slices:d}')

    return n_ham_slices, out_sol_shape


def get_slice_num_hybrid(n: int, 
                         param_stack_shape: Tuple[int, ...], 
                         numeric_doppler_shape: Tuple[int, ...],
                         n_slices: Optional[int] = None,
                         debug: bool = False) -> Tuple[int, Tuple[int, ...]]:
    """
    Estimates memory and determines the number of slices for the analytic solver.

    This version is tailored to the memory footprint of the analytic algorithm,
    which includes large intermediate arrays like the propagator and eigenvector stacks.
    
    Parameters
    ----------
    n : int
        Size of the system basis.
    param_stack_shape : tuple of int
        Tuple of sizes for the sensor's parameter axes (e.g., from L0.shape[:-2]).
    numeric_doppler_shape : tuple of int
        Tuple of sizes for the numeric doppler axes. Pass an empty tuple for the 1D case.
    n_slices : int, optional
        Manually override the minimum number of slices. If None, it's determined automatically.
    debug : bool, optional
        If True, prints detailed memory usage information.

    Returns
    -------
    n_param_slices : int
        Number of slices to use when iterating over the parameter stack.
    out_sol_shape : tuple of int
        Shape of the final, fully-solved solution array.
    """
    if n_slices is None:
        n_slices = 1

    # --- 1. Calculate Mandatory Memory (arrays that exist fully, regardless of slicing) ---
    total_mem = psutil.virtual_memory().available
    
    # The full L0 matrix with all parameter stacks
    l0_full_shape = (*param_stack_shape, n**2, n**2)
    l0_full_mem = memory_size(l0_full_shape, 16) # complex128

    # The final output solution array
    out_sol_shape = (*param_stack_shape, n**2)
    out_sol_mem = memory_size(out_sol_shape, 8)
    
    mand_mem = l0_full_mem + out_sol_mem*2 # Sol is complex before final check

    # --- 2. Calculate Memory for a Single Slice ---
    # This is the memory needed to process ONE parameter point through the hybrid algorithm.
    
    # Shape of the stacks over the numeric velocity grid
    num_dop_stack_shape = (*numeric_doppler_shape, n**2, n**2)
    num_dop_vec_shape = (*numeric_doppler_shape, n**2)

    # Key arrays created inside the loop for a single slice
    mem_l_base = memory_size(num_dop_stack_shape, 16)  # L_base_slice
    mem_rho0 = memory_size(num_dop_vec_shape, 16)      # rho0_slice
    mem_l0m = memory_size(num_dop_stack_shape, 16)      # L0m_slice (propagator)
    mem_eigvecs = memory_size(num_dop_stack_shape, 16)  # r_eigvecs
    mem_rho_dopp = memory_size(num_dop_vec_shape, 16)   # rho_dopp_slice

    # Sum them up and add a safety buffer (e.g., 1.5x) for temporary copies
    single_slice_mem = (mem_l_base + mem_rho0 + mem_l0m + mem_eigvecs + mem_rho_dopp) * 1.5

    # --- 3. Determine the Number of Slices ---
    available_mem = total_mem - mand_mem
    
    if available_mem < single_slice_mem:
        raise RydiquleError(f'System is too large to solve. Need at least {single_slice_mem/1024**3} GiB')

    num_param_points = np.prod(param_stack_shape, dtype=np.ulonglong)
    if num_param_points == 0: 
        num_param_points = 1 # Handle case with no parameter stacks

    # Minimum slices needed based on memory
    full_sol_mem = single_slice_mem * num_param_points
    min_slices_needed = np.ceil(full_sol_mem / available_mem)
    
    # The number of slices is the larger of what's needed and what the user requested
    n_param_slices = int(max(min_slices_needed, n_slices))
    
    if debug:
        print('--- Analytic Solver Memory Debug ---')
        print(f'Total available RAM: {total_mem/1024**3:.4g} GiB')
        print(f'Min Req memory to solve: {single_slice_mem/1024**3:.4g} GiB')
        print(f'Req memory for full solve: {full_sol_mem/1024**3:.4g} GiB')
        print(f'\tFull output solution size: {out_sol_mem/1024**3:.5g} GiB')
        print(f'Available memory for sliced calculations: {available_mem/1024**3:.4g} GiB')
        print(f'Calculated minimum slices needed: {min_slices_needed}')
        print(f'Final number of slices to be used: {n_param_slices}')
        print('------------------------------------')
        
    return n_param_slices, out_sol_shape