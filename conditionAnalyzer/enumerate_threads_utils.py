import sympy as sp
import numpy as np


def parfor_cond(func, ind_ranges, dummy_funcs):
    """Iterate over all threads in the grid and evaluate the function func."""
    count = 0
    for i0 in range(ind_ranges[0], ind_ranges[1]):
        for i1 in range(ind_ranges[2], ind_ranges[3]):
            for i2 in range(ind_ranges[4], ind_ranges[5]):
                for i3 in range(ind_ranges[6], ind_ranges[7]):
                    for i4 in range(ind_ranges[8], ind_ranges[9]):
                        for i5 in range(ind_ranges[10], ind_ranges[11]):
                            if func(i0, i1, i2, i3, i4, i5, *dummy_funcs):
                                count += 1
    return count


def construct_valid_index_ranges(ind_ranges: list, block_dim: list):
    """Construct numpy array representing the possible values of thread and block indices in all three dimensions."""
    idx0 = np.column_stack(np.divmod(np.arange(ind_ranges[0], ind_ranges[1]), block_dim[0]))
    idx1 = np.column_stack(np.divmod(np.arange(ind_ranges[2], ind_ranges[3]), block_dim[1]))
    idx2 = np.column_stack(np.divmod(np.arange(ind_ranges[4], ind_ranges[5]), block_dim[2]))
    idx0_constrained = idx0[(idx0[:, 1] >= ind_ranges[6]) & (idx0[:, 1] < ind_ranges[7])]
    idx1_constrained = idx1[(idx1[:, 1] >= ind_ranges[8]) & (idx1[:, 1] < ind_ranges[9])]
    idx2_constrained = idx2[(idx2[:, 1] >= ind_ranges[10]) & (idx2[:, 1] < ind_ranges[11])]

    return [idx0_constrained, idx1_constrained, idx2_constrained]


# TODO: use np.column_stack only for the smaller two ranges
#       idea: func is lambdified with "numpy" backend -> apply largest ranges directly as np.arrays
#       additionally, the if conditions might be replaced by masking on the arrays
#       pass dependencies and according to them insert the arrays into func
def parfor_cond_np(func, ind_ranges: list, dummy_funcs, block_dim: list, dependencies: list,
                   indvar_range: np.ndarray = None):
    count = 0
    idx_constrained = construct_valid_index_ranges(ind_ranges, block_dim)
    largest_dep_index = 2 - dependencies[::-1].index(1)
    plugin = idx_constrained[largest_dep_index]
    for_inds = [i for i in range(3) if i != largest_dep_index]

    # iterate only over indices the condition depends on
    idx_range = [idx_constrained[i] if dependencies[i] else np.array([[0, 0]]) for i in for_inds]

    cta_args = 3 * [None]
    cta_args[largest_dep_index] = plugin[:, 0]
    thread_args = 3 * [None]
    thread_args[largest_dep_index] = plugin[:, 1]
    for x in idx_range[0]:
        b0, t0 = x
        for y in idx_range[1]:
            b1, t1 = y
            cta_args[for_inds[0]] = b0
            cta_args[for_inds[1]] = b1
            thread_args[for_inds[0]] = t0
            thread_args[for_inds[1]] = t1

            if indvar_range is None:
                res = func(*cta_args, *thread_args, *dummy_funcs)
            else:
                res = 0
                for i_val in indvar_range:
                    res += np.sum(func(*cta_args, *thread_args, *dummy_funcs, i_val))
            count += np.sum(res)

    if indvar_range is None:
        for i in for_inds:
            if dependencies[i] == 0:
                count *= len(idx_constrained[i])
    else:
        count /= (len(plugin) / len(idx_range[0]) / len(idx_range[1]))
    return count


# Since sympy does not support bitwise operations, they are handled with dummy
# functions during symbolic treatment. To evaluate them, the following
# "apply" functions are defined and inserted.
def apply_and(*args):
    if len(args) > 2:
        return args[0] & apply_and(*args[1:])
    elif len(args) == 2:
        return args[0] & args[1]
    else:
        return args


def apply_or(*args):
    if len(args) > 2:
        return args[0] | apply_or(*args[1:])
    elif len(args) == 2:
        return args[0] | args[1]
    else:
        return args


def apply_xor(*args):
    if len(args) > 2:
        return args[0] ^ apply_xor(*args[1:])
    elif len(args) == 2:
        return args[0] ^ args[1]
    else:
        return args


# TODO:
#  These functions can possibly be simplified and expressed as ~apply_and(*args) (and accordingly for nor and nxor).
#  TEST IT!
def apply_nand(*args):
    if len(args) > 2:
        return ~(args[0] & apply_and(*args[1:]))
    elif len(args) == 2:
        return ~apply_and(*args)
    else:
        return args


def apply_nor(*args):
    if len(args) > 2:
        return ~(args[0] | apply_or(*args[1:]))
    elif len(args) == 2:
        return ~apply_or(*args)
    else:
        return args


def apply_nxor(*args):
    if len(args) > 2:
        return ~(args[0] ^ apply_xor(*args[1:]))
    elif len(args) == 2:
        return ~apply_xor(*args)
    else:
        return args


def do_sel(cond, true_val, false_val):
    res = np.where(cond, true_val, false_val)
    return res


def do_ex2(x):
    return 2**x


def do_lg2(x):
    return np.log2(x)


def do_ceil(x):
    return np.ceil(x)


def do_floor(x):
    return np.floor(x)


def do_shl(x, shift):
    return x << shift


def do_lshr(x, shift):
    return x >> shift


class SpecialFunctionDummies:
    def __init__(self):
        f_and = sp.Function('f_and')
        f_or = sp.Function('f_or')
        f_xor = sp.Function('f_xor')
        f_nand = sp.Function('f_nand')
        f_nor = sp.Function('f_nor')
        f_nxor = sp.Function('f_nxor')
        sel = sp.Function('sel')
        ex2 = sp.Function('ex2')
        lg2 = sp.Function('lg2')
        ceil = sp.Function('ceil')
        floor = sp.Function('floor')
        shl = sp.Function('shl')
        lshr = sp.Function('lshr')
        self.dummy_list = [f_and, f_or, f_xor, f_nand, f_nor, f_nxor, sel, ex2, lg2, ceil, floor, shl, lshr]
        self.function_list = [apply_and, apply_or, apply_xor, apply_nand, apply_nor, apply_nxor,
                              do_sel, do_ex2, do_lg2, do_ceil, do_floor, do_shl, do_lshr]
        # TODO: define all explicit functions as methods and generate the self.function_list with the following line:
        # self.function_list = [func for func in dir(SpecialFunctionDummies)
        #                       if callable(getattr(test, func)) and not func.startswith("__")]
