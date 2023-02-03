from collections import defaultdict
from typing import List  # , Tuple   # , Any

from parse_utils import parse_condition, is_loop, LoopString, dependencies  # , Recurrence
import sympy as sp
import numpy as np
import math
import functools  # for reduce, maybe change to from functools import reduce
from operator import itemgetter  # used for sorting in function ThreadGroup.enumerate_threads_explicitly
import multiprocessing as mp
import enumerate_threads_utils as enum
from config_utils import IndvarInfo


_func = None


def worker_init(func):
    global _func
    _func = func


def worker(x):
    return _func(x)


def xmap(func, iterable, processes=None):
    with mp.Pool(processes, initializer=worker_init, initargs=(func,)) as p:
        return p.map(worker, iterable)


class Dim3:

    def __init__(self, x_var=None, y_var=None, z_var=None):
        self.x = x_var
        self.y = y_var
        self.z = z_var

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__,
                                       self.x,
                                       self.y,
                                       self.z)

    def get(self, index: int):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        else:
            raise IndexError

    def to_list(self):
        return [self.x, self.y, self.z]


class Boundaries:

    def __init__(self, block_dim: Dim3, grid_dim: Dim3):
        self.cta_bounds = Dim3([0, block_dim.x], [0, block_dim.y], [0, block_dim.z])
        self.global_bounds = Dim3([0, block_dim.x * grid_dim.x],
                                  [0, block_dim.y * grid_dim.y],
                                  [0, block_dim.z * grid_dim.z])
        self.block_dim = block_dim
        self.grid_dim = grid_dim

    def __eq__(self, other):
        return self.block_dim == other.block_dim and self.grid_dim == other.grid_dim

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                   self.block_dim,
                                   self.grid_dim)

    def cta_length(self, coord: int):
        return self.cta_bounds.get(coord)[1] - self.cta_bounds.get(coord)[0]

    def global_length(self, coord: int):
        return self.global_bounds.get(coord)[1] - self.global_bounds.get(coord)[0]

    def all_lengths(self, coord: int):
        return self.cta_length(coord), self.global_length(coord)


class MaximumIndexInfo:

    def __init__(self, max_val, max_ind, numeric_indices):
        self.val = max_val
        self.ind = max_ind
        self.index_list = numeric_indices


class PatternHelper:
    """
    Helper class holding three wildcards, the list of patterns to test and the dummy function f_and.
    Using this class avoids multiple redefinitions of these objects.
    """

    def __init__(self):
        indices = get_symbolic_grid_index_list()
        self.q = sp.Wild('q')
        self.sign = sp.Wild('sign', exclude=indices[3::])
        self.factor = sp.Wild('factor', exclude=indices[0:3:])
        self.pattern_list = get_pattern_list(self.q, self.sign, self.factor)
        self.f_and = sp.Function('f_and')
        self.f_nand = sp.Function('f_nand')


def get_maximum_length_and_index(dim_lengths: list, num_cpu: int):
    """Sort the bounds such that the largest interval belongs to the innermost loop when enumerating.
     Rationale: It is advantageous for nested loops to call the outer loops less frequent than the inner
                loops since the overhead of creating a loop is reduced.
     Example: call outer loop 10 times, inner loop 1000 times -> inner loop has to be created 10 times
              vs
              call outer loop 1000 times, inner loop 10 times -> inner loop has to be created 1000 times
    """
    max_dim_length = max(dim_lengths)
    max_dim_length_distributed = max_dim_length / num_cpu
    index_of_max = dim_lengths.index(max_dim_length)
    dim_lengths[index_of_max] = int(max_dim_length_distributed)  # This underestimates...
    all_lengths = list(zip(range(6), dim_lengths))
    all_lengths.sort(key=itemgetter(1))    # sort the zipped according to dim_length
    numeric_indices = [ind[0] for ind in all_lengths]  # grid indices as numbers

    return MaximumIndexInfo(max_dim_length_distributed, index_of_max, numeric_indices)


def distribute_index_ranges(num_cpu: int, bounds: list, max_length_info: MaximumIndexInfo):
    index_bounds = []
    for n in range(num_cpu):
        bounds_cpu = [0] * 12
        for i in range(6):
            bound_index = max_length_info.index_list[i]
            if bound_index == max_length_info.ind:
                bounds_cpu[2 * i] = int(n * max_length_info.val)
                bounds_cpu[2 * i + 1] = int((n + 1) * max_length_info.val)
            else:
                bounds_cpu[2 * i] = bounds[bound_index][0]
                bounds_cpu[2 * i + 1] = bounds[bound_index][1]
        index_bounds.append(bounds_cpu)
    return index_bounds


def get_total_num_threads(grid_dim: Dim3, block_dim: Dim3):
    """
    Compute the total number of threads on the kernel grid.
    @param grid_dim: grid dimensions
    @param block_dim: block (=CTA) dimensions
    @return: total number of threads
    """
    ctas_per_grid = grid_dim.x * grid_dim.y * grid_dim.z
    threads_per_cta = block_dim.x * block_dim.y * block_dim.z
    return ctas_per_grid * threads_per_cta


class ThreadGroup:

    # constructor
    def __init__(self, path: list = None, conditions: list = None, loops: list = None, bounds: Boundaries = None):
        if loops is None:
            loops = []
        self.path = path
        self.conditions = conditions
        self.indvar_conditions = set()
        self.loops = loops
        self.bounds = bounds
        self.num_threads = 0
        self.invert_flag = False
        self.exceptions = ([], [], [], [], [], [])
        self.exception_indices = []
        self.exceptions_conditions = set()
        sfd = enum.SpecialFunctionDummies()
        self.func_list = sfd.dummy_list
        self.funcs = sfd.function_list

    def __eq__(self, other):
        return self.path == other.path and self.conditions == other.conditions and self.loops == other.loops

    def __repr__(self):
        return "{}(path={})".format(self.__class__.__name__,
                                    self.path)

    def assign_loop_ids(self, indvars):
        for loop in self.loops:
            loop.get_loop_id(indvars)

    def evaluate_loop_tripcounts(self, parameters: dict, indvars: list):
        if not self.loops:
            print('no loops')
        dependent_loops = []
        for i, loop in enumerate(self.loops):
            # print(loop)
            if 'PHI' in loop.condition.tripcount:
                # print('indvar dependency detected')
                dependent_loops.append(i)
                continue
            try:
                loop.compute_tripcount(self.bounds, parameters)
            except TypeError:
                print('loop trip count cannot be computed')
        for i in dependent_loops:
            dim_dict = self.construct_symbolic_dimension_dict()
            self.loops[i].compute_indvar_dependent_tripcount(dim_dict, parameters, indvars, self.loops)

    def determine_num_threads(self, parameters: dict):
        """
        This method performs the analysis of the parameters. For conditions following the pattern
          bdx * bix + tix <> p
        a truncation of the whole kernel grid can be achieved. For conditions which do not match one of the patterns the
        remaining grid is split among the available CPU cores and the conditions are explicitly evaluated.
        @param parameters: dict of kernel parameters and their values
        @return:
        """
        if self.bounds is None:
            raise ValueError('boundaries not specified')

        pattern = PatternHelper()
        complex_conditions = set()
        for cond in self.conditions:
            if 'PHI ' in cond or 'PHI)' in cond:
                self.indvar_conditions.add(cond)
                continue
            cond_parsed = parse_condition(cond, self.path)
            cond_sympy = sp.sympify(insert_placeholder_functions(cond_parsed))
            success = self.analyze_condition(cond_sympy, parameters, pattern, complex_conditions, is_top_level=True)

            if success == 1:
                print("condition cannot be satisfied")
                self.num_threads = 0
                return

        cta_lengths = [self.bounds.cta_length(i) for i in range(3)]
        global_lengths = [self.bounds.global_length(i) for i in range(3)]
        if not is_well_defined(cta_lengths, global_lengths):
            print("bounds are not well defined")
            self.num_threads = 0
            return

        if complex_conditions:
            self.enumerate_threads_explicitly(complex_conditions, parameters)
        else:
            self.compute_num_threads_from_bounds()

    def analyze_condition(self, cond, parameters, pattern: PatternHelper, complex_conditions: set, is_top_level: bool):
        if isinstance(cond, sp.Rel):
            self.perform_pattern_test_and_update(cond, parameters, pattern, complex_conditions)
            success = 0
        else:
            success = self.decompose_condition_into_atomics(cond, parameters, pattern, complex_conditions, is_top_level)

        return success

    def decompose_condition_into_atomics(self, cond, parameters, pattern: PatternHelper, complex_conditions: set,
                                         is_top_level: bool):
        success = 0
        is_and = cond.func == pattern.f_and
        is_nand = cond.func == pattern.f_nand
        if is_and:
            for arg in cond.args:
                success = self.analyze_condition(arg, parameters, pattern, complex_conditions, False)
                if success == 1:
                    break
        elif cond.is_Function or is_nand:
            complex_conditions.add(cond)
        else:
            success = 0 if cond.subs(parameters) else 1

        return success

    def perform_pattern_test_and_update(self, cond, parameters, pattern: PatternHelper, complex_conditions: set):
        cond_sim = simplify_condition(cond)
        match_res = test_condition_for_pattern(cond_sim, pattern.pattern_list)
        one_match, ind_match = exactly_one_pattern_matches(match_res)
        if not one_match:
            complex_conditions.add(cond_sim)
            return

        block_dim = self.bounds.block_dim
        dim = ind_match % 3
        match = match_res[ind_match]
        relational = get_relational(cond_sim, match[pattern.sign])
        global_update = False
        cta_index_constraint = False
        eval_factor = self.evaluate_matches(match[pattern.factor], parameters)
        eval_sign = self.evaluate_matches(match[pattern.sign], parameters)
        eval_q = self.evaluate_matches(match[pattern.q], parameters)
        if ind_match < 3:
            if match[pattern.factor] != 0:
                mul_factor_sign = eval_factor * eval_sign
                if mul_factor_sign > 0:
                    cta_ind, thread_ind = divmod(-eval_q, eval_factor * block_dim.get(dim))
                    new_bound = cta_ind * block_dim.get(dim) + min(thread_ind*eval_sign, block_dim.get(dim))
                    global_update = True
                else:  # TODO
                    complex_conditions.add(cond_sim)
                    return
            else:
                # update cta_bounds
                thread_ind = -eval_q // eval_sign
                new_bound = thread_ind
        else:  # condition on cta index
            cta_ind = eval_q // eval_factor
            new_bound = block_dim.get(dim) * cta_ind
            global_update = True
            cta_index_constraint = True
            # Note that upon updating the global bounds, (un)equalities affect entire CTAs, not just two
            # consecutive global indices.

        # Rationale: If the condition is not an negated equality (!=) proceed with updating the boundaries.
        # Otherwise, store the excepted index in a list. To distinguish between single global indices and
        # constraints on the CTA indices, a signed flag is introduced.
        # print("bounds before updating", self.bounds.global_bounds, self.bounds.cta_bounds)
        if relational != 1:
            update_bounds(global_update, relational, dim, new_bound, self.bounds, cta_index_constraint)
        else:
            index = 0 if global_update else 1
            cta_flag = -1 if cta_index_constraint else 1
            self.exceptions[index*3+dim].append(cta_flag * new_bound)
            self.exception_indices.append((index, dim))
            self.exceptions_conditions.add(cond_sim)

    def evaluate_matches(self, match: sp.Symbol, parameters: dict):
        if isinstance(match, int) or isinstance(match, float):
            return match
        bdim = self.bounds.block_dim.to_list()
        gdim = self.bounds.grid_dim.to_list()
        expl_dims = bdim + gdim
        sym_dims = get_symbolic_grid_dims_list()
        dims = {sym_dims[i]: expl_dims[i] for i in range(6)}
        f = sp.lambdify(self.func_list, match.subs(parameters).subs(dims))
        return f(*self.funcs)

    def get_indvar_values(self, indvar_list: list):
        indvars_with_phis = []
        last_valid_vals = {}
        indvar_values = len(indvar_list) * [None]
        for idx, iv in enumerate(indvar_list):
            if iv.loop_id not in (loop.loop_id for loop in self.loops):
                continue
            if 'PHI' in iv.rec.raw_str:
                indvars_with_phis.append(idx)
                continue
            inst = '+' if iv.rec.inst == 'add' else '-'
            start_val = sp.sympify(iv.rec.start)
            step_val = sp.sympify(iv.rec.step)
            final_val = self.compute_indvar_final_value(iv, last_valid_vals)
            last_valid = final_val + sp.sympify('-' + inst + iv.rec.step)
            last_valid_vals[sp.Symbol(iv.name+'PHI')] = last_valid
            indvar_values[idx] = (start_val, step_val, final_val)

        for idx in indvars_with_phis:
            if indvar_list[idx].loop_id not in (loop.loop_id for loop in self.loops):
                continue
            start, step, inst, end = indvar_list[idx].rec.evaluate_PHINodes(self.path)
            start_val = sp.sympify(start).subs(last_valid_vals)
            step_val = sp.sympify(step).subs(last_valid_vals)
            final_val = self.compute_indvar_final_value(indvar_list[idx], last_valid_vals)
            indvar_values[idx] = (start_val, step_val, final_val)

        return indvar_values

    def compute_indvar_final_value(self, indvar_info, last_valids):
        start, step, inst, end = indvar_info.rec.evaluate_PHINodes(self.path)
        inst_dict = {'add': '+', 'sub': '-', 'mul': '*', 'div': '//', 'shl': '*', 'lshr': '//'}
        tc_op = '*' if (inst == 'add' or inst == 'sub') else '**'
        if inst == 'shl' or inst == 'lshr':
            step = '(2**' + step + ')'
        try:
            inst = inst_dict[inst]
        except KeyError:
            print("unknown instruction")
        if indvar_info.rec.is_simple:
            tripcount = -1
            for loop in self.loops:
                if indvar_info.loop_id == loop.loop_id:
                    correct_foot_control = 1 if loop.condition.is_guarded else 0
                    tripcount = loop.tripcount + correct_foot_control
                    break
            if tripcount == -1:
                print('Error: no loop found', indvar_info.loop_id)
            final_val = sp.sympify(start + inst + '(' + step + tc_op + '(' + str(tripcount) + '))').subs(last_valids)
        else:
            final_val = sp.sympify(end).subs(last_valids)
        return final_val

    def find_indvars_in_condition(self, cond: sp.Symbol, relevant_indvars: list,
                                  indvar_sym: list, indvar_list: list, index_list: list):
        for sym in cond.free_symbols:
            if 'PHI' in sym.name:
                indvar_sym.append(sym)
                for idx, iv_info in enumerate(indvar_list):
                    if sym.name[:-3] == iv_info.name:
                        relevant_indvars.append(iv_info)
                        index_list.append(idx)
                        break

    def analyze_indvar_conditions(self, indvar_list: list, parameters: dict, dims: dict):
        result_true = []
        indvar_values = self.get_indvar_values(indvar_list)
        for cond in self.indvar_conditions:
            cond_parsed = parse_condition(cond, self.path)
            cond_sympy = sp.sympify(insert_placeholder_functions(cond_parsed))
            cond_sim = cond_sympy.subs(parameters).subs(dims)
            # 1) find indvar and the associated loop
            relevant_indvars = []  # type: List[IndvarInfo]
            index_list = []
            indvar_sym = []
            self.find_indvars_in_condition(cond_sim, relevant_indvars, indvar_sym, indvar_list, index_list)

            if len(indvar_sym) > 1:
                print('Seems like there is more than one indvar in this condition.')
                print('This will be implemented if necessary.')
                return 0

            grid_index_list = get_symbolic_grid_index_list()
            dep = dependencies(cond_sim, grid_index_list)

            if dep == [0, 0, 0, 0, 0, 0]:
                f = sp.lambdify(self.func_list + indvar_sym, cond_sim, "numpy")
                # 2) compute final value if necessary
                for idx, iv in enumerate(relevant_indvars):
                    # 4) apply condition and count Trues
                    tripcount = self.get_tripcount_for_loop(iv.loop_id)
                    vals = tuple(indvar_values[index_list[idx]])
                    val_range = construct_indvar_arange(iv, vals, parameters, dims, tripcount)
                    result_true.append(len(val_range[f(*self.funcs, val_range)]) / tripcount)
                    # 5) somehow store the result
            else:
                bounds_flat, block_dim, dep_sorted, index_list_sorted = get_sorted_arguments(self.bounds, dep,
                                                                                             grid_index_list)
                f = sp.lambdify(index_list_sorted + self.func_list + indvar_sym, cond_sim, "numpy")
                for idx, iv in enumerate(relevant_indvars):
                    tripcount = self.get_tripcount_for_loop(iv.loop_id)
                    vals = tuple(indvar_values[index_list[idx]])
                    val_range = construct_indvar_arange(iv, vals, parameters, dims, tripcount)
                    count = enum.parfor_cond_np(f, bounds_flat, self.funcs, block_dim, dep_sorted, val_range)
                    result_true.append(count / tripcount)
            # How to deal with mixed conditions?
            # 1) get indices the condition depends on
            # 2) create the lambda
            # 3) create the ranges: indvar and indices (global index + cta index as constraint)
            # 3.1) apply the cartesian product to the ranges (this may fail due to memory bounds)
            # 4) insert the range array into the lambda
            # 5) np.sum the result and divide it by tripcount*len(index_ranges)
        return result_true

    def get_tripcount_for_loop(self, loop_id: int):
        """
        Find loop in the ThreadGroup and return its tripcount. An IndexError is raised if there is no loop
        with the ID loop_id.
        @param loop_id: ID of the loop
        @return: tripcount of the loop with the requested ID
        """
        tripcount = None
        for loop in self.loops:
            if loop_id == loop.loop_id:
                correct_foot_control = 1 if loop.condition.is_guarded else 0
                tripcount = loop.tripcount + correct_foot_control
                break
        if tripcount is None:
            raise IndexError('Loop ID not valid.')
        return tripcount

    # TODO
    #  Work In Progress!
    def enumerate_threads_explicitly(self, conditions: set, parameters: dict):
        """
        Explicitly evaluate conditions using (relevant) thread and CTA indices.
        @param conditions: list of SymPy expression denoting the branch conditions
        @param parameters: kernel arguments
        """
        print('count for', self.path)
        index_list = get_symbolic_grid_index_list()
        dimension_dict = self.construct_symbolic_dimension_dict()

        all_conds = conditions|self.exceptions_conditions
        full_cond = functools.reduce(lambda x, y: self.func_list[0](x, y), all_conds)
        full_cond_sim = full_cond.subs(dimension_dict).subs(parameters)  # maybe simplify further
        dep = dependencies(full_cond_sim, index_list)

        if dep != [0, 0, 0, 0, 0, 0]:
            bounds_flat, block_dim, dep_dims_sorted, index_list_sorted = get_sorted_arguments(self.bounds,
                                                                                              dep,
                                                                                              index_list)
            f = sp.lambdify(index_list_sorted + self.func_list, full_cond_sim, "numpy")

            def parfor_cond_wrapper(ind_ranges):
                return enum.parfor_cond_np(f, ind_ranges, self.funcs, block_dim, dep_dims_sorted)

            count = parfor_cond_wrapper(bounds_flat)

            if not self.invert_flag:
                self.num_threads = count
            else:
                total_num_threads = get_total_num_threads(self.bounds.grid_dim, self.bounds.block_dim)
                self.num_threads = total_num_threads - count

        else:  # condition is independent of indices
            f = sp.lambdify(self.func_list, full_cond_sim, "numpy")
            if f(*self.funcs):
                self.compute_num_threads_from_bounds()
            else:
                self.num_threads = 0

    def compute_num_threads_from_bounds(self):
        # check first if the cta_bounds have been modified & check for ill-defined boundaries (lower > upper)
        cta_lengths = [self.bounds.cta_length(i) for i in range(3)]
        global_lengths = [self.bounds.global_length(i) for i in range(3)]
 
        total_num_threads = get_total_num_threads(self.bounds.grid_dim, self.bounds.block_dim)
        global_bounds_coords = [divmod(el, self.bounds.block_dim.get(i))
                                for i in range(3)
                                for el in self.bounds.global_bounds.get(i)]

        if cta_lengths == self.bounds.block_dim.to_list():   # [self.bounds.block_dim.get(i) for i in range(3)]:
            num_threads_per_dim = global_lengths
        else:
            global_bounds_coords_list = [list(el) for el in global_bounds_coords]
            overhang = [0, 0, 0]
            check_alignment(global_bounds_coords_list, overhang, self.bounds.cta_bounds.to_list())
            num_cta = [global_bounds_coords_list[2 * i + 1][0] - global_bounds_coords_list[2 * i][0] + 1
                       for i in range(3)]
            num_threads_per_dim = [overhang[i] + num_cta[i] * cta_lengths[i] for i in range(3)]

        threads_in_bounds = num_threads_per_dim[0] * num_threads_per_dim[1] * num_threads_per_dim[2]
        self.num_threads = threads_in_bounds if not self.invert_flag else total_num_threads - threads_in_bounds
        self.num_threads -= self.compute_correction(num_threads_per_dim, global_bounds_coords)

    def get_grid_dims_from_bounds(self):
        grid_bounds = [[self.bounds.global_bounds.get(i)[0] // self.bounds.block_dim.get(i),
                        math.ceil(self.bounds.global_bounds.get(i)[1] / self.bounds.block_dim.get(i))]
                       for i in range(3)]
        return grid_bounds

    def construct_symbolic_dimension_dict(self):
        sym_dim_list = get_symbolic_grid_dims_list()
        dimension_dict = dict(zip(sym_dim_list,
                                  self.bounds.block_dim.to_list() + self.bounds.grid_dim.to_list()))
        return dimension_dict

    def compute_correction(self, num_threads_per_dim: list, global_bounds_coordinates: list):
        correction = 0
        for el in self.exception_indices:
            ind = el[0]
            dim = el[1]
            ex_list = self.exceptions[ind * 3 + dim]
            [(bi_lo, ti_lo), (bi_hi, ti_hi)] = global_bounds_coordinates[dim * 2:(dim + 1) * 2:]
            for ex in ex_list:
                plane = 0
                if ind == 0:
                    cta_index_constraint = ex < 0
                    ex = abs(ex)   # -ex if cta_index_constraint else ex
                    if cta_index_constraint:
                        if bi_lo < ex < bi_hi:
                            plane = self.bounds.cta_length(dim)
                        elif ex == bi_lo:
                            plane = self.bounds.cta_bounds.get(dim)[1] - max(ti_lo, self.bounds.cta_bounds.get(dim)[0])
                        elif ex == bi_hi:   # TODO: this might be wrong
                            plane = min(ti_hi, self.bounds.cta_bounds.get(dim)[1]) - self.bounds.cta_bounds.get(dim)[0]
                    else:
                        if self.bounds.global_bounds.get(dim)[0] <= ex < self.bounds.global_bounds.get(dim)[1]:
                            plane = 1

                else:  # ind == 1
                    if self.bounds.cta_bounds.get(dim)[0] <= ex < self.bounds.cta_bounds.get(dim)[1]:
                        if ex < ti_lo:
                            num_ctas -= 1
                        if ex > ti_hi:
                            num_ctas -= 1
                        plane = num_ctas
                if plane <= 0:
                    continue
                for j in range(3):
                    plane *= num_threads_per_dim[j] if j != dim else 1
                correction += plane
        return correction


def sort_global_indices(bounds: Boundaries):
    dim_lengths = [bounds.global_length(i) for i in range(3)]
    dim_lengths_sorted = list(zip([0, 1, 2], dim_lengths))
    dim_lengths_sorted.sort(key=itemgetter(1))
    return [ind[0] for ind in dim_lengths_sorted]


def sort_by_global_length(a_list: list, sorted_indices: list):
    return [a_list[i] for i in sorted_indices]


def sort_boundaries(bounds: Boundaries, sorted_indices: list):
    global_dims = bounds.global_bounds.to_list()
    cta_dims = bounds.cta_bounds.to_list()
    global_dims_sorted = sort_by_global_length(global_dims, sorted_indices)
    cta_dims_sorted = sort_by_global_length(cta_dims, sorted_indices)
    dims = global_dims_sorted + cta_dims_sorted
    return [item for sublist in dims for item in sublist]


def get_sorted_arguments(bounds: Boundaries, dep_list: list, index_list: list):
    indices_sorted = sort_global_indices(bounds)
    index_list_sorted = sort_by_global_length(index_list[:3], indices_sorted) + sort_by_global_length(
        index_list[3:], indices_sorted)
    dep_dims = [dep_list[i] | dep_list[i + 3] for i in range(3)]
    dep_dims_sorted = sort_by_global_length(dep_dims, indices_sorted)  # [dep_dims[i] for i in indices_sorted]
    bounds_flat = sort_boundaries(bounds, indices_sorted)
    block_dim = sort_by_global_length(bounds.block_dim.to_list(), indices_sorted)
    return bounds_flat, block_dim, dep_dims_sorted, index_list_sorted


def eval_expressions(*args):
    sfd = enum.SpecialFunctionDummies()
    res = []
    for arg in args:
        lambda_arg = sp.lambdify(sfd.dummy_list, arg)
        arg_eval = lambda_arg(*sfd.function_list)
        res.append(arg_eval)
    return res


def get_indvar_range(start_val, final_val, step_val, inst: str, tripcount: int):
    if inst == 'add' or inst == 'sub':
        val_range = np.arange(start_val, final_val, step_val, dtype=int)
    elif inst == 'mul' or inst == 'shl':
        base = step_val if inst == 'mul' else 2 ** step_val
        start = 0  # math.log(start_val, base)
        stop = math.floor(math.log(final_val / start_val, base))
        num = tripcount
        val_range = int(start_val) * np.logspace(start, stop, int(num), base=base, dtype=int)
    elif inst == 'div' or inst == 'lshr':
        base = step_val if inst == 'div' else 2 ** step_val
        val_range = np.array([start_val // base ** i for i in range(tripcount)], dtype=int)
    else:
        raise NotImplementedError('Specified instruction either not implemented or invalid.')
    return val_range


def construct_indvar_arange(iv: IndvarInfo, iv_vals: tuple, parameters: dict, dims: dict, tripcount: int) -> np.ndarray:
    """
    Construct a numpy array containing all indices an indvar takes.
    @param iv: IndvarInfo object corresponding to an indvar
    @param iv_vals: tuple of start, step and final values
    @param parameters: kernel parameters
    @param dims: kernel grid dimensions
    @param tripcount: tripcount of the associated loop
    @return: numpy.ndarray containing all possible values of the indvar
    """
    start_val, step_val, final_val = tuple(val.subs(parameters).subs(dims) for val in iv_vals)
    tripcount = int(tripcount)
    inst = iv.rec.inst
    try:
        val_range = get_indvar_range(start_val, final_val, step_val, inst, tripcount)
    except TypeError:
        start_val, final_val, step_val = eval_expressions(start_val, final_val, step_val)
        val_range = get_indvar_range(start_val, final_val, step_val, inst, tripcount)

    return val_range


def get_symbolic_grid_index_list():
    bix, biy, biz = sp.symbols('bix:z')
    tix, tiy, tiz = sp.symbols('tix:z')
    return [bix, biy, biz, tix, tiy, tiz]


# TODO: Check order!
def get_symbolic_grid_dims_list():
    bdx, bdy, bdz = sp.symbols('bdx:z')
    gdx, gdy, gdz = sp.symbols('gdx:z')
    return [bdx, bdy, bdz, gdx, gdy, gdz]


def get_pattern_list(q: sp.Wild, sign: sp.Wild, factor: sp.Wild):
    indices = get_symbolic_grid_index_list()  # [bix, biy, biz, tix, tiy, tiz]
    dims = get_symbolic_grid_dims_list()  # [bdx, bdy, bdz, gdx, gdy, gdz]
    pattern_list = [factor * dims[i] * indices[i] + sign * indices[i + 3] + q
                    for i in range(3)]

    # WIP
    index_patterns = [factor * indices[i] + q for i in range(3)]
    return pattern_list + index_patterns


# DEPRECATED
def get_symbolic_placeholder_func_list():
    f_and = sp.Function('f_and')
    f_or = sp.Function('f_or')
    f_xor = sp.Function('f_xor')
    f_nand = sp.Function('f_nand')
    f_nor = sp.Function('f_nor')
    f_nxor = sp.Function('f_nxor')
    return [f_and, f_or, f_xor, f_nand, f_nor, f_nxor]


# DEPRECATED
def get_explicit_placeholder_func_list():
    return [enum.apply_and, enum.apply_or, enum.apply_xor, enum.apply_nand, enum.apply_nor, enum.apply_nxor]


# maybe improve using regex:
# https://stackoverflow.com/questions/6116978/how-to-replace-multiple-substrings-of-a-string
# might be moved to parse_utils
def insert_placeholder_functions(cond_str: str):
    rep = (('And', 'f_and'), ('Or', 'f_or'), ('~f_and', 'f_nand'), ('~f_or', 'f_nor'))
    for r in rep:
        cond_str = cond_str.replace(*r)
    return cond_str


# improved flow of ThreadGroup.determine_num_threads:
# After parsing, call simplify_condition to shift all terms on the LHS.
# Call test_condition_for_pattern.
# Then, if exactly_one_pattern_matches(), get_matched_pattern which return the pattern and its dimensional index.
# The last step before updating the global bounds is to get_relational.
# Depending on the sign of the pattern (+ or -), the relational needs to get inverted (namely if the sign is negative).
def simplify_condition(cond):
    """
    Remove constants on both sides of the conditional relation by transforming the condition such that RHS of the
    relation equals zero.
    """
    assert isinstance(cond, sp.Rel)
    cond_sim = cond.func(cond.lhs - cond.rhs, 0)
    return cond_sim


def test_condition_for_pattern(cond, pattern_list):
    """

    @param cond:
    @param pattern_list:
    @return:
    """
    res = []
    for patt in pattern_list:
        res.append(cond.lhs.match(patt))
    return res


# TODO: check if parse_utils.dependencies can be used somehow
def match_depends_on_indices(match: dict):
    """Check if the match depends on grid or CTA indices."""
    indices = get_symbolic_grid_index_list()
    for key in match:
        for index in indices:
            if index in match[key].free_symbols:
                return True
    return False


def exactly_one_pattern_matches(match_list):
    """
    Check if exactly one pattern is found in the examined condition.
    @param match_list: list of dictionaries containing the matching results
    @return:
    """
    count = 0
    i_match = None
    # sign = sp.Wild('sign')
    for i, match in enumerate(match_list):
        if match is None:
            continue
        if not match_depends_on_indices(match):
            count += 1
            i_match = i
    return count == 1, i_match


# seems to be obsolete
def get_matched_pattern(match_list):
    """
    Get the first entry in match_list that is not None. Note that a reasonable usage of this function requires a
    successful result of the function exactly_one_pattern_matches.
    @param match_list:
    @return:
    """
    sign = sp.Wild('sign')
    for ind, match in enumerate(match_list):
        if match[sign] != 0:
            return ind, match
    raise ValueError('No pattern matches.')


def contains_unsigned_only(l_var: list):
    for el in l_var:
        if el < 0:
            return False
    return True


# TODO: possibly add to class Boundaries
def is_well_defined(cta_lengths: list, global_lengths: list):
    if contains_unsigned_only(cta_lengths) and contains_unsigned_only(global_lengths):
        return True
    else:
        return False


def get_relational(condition, sign):
    func = condition.func if sign > 0 else condition.reversed.func
    return {
        sp.Eq: 0,
        sp.Ne: 1,
        sp.Lt: 2,
        sp.Gt: 3,
        sp.Le: 4,
        sp.Ge: 5
    }.get(func, 0)


def update_global_bounds(relational, dimension, new_bound, old_bounds: Boundaries, cta_index_constraint):
    block_dim = old_bounds.block_dim
    if relational == 0:  # or relational == 1:  # equality and unequality
        shift = block_dim.get(dimension) if cta_index_constraint else 1
        prev_low = old_bounds.global_bounds.get(dimension)[0]
        prev_high = old_bounds.global_bounds.get(dimension)[1]
        old_bounds.global_bounds.get(dimension)[0] = new_bound if new_bound - prev_low > 0 else 0
        old_bounds.global_bounds.get(dimension)[1] = new_bound + shift if prev_high - new_bound > 0 else -1
    elif relational == 2 or relational == 4:  # less than relation
        shift = 0 if relational == 2 else 1
        prev = old_bounds.global_bounds.get(dimension)[1]
        old_bounds.global_bounds.get(dimension)[1] = min(prev, new_bound + shift)
    elif relational == 3 or relational == 5:  # greater than relation
        shift = 1 if relational == 3 else 0
        prev = old_bounds.global_bounds.get(dimension)[0]
        old_bounds.global_bounds.get(dimension)[0] = max(prev, new_bound + shift)
    else:
        raise ValueError("illegal value for variable 'relation'")


def do_update(relational, dimension, new_bound, old_bounds: Dim3, block_dim: Dim3, cta_index_constraint: bool):
    cta_factor = block_dim.get(dimension) if cta_index_constraint else 1
    if relational == 0:  # or relational == 1:  # equality and unequality
        prev_low = old_bounds.get(dimension)[0]
        prev_high = old_bounds.get(dimension)[1]
        old_bounds.get(dimension)[0] = new_bound * cta_factor if new_bound - prev_low >= 0 else 67108864  # max grid dim
        old_bounds.get(dimension)[1] = (new_bound + 1) * cta_factor if prev_high - new_bound > 0 else -1
    elif relational == 2 or relational == 4:  # less than relation (or equal if 4)
        shift = 0 if relational == 2 else 1
        prev = old_bounds.get(dimension)[1]
        old_bounds.get(dimension)[1] = min(prev, (new_bound + shift) * cta_factor)
    elif relational == 3 or relational == 5:  # greater than relation (or equal if 5)
        shift = 1 if relational == 3 else 0
        prev = old_bounds.get(dimension)[0]
        old_bounds.get(dimension)[0] = max(prev, (new_bound + shift) * cta_factor)
    else:
        raise ValueError("illegal value for variable 'relation'")


def update_bounds(global_update: bool, relational, dim, new_bound, old_bounds: Boundaries, cta_index_constraint: bool):
    if global_update:
        do_update(relational, dim, new_bound, old_bounds.global_bounds, old_bounds.block_dim, cta_index_constraint)
    else:
        do_update(relational, dim, new_bound, old_bounds.cta_bounds, old_bounds.block_dim, False)


# TODO: Improve style!
def check_alignment(global_bounds_coords, overhang, cta_bounds):
    for i in range(3):
        cta_lo = cta_bounds[i][0]
        cta_hi = cta_bounds[i][1]

        # check lower bound
        # bi = global_bounds_coords[2 * i][0]
        ti = global_bounds_coords[2 * i][1]
        # if ti <= cta_lo:
        # global_bounds_coords[2 * i][1] = cta_lo
        if ti >= cta_hi:
            # global_bounds_coords[2 * i][1] = cta_lo
            global_bounds_coords[2 * i][0] += 1
        elif cta_lo < ti < cta_hi:
            overhang[i] += cta_hi - ti
            global_bounds_coords[2 * i][0] += 1

        global_bounds_coords[2 * i][1] = cta_lo

        # check upper bound
        # bi = global_bounds_coords[2 * i + 1][0]
        ti = global_bounds_coords[2 * i + 1][1]
        if ti <= cta_lo:
            global_bounds_coords[2 * i + 1][0] -= 1
            # global_bounds_coords[2 * i + 1][1] = cta_hi
        # elif ti >= cta_hi:
        #     global_bounds_coords[2 * i + 1][1] = cta_hi
        elif cta_lo < ti < cta_hi:
            overhang[i] += ti - cta_lo
            global_bounds_coords[2 * i + 1][0] -= 1

        global_bounds_coords[2 * i + 1][1] = cta_hi


class CFG:

    # constructor
    def __init__(self):
        self.graph = defaultdict(list)
        self.groups = []

    # Add a new edge from src to dest.
    def add_edge(self, src, dest, condition):
        self.graph[src].append([dest, condition])

    # Print the graph.
    def print_graph(self):
        print(self.graph)

    def find_src_dest_pair_for_condition(self, cond: str):
        ret_src = None
        ret_dest = None
        for src, dest_cond_list in self.graph.items():
            for dest_cond_pair in dest_cond_list:
                if cond == dest_cond_pair[1]:
                    ret_src = src
                    ret_dest = dest_cond_pair[0]
                    break
        return ret_src, ret_dest

    def sort_loop_edges(self):
        """Sort the edges in the CFG corresponding to loop edges
        such that the negated condition is the second entry."""
        for node in self.graph:
            succ_list = self.graph[node]
            if len(succ_list) == 2 and succ_list[0][1][0:2] == '!{':
                succ_list[0], succ_list[1] = succ_list[1], succ_list[0]

    # Recursively find all unexplored paths.
    def find_paths(self, node: int, visited: dict, path: list, path_cond: list, loops: list):

        # Return if the node has been visited as well as all of its successors.
        all_visited = True
        if visited[node]:
            for succ in self.graph[node]:
                succ_node = succ[0]
                all_visited = all_visited and visited[succ_node]
            if all_visited:
                return
            loop_nodes = construct_loop(path, node)
            loop_cond = self.get_loop_condition(node, loop_nodes)
            loops.append(Loop(loop_nodes, loop_cond))
        else:  # node not visited
            path.append(node)

        visited[node] = True
        loop_appended = False
        cond_appended = False
        head_controlled_loop = False

        for successor in self.graph[node]:
            succ_node = successor[0]
            succ_cond = successor[1]

            # test for self-loops
            if succ_node == node:
                if not loop_appended:
                    loops.append(Loop([node], succ_cond))
                    loop_appended = True
                continue

            if succ_cond[0:2] == '!{' and not loop_appended:
                for succ2 in self.graph[node]:
                    if succ2[0] == node:
                        loops.append(Loop([node], succ2[1]))
                        loop_appended = True

            if succ_node is not None:  # final node not reached
                if not is_loop(succ_cond):  # maybe replace by explicit statement (i.e. no function) -> performance?
                    path_cond.append(succ_cond)
                    cond_appended = True
                elif succ_cond[0] == '{' and visited[succ_node] and succ_cond[-1] == '1':    # foot-controlled loop  # AT THIS POINT IS THE PROBLEM!!!
                    loop_nodes = construct_loop(path, succ_node)
                    loop_cond = succ_cond
                    loops.append(Loop(loop_nodes, loop_cond))
                    loop_appended = True
                    continue
                elif succ_cond[0] == '{' and visited[succ_node] and succ_cond[-1] == '0':
                    head_controlled_loop = True
                    continue
                elif succ_cond[0:2] == '!{' and succ_cond[-1] == '0':
                    # check if the loop is entered in this path
                    loop_body = None
                    for s in self.graph[node]:
                        if s[0] != succ_node:
                            loop_body = s[0]
                            break
                    if loop_body is not None and loop_body not in path:
                        cond = generate_loop_exit_condition(succ_cond)
                        path_cond.append(cond)
                        cond_appended = True
                self.find_paths(succ_node, visited, path, path_cond, loops)
            else:
                # Create new lists from path and path_cond to avoid shallow copy issues.
                # Possibly use deepcopy
                # print('complete', path)
                new_group = ThreadGroup(list(path), list(path_cond), list(loops))
                self.groups.append(new_group)

            if cond_appended:
                path_cond.pop()

        visited[node] = head_controlled_loop
        # Pop if a self-loop has been appended
        if loop_appended:
            loops.pop()

        if all_visited:
            path.pop()
        else:
            loops.pop()

    # Initialize the path exploration.
    def explore_paths(self):
        path = []
        path_cond = []
        loops = []
        visited = {key: False for key in self.graph.keys()}
        self.find_paths(0, visited, path, path_cond, loops)

    def get_loop_condition(self, node, loop_nodes):
        for succ in self.graph[node]:
            succ_cond = succ[1]
            if succ_cond[0] == '{':
                return succ_cond
        else:    # foot-controlled loop
            loop_control = loop_nodes[-1]
            for succ in self.graph[loop_control]:
                if succ[0] == node and succ[1][0] == '{':
                    return succ[1]

        raise TypeError('Not a loop')


# Note: This quick'n'dirty solution is surprisingly faster than the corresponding solution using
#       itertools.takewhile. Maybe it can be optimized, though.
def construct_loop(path: list, current_node: int):
    res = []
    for node in reversed(path):
        res.append(node)
        if node == current_node:
            break
    res.reverse()
    return res


def generate_loop_exit_condition(condition: str):
    slice_begin = 0 if condition[0] == '{' else 1
    ls = LoopString(condition[slice_begin:])
    start = ls.start
    end = ls.end
    inst = ls.inst
    ge = ['add', 'mul', 'shl']
    le = ['sub', 'div', 'lshr']
    if inst in ge:
        predicate = '>='
    elif inst in le:
        predicate = '<='
    else:
        raise KeyError('instruction cannot be identified')
    cond_res = "{} {} {}".format(start, predicate, end)
    return cond_res


class Loop:

    # constructor
    def __init__(self, nodes: list, condition: str):
        self.nodes = nodes
        self.condition = LoopString(condition)
        self.tripcount = 0
        self.is_index_dependent = False
        self.loop_id = None
        self.main_indvar = ""

    def __eq__(self, other):
        return self.nodes == other.nodes

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                   self.nodes,
                                   self.condition.raw_loop)

    def get_loop_id(self, indvars: list):
        """Compare the recurrences and find the loop ID this way."""
        for iv_info in indvars:
            assert isinstance(iv_info, IndvarInfo)
            if iv_info.rec.raw_str == self.condition.raw_loop:
                self.loop_id = iv_info.loop_id
                self.main_indvar = iv_info.name
                break

    # IMPORTANT NOTE:
    # Despite the naming this is not the tripcount in the sense of how often the loop will be executed,
    # but rather the number of how often the backedge is taken.
    # For head-controlled loops these numbers coincide.
    # However, for foot-controlled loops (do-while) the tripcount is larger by one. Thus the subtraction of the flag.
    def compute_tripcount(self, bounds: Boundaries, parameters: dict):
        block_dim = bounds.block_dim
        grid_dim = bounds.grid_dim
        dim_list = get_symbolic_grid_dims_list()
        dimension_dict = dict(zip(dim_list,
                                  [block_dim.x, block_dim.y, block_dim.z,
                                   grid_dim.x, grid_dim.y, grid_dim.z]))
        sfd = enum.SpecialFunctionDummies()
        func_list = sfd.dummy_list
        funcs = sfd.function_list
        tc_sym = sp.sympify(self.condition.tripcount).subs(dimension_dict).subs(parameters)
        index_list = get_symbolic_grid_index_list()
        deps = dependencies(tc_sym, index_list)

        if deps != [0, 0, 0, 0, 0, 0]:
            self.is_index_dependent = True
            bounds_flat, block_dim, dep_dims_sorted, index_list_sorted = get_sorted_arguments(bounds, deps, index_list)
            guard = sp.sympify(self.condition.guard)
            tc_sym += guard
            tc_num = sp.lambdify(index_list_sorted + func_list, tc_sym, "numpy")
            self.tripcount = enum.parfor_cond_np(tc_num, bounds_flat, funcs, block_dim, dep_dims_sorted)
        else:
            tc_num = sp.lambdify(func_list, tc_sym, "numpy")
            self.tripcount = tc_num(*funcs)

    def compute_indvar_dependent_tripcount(self, dimension_dict: dict, parameters: dict, indvars: list, loops: list):
        sfd = enum.SpecialFunctionDummies()
        func_list = sfd.dummy_list
        funcs = sfd.function_list
        tc_sym = sp.sympify(self.condition.tripcount).subs(dimension_dict).subs(parameters)
        if len(tc_sym.free_symbols) != 1:
            raise TypeError('too many free symbols!')  # find better error type
        indvar_sym = list(tc_sym.free_symbols)[0]
        indvar_name = indvar_sym.name[:-3]

        loop_id = None
        indvar = None
        for iv in indvars:
            if indvar_name == iv.name:
                loop_id = iv.loop_id
                indvar = iv
                break
        if loop_id is None:
            raise TypeError("couldn't find loop")

        tc = None
        for lp in loops:
            if lp.loop_id == loop_id:
                tc = lp.tripcount
                break
        if tc is None:
            raise ValueError("couldn't get trip count")
        start_val = sp.sympify(indvar.rec.start)
        step_val = sp.sympify(indvar.rec.step)
        final_val = sp.sympify(indvar.rec.end)
        val_range = construct_indvar_arange(indvar, (start_val, step_val, final_val), dimension_dict, parameters, tc)

        tc_num = sp.lambdify(func_list + [indvar_sym], tc_sym, "numpy")
        res = tc_num(*funcs, val_range) + 1

        # Weight by the outer loop trip count.
        # This is necessary since every iteration is already examined.
        self.tripcount = (np.sum(res) - (tc + 1)) / (tc + 1)


# TODO: support more increment operations!
def determine_tripcount_for_index(bounds: Boundaries, dim, n, step, foot_flag, start_flag, offset):
    """
    Compute the backedge count (?) if the loop depends on the global index.
    Note that currently (when commenting out foot_flag) this function may only be used in the
    ThreadGroup approach.
    @param bounds:
    @param dim:
    @param n:
    @param step:
    @param foot_flag:
    @param start_flag: if set, the loop starts with the index and runs to n; else other way around
    @param offset:
    @return:
    """
    block_dim = bounds.block_dim.get(dim)
    (global_min, global_max) = tuple(bounds.global_bounds.get(dim))
    (local_min, local_max) = tuple(bounds.cta_bounds.get(dim))
    (bi_lo, ti_lo) = tuple(divmod(global_min, block_dim))
    (bi_hi, ti_hi) = tuple(divmod(global_max, block_dim))
    res = 0
    for b in range(bi_lo, bi_hi+1):
        if start_flag:
            lower = max(local_min + b * block_dim + offset, global_min)
            upper = min(local_max + b * block_dim + offset, n)
        else:
            lower = max(local_min + b * block_dim + offset, n)
            upper = min(local_max + b * block_dim + offset, global_max)
        for i in range(lower, upper):
            res += math.ceil(abs(n - i / step)) - foot_flag
    return res


# simple case: either loop_start or loop_end depends on global index
# -> the other one represents "n" in determine_tripcount_for_index.
# TODO: more complex case, i.e. both start and end depend on global index
def depends_on_global_index(loop_start: sp.Symbol, loop_end: sp.Symbol):
    inds = get_symbolic_grid_index_list()    # [bix, biy, biz, tix, tiy, tiz]
    dims = get_symbolic_grid_dims_list()     # [bdx, bdy, bdz, gdx, gdy, gdz]
    q = sp.Wild('q')
    patt = [q + inds[i] * dims[i] + inds[i+3] for i in range(3)]
    start_matches = [loop_start.match(pattern) for pattern in patt]
    end_matches = [loop_end.match(pattern) for pattern in patt]

    start_flag, i_start = exactly_one_pattern_matches(start_matches)
    end_flag, i_end = exactly_one_pattern_matches(end_matches)
    depends = start_flag | end_flag
    offset = 0
    dim = None

    if start_flag:
        offset = start_matches[i_start][q]
        dim = i_start
    if end_flag:
        offset = end_matches[i_end][q]
        dim = i_end

    return depends, start_flag, offset, dim
