import sympy as sp
import numpy as np
import parse_utils as parse
import config_utils as config
import enumerate_threads_utils as enum
import cfg
import time
import os
from joblib import Parallel, delayed
import multiprocessing


def read_transition_matrix(filename: str):
    trans_mat_raw = np.genfromtxt(filename, unpack=False, delimiter=';', dtype=str, skip_header=2, comments=None)
    trans_mat_index = trans_mat_raw[:, [0, 1]].astype(np.int)
    trans_mat_condition = trans_mat_raw[:, 2]

    num_blocks = np.amax(trans_mat_index).item() + 1  # use item() to cast from np.int to built-in int
    num_entries = len(trans_mat_condition)
    symbol_list = sp.symbols('s0:' + str(num_entries))

    trans_mat_sym = sp.eye(num_blocks)
    condition_dict = {}

    for i in range(num_entries):
        col = trans_mat_index[i, 1]
        row = trans_mat_index[i, 0]
        if trans_mat_condition[i] != "1":
            condition_dict[symbol_list[i]] = trans_mat_condition[i]
            if "&&" in trans_mat_condition[i]:
                dep_cond = trans_mat_condition[i].rsplit(sep=" && ", maxsplit=1)[0]
                index = parse.find_string_index(trans_mat_condition, dep_cond, num_entries)
                if index != -1:
                    trans_mat_sym[row, col] -= symbol_list[i] / symbol_list[index]  # explicit minus sign to account
                    # for the cancelling of the
                    # individual signs in the fraction
                    continue
            trans_mat_sym[row, col] += symbol_list[i]
        else:
            trans_mat_sym[row, col] = -1

    return trans_mat_sym, condition_dict


def find_indvars_in_condition(cond: sp.Symbol, relevant_indvars: list, indvar_sym: list,
                              indvar_list: list, index_list: list):
    for sym in cond.free_symbols:
        if 'PHI' in sym.name:
            indvar_sym.append(sym)
            for idx, iv_info in enumerate(indvar_list):
                if sym.name[:-3] == iv_info.name:
                    relevant_indvars.append(iv_info)
                    index_list.append(idx)
                    break


class MatrixAnalysis:
    def __init__(self, trans_mat_file: str, host_file: str, start_lineno: int, analyze_all: bool):
        self.kernel_name = config.get_kernel_name(trans_mat_file)
        print(self.kernel_name)
        with open(trans_mat_file) as f:
            line = f.readline()
            launch_args = config.get_launch_arguments(line, get_name=True)  # type: dict
            next_line = f.readline()
            self.has_cfg = bool(f.readline())
        config.convert_to_symbol(launch_args)

        self.dims, self.parameter_dict, self.lineno = config.get_grid_and_parameters(self.kernel_name,
                                                                                     host_file,
                                                                                     launch_args,
                                                                                     start_lineno,
                                                                                     analyze_all)
        self.total_num_threads = np.prod(self.dims)
        self.block_freqs = None
        self.t_fill = 0
        self.t_solve = 0

        if self.has_cfg:
            self.trans_mat_sym, self.condition_dict = read_transition_matrix(trans_mat_file)
            self.num_blocks = self.trans_mat_sym.shape[0]
            self.associate_conditions = dict()
            self.condition_keys = dict()
            self.get_associate_conditions_columnwise()
            self.get_key_list()

            self.indvar_list = config.get_indvars(next_line)
            # In order to use the methods of the Loop class, we need a Boundaries object.
            grid_dim = cfg.Dim3(self.dims[0], self.dims[1], self.dims[2])
            block_dim = cfg.Dim3(self.dims[3], self.dims[4], self.dims[5])
            self.bounds = cfg.Boundaries(block_dim, grid_dim)

            sym_dim_list = cfg.get_symbolic_grid_dims_list()
            self.dimension_dict = dict(zip(sym_dim_list, block_dim.to_list() + grid_dim.to_list()))

            # Provide a list of functions not supported by SymPy.
            sfd = enum.SpecialFunctionDummies()
            self.func_list = sfd.dummy_list
            self.funcs = sfd.function_list

            self.loops = dict()
            self.get_loops()
            self.frequency_dict = dict()

            self.solve_numeric = bool(int(os.getenv('SOLVE_NUMERIC', default=1)))
            self.sequential = bool(int(os.getenv('SEQUENTIAL', default=1)))

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.kernel_name)

    def get_associate_conditions(self):
        key_list = self.condition_dict.keys()
        for key in key_list:
            if self.condition_dict[key][0] == '!':
                for key_i in key_list:
                    if self.condition_dict[key_i] == self.condition_dict[key][1:]:
                        self.associate_conditions[key_i] = key

    def get_associate_conditions_columnwise(self):
        for i in range(self.num_blocks):
            col = self.trans_mat_sym.col(i)
            symbols = list(col.free_symbols)
            if len(symbols) == 2:
                cond0 = self.condition_dict[symbols[0]]
                if cond0[0] == '!':
                    self.associate_conditions[symbols[1]] = symbols[0]
                else:
                    self.associate_conditions[symbols[0]] = symbols[1]

    def get_key_list(self):
        key_list = self.condition_dict.keys()
        # Find keys which are not stored in associate_conditions (neither key nor value).
        # These are those which contain partial negations.
        extra_keys = []
        for key in key_list:
            if key not in self.associate_conditions.keys() and key not in self.associate_conditions.values():
                extra_keys.append(key)

        self.condition_keys = list(self.associate_conditions.keys()) + extra_keys

    def get_loops(self):
        for key in self.condition_keys:
            condition = self.condition_dict[key]
            if parse.is_loop(condition):
                loop = cfg.Loop([], condition)
                loop.get_loop_id(self.indvar_list)
                self.loops[key] = loop

    def evaluate_loop_tripcounts(self):
        if not self.loops:
            print('no loops')
        dependent_loops = []
        for i, loop in self.loops.items():
            if 'PHI' in loop.condition.tripcount:
                dependent_loops.append(i)
                continue
            try:
                loop.compute_tripcount(self.bounds, self.parameter_dict)
            except TypeError:
                print('loop trip count cannot be computed')
        for i in dependent_loops:
            self.loops[i].compute_indvar_dependent_tripcount(self.dimension_dict,
                                                             self.parameter_dict,
                                                             self.indvar_list,
                                                             self.loops.values())

    def get_indvar_values(self, indvar_list: list):
        last_valid_vals = {}
        indvar_values = len(indvar_list) * [(0, 0, 0)]
        for idx, iv in enumerate(indvar_list):
            if iv.loop_id not in (loop.loop_id for loop in self.loops.values()):
                continue
            if 'PHI' in iv.rec.raw_str:
                raise NotImplementedError('PHINodes are not supported yet.')
            inst = '+' if iv.rec.inst == 'add' else '-'    # TODO: support further increment operations!
            start_val = sp.sympify(iv.rec.start)
            step_val = sp.sympify(iv.rec.step)
            final_val = self.compute_indvar_final_value(iv, last_valid_vals)
            last_valid = final_val + sp.sympify('-' + inst + iv.rec.step)
            last_valid_vals[sp.Symbol(iv.name+'PHI')] = last_valid
            indvar_values[idx] = (start_val, step_val, final_val)

        return indvar_values

    def compute_indvar_final_value(self, indvar_info, last_valids):
        start, step, inst, end = indvar_info.rec.evaluate_PHINodes([])
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
            for loop in self.loops.values():
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

    def analyze_indvar_conditions(self, indvar_conditions: dict):
        result_true = dict()
        for key, cond in indvar_conditions.items():
            cond_parsed = parse.parse_condition(cond)
            cond_sympy = sp.sympify(cfg.insert_placeholder_functions(cond_parsed))
            cond_sim = cond_sympy.subs(self.parameter_dict).subs(self.dimension_dict)
            # 1) find indvar and the associated loop
            relevant_indvars = []
            index_list = []
            indvar_sym = []
            find_indvars_in_condition(cond_sim, relevant_indvars, indvar_sym, self.indvar_list, index_list)
            indvar_values = self.get_indvar_values(relevant_indvars)
            if len(indvar_sym) > 1:
                print('Seems like there is more than one indvar in this condition.')
                print('This will be implemented if necessary.')
                return 0

            grid_index_list = cfg.get_symbolic_grid_index_list()
            dep = parse.dependencies(cond_sim, grid_index_list)

            if dep == [0, 0, 0, 0, 0, 0]:
                f = sp.lambdify(self.func_list + indvar_sym, cond_sim, "numpy")
                # 2) compute final value if necessary
                for idx, iv in enumerate(relevant_indvars):
                    # 4) apply condition and count Trues
                    tripcount = self.get_tripcount_for_loop(iv.loop_id)
                    vals = tuple(indvar_values[idx])
                    val_range = cfg.construct_indvar_arange(iv, vals, self.parameter_dict, self.dimension_dict,
                                                            tripcount)
                    result_true[key] = (len(val_range[f(*self.funcs, val_range)]) / tripcount)
            else:
                bounds_flat, block_dim, dep_sorted, index_list_sorted = cfg.get_sorted_arguments(self.bounds, dep,
                                                                                                 grid_index_list)
                f = sp.lambdify(index_list_sorted + self.func_list + indvar_sym, cond_sim, "numpy")
                for idx, iv in enumerate(relevant_indvars):
                    tripcount = self.get_tripcount_for_loop(iv.loop_id)
                    vals = tuple(indvar_values[idx])
                    val_range = cfg.construct_indvar_arange(iv, vals, self.parameter_dict, self.dimension_dict,
                                                            tripcount)
                    count = enum.parfor_cond_np(f, bounds_flat, self.funcs, block_dim, dep_sorted, val_range)
                    result_true[key] = count / tripcount
        return result_true

    def get_tripcount_for_loop(self, loop_id: int):
        """
        Find loop in the ThreadGroup and return its tripcount. An IndexError is raised if there is no loop
        with the ID loop_id.
        @param loop_id: ID of the loop
        @return: tripcount of the loop with the requested ID
        """
        tripcount = None
        for loop in self.loops.values():
            if loop_id == loop.loop_id:
                correct_foot_control = 1 if loop.condition.is_guarded else 0
                tripcount = loop.tripcount + correct_foot_control
                break
        if tripcount is None:
            raise IndexError('Loop ID not valid.')
        return tripcount

    def analyze_condition(self, key: sp.Symbol, indvar_conditions: dict):
        condition = self.condition_dict[key]
        if parse.is_loop(condition):
            loop = self.loops[key]
            if not loop.is_index_dependent:
                tc = loop.tripcount
            else:
                tc = loop.tripcount / self.total_num_threads
                if loop.condition.is_guarded:
                    tc -= 1
            numerator = 1 if condition[0] == '!' else tc
            denominator = tc + 1
            if denominator == 0:
                loop_freq = 0
            else:
                loop_freq = numerator / denominator

            self.frequency_dict[key] = -loop_freq
            self.frequency_dict[self.associate_conditions[key]] = -(1 - loop_freq)
        elif 'PHI' not in condition:  # ordinary condition
            cond_parsed = parse.parse_condition(condition)
            cond_sym = sp.sympify(cfg.insert_placeholder_functions(cond_parsed))
            cond_sym_sim = cond_sym.subs(self.dimension_dict).subs(self.parameter_dict)
            index_list = cfg.get_symbolic_grid_index_list()
            dep = parse.dependencies(cond_sym_sim, index_list)

            if dep != [0, 0, 0, 0, 0, 0]:
                bounds_flat, block_dim, dep_dims_sorted, index_list_sorted = cfg.get_sorted_arguments(self.bounds,
                                                                                                      dep,
                                                                                                      index_list)
                f = sp.lambdify(index_list_sorted + self.func_list, cond_sym_sim, "numpy")
                count = enum.parfor_cond_np(f, bounds_flat, self.funcs, block_dim, dep_dims_sorted)
                cond_freq = count / self.total_num_threads
            else:  # condition is independent of indices
                f = sp.lambdify(self.func_list, cond_sym_sim, "numpy")
                cond_freq = 1 if f(*self.funcs) else 0

            self.frequency_dict[key] = -cond_freq
            if key in self.associate_conditions.keys():
                self.frequency_dict[self.associate_conditions[key]] = -(1 - cond_freq)
        elif 'PHI' in condition and '$' not in condition:  # indvar condition
            indvar_conditions[key] = condition
        else:
            raise NotImplementedError('PHINode dependent conditions are not supported yet.')

    # Strategy:
    # 1) determine frequency for loops and non-indvar conditions
    # 2) determine frequency for indvar conditions
    def determine_transition_frequencies(self):
        indvar_conditions = dict()
        self.evaluate_loop_tripcounts()

        if self.sequential:
            print("determine sequentially")
            for key in self.condition_keys:
                self.analyze_condition(key, indvar_conditions)
        else:
            cpu_count = multiprocessing.cpu_count()
            print(f"determine parallel with {cpu_count} jobs")
            Parallel(n_jobs=cpu_count, require='sharedmem')(delayed(self.analyze_condition)(key, indvar_conditions)
                                                            for key in self.condition_keys)
        if indvar_conditions:
            indvar_res = self.analyze_indvar_conditions(indvar_conditions)
            for key, res in indvar_res.items():
                self.frequency_dict[key] = -res
                if key in self.associate_conditions.keys():
                    self.frequency_dict[self.associate_conditions[key]] = -(1 - res)

    def solve_matrix(self):
        if self.solve_numeric:
            print("Solve numerically")
            trans_mat_num = np.array(self.trans_mat_sym.subs(self.frequency_dict), dtype=float)
            input_vector = np.zeros(self.num_blocks)
            input_vector[0] = self.total_num_threads
            try:
                self.block_freqs = np.linalg.solve(trans_mat_num, input_vector)
            except np.linalg.LinAlgError:
                print("compute approximate solution")
                self.block_freqs = np.linalg.lstsq(trans_mat_num, input_vector)
        else:
            print("Solve symbolically")
            input_vector = sp.zeros(self.num_blocks, 1)
            num_threads = sp.symbols('nt')
            input_vector[0] = num_threads
            sol_sym = sp.linsolve((self.trans_mat_sym, input_vector))
            sol_num = list(sol_sym.subs(self.frequency_dict).subs(num_threads, self.total_num_threads))
            self.block_freqs = np.array(sol_num).astype(np.float64).flatten()

    def perform_analysis(self):
        if self.has_cfg:
            t = time.perf_counter()
            self.determine_transition_frequencies()
            self.t_fill = time.perf_counter() - t
            t = time.perf_counter()
            self.solve_matrix()
            self.t_solve = time.perf_counter() - t
        else:
            self.block_freqs = [self.total_num_threads]
