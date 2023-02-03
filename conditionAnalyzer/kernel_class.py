import numpy as np
from cfg import CFG, Dim3, Boundaries, ThreadGroup
import config_utils as config
from copy import deepcopy


# Strategy:
# 1) construct a Kernel object
#    -> build CFG
#    -> find paths and ThreadGroups
#    -> read grid dimensions and additional parameters from host file
# 2) for each ThreadGroup ...
#    ... determine num_threads
#    ... compute loop tripcounts if necessary
# 3) walk through the path of each ThreadGroup and add num_threads to the execution frequency of each BasicBlock
#    in the path, multiplied by the tripcount if necessary


class Kernel:
    """
    This is the entering point for the analysis. The output file of the device pass is read in and from this,
    the CFG is constructed.
    """

    # constructor
    def __init__(self, device_file: str, host_file: str, start_lineno: int, analyze_all: bool):
        self.cfg = CFG()
        self.kernel_name = config.get_kernel_name(device_file) 

        with open(device_file) as f:
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

        if self.has_cfg:
            self.indvars = config.get_indvars(next_line)
            self.indvar_block_freq = dict()
            trans_mat_raw = np.genfromtxt(device_file, unpack=False, delimiter=';', dtype=str, skip_header=2)
            trans_mat_index = trans_mat_raw[:, [0, 1]].astype(np.int)
            trans_mat_condition = trans_mat_raw[:, 2]

            self.num_blocks = np.amax(trans_mat_index).item() + 1  # use item() to cast from np.int to built-in int
            self.block_freqs = [0] * self.num_blocks
            num_entries = len(trans_mat_condition)

            for i in range(num_entries):
                row = trans_mat_index[i, 0]
                col = trans_mat_index[i, 1]
                self.cfg.add_edge(col, row, trans_mat_condition[i])
                if row not in trans_mat_index[:, 1] and row not in self.cfg.graph:
                    self.cfg.add_edge(row, None, '1')

            self.cfg.sort_loop_edges()
            self.cfg.explore_paths()

            bounds = Boundaries(Dim3(self.dims[3], self.dims[4], self.dims[5]),
                                Dim3(self.dims[0], self.dims[1], self.dims[2]))
            for group in self.cfg.groups:
                group.bounds = deepcopy(bounds)
                group.assign_loop_ids(self.indvars)

    def analyze_groups(self):
        """
        For all ThreadGroup objects derived from the CFG, determine the number of satisfying threads
        and the backedge taken count of loops, if necessary.
        Add the result to all Basic Blocks of the kernel.
        @return:
        """
        if self.has_cfg:
            print("num groups", len(self.cfg.groups))
            for group in self.cfg.groups:
                group.determine_num_threads(self.parameter_dict)
                if group.num_threads != 0:
                    group.evaluate_loop_tripcounts(self.parameter_dict, self.indvars)
                    dim_dict = group.construct_symbolic_dimension_dict()
                    indvar_res = group.analyze_indvar_conditions(self.indvars, self.parameter_dict, dim_dict)
                    if 0 in indvar_res:
                        group.num_threads = 0
                        continue

                    for res in indvar_res:
                        group.num_threads *= res
                self.add_num_threads(group)
        else:
            total_num_threads = np.prod(self.dims)
            self.block_freqs = [total_num_threads]

    def add_num_threads(self, group: ThreadGroup):
        """
        Add the number of satisfying threads in group to each Basic Block in the path.
        This number is multiplied by the loop tripcount if necessary.
        @param group:
        @return:
        """
        for block in group.path:
            num = group.num_threads
            is_in_idx_dependent_loop = False
            for loop in group.loops:
                if block in loop.nodes:
                    if loop.is_index_dependent and group.num_threads != 0:
                        if not is_in_idx_dependent_loop:    # avoid multiple divisions for nested index-dependent loops
                            num /= group.num_threads
                            is_in_idx_dependent_loop = True
                        num *= loop.tripcount
                    elif loop.condition.is_guarded or block == loop.nodes[0]:
                        num *= (loop.tripcount + 1)
                    else:
                        num *= loop.tripcount
            self.block_freqs[block] += num
