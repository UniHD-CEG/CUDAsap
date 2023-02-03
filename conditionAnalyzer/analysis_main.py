import os
import time
from argparse import ArgumentParser
from kernel_class import Kernel
from matrix_analysis import MatrixAnalysis


def process_args():
    parser = ArgumentParser()
    parser.add_argument('-d', '--device_file', metavar='<CFG file>', type=str, required=True,
                        help='CFG file (from device pass)')
    parser.add_argument('-p', '--parameter_file', metavar='<parameter file>', type=str, required=True,
                        help='kernel parameter file (possibly from host runtime)')
    parser.add_argument('-a', '--all', required=False, action='store_true',
                        help='analyze all configuration entries in the parameter file for the given kernel')
    return parser.parse_args()


def dump_results(kernel, t_elapsed):
    print("*** RESULTS ***")
    grid_dim = str(kernel.dims[0]) + " x " + str(kernel.dims[1]) + " x " + str(kernel.dims[2])
    block_dim = str(kernel.dims[3]) + " x " + str(kernel.dims[4]) + " x " + str(kernel.dims[5])
    print("grid: ", grid_dim, ",", block_dim)
    for i, freq in enumerate(kernel.block_freqs):
        print("  block ", i, ":", freq)
    print("  total executions: ", sum(kernel.block_freqs))
    print("  elapsed time: {:.6f}s".format(t_elapsed))
    if isinstance(kernel, MatrixAnalysis):
        print("  time to fill the matrix: {:.6f}s".format(kernel.t_fill))
        print("  time to solve the equations: {:.6f}s".format(kernel.t_solve))


def write_results_to_file(kernel, t_elapsed, analysis_switch):
    with open('block_freq.txt', mode='a') as f:
        f.write(kernel.kernel_name)
        f.write(',' + str(analysis_switch))
        for dim in kernel.dims:
            f.write(',' + str(dim))
        for name, par in kernel.parameter_dict.items():
            f.write(',[' + str(name) + ']' + str(par))
        f.write(",{:.6f}".format(t_elapsed))
        if analysis_switch == 1:
            f.write(",{:.6f}".format(kernel.t_fill))
            f.write(",{:.6f}".format(kernel.t_solve))
        for freq in kernel.block_freqs:
            f.write(',' + str(int(round(freq))))
        f.write('\n')


def file_len(file: str):
    with open(file, 'r') as f:
        for i, l in enumerate(f):
            pass
    return i + 1


if __name__ == '__main__':
    args = process_args()
    device_file = args.device_file
    host_file = args.parameter_file
    analyze_all = args.all
    analysis_switch = int(os.getenv('ANALYSIS_TYPE', default=1))
    num_lines = file_len(host_file)
    start_lineno = 0

    while start_lineno < num_lines:
        if analysis_switch != 1:
            print('Thread Group Approach')
            t_start = time.perf_counter()
            kernel = Kernel(device_file, host_file, start_lineno, analyze_all)
            kernel.analyze_groups()
            t_end = time.perf_counter()
        else:
            print('Adjacency Matrix Approach')
            t_start = time.perf_counter()
            kernel = MatrixAnalysis(device_file, host_file, start_lineno, analyze_all)
            if kernel.dims == [0, 0, 0, 0, 0, 0]:
                break
            kernel.perform_analysis()
            t_end = time.perf_counter()
        t_elapsed = t_end - t_start

        dump_results(kernel, t_elapsed)
        write_results_to_file(kernel, t_elapsed, analysis_switch)
        start_lineno = kernel.lineno + 1
