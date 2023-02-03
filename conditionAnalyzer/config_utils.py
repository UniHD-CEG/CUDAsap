# imports
import numpy as np
import sympy as sp
from parse_utils import Recurrence


def get_number_or_nan(string):
    """Parse str to int. Return nan on failure."""
    try:
        if '.' in string:
            return float(string)
        else:
            return int(string)
    except ValueError:
        return float('nan')


def get_parameter(string, get_name: bool = False) -> dict:
    """Parse kernel launch parameter.
    Syntax: [i]par, where par is the i-th entry in the kernel sensitivity list.
    """
    assert string[0] == '['
    right_par = string.find(']')
    assert right_par != -1
    p_ind = string[1:right_par]
    ret_val = string[right_par+1:] if get_name else get_number_or_nan(string[right_par + 1:])
    return {p_ind: ret_val}


def get_launch_arguments(line: str, get_name: bool, split_list: list = None, delimiter: str = ';') -> dict:
    """Get the names or values and indices of kernel launch arguments."""
    if split_list is None:
        line_split = line.split(delimiter)
    else:
        line_split = split_list
    args = {}
    for i in range(len(line_split)-1):
        args.update(get_parameter(line_split[i], get_name))
    return args


def convert_to_symbol(par_dict: dict) -> None:
    """Convert the value entries of the parameter dictionary to sympy.Symbol."""
    for idx, par_name in par_dict.items():
        par_dict[idx] = sp.Symbol(par_name)


def concat(x, y):
    """Concatenate the two input strings representing a 32-bit integer each to a 64-bit integer."""
    x_bin = bin(int(x))
    y_bin = bin(int(y))
    x_width = len(x_bin[2:])
    pad = (32-x_width) * '0'
    res_bin = y_bin[2:] + pad + x_bin[2:]
    res = str(int(res_bin, 2))
    return res


def check_nans(number_list: list, list_name: str, is_int: bool):
    """Check for nans in number_list and request user input if an entry is nan."""
    for i in range(len(number_list)):
        if np.isnan(number_list[i]):
            if list_name == 'grid' and (i == 0 or i == 2):
                print("Component "+str(i)+" of "+list_name+" is not a number.")
                temp_x = input("Enter x: ")
                temp_y = input("Enter y: ")
                temp = concat(temp_x, temp_y)
            else:
                temp = input("Component "+str(i)+" of "+list_name+" is not a number. Please enter it now: ")
            number_list[i] = int(temp) if is_int else float(temp)


def check_nans_in_dict(number_dict: dict, dict_name: str):
    for key, val in number_dict.items():
        if np.isnan(val):
            temp = input("Component " + key + " of " + dict_name + " is not a number. Please enter it now: ")
            if '.' in temp:   # look for decimal point
                is_int = False
            else:
                is_int = True
            number_dict[key] = int(temp) if is_int else float(temp)


def convert_xy_dims(grid, ext_grid):
    """In CUDA 8.0, the x- and y-dimensions of the grid and the blocks are concatenated to a 64-bit integer.
    In this function, they are restored as 32-bit integers and stored in an extended dimension list, together with
    the respective z-component.
    """
    ext_grid[2] = grid[1]
    ext_grid[5] = grid[3]
    for i in [0, 2]:
        binary = bin(grid[i]).lstrip('0b')
        size = len(binary)
        assert size <= 64, "dimension is too large"
        x_dim = int(binary[size-32:], 2)
        y_dim = int(binary[:size-32], 2)
        ext_grid[i//2 * 3] = x_dim
        ext_grid[i//2 * 3 + 1] = y_dim


def get_kernel_name(device_file: str):
    last_slash_pos = device_file.rfind('/')
    kernel_name = device_file[last_slash_pos + 1:-4]
    if kernel_name[-3:] == "cfg":
        kernel_name = kernel_name[:-4]
    return kernel_name


def get_grid_and_parameters(kernel_name: str, host_file: str, launch_args: dict, start_lineno: int, analyze_all: bool):
    """
    Read the kernel grid configuration as well as the parameters from the output of the host pass.
    @param kernel_name: kernel to examine
    @param host_file: filename of the host pass output
    @param launch_args: dict where each launch argument (as sympy.Symbol) is assigned to its index
    @param start_lineno: line number to start the search for kernel config
    @return dims, params, i: grid dimensions, parameter dict and line number
    """
    dims = 6 * [0]    # [gdx, gdy, gdz, bdx, bdy, bdz]
    params = {}
    with open(host_file) as f:
        # TODO: save each hit as a separate configuration to test
        for i, line in enumerate(f):
            if i < start_lineno:
                continue
            if line.find(kernel_name) != -1:
                line_split = line.split(',')
                grid = [get_number_or_nan(line_split[i]) for i in range(1, 5)]
                par = get_launch_arguments(line='', get_name=False, split_list=line_split[5:])

                check_nans(grid, "grid", True)
                check_nans_in_dict(par, "par")
                convert_xy_dims(grid, dims)

                params = {}
                for key, val in launch_args.items():
                    params.update({val: par[key]})
                if analyze_all:
                    break

    return dims, params, i


class IndvarInfo:
    """Container class holding information about an indvar."""
    def __init__(self, name: str, loop_id: int, recurrence: Recurrence):
        self.name = name
        self.loop_id = loop_id
        self.rec = recurrence

    def __repr__(self):
        return '{}({}, {}, {})'.format(self.__class__.__name__,
                                       self.name,
                                       self.loop_id,
                                       self.rec)


def get_indvars(line: str):
    line_split = line.split(sep=';')
    indvars = []
    for el in line_split:
        if not el[0] == '[':
            continue
        rec_begin = el.find('{')
        rec = Recurrence(el[rec_begin::])
        sep_name_id = el.find(',', 0, rec_begin)
        name = el[1:sep_name_id:]
        loop_id = el[sep_name_id+1:rec_begin-1:]
        indvars.append(IndvarInfo(name, int(loop_id), rec))
    return indvars
