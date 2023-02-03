# imports
from typing import Dict
import math
import sympy as sp


# The additional whitespaces are used as padding -> no wrong replacement
binaryOperators = {
    " add " : " + ",
    " fadd ": " + ",
    " sub " : " - ",
    " fsub ": " - ",
    " mul " : " * ",
    " fmul ": " * ",
    # " udiv ": " // ",
    # " sdiv ": " // ",
    " fdiv ": " / ",
    " urem ": " % ",
    " srem ": " % ",
    " frem ": " % ",
    " and " : " & ",
    " or "  : " | ",
    " xor " : " ^ ",
    "!"     : "~"
}

# Implement bitwise ops in sympy:
#  - define dummy sympy function, e.g. f_add, f_or, f_xor
#  - find &, |, ^ in condition string, find lhs and rhs, replace by corresponding dummy functino
#  - sympify
#  - lambdify with dummy function as function argument -> replace by lambda / original operator


def replace_operators(cond):
    """Replace verbose operator names by the corresponding operator in Python notation

    :param cond: string  which contains the branch condition
    :return condRep: string with replaced operators
    """
    cond_rep = cond
    for opName, op in binaryOperators.items():
        assert isinstance(cond_rep, str)
        cond_rep = cond_rep.replace(opName, op)
    return cond_rep


def find_parens(s):
    """Find matching parentheses in a string s.

    Token from https://stackoverflow.com/questions/29991917/indices-of-matching-parentheses-in-python

    :param s: string to examine
    :return: dictionary where {key: value} = {position_left_parenthesis: position_right_parenthesis}
    """
    toret = {}  # type: Dict[int, int]
    pstack = []

    for i, c in enumerate(s):
        if c == '(':
            pstack.append(i)
        elif c == ')':
            if len(pstack) == 0:
                raise IndexError("No matching closing parens at: " + str(i))
            toret[pstack.pop()] = i

    if len(pstack) > 0:
        raise IndexError("No matching opening parens at: " + str(pstack.pop()))

    return toret


def find_string_index(cond_list, dep_cond, n_entries):
    """Find the index if dep_cond in cond_list.
    In the context of the branch analysis, this is needed for correct handling of conditional probabilities.
    """
    for i in range(n_entries):
        if cond_list[i] == dep_cond:
            return i
    return -1


def replace_int_division(s, paren_pos, div="udiv"):
    """Replace "sdiv" and "udiv" such that Sympy can interpret it as integer division, i.e. using floor().

    :param s: string containing an integer division to replace
    :param div: string to indicate if unsigned or signed integer division is to be replaced
    :param paren_pos: dictionary containing the positions of matching parentheses
    :return res: string with replaced integer division
    """
    if s == div:
        return 'div'
    pos = s.find(div)
    part = s.partition(div)
    lhs = part[0]
    pos_left_par = lhs.rfind("(")

    while paren_pos[pos_left_par] < pos:
        pos_left_par = lhs.rfind("(", 0, pos_left_par)

    pos_right_par = paren_pos[pos_left_par]
    lhs = part[0][pos_left_par + 1:]
    rhs = part[2][1:pos_right_par - pos - len(div)]
    res = s[:pos_left_par] + "floor(" + lhs + "/" + rhs + s[pos_right_par:]  # type: str

    return res


def replace_wrapper(cond):
    """Replace simple operators as well as integer divisions.

    :param cond: string containing verbose operator names
    :return replaced: string with operator names replaced by the corresponding operators
    """
    replaced = replace_operators(cond)
    paren_pos = find_parens(replaced)
    for div in ("sdiv", "udiv"):
        while div in replaced:
            replaced = replace_int_division(replaced, paren_pos, div)
            paren_pos = find_parens(replaced)

    return replaced


def find_bin_op_begin(cond, paren_pos, bin_op):
    """Find the beginning index in the string cond containing a binary operator."""
    assert isinstance(cond, str)
    assert isinstance(bin_op, str)
    pos = cond.find(bin_op)
    op_begin = []
    while pos != -1:
        key_max = 0
        for key in paren_pos:
            if key < pos and key > key_max and paren_pos[key] > pos:
                key_max = key
        op_begin.append(key_max)
        pos = cond.find(bin_op, pos + 2)

    return op_begin


def find_eq_begin(eq, parens_pos, is_equal):
    """Find the beginning indices of all (negated) equations in eq

    :param eq: input string
    :param parens_pos: dictionary of the positions of matching parentheses in eq
    :param is_equal: boolean to switch between equation and negated equation
    :return: list of start indices
    """
    cmp = "==" if is_equal else "~="
    return find_bin_op_begin(eq, parens_pos, cmp)


def replace_logic_and_bit_operators(cond):
    """Replace each instance of a logic or bit shift operator by a dummy function taking two arguments (lhs, rhs)."""
    assert isinstance(cond, str)
    ops = {'& ': 'f_and',
           '| ': 'f_or',
           '^ ': 'f_xor',
           'shl ': 'shl',
           'lshr ': 'lshr'}
    for op in ops:
        pos = cond.find(op)
        while pos != -1:
            paren_pos = find_parens(cond)
            key_max = 0
            for key in paren_pos:
                if key < pos and key > key_max and paren_pos[key] > pos:
                    key_max = key
            end = paren_pos[key_max]
            sub_cond = cond[key_max:end + 1]
            split_sub = sub_cond.split(' ' + op, maxsplit=1)
            cond = cond.replace(sub_cond, ops[op] + split_sub[0] + ',' + split_sub[1])
            # print("repl cond: " + cond)
            pos = cond.find(op, pos+len(op))

    return cond


def split_equation(eq, is_equal):
    """Split string if it is an equation and convert it into a sympifiable expression.

    :param eq: input string
    :param is_equal: boolean to switch between equation and negated equation
    :return: string eq with Eq(lhs, rhs) replacing equations if eq is an equation; else eq
    """
    paren_pos = find_parens(eq)
    eq_begin_list = find_eq_begin(eq, paren_pos, is_equal)
    split_cmp = " == " if is_equal else " ~= "
    sym_func = "Eq" if is_equal else "Ne"

    eq_dict = {}
    for begin in eq_begin_list:
        end = paren_pos[begin]
        sub_eq = eq[begin:end+1]
        split_eq = sub_eq.split(split_cmp)
        eq_dict[sub_eq] = sym_func+split_eq[0]+","+split_eq[1]

    for key in eq_dict:
        eq = eq.replace(key, eq_dict[key])

    return eq


def split_condition(cond_str, sep):
    """Split the condition string according to the separator. Two cases are allowed:
    1) sep=" || ": split the condition and call the function again for the second case,
    2) sep=" && ": split the condition further, replace the operator names and convert equations to Eq()

    :param cond_str: input string
    :param sep: separator specifying where to split the string
    :return: simplified condition string which can be sympified
    """
    splitted = cond_str.split(sep)
    res = []
    if len(splitted) > 1:
        if sep == " || ":
            res.append("Or(")
        elif sep == " && ":
            res.append("And(")

    for s in splitted:
        if sep == " || ":
            s = split_condition(s, " && ")
        elif sep == " && ":
            s = replace_wrapper(s)
            s = split_equation(s, True)
            s = split_equation(s, False)
            s = replace_logic_and_bit_operators(s)

        res.append(s)
        res.append(",")

    del res[-1]    # delete the last "," for sympify
    res_cond = ""
    if len(splitted) > 1:
        res.append(")")
        res_cond = "".join(res)
    else:
        res_cond = res[0]

    return res_cond


def parse_condition(condition: str, path: list = []):
    """Wrapper function to call split_condition.
    Strategy:
        1) use cond.split(sep=" || ") to split whole condition into individual pieces -> connect with Or(arg0, arg1, ...)
        2) use subcond.split(sep=" && ") on the individual pieces -> connect with And(arg0, arg1, ...)
        3) use subsubcond.split(sep=" == ") and sep=" != " to identify (in)equalities
            -> replace by Eq(), Ne() in order to make it parseable with sympy
        4) Finally, get all PHINodes and evaluate them, if a path is specified.

    :param condition: condition to parse
    :param path: optional parameter used for the evaluation of PHINodes
    :return: condition as a sympifiable expression
    """
    assert isinstance(condition, str)
    cond_parsed = split_condition(condition, sep=" || ")
    if path:
        phi_list = find_PHINode(cond_parsed, find_parens(cond_parsed))
        for phi in phi_list:
            phi_eval = phi.evaluate(path)
            cond_parsed = cond_parsed.replace(phi.raw_str, phi_eval)
    cond_parsed = cond_parsed.replace('~0', '-1').replace('~-1', '0').replace('~1', '0').replace('#', ',')
    return cond_parsed


def is_loop(condition):
    """Checks if the argument string is a loop in recurrence form.

    :param condition: string to test
    :return: True if condition is a loop, else False
    """
    # return True if condition[0] == '{' or condition[1] == '{' else False
    return '{' in condition[0:2]


def dependencies(expr: sp.Symbol, symbol_list_: list):
    """Examines dependencies of the Sympy expression expr according to the symbols in symbol_list_

    :param expr: Sympy expression
    :param symbol_list_: list of Sympy symbols
    :return dep_list: bitvector denoting dependencies ('1' for dependence, '0' for independence)
    """
    expr_symbols = expr.free_symbols
    dep_list = [1 if symbol_list_[i] in expr_symbols else 0 for i in range(len(symbol_list_))]
    return dep_list


class Recurrence:
    """Class to hold information about loop recurrences.
    Note: This class may replace LoopString (at least in parts) in the future."""
    def __init__(self, rec_str: str, path: list = []):
        self.raw_str = rec_str
        self.is_simple = '},' not in self.raw_str
        first_comma = self.raw_str.find(',')
        second_comma = self.raw_str.find(',', first_comma+1)
        bracket_close = self.raw_str.find('}')
        self.start = parse_condition(self.raw_str[1:first_comma], path)
        self.step = parse_condition(self.raw_str[second_comma + 1:bracket_close], path)
        self.inst = parse_condition(self.raw_str[first_comma + 1:second_comma], path)
        self.end = ""
        self.guard = ""
        if not self.is_simple:
            third_comma = self.raw_str.find(',', second_comma + 1)
            last_comma = self.raw_str.rfind(',')
            self.end = parse_condition(self.raw_str[third_comma + 1:last_comma], path)
            self.guard = str(self.raw_str[last_comma + 1:])

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.raw_str)

    def evaluate_PHINodes(self, path: list):
        """If 'path' was not specified when constructing the object, PHINodes are not evaluated yet.
        This is remedied using this function."""
        if 'PHI' in self.raw_str:
            start_eval = parse_condition(self.start, path)
            step_eval = parse_condition(self.step, path)
            inst_eval = parse_condition(self.inst, path)
            end_eval = parse_condition(self.end, path)
        else:
            start_eval = self.start
            step_eval = self.step
            inst_eval = self.inst
            end_eval = self.end
        return start_eval, step_eval, inst_eval, end_eval


class PHINode:

    def __init__(self, phi_str: str):
        self.raw_str = phi_str
        self.block_value_dict = {}
        pos_phi = self.raw_str.find('PHI')
        self.name = self.raw_str[:pos_phi]
        self.is_specified = '$(' in self.raw_str
        self.block_id = None

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.name)

    def __eq__(self, other):
        return self.raw_str == other.raw_str

    def parse(self):
        if not self.is_specified:
            print('Cannot deduce more information on PHINode', self.name)
            return
        pos_id_end = self.raw_str.find('$')
        self.block_id = int(self.raw_str[len(self.name)+4:pos_id_end])
        paren_pos = find_parens(self.raw_str)
        del paren_pos[len(self.name)+3]    # delete pair of outermost parentheses since all delimiters are in between
        delim_pos = []
        pos = 0
        while pos != -1:
            pos = self.raw_str.find(':', pos+1)
            delim_pos.append(pos)
        # TODO: what about nested PHINodes?
        for left, right in paren_pos.items():
            for delim in delim_pos:
                if left < delim < right:
                    block = int(self.raw_str[left+1:delim])
                    value = self.raw_str[delim+1:right]
                    self.block_value_dict[block] = value

    def evaluate(self, path: list):
        if not self.block_id in path:
            raise KeyError("PHINode is not in path.")
        index_id = path.index(self.block_id)
        from_block = path[index_id-1]
        return self.block_value_dict[from_block]


def find_PHINode(raw_str: str, paren_pos: dict):
    phi_pos = 0
    all_phis = []
    while phi_pos != -1:
        phi_pos = raw_str.find('PHI', phi_pos+1)
        if phi_pos != -1 and phi_pos+3 in paren_pos.keys():
            name_start = get_alnum_start_position(raw_str, phi_pos, -1)
            phi = PHINode(raw_str[name_start:paren_pos[phi_pos+3]+1])
            phi.parse()
            all_phis.append(phi)
    return all_phis


def get_alnum_start_position(raw_str: str, pos: int, direction: int = 1):
    if direction > 0:
        step = 1
    elif direction < 0:
        step = -1
    else:
        raise ValueError("direction must be either larger or smaller than zero.")

    pos_iter = pos
    while raw_str[pos_iter].isalnum():
        pos_iter += step

    return pos_iter - step


class LoopString:
    """Class to store relevant loop properties. It also computes and holds the tripcount.

    """
    def __init__(self, loop_str: str, path: list = []):
        self.raw_loop = loop_str
        first_comma = self.raw_loop.find(",")
        second_comma = self.raw_loop.find(",", first_comma + 1)
        third_comma = self.raw_loop.find(",", second_comma + 1)
        last_comma = self.raw_loop.rfind(",")

        self.start = parse_condition(self.raw_loop[1:first_comma], path)
        self.end   = parse_condition(self.raw_loop[third_comma+1:last_comma], path)
        self.step  = parse_condition(self.raw_loop[second_comma+1:third_comma-1], path)
        self.inst  = parse_condition(self.raw_loop[first_comma+1:second_comma], path)
        self.guard = str(self.raw_loop[last_comma+1:])

        self.is_guarded = True if self.guard == '1' else False
        if self.inst == "sdiv" or self.inst == "udiv":
            self.inst = "div"

        self.tripcount = ""
        if self.inst == "add" or self.inst == "sub":
            assert self.step != "0"
            self.tripcount = "ceiling(Abs((" + self.end + "-" + self.start + ")/" + self.step + "))" + "-" + self.guard
        elif self.inst == "mul" or self.inst == "shl":
            assert self.start != "0"
            base = self.step if self.inst == "mul" else "2**"+self.step
            self.tripcount = "ceiling(log(" + self.end + "/" + self.start + "," + base + "))" + "-" + self.guard
        elif self.inst == "div" or self.inst == "lshr":
            assert self.start != "0"
            base = self.step if self.inst == "div" else "(2**" + self.step + ')'
            self.tripcount = "floor(log(" + self.start + "/" + "(" + self.end + "+1)" + "," + base + ") +1)" \
                             + "-" + self.guard

        else:
            raise NotImplementedError("Increment operation not implemented.")


class LoopInfo:
    def __init__(self, loop_str):
        neg_flag = loop_str[0] == "!"
        if neg_flag:
            self.loop = LoopString(loop_str[1:])
            self.freq = "1/("+self.loop.tripcount+"+1)"
        else:
            self.loop = LoopString(loop_str)
            self.freq = self.loop.tripcount+"/(" + self.loop.tripcount + "+1)"
