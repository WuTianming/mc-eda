from typing import Tuple
import pyrtl
from pyrtl import Register, Const, Input, Output, WireVector, LogicNet, concat_list, concat
import networkx as nx

def one_bit_add(a, b, carry_in):
    assert len(a) == len(b) == 1  # len returns the bitwidth
    sum = a ^ b ^ carry_in  # operators on WireVectors build the hardware
    carry_out = a & b | a & carry_in | b & carry_in
    return sum, carry_out

def ripple_add(a, b, carry_in=0):
    a, b = pyrtl.match_bitwidth(a, b)
    if len(a) == 1:
        sumbits, carry_out = one_bit_add(a, b, carry_in)
    else:
        lsbit, ripplecarry = one_bit_add(a[0], b[0], carry_in)
        msbits, carry_out = ripple_add(a[1:], b[1:], ripplecarry)
        sumbits = pyrtl.concat(msbits, lsbit)
    return sumbits, carry_out

def demo_adder(BIT=3):
    # instantiate an adder into a 3-bit counter
    input_A = Input(bitwidth=BIT, name='input_A')
    input_B = Input(bitwidth=BIT, name='input_B')
    final_out = Output(bitwidth=BIT+1, name='final_out')
    # sum, carry_out = ripple_add(input_A, Const("3'b1", name='constant'))
    sum, carry_out = ripple_add(input_A, input_B)

    # final_out <<= input_A + input_B
    final_out <<= concat(sum, carry_out)

def demo_adder_resetter(BIT=3):
    # instantiate an adder into a 3-bit counter
    reset_hi = Input(bitwidth=1, name='reset_hi')
    reset_lo = Input(bitwidth=1, name='reset_lo')
    input_A = Input(bitwidth=BIT, name='input_A')
    input_B = Input(bitwidth=BIT, name='input_B')
    all_0   = Const(0, bitwidth=BIT+1)
    all_1   = ~all_0
    sum, carry_out = ripple_add(input_A, input_B)
    final_out = Output(bitwidth=BIT+1, name='final_out')
    final_out <<= pyrtl.select(reset_hi, all_1, pyrtl.select(reset_lo, all_0, concat(carry_out, sum)))

def demo_max(BIT=2):
    from functools import reduce
    def max_n(*inputs):
        def max_2(x,y):
            return pyrtl.select(x>y, x, y)
        return reduce(max_2, inputs)
    input_A = Input(bitwidth=BIT, name='input_A')
    input_B = Input(bitwidth=BIT, name='input_B')
    input_C = Input(bitwidth=BIT, name='input_C')
    final_out = Output(bitwidth=BIT, name='final_out')
    final_out <<= max_n(input_A, input_B, input_C)

def demo_ALU(n_bits=4):
    # Define 8-bit input wires
    A = pyrtl.Input(n_bits, '1_A')
    B = pyrtl.Input(n_bits, '2_B')
    ALU_Sel = pyrtl.Input(4, '0_ALU_Sel')

    # Define 8-bit output wire and carry-out flag
    ALU_Out = pyrtl.Output(n_bits, 'ALU_Out')
    CarryOut = pyrtl.Output(1, 'CarryOut')

    sum_result = pyrtl.WireVector(n_bits+1, 'sum_result')
    sum_result <<= A.zero_extended(n_bits+1) + B.zero_extended(n_bits+1)
    CarryOut <<= sum_result[n_bits]

    alu_result = pyrtl.WireVector(n_bits, 'alu_result')
    with pyrtl.conditional_assignment:
        with ALU_Sel == 0b0000:
            alu_result |= A + B  # Addition
        with ALU_Sel == 0b0001:
            alu_result |= A - B  # Subtraction
        with ALU_Sel == 0b0010:
            alu_result |= A * B  # Multiplication
        # with ALU_Sel == 0b0011:
        #     alu_result |= A // B  # Division
        # with ALU_Sel == 0b0100:
        #     alu_result |= pyrtl.shift_left_logical(A, 1)  # Logical shift left
        # with ALU_Sel == 0b0101:
        #     alu_result |= pyrtl.shift_right_logical(A, 1)  # Logical shift right
        # with ALU_Sel == 0b0110:
        #     alu_result |= pyrtl.concat(A[6:0], A[7])  # Rotate left
        # with ALU_Sel == 0b0111:
        #     alu_result |= pyrtl.concat(A[0], A[7:1])  # Rotate right
        # with ALU_Sel == 0b1000:
        #     alu_result |= A & B  # Logical AND
        # with ALU_Sel == 0b1001:
        #     alu_result |= A | B  # Logical OR
        # with ALU_Sel == 0b1010:
        #     alu_result |= A ^ B  # Logical XOR
        # with ALU_Sel == 0b1011:
        #     alu_result |= ~(A | B)  # Logical NOR
        # with ALU_Sel == 0b1100:
        #     alu_result |= ~(A & B)  # Logical NAND
        # with ALU_Sel == 0b1101:
        #     alu_result |= ~(A ^ B)  # Logical XNOR
        # with ALU_Sel == 0b1110:
        #     alu_result |= pyrtl.select(A>B, 1, 0)  # Greater comparison
        # with ALU_Sel == 0b1111:
        #     alu_result |= pyrtl.select(A==B, 1, 0)  # Equal comparison
        with pyrtl.otherwise:
            alu_result |= A + B  # Default case

    ALU_Out <<= alu_result


demo_adder(BIT=8)

# demo_adder()
# demo_max(BIT=1)
# demo_ALU(n_bits=8)

# input_A = Input(bitwidth=1, name='input_A')
# input_B = Input(bitwidth=1, name='input_B')
# final_out = Output(bitwidth=1, name='final_out')
# final_out <<= input_A.nand(input_B)

# synthesize into NAND gates
pyrtl.optimize()  # pre-synth optimize doesn't do much, but helped in the demo_adder(BIT=1)
pyrtl.synthesize()
pyrtl.optimize()
pyrtl.nand_synth()
pyrtl.optimize()
# pyrtl.common_subexp_elimination(pyrtl.working_block())
# pyrtl.constant_propagation(pyrtl.working_block())

with open("adder.dot", "w") as f:
    f.write(pyrtl.block_to_graphviz_string())

def namer(thing, is_edge=True):
    """ Returns a "good" string for thing in printed graphs. """
    if is_edge:
        if thing.name is None or thing.name.startswith('tmp'):
            return ''
        else:
            return '/'.join([thing.name, str(len(thing))])
    elif isinstance(thing, Const):
        return str(thing.val)
    elif isinstance(thing, WireVector):
        return thing.name or '??'
    elif isinstance(thing, LogicNet):
        return thing.op + str(thing.op_param or '')
    else:
        # print('@@', type(thing))
        return thing.op + str(thing.op_param or '')

asm = { 'n': 'NAND', '~': 'NOT', 'w': 'WIRE', 'c': 'CONCAT', 's': 'SUBSET' }

def print_circuit(gates, idmap):
    for g0 in gates:
        args_print = []
        for a in gates[g0]:
            if isinstance(a, str):
                args_print.append(f'str=\"{a}\"')
            elif isinstance(a, int):
                args_print.append(f'int={a}')
            elif isinstance(a, Input):
                args_print.append(str(a))
            elif isinstance(a, Tuple) and isinstance(a[0], Input):
                args_print.append((str(a[0]), a[1]))
            else:
                # if (idmap[a] in [22, 28, 9, 21]):
                #     print(a, idmap[a], '@@')
                args_print.append(idmap[a])

        if isinstance(g0, LogicNet):
            # print(asm[g0.op], idmap[g0], '<--', list(idmap[a] for a in gates[g0]))
            print(asm[g0.op], idmap[g0], '<--', args_print)
        elif isinstance(g0, Output):
            print('OUTPUT', idmap[g0], '<--', args_print)

def get_nand_circuit_0(block=None):
    g = pyrtl.net_graph(block)      # edges = graph[source][dest]

    # See core.py:244

    gates = {}      # entry looks like dst(LogicNet): [operand_1, operand_2]
    inputs = set()
    consts = set()
    outputs = {}    # dst(LogicNet): [the gate producing the result, or concat]
    # SEE visualization.py:214
    # 'n' means NAND
    # '~' means Invert
    # 'w' means wire
    # 'c' means Concat
    # 's' means Subset.
    #     node.op_param is a tuple of the selected bits to pull from the argument wire,
    #     so it could look something like (0,0,0,0,0,0,0), meaning dest wire is going
    #     to be a concatenation of the zero-th bit of the argument wire, 7 times.

    wire_to_src = {}

    for src in g:
        for dst in g[src]:
            for edge in g[src][dst]:
                # print(type(edge), type(src), type(dst))
                if isinstance(src, Input):
                    inputs.add(src)
                elif isinstance(src, Const):
                    consts.add(src)

                if isinstance(dst, LogicNet):
                    if dst not in gates:
                        gates[dst] = []
                    gates[dst].append(src)  # NOTE: how do we make sure CONCAT is in order?
                    if dst.op == 'c':
                        wire_to_src[edge] = src
                    #     print([str(u) for u in dst.args])
                elif isinstance(dst, Output):
                    if dst not in outputs:
                        outputs[dst] = [src]
                    else:
                        raise ValueError()
                else:
                    raise ValueError()

    # reorder the CONCATs
    for g0 in gates:
        if g0.op == 'c':
            ordered_concat = []
            for arg in g0.args:         # WireVector
                argx = wire_to_src[arg] # LogicNet
                ordered_concat.append(argx)
            gates[g0] = ordered_concat

    inc = 0
    idmap = {}  # map node -> index
    def mapper(u):
        nonlocal inc
        if u not in idmap:
            idmap[u] = inc
            inc += 1
        return idmap[u]

    for g0 in inputs:
        mapper(g0)
    for g0 in consts:
        mapper(g0)
    for g0 in outputs:
        mapper(g0)              # <- this is just a label for the output
        mapper(outputs[g0][0])  # <- this is the actual output
    for g0 in gates:
        mapper(g0)
        for arg in gates[g0]:   # LogicNet, wire.Const, wire.Input
            mapper(arg)

    # NOTE: This is already implemented in passes.py :: _remove_wire_nets()
    # After that pass, the only 'w' are those directly connected to the output.
    # NOTE again: 'w' to output is also dealt with in passes.py :: direct_connect_outputs()
    # DISAMBIGUATION: _remove_wire_nets() doesn't remove wirevectors.
    wire_reduce = {}

    for g0 in gates:
        if isinstance(g0, LogicNet):
            if g0.op == 'w':
                prev = gates[g0][0]
                while prev.op == 'w':
                    prev = gates[prev][0]
                wire_reduce[g0] = prev
    for u,v in wire_reduce.items():
        gates.pop(u)
    for g0 in gates:
        gates[g0] = [(wire_reduce.get(u, u)) for u in gates[g0]]
    for g0 in outputs:
        # print([u.op for u in outputs[g0]])
        outputs[g0] = [(wire_reduce.get(u, u)) for u in outputs[g0]]
        # print([u.op for u in outputs[g0]])

    print_circuit(gates, idmap)
    for g0 in outputs:
        print('OUTPUT', idmap[g0], '<--', list(idmap[a] for a in outputs[g0]))
    for g0 in inputs:
        print("INPUT", idmap[g0])
    for g0 in consts:
        print("CONST", idmap[g0], '=', g0.val)
    return idmap, inputs, outputs, consts, gates

def get_nand_circuit_1(block=None):
    idmap, inputs, outputs, consts, gates = get_nand_circuit_0(block)
    print("@@ Outputs ", outputs)
    print("@@\n\n")

    # 1. break multi-bit items into separate bits (remove concat/subset)
    # 2. merge Chain NANDs into multi-input NAND
    # 3. reverse all input/output/consts to generate the dual graph
    # 4. inter-layer planning
    # This function implements only 1.

    # i'm supposed to do a toposort but im lazy
    complete_set = set()

    while True:
        # print("")

        outer_complete = True
        while True:
            has_update = False
            for g0 in gates.keys():
                if g0.op == 'c' and g0 not in complete_set:    # CONCAT
                    outer_complete = False
                    full_list = []
                    success = True
                    for arg in gates[g0]:
                        if isinstance(arg, LogicNet):
                            if arg.op == 'c' and arg not in complete_set:
                                # print(g0, "11 pending unexpanded", arg)
                                success = False
                                break   # cannot expand CONCAT, if preceding CONCAT isn't expanded
                            if arg.op == 's' and arg not in complete_set:
                                # print(g0, "12 pending unexpanded", arg)
                                success = False
                                break   # cannot expand CONCAT, if preceding CONCAT isn't expanded
                    if not success:
                        continue
                    for arg in gates[g0]:
                        if isinstance(arg, LogicNet):
                            if arg.op == 'c':
                                for arg1 in gates[arg]:
                                    full_list.append(arg1)
                            elif arg.op == 's':
                                for arg1 in gates[arg]:
                                    full_list.append(arg1)
                            else:
                                full_list.append(arg)
                        elif isinstance(arg, Input):
                            for bit in range(arg.bitwidth):
                                full_list.append((arg, bit))
                        elif isinstance(arg, Const):
                            for bit in range(arg.bitwidth):
                                bit_v = (1 if (arg.val & (1 << bit)) else 0)
                                full_list.append(bit_v)
                    gates[g0] = full_list
                    print("Updated C gate {} to {}".format(g0, [str(u) for u in gates[g0]]))
                    has_update = True
                    complete_set.add(g0)
            if not has_update:
                break

        # print("")

        # now that we've expanded all 'CONCAT' instructions, we can just go ahead and take the SUBSETs
        while True:
            has_update = False
            for g0 in gates.keys():
                if g0.op == 's' and g0 not in complete_set:    # SUBSET
                    outer_complete = False
                    full_list = []
                    assert len(gates[g0]) == 1
                    arg = gates[g0][0]
                    if isinstance(arg, LogicNet):
                        if arg.op == 'c':
                            if arg not in complete_set:
                                # print(g0, "21 pending unexpanded", arg)
                                continue
                            for arg1 in gates[arg]:
                                full_list.append(arg1)  # here, arg1 may be str or int due to previous expansion
                        elif arg.op == 's':
                            if arg not in complete_set:
                                # print(g0, "22 pending unexpanded", arg)
                                continue
                            for arg1 in gates[arg]:
                                full_list.append(arg1)
                        else:
                            raise ValueError()
                    elif isinstance(arg, Input):
                        for bit in range(arg.bitwidth):
                            full_list.append((arg, bit))
                    elif isinstance(arg, Const):
                        for bit in range(arg.bitwidth):
                            bit_v = (1 if (arg.val & (1 << bit)) else 0)
                            full_list.append(bit_v)
                    else:
                        raise ValueError()

                    select = g0.op_param
                    select_list = []
                    for v in select:
                        select_list.append(full_list[v])
                    gates[g0] = select_list
                    print("Updated S gate {} to {}".format(g0, [str(u) for u in gates[g0]]))
                    has_update = True
                    complete_set.add(g0)
            if not has_update:
                break

        if outer_complete:
            break

    # expand the list into output
    for wire in outputs:
        full_list = []
        for arg in outputs[wire]:
            if isinstance(arg, LogicNet):
                if arg.op == 'c':
                    for arg1 in gates[arg]:
                        full_list.append(arg1)
                elif arg.op == 's':
                    full_list += gates[arg]
                else:
                    full_list.append(arg)
            elif isinstance(arg, Input):
                for bit in range(arg.bitwidth):
                    full_list.append((arg, bit))
            elif isinstance(arg, Const):
                for bit in range(arg.bitwidth):
                    bit_v = (1 if (arg.val & (1 << bit)) else 0)
                    full_list.append(bit_v)
            else:
                print('CHECK:', arg)
        outputs[wire] = full_list
        print("Updated output {} to {}".format(wire, [str(u) for u in outputs[wire]]))

    # expand the list into receiving gates
    for g0 in gates:
        if g0.op == 's' or g0.op == 'c':
            continue
        full_list = []
        for arg in gates[g0]:
            if isinstance(arg, LogicNet):
                if arg.op == 'c':
                    # for arg1 in gates[arg]:
                    #     full_list.append(arg1)
                    raise ValueError()
                elif arg.op == 's':
                    full_list += gates[arg]
                else:
                    full_list.append(arg)
            elif isinstance(arg, Input):
                for bit in range(arg.bitwidth):
                    full_list.append((arg, bit))
            elif isinstance(arg, Const):
                for bit in range(arg.bitwidth):
                    bit_v = (1 if (arg.val & (1 << bit)) else 0)
                    full_list.append(bit_v)
            else:
                print('CHECK:', arg)
        gates[g0] = full_list

    # now, just like how wires are removed, remove the SUBSETs and CONCATs
    # keep CONCATs that are connected to the output because we didn't deal with output yet

    to_remove = []
    for g0 in gates:
        if g0.op == 's':
            to_remove.append(g0)
        if g0.op == 'c':
            to_remove.append(g0)
    for r in to_remove:
        gates.pop(r)

    print()
    print()
    print()

    # netlist = []
    # netlist_inputs = [] # inputs are put in the front

    # # toposort
    # queue = []
    # visited = set()
    # for g in outputs:
    #     for arg in outputs[g]:
    #         queue.append(arg)
    #         visited.add(arg)
    # while len(queue) > 0:
    #     g = queue[0]
    #     queue = queue[1:]
    #     if isinstance(g, Tuple) and isinstance(g[0], Input):
    #         netlist_inputs.insert(0, g)
    #         print("Input", g, type(g))
    #     elif isinstance(g, int):
    #         netlist.insert(0, g)
    #         print("Const", g)
    #     else:
    #         netlist.insert(0, g)
    #         for arg in gates[g]:
    #             if arg not in visited:
    #                 queue.append(arg)
    #                 visited.add(arg)
    # netlist = netlist_inputs + netlist

    # inc = 0
    # idmap = {}  # map node -> index
    # def mapper(u):
    #     nonlocal inc
    #     if u not in idmap:
    #         idmap[u] = inc
    #         inc += 1
    #     return idmap[u]
    # pass
    # for g in netlist:
    #     mapper(g)
    # for g in outputs:
    #     mapper(g)

    print_circuit(gates, idmap)
    print_circuit(outputs, idmap)

    # return netlist, gates, outputs, idmap
    return gates, outputs, idmap

def floor_planning(block=None, max_width=32):
    # 2. merge Chain NANDs into multi-input NAND
    # 3. reverse all input/output/consts to generate the dual graph
    # 4. inter-layer planning
    gates, outputs, idmap = get_nand_circuit_1(block)
    remain = list(gates.keys())

    cur_layer = set()   # [ a, b, c, ... ]
    dependents = {}     # { a: [b, c, ...] }
    to_consume = {}     # { a: [b, c, ...] }
    cur_layer_id = 0
    layer_ids = {}      # the layer ID when earliest added

    cur_floor = dict()
    # floor_0 = { id_0 : set(NOR input ids on last layer) --OR-- "hold", meaning hold the signal }
    floor_plan = []

    for g in gates:
        for arg in gates[g]:
            if arg not in dependents:
                dependents[arg] = set()
            dependents[arg].add(g)
    for g in outputs:
        for arg in outputs[g]:
            if arg not in dependents:
                dependents[arg] = set()
            dependents[arg].add(g)
    for g in gates:
        if g not in dependents:
            print(f"Warning: {g} has no dependents")
            exit(1)

    # place all inputs and constants
    for g in gates:
        for arg in gates[g]:
            if isinstance(arg, Tuple) and isinstance(arg[0], Input):
                if arg not in cur_layer:
                    layer_ids[arg] = cur_layer_id
                    cur_layer.add(arg)
                    cur_floor[arg] = 'input'
                    to_consume[arg] = dependents[arg]
            elif isinstance(arg, int):
                if arg not in cur_layer:
                    layer_ids[arg] = cur_layer_id
                    cur_layer.add(arg)
                    cur_floor[arg] = 'const'
                    to_consume[arg] = dependents[arg]

    # fresh start at a new layer, but carry over all initial wires.
    cur_layer_id += 1
    floor_plan.append(cur_floor.copy())
    for k in cur_floor:
        cur_floor[k] = 'hold'

    while len(remain) > 0:
        print("Remaining", len(remain))
        print("len(cur_layer) =", len(cur_layer))
        found = []
        for idx, g in enumerate(remain):    # first try to find one that consumes a signal
            success = True
            for arg in gates[g]:
                if arg not in cur_layer:
                    success = False
                    break
                if layer_ids[arg] == cur_layer_id:
                    success = False
                    break
            if success:
                n_consume = 0
                for old_wire in list(to_consume.keys()):
                    if g in to_consume[old_wire] and len(to_consume[old_wire]) == 1:
                        n_consume += 1
                # lower the priority of output wires, because they will persist
                if len(dependents[g]) == 1 and isinstance(list(dependents[g])[0], Output):
                    n_consume += (-1)
                found.append((n_consume, g))
                # print(f"found {g} consuming {n_consume} wires")

        if len(found) > 0:
            found.sort(key=lambda tup: -tup[0])
            g = found[0][1]
            print(f"Inserting {g}, which should consume {found[0][0]} old wires")
            for old_wire in list(to_consume.keys()):
                if g in to_consume[old_wire]:
                    to_consume[old_wire].remove(g)
                    if len(to_consume[old_wire]) == 0:
                        print(f"{old_wire} is CONSUMED!")
                        cur_layer.remove(old_wire)
                        to_consume.pop(old_wire)
                        cur_floor.pop(old_wire)
            for arg in gates[g]:
                print(f"  Arg: {arg}; lifting from layer {layer_ids[arg]} --> {cur_layer_id}")

            cur_layer.add(g)
            cur_floor[g] = set(gates[g])
            layer_ids[g] = cur_layer_id
            to_consume[g] = dependents[g].copy()
            remain.remove(g)
            print(f"putting {g} on layer {cur_layer_id}\n")
            if len(remain) > 0 and len(cur_layer) < max_width:
                continue

        print("Saving current layer into floor_plan...")
        # floor_0 = { id_0 : set(NOR input ids on last layer) --OR-- "hold", meaning hold the signal }
        floor_plan.append(cur_floor.copy())
        for k in cur_floor:
            cur_floor[k] = 'hold'

        print(f" >>> Incrementing layer count to {cur_layer_id + 1}\n")
        cur_layer_id += 1

    # print(to_consume)
    # print(floor_plan)
    # print(cur_floor)
    floor_lanes = []
    def sort_custom(floor, idx):
        value = list(floor.keys())
        def key_func(k):
            gate = floor[k] # either str or set()
            if gate == 'input':
                # print("9 input " + str(k))
                return "9 input " + str(k)
            else:
                def indexof(net, netlist):
                    for idx, n2 in enumerate(netlist):
                        if n2 is net:
                            return idx
                    return -1
                for o in outputs:
                    if k in set(outputs[o]):
                        return "1 output " + str(o) + f" [{indexof(k, outputs[o])}]"
                return "3 gate " + str(k)
        return sorted(value, key=key_func, reverse=(idx % 2 == 1))
    for idx,floor in enumerate(floor_plan):
        lanes = dict()
        inc = 0
        if idx == len(floor_plan) - 1:
            idx = 0
        for g in sort_custom(floor, idx):
            lanes[g] = inc
            inc += 1
        floor_lanes.append(lanes)
    floor_plan_in_num = []
    for idx,floor in enumerate(floor_plan):
        floor_in_num = dict()
        for g in floor:
            g_num = floor_lanes[idx][g]
            # print(type(floor[g]))
            if floor[g] == 'input':
                floor_in_num[g_num] = [str(g[0]), g[1]]
            elif floor[g] == 'const':
                floor_in_num[g_num] = ["const", g.val]
            elif floor[g] == 'hold':
                floor_in_num[g_num] = ["hold", floor_lanes[idx-1][g], ]
            else:
                floor_in_num[g_num] = []
                for gg in floor[g]:
                    floor_in_num[g_num].append(floor_lanes[idx-1][gg])
            for o in outputs:
                def indexof(net, netlist):
                    for idx, n2 in enumerate(netlist):
                        if n2 is net:
                            return idx
                    return -1
                if g in set(outputs[o]):
                    floor_in_num[g_num].append("Output: " + str(o) + f" [{indexof(g, outputs[o])}]")
                    break
        floor_plan_in_num.append(floor_in_num)

    return floor_lanes, floor_plan_in_num

def floor_plan_stats(floor_plan):
    n_layers = len(floor_plan)
    n_in, n_out = len(floor_plan[0]), len(floor_plan[-1])
    tot_gates = sum([len(f) for f in floor_plan])
    # max/min/avg circuit width
    max_w, min_w, avg_w = max([len(f) for f in floor_plan]), min([len(f) for f in floor_plan]), tot_gates/n_layers
    # total and average of "hold" gates
    tot_hold, avg_hold = 0,0
    for floor in floor_plan:
        tot_hold += len([g for g in floor if floor[g][0] == 'hold'])
    avg_hold = tot_hold / n_layers

    print()
    print(" ======== FLOOR PLAN STATISTICS ========")
    print(f"   Number of layers       : {n_layers}")
    print(f"   Input width            : {n_in}")
    print(f"   Output width           : {n_out}")
    print(f"   Total gates            : {tot_gates}")
    print(f"   Max circuit width      : {max_w}")
    print(f"   Min circuit width      : {min_w}")
    print(f"   Avg circuit width      : {avg_w:.2f}")
    print(f"   Total 'hold' gates     : {tot_hold}")
    print(f"   Avg 'hold' gates/layer : {avg_hold:.2f}")
    print(f"   hold / total gates     : {tot_hold / tot_gates:.2f}")
    print(" ======================================")
    print()

def nand_block_to_torch_tensor(block=None, max_width=16):
    # (done) when a NOT gate is only feeding other NOT gates, the subtree can be removed. make this a synthesizing pass
    floor_lanes, floor_plan_in_num = floor_planning(block, max_width=max_width)
    print("\n\n == FLOOR PLAN ==")
    for floor in floor_plan_in_num:
        print(floor)
    floor_plan_stats(floor_plan_in_num)

    import pickle
    with open("max.pkl", "wb") as f:
        pickle.dump(floor_plan_in_num, f)

# nand_block_to_torch_tensor(max_width=32)
nand_block_to_torch_tensor(max_width=150)