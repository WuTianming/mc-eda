from typing import Dict, List, Tuple
from enum import Enum

class LaneOp(Enum):
    INVALID = 0
    INPUT = 1
    NAND = 2
    HOLD = 3
    DELAY_LINE = 4

class Lane:
    def __init__(self):
        self.lane_id = None
        self.lane_op = LaneOp.INVALID
        self.lane_operands = []

        self.is_output = False
        self.output_name = None
        self.input_which_bit = None

        self.is_input = False
        self.input_name = None
        self.input_which_bit = None

    def __init__(self, lane_id, lane_op, lane_operands, output_name=None, output_which_bit=None):
        if lane_op == LaneOp.INPUT:
            self.is_input = True
            self.input_name = lane_operands[0]
            self.input_which_bit = lane_operands[1]
        else:
            self.is_input = False
            self.input_name = None
            self.input_which_bit = None

        self.lane_id = lane_id
        self.lane_op = lane_op
        self.lane_operands = lane_operands

        if output_name is not None:
            self.is_output = True
            self.output_name = output_name
            self.output_which_bit = output_which_bit
        else:
            self.is_output = False
            self.output_name = None
            self.output_which_bit = None
    def __repr__(self):
        return self.__str__()
    def __str__(self):
        # very succinct, qasm style
        ret = ""
        if self.is_input:
            ret = f"{self.lane_op.name} {self.input_name}[{self.input_which_bit}]"
        else:
            ret = f"{self.lane_op.name} " + str(self.lane_operands)
        if self.is_output:
            ret += f" --> {self.output_name}[{self.output_which_bit}]"
        return ret

class Layer:
    def __init__(self):
        self.layer_id = None
        self.lanes: Dict[int, Lane] = {}
    # def __repr__(self):
    #     return "Layer ID: " + str(self.layer_id) + ", Lanes: " + str(self.lanes)
    def __str__(self):
        return dict(sorted(list(self.lanes.items()))).__str__()
    def __getitem__(self, lane_id) -> Lane:
        return self.lanes[lane_id]
    def __setitem__(self, lane_id, lane):
        self.lanes[lane_id] = lane
    def __len__(self):
        return len(self.lanes)
    def __iter__(self):
        # for iter: always iterate keys in order
        return iter(sorted(self.lanes.keys()))
    def items(self):
        return list(sorted(self.lanes.items()))
    def __contains__(self, lane_id):
        return lane_id in self.lanes

class FloorPlan:
    def __init__(self):
        self.layers: List[Layer] = []
    def __getitem__(self, layer_id) -> Layer:
        return self.layers[layer_id]
    def __setitem__(self, layer_id, layer):
        self.layers[layer_id] = layer
    def __len__(self):
        return len(self.layers)
    def __iter__(self):
        return iter(self.layers)
    def print_stats(self):
        n_layers = len(self.layers)
        n_in, n_out = len(self[0]), len(self[-1])
        tot_gates = sum([len(f) for f in self])
        # max/min/avg circuit width
        max_w, min_w, avg_w = max([len(f) for f in self]), min([len(f) for f in self]), tot_gates/n_layers
        # total and average of "hold" gates
        tot_hold, avg_hold = 0,0
        for floor in self:
            tot_hold += len([g for g in floor if floor[g].lane_op == LaneOp.HOLD])
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
