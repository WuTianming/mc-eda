import amulet
from amulet import Block, StringTag
from amulet.api.chunk import Chunk
ST = StringTag
from amulet.api.errors import ChunkLoadError, ChunkDoesNotExist

from data_structures import LaneOp, Lane, Layer, FloorPlan

# prefix_path = "/Users/wtm/playground/minecraft/.minecraft/saves/PythonGenerated-2"
prefix_path = "/Users/wtm/playground/minecraft/.minecraft/versions/1.18.2-Fabric/saves/PythonGenerated-2"
level = amulet.load_level(prefix_path)

"""
     N (-Z)
W (-X)    E (+X)
     S (+Z)
"""

def Blk(name, *args):
    return Block("universal_minecraft", name, *args)

air = Blk("air")
diamond = Blk("diamond_block")
gold = Blk("gold_block")
# glass = Blk("glass")
glass = Blk("stained_glass")
grass = Blk("grass_block")
piston_down = Blk("piston", {"extended": ST("false"), "facing": ST("down")})
spiston_down = Blk("sticky_piston", {"extended": ST("false"), "facing": ST("down")})
stone = Blk("stone")
wire = {
    "-": Blk("redstone_wire", {'east': ST("side"), 'north': ST("none"), 'south': ST("none"), 'west': ST("side"), 'power': ST("0")}),
    "|": Blk("redstone_wire", {'east': ST("none"), 'north': ST("side"), 'south': ST("side"), 'west': ST("none"), 'power': ST("0")}),
    "+": Blk("redstone_wire", {'east': ST("side"), 'north': ST("side"), 'south': ST("side"), 'west': ST("side"), 'power': ST("0")}),
    "ne": Blk("redstone_wire", {'east': ST("side"), 'north': ST("side"), 'south': ST("none"), 'west': ST("none"), 'power': ST("0")}),
    "nw": Blk("redstone_wire", {'east': ST("none"), 'north': ST("side"), 'south': ST("none"), 'west': ST("side"), 'power': ST("0")}),
    "se": Blk("redstone_wire", {'east': ST("side"), 'north': ST("none"), 'south': ST("side"), 'west': ST("none"), 'power': ST("0")}),
    "sw": Blk("redstone_wire", {'east': ST("none"), 'north': ST("none"), 'south': ST("side"), 'west': ST("side"), 'power': ST("0")}),
}
repeater = {
    "w": Blk("repeater", {'delay': ST("1"), 'facing': ST("west"),  'locked': ST("false"), 'powered': ST("false")}),
    "e": Blk("repeater", {'delay': ST("1"), 'facing': ST("east"),  'locked': ST("false"), 'powered': ST("false")}),
    "n": Blk("repeater", {'delay': ST("1"), 'facing': ST("north"), 'locked': ST("false"), 'powered': ST("false")}),
    "s": Blk("repeater", {'delay': ST("1"), 'facing': ST("south"), 'locked': ST("false"), 'powered': ST("false")}),
}
comparator = {
    "w": Blk("comparator", {'facing': ST("west"),  'mode': ST("compare"), 'powered': ST("false")}),
    "e": Blk("comparator", {'facing': ST("east"),  'mode': ST("compare"), 'powered': ST("false")}),
    "n": Blk("comparator", {'facing': ST("north"), 'mode': ST("compare"), 'powered': ST("false")}),
    "s": Blk("comparator", {'facing': ST("south"), 'mode': ST("compare"), 'powered': ST("false")}),
}
lever = [
    Blk("lever", {'face': ST("floor"), 'facing': ST("east"), 'powered': ST("false")}),
    Blk("lever", {'face': ST("floor"), 'facing': ST("east"), 'powered': ST("true")}),
]
torch = {
    "w": Blk("redstone_torch", {'facing': ST("west"), 'lit': ST("true")}),
    "e": Blk("redstone_torch", {'facing': ST("east"), 'lit': ST("true")}),
    "n": Blk("redstone_torch", {'facing': ST("north"), 'lit': ST("true")}),
    "s": Blk("redstone_torch", {'facing': ST("south"), 'lit': ST("true")}),
    "u": Blk("redstone_torch", {'facing': ST("up"), 'lit': ST("true")}),
}
hopper = {
    "w": Blk("hopper", {'enabled': ST("true"), 'facing': ST("west")}),
    "e": Blk("hopper", {'enabled': ST("true"), 'facing': ST("east")}),
    "n": Blk("hopper", {'enabled': ST("true"), 'facing': ST("north")}),
    "s": Blk("hopper", {'enabled': ST("true"), 'facing': ST("south")}),
}
lamp = [
    Blk("redstone_lamp", {'lit': ST("false")}),
    Blk("redstone_lamp", {'lit': ST("true")}),
]
def get_noteblock(note=0):
    return Blk("note_block", {'instrument': ST("bell"), 'note': ST(note % 25), 'powered': ST("false")})
def get_consonant_note(p, q):
    # D F A B = 2, 5, 9, 11
    # return [2, 5, 9, 11, 2+12, 5+12, 9+12, 11+12][int(p/q*8)]
    return [2, 6, 9, 2+12, 6+12, 9+12][p%6]
wire_inv = {v: k for k, v in wire.items()}
repeater_inv = {v: k for k, v in repeater.items()}
torch_inv = {v: k for k, v in torch.items()}

def rot_clockwise(tag, k):
    def rot(tag):
        return { "+":"+", "-":"|", "|":"-","ne":"se","nw":"ne","se":"sw","sw":"nw","w":"n","n":"e","e":"s","s":"w" }.get(tag, tag)
    for _ in range(k%4):
        tag = rot(tag)
    return tag

y_lvl = 4   # lowest air block = 4

loaded_chunks = dict()

def get_chunk(cx, cz, clean=False):
    if clean:
        loaded_chunks[(cx, cz)] = level.create_chunk(cx, cz, "minecraft:overworld")
    else:
        if (cx, cz) not in loaded_chunks:
            try:
                new_chunk = level.get_chunk(cx, cz, "minecraft:overworld")
            except ChunkDoesNotExist:
                new_chunk = level.create_chunk(cx, cz, "minecraft:overworld")
            loaded_chunks[(cx, cz)] = new_chunk
    return loaded_chunks[(cx, cz)]

def clrchunk(cx, cz):
    chunk = get_chunk(cx, cz, clean=False)
    for x in range(16):
        for z in range(16):
            chunk.set_block(x, y_lvl-1, z, grass)
            for y in range(64):
                chunk.set_block(x, y_lvl+y, z, air)

# TODO:
# 1. add rotation and write 4 funcs into 1
# 2. implement stair-like folding to save vertical space
# 3. change all diamonds without torches into glass
# 4. initialize the wires and torches with correct values, or add some kind of fuzzer
# 5. add an output monitor

x_offset = 0
y_offset = 0
z_offset = 0
rot = 0

def clockwise_rotate(x, z, k):
    char_to_k = {"n":0, "e":1, "s":2, "w":3}
    return [(x, z), (-z, x), (-x, -z), (z, -x)][char_to_k.get(k, k%4)]

def putblk(x, y, z, blk=None):
    global x_offset, y_offset, z_offset, rot

    if blk is None:
        blk = putblk.blk
    else:
        putblk.blk = blk
    x,z = clockwise_rotate(x,z,rot)
    x += x_offset
    y += y_offset
    z += z_offset
    blk_rot = blk
    if blk in wire_inv:
        tag = wire_inv[blk]
        blk_rot = wire[rot_clockwise(tag, rot)]
    elif blk in repeater_inv:
        tag = repeater_inv[blk]
        blk_rot = repeater[rot_clockwise(tag, rot)]
    elif blk in torch_inv:
        tag = torch_inv[blk]
        blk_rot = torch[rot_clockwise(tag, rot)]
    get_chunk(x // 16, z // 16).set_block(x % 16, y_lvl + y, z % 16, blk_rot)

class Offset:
    def __init__(self, dx, dy, dz, dk=0):
        global rot
        self.dx, self.dz = clockwise_rotate(dx, dz, rot)
        self.dy = dy
        self.dk = dk

    def __enter__(self):
        global x_offset, y_offset, z_offset, rot
        x_offset += self.dx
        y_offset += self.dy
        z_offset += self.dz
        rot += self.dk

    def __exit__(self, exc_type, exc_value, traceback):
        global x_offset, y_offset, z_offset, rot
        x_offset -= self.dx
        y_offset -= self.dy
        z_offset -= self.dz
        rot -= self.dk

def put_gate(x, y, z, inv):
    putblk(x, y-inv, z, air)
    putblk(x, y-inv+1, z, diamond)
    putblk(x, y-inv+2, z, spiston_down)
    if inv == 1:
        putblk(x, y+2, z, stone)
    putblk(x, y+3, z, stone)
    putblk(x, y+4, z, wire['-'])

def put_hopper_clock(x, y, z):
    with Offset(x, y, z):
        putblk(0, 0, 0, hopper['e'])
        putblk(1, 0, 0, hopper['e'])
        putblk(2, 0, 0, hopper['s'])
        putblk(2, 0, 1, hopper['s'])
        putblk(2, 0, 2, hopper['w'])
        putblk(1, 0, 2, hopper['w'])
        putblk(0, 0, 2, hopper['n'])
        putblk(0, 0, 1, hopper['n'])

# repeater_gap = 8
repeater_gap = 16

def put_nor_layer_0(x, y, z, wires, input_len, output_len, inv, tmap, initial=False, note=-1):
    with Offset(x - 2*output_len - 2, y, z):
        for i in range(2*wires):
            idx = wires - i//2 - 1
            for j in range(2*output_len):   # from the end to the start
                if i%2 == 1:
                    # if idx == 0:
                    #     putblk(j, 0, i, gold)
                    # else:
                    # putblk(j, 0, i, diamond)
                    if j%2 == 0 and (j//2) in tmap[idx]:
                        putblk(j, 0, i, diamond)
                        # the resetter wire
                        if idx == 0 and j == 2*output_len - 2:
                            putblk(j, 0, i+1, lamp[0])
                    else:
                        putblk(j, 0, i, glass)
                    putblk(j, 1, i, wire['-'])
                    if (2*output_len-j+1) % repeater_gap == 0:
                        putblk(j, 1, i, repeater["e"])
                elif j%2 == 0:
                    if (j//2) in tmap[idx]:
                        putblk(j, 0, i, torch["n"])
                    else:
                        putblk(j, 0, i, air)
            if i%2 == 1:
                with Offset(2*output_len, 0, 0):
                    for j in range(2*input_len+2):
                        # putblk(j, 0, i, diamond)
                        putblk(j, 0, i, glass)
                        putblk(j, 1, i, wire['-'])
                        if (j+1) % repeater_gap == 0:
                            putblk(j, 1, i, repeater["e"])
                    if inv[idx]:
                        putblk(-1, 1, i, torch["w"])
                    else:
                        putblk(-1, 1, i, repeater["e"])
                    putblk(0, 1, i, gold)
                    if note != -1:
                        putblk(0, 2, i, get_noteblock(note))
                    if initial:
                        putblk(0, 2, i, lever[0])
                        putblk(1, 1, i, air)
                        putblk(1, 0, i, air)
                    else:
                        putblk(1, 1, i, repeater["e"])

def put_nor_array(x, y, z, floor_plan: FloorPlan):
    wire_counts = [0,] + [len(f) for f in floor_plan] + [6, ]
    depth = len(wire_counts) - 2
    with Offset(x, y, z):
        for d in range(depth):
            # we need to put next layer's torches on this layer
            # but the inverter ("hold") depends on this layer
            inverter_map = floor_plan[d]
            inverter_mask = []
            for t in range(len(inverter_map)):
                if inverter_map.lanes[t].lane_op == LaneOp.HOLD or (
                   inverter_map[t].is_input and inverter_map[t].input_name == 'Resetter'):
                    inverter_mask.append(True)
                else:
                    inverter_mask.append(False)

            tmap = []
            for t in range(len(inverter_map)):
                tmap.append(set())
            if d < depth - 1:
                torches_map = floor_plan[d+1]
                for t in range(len(torches_map)):
                    if torches_map[t].lane_op == LaneOp.HOLD:
                        for arg in torches_map[t].lane_operands:
                            tmap[arg].add(t)
                    elif torches_map[t].lane_op == LaneOp.INPUT:
                        print("should be input, but input shouldn't be in the next layer anytime")
                    else:
                        for arg in torches_map[t].lane_operands:
                            tmap[arg].add(t)

            # print(inverter_mask)
            # print(tmap)

            spiral_type = (d % 4)
            i_len = wire_counts[d]
            o_len = wire_counts[d+2]
            wires = wire_counts[d+1]

            # set to -1 for omitting noteblocks
            note = get_consonant_note(d, depth)

            if spiral_type == 0:
                with Offset(1, 0, 1, 0):
                    put_nor_layer_0(0, -2*d, 0, wires, i_len, o_len, inverter_mask, tmap, note=note, initial=(d==0))
            elif spiral_type == 1:
                with Offset(-2, 0, 1, 1):
                    put_nor_layer_0(0, -2*d, 0, wires, i_len, o_len, inverter_mask, tmap, note=note)
            elif spiral_type == 2:
                with Offset(-2, 0, -2, 2):
                    put_nor_layer_0(0, -2*d, 0, wires, i_len, o_len, inverter_mask, tmap, note=note)
            elif spiral_type == 3:
                with Offset(1, 0, -2, 3):
                    put_nor_layer_0(0, -2*d, 0, wires, i_len, o_len, inverter_mask, tmap, note=note)

            if d == 0:
                # put a hopper clock
                with Offset(0, 0, wires*2):
                    put_hopper_clock(2, 1, 0)
                    putblk(1, 0, 0, glass)
                    putblk(1, 1, 0, comparator['e'])
                    putblk(0, 0, 0, glass)
                    putblk(0, 1, 0, wire['-'])


def work():
    for cx in range(-5, 5):
        for cz in range(-5, 5):
            get_chunk(cx, cz, clean=True)
            clrchunk(cx, cz)

    import pickle
    with open("circuit.pkl", "rb") as f:
        floor_plan: FloorPlan = pickle.load(f)

    def gen_dummy_floor_plan(width: int, height: int) -> FloorPlan:
        floor_plan = FloorPlan()
        input_layer = Layer()
        input_layer.layer_id = 0
        for i in range(width):
            lane = Lane(lane_id=i, lane_op=LaneOp.INPUT, lane_operands=[f'input_{i}', 0])
            input_layer[i] = lane
        floor_plan.layers.append(input_layer)
        for layer_idx in range(1, height):
            hold_layer = Layer()
            hold_layer.layer_id = layer_idx
            for i in range(width):
                lane = Lane(lane_id=i, lane_op=LaneOp.HOLD, lane_operands=[i])
                hold_layer[i] = lane
            floor_plan.layers.append(hold_layer)
        return floor_plan

    def augment_floor_plan(floor_plan: FloorPlan):
        def inc_listen_by_1(lane: Lane):
            if lane.lane_op == LaneOp.INPUT:
                return lane
            else:
                lane.lane_operands = [u+1 for u in lane.lane_operands]
                return lane
        def inc_lane_by_1(layer: Layer):
            tmp = [(u+1, v) for (u,v) in layer.lanes.items()]
            for u,v in tmp:
                v.lane_id += 1
            layer.lanes = dict(tmp)
        inc_lane_by_1(floor_plan.layers[0])
        floor_plan.layers[0][0] = Lane(0, LaneOp.INPUT, ["Resetter", 0])
        for f in range(1, len(floor_plan)):
            for u,v in floor_plan[f].items():
                floor_plan[f][u] = inc_listen_by_1(v)
            if f == len(floor_plan) - 1:
                break
            inc_lane_by_1(floor_plan[f])
            floor_plan.layers[f][0] = Lane(0, LaneOp.HOLD, [0])

    augment_floor_plan(floor_plan)
    for f in floor_plan.layers:
        print(f)
    floor_plan.print_stats()

    def platform(y=0):
        putblk(0, y, 0, gold); putblk(0, y, -1); putblk(-1, y, 0); putblk(-1, y, -1)

    lift = 20
    with Offset(0, lift, 0):
        platform(1)
        with Offset(0, len(floor_plan) * 2, 0):
            platform()
            put_nor_array(0, 0, 0, floor_plan=floor_plan)

    for (cx,cz), chunk in loaded_chunks.items():
        level.put_chunk(chunk, "minecraft:overworld")
    level.save()
    level.close()

def inspect(cx, cz):
    chunk = get_chunk(cx, cz, clean=False)
    for x in range(16):
        for y in range(180, 220):
            for z in range(16):
                b = chunk.get_block(x,y,z)
                if len(b.properties) > 0:
                    print(b.base_name, b.properties)
                # print(b.base_name)

work()
# inspect(0, 0)