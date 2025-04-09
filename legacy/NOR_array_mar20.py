import amulet
from amulet import Block, StringTag
ST = StringTag
from amulet.api.errors import ChunkLoadError, ChunkDoesNotExist

prefix_path = "/Users/wtm/playground/minecraft/.minecraft/saves/PythonGenerated-2"
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
glass = Blk("glass")
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
    "w": Blk("repeater", {'delay': ST("1"), 'facing': ST("west"), 'locked': ST("false"), 'powered': ST("false")}),
    "e": Blk("repeater", {'delay': ST("1"), 'facing': ST("east"), 'locked': ST("false"), 'powered': ST("false")}),
    "n": Blk("repeater", {'delay': ST("1"), 'facing': ST("north"), 'locked': ST("false"), 'powered': ST("false")}),
    "s": Blk("repeater", {'delay': ST("1"), 'facing': ST("south"), 'locked': ST("false"), 'powered': ST("false")}),
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

y_lvl = 4   # lowest air block = 4

loaded_chunks = dict()

def get_chunk(cx, cz, clean=True):
    if (cx, cz) not in loaded_chunks:
        new_chunk =  level.get_chunk(cx, cz, "minecraft:overworld")
        if clean:
            clrchunk(new_chunk)
        loaded_chunks[(cx, cz)] = new_chunk
    return loaded_chunks[(cx, cz)]

def clrchunk(chunk):
    for x in range(16):
        for z in range(16):
            chunk.set_block(x, y_lvl-1, z, grass)
            for y in range(64):
                chunk.set_block(x, y_lvl+y, z, air)
            for y in range(128, 204):
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
    if blk is None:
        blk = putblk.blk
    else:
        putblk.blk = blk
    x += x_offset
    y += y_offset
    z += z_offset
    x,z = clockwise_rotate(x,z,rot)
    get_chunk(x // 16, z // 16).set_block(x % 16, y_lvl + y, z % 16, blk)

class Offset:
    def __init__(self, dx, dz):
        self.dx = dx
        self.dz = dz
        self.dy = 0
    def __init__(self, dx, dy, dz):
        self.dx = dx
        self.dz = dz
        self.dy = dy

    def __enter__(self):
        global x_offset, y_offset, z_offset
        x_offset += self.dx
        y_offset += self.dy
        z_offset += self.dz

    def __exit__(self, exc_type, exc_value, traceback):
        global x_offset, y_offset, z_offset
        x_offset -= self.dx
        y_offset -= self.dy
        z_offset -= self.dz

def put_gate(x, y, z, inv):
    putblk(x, y-inv, z, air)
    putblk(x, y-inv+1, z, diamond)
    putblk(x, y-inv+2, z, spiston_down)
    if inv == 1:
        putblk(x, y+2, z, stone)
    putblk(x, y+3, z, stone)
    putblk(x, y+4, z, wire['-'])

repeater_gap = 8

def put_nor_layer_0(x, y, z, wires, input_len, output_len, inv, tmap, initial=False):
    """
    *   *
    [][][]*\]]
    *   *   ^repeater
    [][][][[]]

    "input_len"  means last layer wire count
    "output_len" means next layer wire count
    """
    with Offset(x, y, z):
        for i in range(2*wires):
            for j in range(2*output_len):
                if i%2 == 1:
                    putblk(j, 0, i, diamond)
                    putblk(j, 1, i, wire['-'])
                    if (j+1) % repeater_gap == 0:
                        putblk(j, 1, i, repeater["e"])
                elif j%2 == 0 and (j//2) in tmap[i//2]:
                    putblk(j, 0, i, torch["n"])
            if i%2 == 1:
                for j in range(2*input_len):
                    putblk(2*output_len+j, 0, i, diamond)
                    putblk(2*output_len+j, 1, i, wire['-'])
                    if (j+1) % repeater_gap == 0:
                        putblk(2*output_len+j, 1, i, repeater["e"])
                if inv[i//2]:
                    putblk(2*output_len-1, 1, i, torch["w"])
                else:
                    putblk(2*output_len-1, 1, i, repeater["e"])
                putblk(2*output_len, 1, i, gold)
                if initial:
                    putblk(2*output_len, 2, i, lever[0])
                else:
                    putblk(2*output_len+1, 1, i, repeater["e"])

def put_nor_layer_1(x, y, z, wires, input_len, output_len, inv, tmap):
    """
    []* []*
    []  []
    []* []*
    []  []
    """
    with Offset(x, y, z):
        for i in range(2*wires):
            for j in range(2*output_len):
                if i%2 == 0:
                    putblk(i, 0, j, diamond)
                    putblk(i, 1, j, wire['|'])
                    if (j+1) % repeater_gap == 0:
                        putblk(i, 1, j, repeater["s"])
                elif j%2 == 0 and (j//2) in tmap[i//2]:
                    putblk(i, 0, j, torch["e"])
            if i%2 == 0:
                for j in range(2*input_len):
                    putblk(i, 0, 2*output_len+j, diamond)
                    putblk(i, 1, 2*output_len+j, wire['|'])
                    if (j+1) % repeater_gap == 0:
                        putblk(i, 1, 2*output_len+j, repeater["s"])
                if inv[i//2]:
                    putblk(i, 1, 2*output_len-1, torch["n"])
                else:
                    putblk(i, 1, 2*output_len-1, repeater["s"])
                putblk(i, 1, 2*output_len, gold)
                putblk(i, 1, 2*output_len+1, repeater["s"])

def put_nor_layer_2(x, y, z, wires, input_len, output_len, inv, tmap):
    """
    [][][][]
      *   *
    [][][][]
      *   *
    """
    with Offset(x, y, z):
        for i in range(2*wires):
            for j in range(2*output_len):
                if i%2 == 0:
                    putblk(j, 0, i, diamond)
                    putblk(j, 1, i, wire['-'])
                    if (j+2) % repeater_gap == 0:
                        putblk(j, 1, i, repeater["w"])
                elif j%2 == 1 and (j//2) in tmap[i//2]:
                    putblk(j, 0, i, torch["s"])
            if i%2 == 0:
                for j in range(2*input_len):
                    putblk(-1-j, 0, i, diamond)
                    putblk(-1-j, 1, i, wire['-'])
                    if (j+3) % repeater_gap == 0:
                        putblk(-1-j, 1, i, repeater["w"])
                if inv[i//2]:
                    putblk(0, 1, i, torch["e"])
                else:
                    putblk(0, 1, i, repeater["w"])
                putblk(-1, 1, i, gold)
                putblk(-2, 1, i, repeater["w"])

def put_nor_layer_3(x, y, z, wires, input_len, output_len, inv, tmap):
    """
      []  []
     *[] *[]
      []  []
     *[] *[]
    """
    with Offset(x, y, z):
        for i in range(2*wires):
            for j in range(2*output_len):
                if i%2 == 1:
                    putblk(i, 0, j, diamond)
                    putblk(i, 1, j, wire['|'])
                    if (j+2) % repeater_gap == 0:
                        putblk(i, 1, j, repeater["n"])
                elif j%2 == 1 and (j//2) in tmap[i//2]:
                    putblk(i, 0, j, torch["w"])
            if i%2 == 1:
                for j in range(2*input_len):
                    putblk(i, 0, -j-1, diamond)
                    putblk(i, 1, -j-1, wire['|'])
                    if (j+3) % repeater_gap == 0:
                        putblk(i, 1, -j-1, repeater["n"])
                if inv[i//2]:
                    putblk(i, 1, 0, torch["s"])
                else:
                    putblk(i, 1, 0, repeater["n"])
                putblk(i, 1, -1, gold)
                putblk(i, 1, -2, repeater["n"])


def put_nor_array(x, y, z, floor_plan=None):
    wire_counts = [0,] + [len(f) for f in floor_plan] + [6, ]
    depth = len(wire_counts) - 2
    with Offset(x, y, z):
        dx,dy,dz = 0,0,0
        for d in range(depth):
            # we need to put next layer's torches on this layer
            # but the inverter ("hold") depends on this layer
            inverter_map = floor_plan[d]
            inverter_mask = []
            for t in range(len(inverter_map)):
                if inverter_map[t][0] == 'hold':
                    inverter_mask.append(True)
                else:
                    inverter_mask.append(False)

            tmap = []
            for t in range(len(inverter_map)):
                tmap.append(set())
            if d < depth - 1:
                torches_map = floor_plan[d+1]
                for t in range(len(torches_map)):
                    if torches_map[t][0] == 'hold':
                        tmap[torches_map[t][1]].add(t)
                    elif isinstance(torches_map[t][0], str):
                        print("should be input, but input shouldn't be in the next layer anytime")
                    else:
                        for var in torches_map[t]:
                            if isinstance(var, int):
                                tmap[var].add(t)

            # print(inverter_mask)
            # print(tmap)

            spiral_type = (d % 4)
            worker = [put_nor_layer_0, put_nor_layer_1, put_nor_layer_2, put_nor_layer_3][spiral_type]
            i_len = wire_counts[d] + 1
            o_len = wire_counts[d+2]
            wires = wire_counts[d+1]
            if spiral_type == 0:
                dx += -2 * o_len -2 # because of the i_len+1
            elif spiral_type == 1:
                dz += -2 * o_len -2
            elif spiral_type == 2:
                dx -= -2 * i_len
            elif spiral_type == 3:
                dz -= -2 * i_len
            print(dx, dz)
            # global diamond
            # diamond0 = diamond
            # diamond = glass
            if d == 0:
                worker(dx, dy-2*d, dz, wires, i_len, o_len, inverter_mask, tmap, initial=True)
            else:
                worker(dx, dy-2*d, dz, wires, i_len, o_len, inverter_mask, tmap)
            # diamond = diamond0


def work():
    for cx in range(-3, 3):
        for cz in range(-3, 3):
            clrchunk(get_chunk(cx, cz))

    import pickle
    with open("max.pkl", "rb") as f:
        floor_plan = pickle.load(f)
    put_nor_array(0, 200, 0, floor_plan=floor_plan)

    for (cx,cz), chunk in loaded_chunks.items():
        level.put_chunk(chunk, "minecraft:overworld")
    level.save()
    level.close()

def inspect(cx, cz):
    chunk = get_chunk(cx, cz, clean=False)
    for x in range(16):
        for y in range(16):
            for z in range(16):
                b = chunk.get_block(x,y,z)
                # if len(b.properties) > 0:
                #     print(b.base_name, b.properties)
                print(b.base_name)

work()
# inspect(0, 0)