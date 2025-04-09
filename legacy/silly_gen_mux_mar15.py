import amulet
from amulet import Block, StringTag
ST = StringTag
from amulet.api.errors import ChunkLoadError, ChunkDoesNotExist

prefix_path = "/Users/wtm/playground/minecraft/.minecraft/saves/PythonGenerated"
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
grass = Blk("grass_block")
piston_down = Blk("piston", {"extended": ST("false"), "facing": ST("down")})
spiston_down = Blk("sticky_piston", {"extended": ST("false"), "facing": ST("down")})
stone = Blk("stone")
wire = {
    "-": Blk("redstone_wire", {'east': ST("side"), 'north': ST("none"), 'south': ST("none"), 'west': ST("side"), 'power': ST("0")}),
    "|": Blk("redstone_wire", {'east': ST("none"), 'north': ST("side"), 'south': ST("side"), 'west': ST("none"), 'power': ST("0")}),
    "+": Blk("redstone_wire", {'east': ST("side"), 'north': ST("side"), 'south': ST("side"), 'west': ST("side"), 'power': ST("0")}),
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

y_lvl = 4   # lowest air block = 4

loaded_chunks = dict()

def get_chunk(cx, cz):
    if (cx, cz) not in loaded_chunks:
        loaded_chunks[(cx, cz)] = level.get_chunk(cx, cz, "minecraft:overworld")
    return loaded_chunks[(cx, cz)]

def clrchunk(chunk):
    for x in range(16):
        for z in range(16):
            chunk.set_block(x, y_lvl-1, z, grass)
            for y in range(16):
                chunk.set_block(x, y_lvl+y, z, air)

x_offset = 0
y_offset = 0
z_offset = 0

def putblk(x, y, z, blk=None):
    if blk is None:
        blk = putblk.blk
    else:
        putblk.blk = blk
    x += x_offset
    y += y_offset
    z += z_offset
    get_chunk(x // 16, z // 16).set_block(x, y_lvl + y, z, blk)

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

def put_lut4(x, y, z, data):
    """
    control port: (x,z) and (x,z+2)
    """

    assert len(data) == 4, "Must provide 4 bit data for LUT"
    with Offset(x+1, y, z-2):
        putblk(0, 1, 0, diamond)
        putblk(1, 1, 0)
        putblk(2, 1, 0)
        putblk(3, 1, 0)
        putblk(0, 2, 0, lever[data[0]])
        putblk(1, 2, 0, lever[data[1]])
        putblk(2, 2, 0, lever[data[2]])
        putblk(3, 2, 0, lever[data[3]])

        putblk(0, 1, 1, diamond)
        putblk(1, 1, 1)
        putblk(2, 1, 1)
        putblk(3, 1, 1)
        putblk(0, 2, 1, repeater['n'])
        putblk(1, 2, 1)
        putblk(2, 2, 1)
        putblk(3, 2, 1)
        put_gate(0, 2, 2, 0)
        put_gate(1, 2, 2, 0)
        put_gate(2, 2, 2, 1)
        put_gate(3, 2, 2, 1)
        putblk(3, 6, 2, air)
        putblk(3, 5, 2, wire["-"])
        putblk(4, 4, 2, wire["-"])
        putblk(4, 3, 2, stone)
        putblk(5, 3, 2, wire["-"])
        putblk(5, 2, 2, stone)

        putblk(-1, 5, 2, diamond)
        putblk(-1, 6, 2, lever[0])

        putblk(0, 1, 3, diamond)
        putblk(1, 1, 3)
        putblk(2, 1, 3)
        putblk(3, 1, 3)
        putblk(0, 2, 3, repeater['n'])  # north in, south out
        putblk(1, 2, 3)
        putblk(2, 2, 3)
        putblk(3, 2, 3)
        put_gate(0, 2, 4, 0)
        put_gate(1, 2, 4, 1)
        put_gate(2, 2, 4, 0)
        put_gate(3, 2, 4, 1)
        putblk(3, 6, 4, air)
        putblk(3, 5, 4, wire["-"])
        putblk(4, 4, 4, wire["-"])
        putblk(4, 3, 4, stone)
        putblk(5, 3, 4, wire["-"])
        putblk(5, 2, 4, stone)

        putblk(-1, 5, 4, diamond)
        putblk(-1, 6, 4, lever[0])

def crossbar_switch(x, y, z):
    with Offset(x, y, z):
        for i in range(8):
            # putblk()
            pass

def work():
    for cx in [-1, 0, 1]:
        for cz in [-1, 0, 1]:
            clrchunk(get_chunk(cx, cz))

    put_lut4(0, 0, 0, [0,1,0,0])
    crossbar_switch(30, 0, 30)

    for (cx,cz), chunk in loaded_chunks.items():
        level.put_chunk(chunk, "minecraft:overworld")
    level.save()
    level.close()

def inspect(cx, cz):
    chunk = get_chunk(cx, cz)
    for x in range(16):
        for y in range(16):
            for z in range(16):
                b = chunk.get_block(x,y,z)
                if len(b.properties) > 0:
                    print(b.base_name, b.properties)

work()
# inspect(0, 0)