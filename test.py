import amulet
from amulet import Block, StringTag
from amulet.api.errors import ChunkLoadError, ChunkDoesNotExist

# load the level
# this will automatically find the wrapper that can open the world and set everything up for you.
prefix_path = "/Users/wtm/playground/minecraft/.minecraft/saves/PythonGenerated"
level = amulet.load_level(prefix_path)

# read/write the world data here
# read a chunk
chunk = level.get_chunk(0, 0, "minecraft:overworld")
print(chunk)
print(chunk.coordinates)
print(chunk.get_block(0, 0, 0))
print(chunk.get_block(0, 1, 0))
print(chunk.get_block(0, 2, 0))
print(chunk.get_block(0, 3, 0))

# make changes to the chunk here
# chunk.set_block(0, 3, 0, amulet.Block("universal_minecraft", "grass_block", {"snowy": StringTag("true")}))
chunk.set_block(0, 3, 0, amulet.Block("universal_minecraft", "diamond_block"))

print(chunk.get_block(0, 0, 0))
print(chunk.get_block(0, 1, 0))
print(chunk.get_block(0, 2, 0))
print(chunk.get_block(0, 3, 0))
level.put_chunk(chunk, "minecraft:overworld")

# save the changes to the world
level.save()

# close the world
level.close()
