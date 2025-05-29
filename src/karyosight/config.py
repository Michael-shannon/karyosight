# config.py
from pathlib import Path

MASTER = Path(r"D:\LUNGBUDLARGETEST___tif")  # master directory

# RAW_DIR         = Path(r"D:\LUNGBUDMASTERTEST_3") / "analyze"
# STITCHED_DIR    = Path(r"D:\LUNGBUDMASTERTEST_3") / "stitched"
# CROPPED_DIR = Path(r"D:\LUNGBUDMASTERTEST_3") / "cropped"

RAW_DIR         = MASTER / "analyze"
STITCHED_DIR    = MASTER / "stitched"
CROPPED_DIR = MASTER / "cropped"


CONDITION_PREF  = "Condition_"
MULTIVIEW_CMD   = "multiview-stitcher"      # your CLI tool
# BATCH_GRID_SHAPE= (2, 2)                    # split each full-grid into 2×2 sub-grids
OVERLAP_PERCENT = 10                        # stitcher overlap hint
OUTPUT_EXT      = ".ome.tiff"               # output format

PIXEL_SIZE_X   = 0.23903098823163  # µm per pixel (X)
PIXEL_SIZE_Y   = 0.23903098823163  # µm per pixel (Y)
PIXEL_SIZE_Z   = 2.0               # µm per Z step

TILE_SIZE_X    = 1024              # px, from SizeX
TILE_SIZE_Y    = 1024              # px, from SizeY

CHANNELS       = ["CH1","CH2","CH3","CH4","CH5","CH6"]
REG_CHANNEL    = "CH1"             # registration channel name

MAX_TILES      = 36              
# BATCH_TILE_OVERLAP = 1

# BATCH_GRID_SHAPE = (6, 6)    # how many tiles per side in each sub-grid
# how many tiles per side in each sub-grid (set to None to auto-calc)
BATCH_GRID_SHAPE = [12,12]

# if BATCH_GRID_SHAPE is None, we’ll estimate from RAM with:
SAFETY_FRACTION  = 0.5   # fraction of available RAM to use
OVERHEAD_FACTOR  = 2.0   # per-tile memory multiplier for Dask/Zarr overhead





BATCH_TILE_OVERLAP = 1         # how many tiles extra on each edge for overlap