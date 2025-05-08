# config.py
from pathlib import Path

RAW_DIR         = Path(r"D:\LUNGBUDMASTERTEST_2") / "analyze"
STITCHED_DIR    = Path(r"D:\LUNGBUDMASTERTEST_2") / "stitched"
CONDITION_PREF  = "Condition_"
MULTIVIEW_CMD   = "multiview-stitcher"      # your CLI tool
BATCH_GRID_SHAPE= (2, 2)                    # split each full-grid into 2×2 sub-grids
OVERLAP_PERCENT = 10                        # stitcher overlap hint
OUTPUT_EXT      = ".ome.tiff"               # output format

PIXEL_SIZE_X   = 0.23903098823163  # µm per pixel (X)
PIXEL_SIZE_Y   = 0.23903098823163  # µm per pixel (Y)
PIXEL_SIZE_Z   = 2.0               # µm per Z step

TILE_SIZE_X    = 1024              # px, from SizeX
TILE_SIZE_Y    = 1024              # px, from SizeY

CHANNELS       = ["CH1","CH2","CH3","CH4","CH5","CH6"]
REG_CHANNEL    = "CH1"             # registration channel name

MAX_TILES      = 10              


