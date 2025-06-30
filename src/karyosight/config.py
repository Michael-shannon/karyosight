# config.py
from pathlib import Path

# MASTER = Path(r"D:\LUNGBUDLARGETEST___tif")  # master directory
# MASTER = Path(r"D:\LUNGBUD_MultiGridTest")  # master directory
# MASTER = Path(r"D:\Lungbud_testsetmaster")  # master directory
# MASTER = Path(r"O:\mshannon\2025\May\Lungbud_MASTER")  # master directory
# MASTER = Path(r"D:\LUNGBUD_master")
# MASTER = Path(r"O:\mshannon\2025\May\LUNGBUD_master2")
MASTER = Path(r"D:\LB_TEST2")
# MASTER = Path(r"D:\LUNGBUD_OIRTEST")  # master directory

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
PIXEL_SIZE_Z   = 1.0               # µm per Z step

TILE_SIZE_X    = 1024              # px, from SizeX
TILE_SIZE_Y    = 1024              # px, from SizeY

# CHANNELS       = ["CH1","CH2","CH3","CH4","CH5","CH6"]
CHANNELS       = ["nucleus","tetraploid","diploid","sox9_alv","sox2_air","brightfield"] # sox9 is alveoli sox2 is airway
REG_CHANNEL    = "nucleus"             # registration channel name

MAX_TILES      = 36              
# BATCH_TILE_OVERLAP = 1

# BATCH_GRID_SHAPE = (6, 6)    # how many tiles per side in each sub-grid
# how many tiles per side in each sub-grid (set to None to auto-calc)
BATCH_GRID_SHAPE = [6,6]

# if BATCH_GRID_SHAPE is None, we'll estimate from RAM with:
SAFETY_FRACTION  = 0.5   # fraction of available RAM to use
OVERHEAD_FACTOR  = 2.0   # per-tile memory multiplier for Dask/Zarr overhead





BATCH_TILE_OVERLAP = 1         # how many tiles extra on each edge for overlap

# Z-CROPPING SETTINGS FOR STITCHING
CROP_Z_BLACK_FRAMES = False    # whether to crop black z-frames after stitching
Z_CROP_THRESHOLD = 0.01        # threshold for detecting empty z-slices (fraction of global max)

# BLACK FRAME REMOVAL SETTINGS FOR CROPPING
REMOVE_BLACK_FRAMES = True            # whether to remove black z-slices during cropping
BLACK_FRAME_THRESHOLD = 0.001         # threshold for black frame detection (0.1% of max intensity)
BLACK_FRAME_METHOD = 'intensity_threshold'  # method: 'intensity_threshold', 'content_ratio', or 'mean_intensity'
MIN_Z_SLICES = 20                      # minimum number of z-slices to keep

# EDGE CROPPED ORGANOID FILTERING SETTINGS
FILTER_EDGE_CROPPED = True            # whether to filter out edge-cropped organoids
EDGE_PROXIMITY_THRESHOLD = 0.05       # distance from image edge as fraction of image size (5%)
STRAIGHT_EDGE_THRESHOLD = 0.7         # minimum ratio of straight edge pixels to total perimeter
BLACK_REGION_THRESHOLD = 0.3          # minimum ratio of black pixels in extended bbox

# ======================================================================================
# CELLPOSE-SAM SEGMENTATION PARAMETERS (OPTIMIZED)
# ======================================================================================
# These parameters were optimized through comprehensive testing
# Recommended settings for nuclei segmentation in lung organoids

# Core Segmentation Parameters (from parameter testing: D50, FT0.4, CT0.0)
OPTIMAL_DIAMETER = 50.0                # pixels - optimal nuclei diameter
OPTIMAL_FLOW_THRESHOLD = 0.4           # flow error threshold (0.0-1.0, higher = stricter)
OPTIMAL_CELLPROB_THRESHOLD = 0.0       # cell probability threshold (-6 to 6, higher = stricter)

# Model and Processing Settings
SEGMENTATION_MODEL_TYPE = 'cpsam'      # Cellpose-SAM model with integrated SAM
USE_3D_SEGMENTATION = True             # 3D segmentation for better accuracy
USE_GPU_SEGMENTATION = True            # GPU acceleration when available

# Speed Optimization Parameters (determined by batch size testing)
OPTIMAL_BATCH_SIZE = 16                # default - will be updated by optimization testing
MAX_BATCH_SIZE = 32                    # maximum safe batch size for memory
MIN_BATCH_SIZE = 4                     # minimum batch size for efficiency

# Anisotropy (automatically calculated from pixel sizes)
SEGMENTATION_ANISOTROPY = PIXEL_SIZE_Z / PIXEL_SIZE_X  # Z-spacing / XY-pixel size

# Channel Configuration
SEGMENTATION_CHANNEL = 0               # channel index for nuclei (DAPI/nucleus channel)
SEGMENTATION_CHANNELS = [0, 0]         # [cytoplasm, nucleus] - both set to nuclei channel

# Output Configuration
SEGMENTED_SAM_DIR = MASTER / "segmented_SAM"  # output directory for SAM segmentations
SAVE_FLOWS = False                     # whether to save flow fields (large files)
SAVE_PROBABILITIES = False             # whether to save cell probabilities (large files)

# Performance and Memory Settings
GPU_MEMORY_FRACTION = 0.8              # fraction of GPU memory to use
ENABLE_MIXED_PRECISION = True          # use mixed precision for speed (if available)
CHUNK_SIZE_3D = (32, 256, 256)         # zarr chunk size for 3D data
COMPRESSION_LEVEL = 3                  # zarr compression level (0-9)

# Quality Control Settings
MIN_NUCLEI_SIZE = 1000                 # minimum nuclei size in pixels (from size filtering optimization)
MAX_NUCLEI_SIZE = 10000                # maximum nuclei size in pixels
ENABLE_QUALITY_FILTERING = True        # filter nuclei by size and shape

# Size Filtering Optimization Results (will be updated by testing)
CONSERVATIVE_MIN_SIZE = 100            # conservative threshold (removes debris, keeps small nuclei)
AGGRESSIVE_MIN_SIZE = 200              # aggressive threshold (removes small nuclei too)

# Out-of-Focus Detection and Removal Settings
ENABLE_FOCUS_FILTERING = True          # whether to filter out-of-focus organoids
FOCUS_DETECTION_METHOD = 'variance'    # method: 'variance', 'gradient', 'laplacian'
FOCUS_THRESHOLD = 0.5                  # threshold for focus detection (will be optimized)
MIN_FOCUS_SCORE = 100.0                # minimum focus score to keep organoid

# Validation and Testing
VALIDATION_ORGANOID_IDX = 0            # organoid index for parameter testing
VALIDATION_CONDITION = None            # condition for testing (None = first available)
TEST_SAMPLE_SIZE = 1                   # number of organoids to test parameters on

# Advanced Settings
CELLPOSE_TILE_NORM = True              # normalize tiles independently
CELLPOSE_INTERP = True                 # interpolate between tiles
CELLPOSE_CLUSTER = False               # use DBSCAN clustering post-processing
CELLPOSE_NET_AVG = True                # average networks for better results

# ======================================================================================

# choose your "max-proj" loader:
#   "tif"  → only *.tif / *.tiff
#   "oir"  → only *.oir via Bio-Formats
#   "both" → TIFFs first, then OIR series
IMAGE_LOADER = "tif"

# Path to your local Fiji.app folder (unzipped from the official download)
FIJI_APP = Path(r"D:\GITHUB_SOFTWARE\Fiji.app")

# D:\GITHUB_SOFTWARE\Fiji.app\ImageJ-win64.exe