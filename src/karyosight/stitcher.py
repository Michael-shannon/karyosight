

########################### CARROT #################################

import os
import zipfile
import json
import pprint
import xml.etree.ElementTree as ET
from pathlib import Path
import tifffile
import numpy as np
import math
import xarray as xr
import dask.array as da
import dask.diagnostics
import ngff_zarr
from multiview_stitcher import spatial_image_utils as si_utils
from multiview_stitcher import (
    fusion,
    io,
    msi_utils,
    vis_utils,
    ngff_utils,
    param_utils,
    registration,
)
from tqdm import tqdm
import matplotlib.pyplot as plt
from dask import delayed

import karyosight.config as cfg

import psutil
import re
import cupy
import math
from pathlib import Path

# class Stitcher:
#     """
#     A class to process tiled TIFF datasets: extract metadata, convert to Zarr, stitch, and export.
#     """
#     def __init__(self, master_folder):
#         """
#         Initialize with a master directory containing multiple 'Condition_' folders.

#         Parameters:
#           - master_folder: Path or str to a directory containing condition subfolders
#         """
#         self.master_folder = Path(master_folder)
#         self.conditions = sorted([
#             p for p in self.master_folder.iterdir()
#             if p.is_dir() and p.name.startswith("Condition_")
#         ])

#     # def extract_metadata(self, condition):
#     #     """
#     #     Extract tiling metadata for a given condition folder.

#     #     Returns a dict with keys: overlap, region_origin, region_size,
#     #     grid_size, tile_shape, tiles
#     #     """
#     #     cond_folder = Path(condition)
#     #     if not cond_folder.is_dir():
#     #         cond_folder = self.master_folder / condition
#     #     info_file = next(cond_folder.glob("*forVSIimages.omp2info"))
#     #     tif_folder = cond_folder

#     #     def strip_ns(tag): return tag.split("}")[-1]
#     #     tree = ET.parse(info_file)
#     #     root = tree.getroot()

#     #     overlap_pct = float(next(e for e in root.iter() if strip_ns(e.tag) == "overlap").text)
#     #     overlap = {"x": overlap_pct/100, "y": overlap_pct/100}

#     #     coords = next(e for e in root.iter() if strip_ns(e.tag) == "coordinates")
#     #     region_origin = {"x": float(coords.attrib["x"]), "y": float(coords.attrib["y"]) }
#     #     region_size   = {"width": float(coords.attrib["width"]), "height": float(coords.attrib["height"]) }

#     #     num_x = int(next(e for e in root.iter() if strip_ns(e.tag) == "numOfXAreas").text)
#     #     num_y = int(next(e for e in root.iter() if strip_ns(e.tag) == "numOfYAreas").text)

#     #     tif_stems = {p.stem for p in tif_folder.glob("*.tif")}
#     #     tiles = []
#     #     for area in (e for e in root.iter() if strip_ns(e.tag) == "area"):
#     #         img  = next(e for e in area if strip_ns(e.tag) == "image").text
#     #         stem = Path(img).stem
#     #         if stem in tif_stems:
#     #             xidx = int(next(e for e in area if strip_ns(e.tag) == "xIndex").text)
#     #             yidx = int(next(e for e in area if strip_ns(e.tag) == "yIndex").text)
#     #             tiles.append({"filename": f"{stem}.tif", "x_index": xidx, "y_index": yidx})
#     #     tiles.sort(key=lambda t: (t["y_index"], t["x_index"]))

#     #     first_tif = tif_folder / tiles[0]["filename"]
#     #     arr = tifffile.imread(first_tif)
#     #     if arr.ndim == 4:
#     #         z, c, y, x = arr.shape
#     #     else:
#     #         z, y, x = arr.shape
#     #     tile_shape = {"z": z, "y": y, "x": x}




#     #     return {
#     #         "overlap": overlap,
#     #         "region_origin": region_origin,
#     #         "region_size": region_size,
#     #         "grid_size": {"cols": num_x, "rows": num_y},
#     #         "tile_shape": tile_shape,
#     #         "tiles": tiles
#     #     }

#     @staticmethod
#     def _estimate_max_subgrid(cmd_folder,
#                               safety_fraction: float,
#                               overhead_factor: float) -> int:
#         """
#         Inspect one TIFF in `cmd_folder` and
#         return the largest square side (in tiles)
#         that fits into RAM.
#         """
#         # 1) pick a sample TIFF
#         sample = next(Path(cmd_folder).glob("*.tif"), None)
#         if sample is None:
#             raise RuntimeError(f"No TIFFs in {cmd_folder}")
#         arr = tifffile.imread(sample)
#         # unpack dims
#         if arr.ndim == 4:
#             z, c, y, x = arr.shape
#         elif arr.ndim == 3:
#             z, y, x = arr.shape; c = 1
#         else:
#             raise RuntimeError(f"Unrecognized shape {arr.shape}")
#         bytes_per_tile = z * y * x * c * arr.dtype.itemsize
#         avail = psutil.virtual_memory().available * safety_fraction
#         eff = bytes_per_tile * overhead_factor
#         max_tiles = int(avail // eff)
#         side = int(np.floor(np.sqrt(max_tiles)))
#         return max(1, side)

#     def extract_metadata(self, condition):
#         """
#         Extract tiling metadata for a given condition folder.

#         Returns a dict with keys: overlap, region_origin, region_size,
#         grid_size, tile_shape, tiles
#         """
#         cond_folder = Path(condition)
#         if not cond_folder.is_dir():
#             cond_folder = self.master_folder / condition
#         info_file = next(cond_folder.glob("*forVSIimages.omp2info"))
#         tif_folder = cond_folder

#         def strip_ns(tag): return tag.split("}")[-1]
#         tree = ET.parse(info_file)
#         root = tree.getroot()

#         overlap_pct = float(next(e for e in root.iter() if strip_ns(e.tag) == "overlap").text)
#         overlap = {"x": overlap_pct/100, "y": overlap_pct/100}

#         coords = next(e for e in root.iter() if strip_ns(e.tag) == "coordinates")
#         region_origin = {"x": float(coords.attrib["x"]), "y": float(coords.attrib["y"]) }
#         region_size   = {"width": float(coords.attrib["width"]), "height": float(coords.attrib["height"]) }

#         num_x = int(next(e for e in root.iter() if strip_ns(e.tag) == "numOfXAreas").text)
#         num_y = int(next(e for e in root.iter() if strip_ns(e.tag) == "numOfYAreas").text)

#         tif_stems = {p.stem for p in tif_folder.glob("*.tif")}
#         tiles = []
#         for area in (e for e in root.iter() if strip_ns(e.tag) == "area"):
#             img  = next(e for e in area if strip_ns(e.tag) == "image").text
#             stem = Path(img).stem
#             if stem in tif_stems:
#                 xidx = int(next(e for e in area if strip_ns(e.tag) == "xIndex").text)
#                 yidx = int(next(e for e in area if strip_ns(e.tag) == "yIndex").text)
#                 tiles.append({"filename": f"{stem}.tif", "x_index": xidx, "y_index": yidx})
#         tiles.sort(key=lambda t: (t["y_index"], t["x_index"]))

#         first_tif = tif_folder / tiles[0]["filename"]
#         arr = tifffile.imread(first_tif)
#         if arr.ndim == 4:
#             z, c, y, x = arr.shape
#         else:
#             z, y, x = arr.shape
#         tile_shape = {"z": z, "y": y, "x": x}
#         # a
#         meta = {
#             "overlap": { "x": overlap_pct/100, "y": overlap_pct/100 },
#             "region_origin": region_origin,
#             "region_size": region_size,
#             "grid_size": {"cols": num_x, "rows": num_y},
#             "tile_shape": tile_shape,
#             "tiles": tiles
#         }

#         # build static sub-grid definitions
#         cols, rows = num_x, num_y
#         bw, bh = cfg.BATCH_GRID_SHAPE
#         tile_overlap = cfg.BATCH_TILE_OVERLAP

#         subgrids = []
#         nx = math.ceil(cols/bw)
#         ny = math.ceil(rows/bh)
#         for sgx in range(nx):
#             for sgy in range(ny):
#                 x0, y0 = sgx*bw,      sgy*bh
#                 x1, y1 = min(cols,x0+bw), min(rows,y0+bh)
#                 xo, yo = max(0, x0-tile_overlap), max(0, y0-tile_overlap)
#                 xe, ye = min(cols, x1+tile_overlap), min(rows, y1+tile_overlap)

#                 tiles_in_batch = [
#                     t for t in tiles
#                     if xo <= t["x_index"] < xe
#                     and yo <= t["y_index"] < ye
#                 ]
#                 tiles_in_batch.sort(key=lambda t:(t["y_index"],t["x_index"]))
#                 subgrids.append({
#                     "id": f"{sgx}_{sgy}",
#                     "x_range": (x0, x1),
#                     "y_range": (y0, y1),
#                     "tiles": tiles_in_batch
#                 })

#         meta["subgrids"] = subgrids
#         meta["num_subgrids"] = {"cols": nx, "rows": ny}

#         return meta

#     @staticmethod
#     def load_maxproj_images(folder):
#         """
#         Generator yielding (filename, max_projection) for each TIFF in `folder`.
#         """
#         folder = Path(folder)
#         tif_paths = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
#         for path in tif_paths:
#             stack = tifffile.imread(path)
#             if stack.ndim == 4:
#                 channel0 = stack[:, 0, ...]
#             elif stack.ndim == 3:
#                 channel0 = stack
#             else:
#                 raise ValueError(f"Unexpected image dims {stack.shape} for {path.name}")
#             yield path.name, np.max(channel0, axis=0)

#     @staticmethod
#     def get_tile_grid_position_from_tile_index(tile_index, num_cols, num_rows=None):
#         """
#         Convert flat tile_index to grid coords (z,y,x).
#         """
#         return {'z': 0, 'y': tile_index // num_cols, 'x': tile_index % num_cols}

#     @staticmethod
#     def convert_tiles_to_zarr(tiles, tif_folder, scale, translations,
#                               overwrite=True, use_gpu=False):
#         """
#         Convert tile TIFFs to OME-Zarr using same basename.
#         If use_gpu=True, wrap arrays in CuPy via Dask.
#         """
#         tif_folder = Path(tif_folder)
#         zarr_folder = tif_folder
#         zarr_paths, msims = [], []

#         for idx, tile in enumerate(tiles):
#             tif_path = tif_folder / tile['filename']
#             arr = tifffile.imread(tif_path)
#             if arr.ndim == 3:
#                 arr = arr[None, ...]
#             elif arr.ndim == 4:
#                 arr = arr.transpose(1, 0, 2, 3)
#             else:
#                 raise ValueError(f"Unexpected arr shape {arr.shape} for {tif_path.name}")

#             if use_gpu:
#                 import cupy
#                 darr = da.from_array(arr, chunks="auto", asarray=cupy.asarray)
#             else:
#                 darr = da.from_array(arr, chunks="auto")

#             sim = si_utils.get_sim_from_array(
#                 darr, dims=["c","z","y","x"],
#                 scale=scale,
#                 translation=translations[idx],
#                 transform_key=io.METADATA_TRANSFORM_KEY
#             )

#             stem = Path(tile['filename']).stem
#             zarr_path = zarr_folder / f"{stem}.zarr"
#             ngff_utils.write_sim_to_ome_zarr(sim, str(zarr_path), overwrite=overwrite)

#             sim2 = ngff_utils.read_sim_from_ome_zarr(str(zarr_path))
#             msim = msi_utils.get_msim_from_sim(sim2)

#             zarr_paths.append(str(zarr_path))
#             msims.append(msim)
#                     # Option A: persist (turns each D
#         return zarr_paths, msims

#     def compute_translations(self, meta, scale):
#         tiles, overlap, ts = meta['tiles'], meta['overlap'], meta['tile_shape']
#         transs = []
#         for t in tiles:
#             tx = t['x_index']*(1-overlap['x'])*ts['x']*scale['x']
#             ty = t['y_index']*(1-overlap['y'])*ts['y']*scale['y']
#             transs.append({'x':tx,'y':ty,'z':0.0})
#         return transs
    
#     def split_tiles_into_batches(self,
#                                  meta: dict,
#                                  batch_shape: tuple[int,int],
#                                  overlap_tiles: int = 1) -> list[dict]:
#         """
#         Breaks the full meta['tiles'] (flattened, row-major) into
#         sub-grid “batches” of size batch_shape, stepped so that
#         each window overlaps its neighbors by overlap_tiles.

#         Returns a list of dicts, each with:
#           - 'x_start','y_start': the top-left tile coords of the window
#           - 'tiles':       the list of tile-dicts in that window
#         """
#         cols = meta['grid_size']['cols']
#         rows = meta['grid_size']['rows']
#         tiles = meta['tiles']

#         batch_w, batch_h = batch_shape
#         step_x = batch_w - overlap_tiles
#         step_y = batch_h - overlap_tiles

#         batches = []
#         seen = set()
#         for y0 in range(0, rows, step_y):
#             for x0 in range(0, cols, step_x):
#                 # clamp window to grid boundaries
#                 x_start = min(x0, cols - batch_w)
#                 y_start = min(y0, rows - batch_h)

#                 key = (x_start, y_start)
#                 if key in seen:
#                     continue
#                 seen.add(key)

#                 # collect only tiles in this sub-grid
#                 sub = [
#                     t for t in tiles
#                     if (x_start <= t['x_index'] < x_start + batch_w)
#                     and (y_start <= t['y_index'] < y_start + batch_h)
#                 ]
#                 sub.sort(key=lambda t: (t['y_index'], t['x_index']))
#                 batches.append({
#                     'x_start': x_start,
#                     'y_start': y_start,
#                     'tiles': sub
#                 })
#         return batches


#     def perform_channel_alignment(self, msims, do_align, tile_ix=0):
#         key = 'affine_metadata'
#         if do_align:
#             key = 'affine_metadata_ch_reg'
#             # channel registration logic here
#         return msims, key

#     # def stitch_tiles(self, msims, transform_key):
#     #     new_key = 'affine_registered'
#     #     with dask.diagnostics.ProgressBar():
#     #         params = registration.register(
#     #             msims,
#     #             registration_binning={'y':1,'x':1},
#     #             reg_channel_index=0,
#     #             transform_key=transform_key,
#     #             new_transform_key=new_key,
#     #             pre_registration_pruning_method="keep_axis_aligned"
#     #         )
#     #     return params, new_key
    
#     def stitch_tiles(self, msims, transform_key, scheduler='threads'): # carrot changed for latest
#         """
#         Run tile-to-tile affine registration, store under 'affine_registered'.
#         Use local threads scheduler by default to avoid sending large graphs.
#         """
#         new_key = 'affine_registered'
#         with dask.diagnostics.ProgressBar():
#             params = registration.register(
#                 msims,
#                 registration_binning={'y':1,'x':1},
#                 reg_channel_index=0,
#                 transform_key=transform_key,
#                 new_transform_key=new_key,
#                 pre_registration_pruning_method="keep_axis_aligned"
#             )
#         return params, new_key    

#     def fuse_and_export(self, msims, out_folder, export_tiff=False):
#         out = Path(out_folder)
#         out.mkdir(exist_ok=True, parents=True)
#         name = out.name
#         zarr_out = out / f"{name}.zarr"
#         fused = fusion.fuse([
#             msi_utils.get_sim_from_msim(m) for m in msims
#         ], transform_key='affine_registered', output_chunksize=256)
#         with dask.diagnostics.ProgressBar():
#             ngff_utils.write_sim_to_ome_zarr(fused, str(zarr_out), overwrite=True)
#         if export_tiff:
#             tif_out = out / f"{name}.tif"
#             with dask.diagnostics.ProgressBar():
#                 io.save_sim_as_tif(str(tif_out), fused)


# ############### EXTRA ############################# Cilantro. Added just before carrot

#     @staticmethod
#     def _load_msim_from_zarr(zarr_path: str):
#         """Read a SIM from disk and wrap it as a multiscale image."""
#         sim = ngff_utils.read_sim_from_ome_zarr(zarr_path)
#         return msi_utils.get_msim_from_sim(sim)

#     def stitch_from_zarr(self, zarr_paths, transform_key):
#         """
#         Instead of taking MSIM objects, take the list of tile .zarr paths (small!)
#         and do the registration entirely on the workers.
#         """
#         # 1) Build a list of *delayed* MSIMs so only the zarr_paths list crosses the wire
#         msims_delayed = [
#             delayed(self._load_msim_from_zarr)(str(p))
#             for p in zarr_paths
#         ]

#         # 2) Perform registration on those delayed MSIMs
#         new_key = 'affine_registered'
#         with dask.diagnostics.ProgressBar():
#             params = registration.register(
#                 msims_delayed,
#                 registration_binning={'y':1,'x':1},
#                 reg_channel_index=0,
#                 transform_key=transform_key,
#                 new_transform_key=new_key,
#                 pre_registration_pruning_method="keep_axis_aligned",
#             )
#         return params, new_key

# ################# CARROT ##############################

class Stitcher:
    """
    A class to process tiled TIFF datasets: extract metadata, convert to Zarr, stitch, and export.
    """
    def __init__(self, master_folder):
        """
        Initialize with a master directory containing multiple 'Condition_' folders.

        Parameters:
          - master_folder: Path or str to a directory containing condition subfolders
        """
        self.master_folder = Path(master_folder)
        self.conditions = sorted([
            p for p in self.master_folder.iterdir()
            if p.is_dir() and p.name.startswith("Condition_")
        ])

    # def extract_metadata(self, condition):
    #     """
    #     Extract tiling metadata for a given condition folder.

    #     Returns a dict with keys: overlap, region_origin, region_size,
    #     grid_size, tile_shape, tiles
    #     """
    #     cond_folder = Path(condition)
    #     if not cond_folder.is_dir():
    #         cond_folder = self.master_folder / condition
    #     info_file = next(cond_folder.glob("*forVSIimages.omp2info"))
    #     tif_folder = cond_folder

    #     def strip_ns(tag): return tag.split("}")[-1]
    #     tree = ET.parse(info_file)
    #     root = tree.getroot()

    #     overlap_pct = float(next(e for e in root.iter() if strip_ns(e.tag) == "overlap").text)
    #     overlap = {"x": overlap_pct/100, "y": overlap_pct/100}

    #     coords = next(e for e in root.iter() if strip_ns(e.tag) == "coordinates")
    #     region_origin = {"x": float(coords.attrib["x"]), "y": float(coords.attrib["y"]) }
    #     region_size   = {"width": float(coords.attrib["width"]), "height": float(coords.attrib["height"]) }

    #     num_x = int(next(e for e in root.iter() if strip_ns(e.tag) == "numOfXAreas").text)
    #     num_y = int(next(e for e in root.iter() if strip_ns(e.tag) == "numOfYAreas").text)

    #     tif_stems = {p.stem for p in tif_folder.glob("*.tif")}
    #     tiles = []
    #     for area in (e for e in root.iter() if strip_ns(e.tag) == "area"):
    #         img  = next(e for e in area if strip_ns(e.tag) == "image").text
    #         stem = Path(img).stem
    #         if stem in tif_stems:
    #             xidx = int(next(e for e in area if strip_ns(e.tag) == "xIndex").text)
    #             yidx = int(next(e for e in area if strip_ns(e.tag) == "yIndex").text)
    #             tiles.append({"filename": f"{stem}.tif", "x_index": xidx, "y_index": yidx})
    #     tiles.sort(key=lambda t: (t["y_index"], t["x_index"]))

    #     first_tif = tif_folder / tiles[0]["filename"]
    #     arr = tifffile.imread(first_tif)
    #     if arr.ndim == 4:
    #         z, c, y, x = arr.shape
    #     else:
    #         z, y, x = arr.shape
    #     tile_shape = {"z": z, "y": y, "x": x}




    #     return {
    #         "overlap": overlap,
    #         "region_origin": region_origin,
    #         "region_size": region_size,
    #         "grid_size": {"cols": num_x, "rows": num_y},
    #         "tile_shape": tile_shape,
    #         "tiles": tiles
    #     }

    @staticmethod
    def _estimate_max_subgrid(cmd_folder,
                              safety_fraction: float,
                              overhead_factor: float) -> int:
        """
        Inspect one TIFF in `cmd_folder` and
        return the largest square side (in tiles)
        that fits into RAM.
        """
        # 1) pick a sample TIFF
        sample = next(Path(cmd_folder).glob("*.tif"), None)
        if sample is None:
            raise RuntimeError(f"No TIFFs in {cmd_folder}")
        arr = tifffile.imread(sample)
        # unpack dims
        if arr.ndim == 4:
            z, c, y, x = arr.shape
        elif arr.ndim == 3:
            z, y, x = arr.shape; c = 1
        else:
            raise RuntimeError(f"Unrecognized shape {arr.shape}")
        bytes_per_tile = z * y * x * c * arr.dtype.itemsize
        avail = psutil.virtual_memory().available * safety_fraction
        eff = bytes_per_tile * overhead_factor
        max_tiles = int(avail // eff)
        side = int(np.floor(np.sqrt(max_tiles)))
        return max(1, side)

    def extract_metadata(self, condition):
        """
        Extract tiling metadata for a given condition folder.

        Returns a dict with keys: overlap, region_origin, region_size,
        grid_size, tile_shape, tiles
        """
        cond_folder = Path(condition)
        if not cond_folder.is_dir():
            cond_folder = self.master_folder / condition
        info_file = next(cond_folder.glob("*forVSIimages.omp2info"))
        tif_folder = cond_folder

        def strip_ns(tag): return tag.split("}")[-1]
        tree = ET.parse(info_file)
        root = tree.getroot()

        overlap_pct = float(next(e for e in root.iter() if strip_ns(e.tag) == "overlap").text)
        overlap = {"x": overlap_pct/100, "y": overlap_pct/100}

        coords = next(e for e in root.iter() if strip_ns(e.tag) == "coordinates")
        region_origin = {"x": float(coords.attrib["x"]), "y": float(coords.attrib["y"]) }
        region_size   = {"width": float(coords.attrib["width"]), "height": float(coords.attrib["height"]) }

        num_x = int(next(e for e in root.iter() if strip_ns(e.tag) == "numOfXAreas").text)
        num_y = int(next(e for e in root.iter() if strip_ns(e.tag) == "numOfYAreas").text)

        tif_stems = {p.stem for p in tif_folder.glob("*.tif")}
        tiles = []
        for area in (e for e in root.iter() if strip_ns(e.tag) == "area"):
            img  = next(e for e in area if strip_ns(e.tag) == "image").text
            stem = Path(img).stem
            if stem in tif_stems:
                xidx = int(next(e for e in area if strip_ns(e.tag) == "xIndex").text)
                yidx = int(next(e for e in area if strip_ns(e.tag) == "yIndex").text)
                tiles.append({"filename": f"{stem}.tif", "x_index": xidx, "y_index": yidx})
        tiles.sort(key=lambda t: (t["y_index"], t["x_index"]))


        # — Diagnostic: compare on‐disk TIFFs vs. XML entries — # NEW
        all_tifs = {p.name for p in tif_folder.glob("*.tif")}
        referenced = {f"{t['filename']}" for t in tiles}
        missing_in_xml  = sorted(all_tifs - referenced)
        missing_on_disk = sorted(referenced - all_tifs)
        if missing_in_xml:
            print(f"⚠️ {len(missing_in_xml)} TIFF(s) in {condition} not in XML:",
                missing_in_xml)
        if missing_on_disk:
            print(f"⚠️ {len(missing_on_disk)} XML entries missing .tif on disk:",
                missing_on_disk)

        first_tif = tif_folder / tiles[0]["filename"]
        arr = tifffile.imread(first_tif)
        if arr.ndim == 4:
            z, c, y, x = arr.shape
        else:
            z, y, x = arr.shape

        tile_shape = {"z": z, "y": y, "x": x}

#################
        if cfg.BATCH_GRID_SHAPE is None:
            bw = bh = self._estimate_max_subgrid(
                tif_folder,
                cfg.SAFETY_FRACTION,
                cfg.OVERHEAD_FACTOR
            )
            print(f"Auto‐chosen BATCH_GRID_SHAPE = ({bw},{bh})")
        else:
            bw, bh = cfg.BATCH_GRID_SHAPE
        tile_overlap = 1  # always 1‐tile border

        cols, rows = num_x, num_y
        nx = math.ceil(cols / bw)
        ny = math.ceil(rows / bh)

        subgrids = []
        for sgx in range(nx):
            for sgy in range(ny):
                # main window
                x0, y0 = sgx*bw, sgy*bh
                x1, y1 = min(cols, x0+bw), min(rows, y0+bh)
                # add 1‐tile overlap
                xo, yo = max(0, x0-tile_overlap), max(0, y0-tile_overlap)
                xe, ye = min(cols, x1+tile_overlap), min(rows, y1+tile_overlap)

                full = []
                for yy in range(yo, ye):
                    for xx in range(xo, xe):
                        # find matching tile or pad
                        t0 = next((t for t in tiles
                                if t["x_index"]==xx and t["y_index"]==yy), None)
                        entry = t0.copy() if t0 else {
                            "filename": None,
                            "x_index": xx,
                            "y_index": yy
                        }
                        full.append(entry)

                # extract the “A…_G…” prefix from any real tile
                prefix = None
                for t in full:
                    if t["filename"]:
                        m = re.match(r"(.+__A\d+_G\d+)_\d+", Path(t["filename"]).stem)
                        if m:
                            prefix = m.group(1)
                            break

                subgrids.append({
                    "id":    f"{prefix or 'sub'}_sg{sgx}_{sgy}",
                    "tiles": full,
                    "x_range": (x0,x1),
                    "y_range": (y0,y1)
                })

        # — Diagnostics: see if any subgrid is empty or padded —
        for sg in subgrids:
            real = [t for t in sg["tiles"] if t["filename"]] # NEW CILANTRO
            total = len(sg["tiles"])
            if not real:
                print(f"⚠️  Subgrid {sg['id']} is EMPTY (0 real tiles).")
            elif len(real) < total:
                print(f"ℹ️  Subgrid {sg['id']} has {len(real)}/{total} real tiles (padded {total-len(real)})")


        meta = {
            "overlap": overlap,
            "region_origin": region_origin,
            "region_size": region_size,
            "grid_size": {"cols": cols, "rows": rows},
            "tile_shape": tile_shape,
            "tiles": tiles,
            "subgrids": subgrids,
            "num_subgrids": {"cols": nx, "rows": ny}
        }
        return meta





# ######################
#         # a
#         meta = {
#             "overlap": { "x": overlap_pct/100, "y": overlap_pct/100 },
#             "region_origin": region_origin,
#             "region_size": region_size,
#             "grid_size": {"cols": num_x, "rows": num_y},
#             "tile_shape": tile_shape,
#             "tiles": tiles
#         }

#         # build static sub-grid definitions
#         cols, rows = num_x, num_y
#         bw, bh = cfg.BATCH_GRID_SHAPE
#         tile_overlap = cfg.BATCH_TILE_OVERLAP

#         subgrids = []
#         nx = math.ceil(cols/bw)
#         ny = math.ceil(rows/bh)
#         for sgx in range(nx):
#             for sgy in range(ny):
#                 x0, y0 = sgx*bw,      sgy*bh
#                 x1, y1 = min(cols,x0+bw), min(rows,y0+bh)
#                 xo, yo = max(0, x0-tile_overlap), max(0, y0-tile_overlap)
#                 xe, ye = min(cols, x1+tile_overlap), min(rows, y1+tile_overlap)

#                 tiles_in_batch = [
#                     t for t in tiles
#                     if xo <= t["x_index"] < xe
#                     and yo <= t["y_index"] < ye
#                 ]
#                 tiles_in_batch.sort(key=lambda t:(t["y_index"],t["x_index"]))
#                 subgrids.append({
#                     "id": f"{sgx}_{sgy}",
#                     "x_range": (x0, x1),
#                     "y_range": (y0, y1),
#                     "tiles": tiles_in_batch
#                 })

#         meta["subgrids"] = subgrids
#         meta["num_subgrids"] = {"cols": nx, "rows": ny}

#         return meta

    @staticmethod
    def load_maxproj_images(folder):
        """
        Generator yielding (filename, max_projection) for each TIFF in `folder`.
        """
        folder = Path(folder)
        tif_paths = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
        for path in tif_paths:
            stack = tifffile.imread(path)
            if stack.ndim == 4:
                channel0 = stack[:, 0, ...]
            elif stack.ndim == 3:
                channel0 = stack
            else:
                raise ValueError(f"Unexpected image dims {stack.shape} for {path.name}")
            yield path.name, np.max(channel0, axis=0)

    @staticmethod
    def get_tile_grid_position_from_tile_index(tile_index, num_cols, num_rows=None):
        """
        Convert flat tile_index to grid coords (z,y,x).
        """
        return {'z': 0, 'y': tile_index // num_cols, 'x': tile_index % num_cols}

    @staticmethod
    # def convert_tiles_to_zarr(tiles, tif_folder, scale, translations,
    #                           overwrite=True, use_gpu=False):
        
    def convert_tiles_to_zarr(tiles, #cilantro
                               tif_folder,
                               scale,
                               translations,
                               tile_shape,
                               subgrid_id: str = None,
                               overwrite=True,
                               use_gpu=False):
        """
        Convert tile TIFFs to OME-Zarr using same basename.
        If use_gpu=True, wrap arrays in CuPy via Dask.
        """
        tif_folder = Path(tif_folder)
        # zarr_folder = tif_folder
        zarr_folder = Path(tif_folder) #cilantro
        print(f"→ [Subgrid {subgrid_id}] writing {len(tiles)} tiles to {zarr_folder}") #cilantro
        zarr_paths, msims = [], []
################### OLD down ######################################
        # for idx, tile in enumerate(tiles):
        #     tif_path = tif_folder / tile['filename']
        #     arr = tifffile.imread(tif_path)
        #     if arr.ndim == 3:
        #         arr = arr[None, ...]
        #     elif arr.ndim == 4:
        #         arr = arr.transpose(1, 0, 2, 3)
        #     else:
        #         raise ValueError(f"Unexpected arr shape {arr.shape} for {tif_path.name}")

        #     if use_gpu:
                
        #         darr = da.from_array(arr, chunks="auto", asarray=cupy.asarray)
        #     else:
        #         darr = da.from_array(arr, chunks="auto")

        #     sim = si_utils.get_sim_from_array(
        #         darr, dims=["c","z","y","x"],
        #         scale=scale,
        #         translation=translations[idx],
        #         transform_key=io.METADATA_TRANSFORM_KEY
        #     )

        #     stem = Path(tile['filename']).stem

        #     name = f"{stem}"
        #     if subgrid_id:
        #         name += f"_{subgrid_id}"
        #     zarr_path = zarr_folder / f"{name}.zarr" #new
        #     # zarr_path = zarr_folder / f"{stem}.zarr" #old 

        #     ngff_utils.write_sim_to_ome_zarr(sim, str(zarr_path), overwrite=overwrite)

        #     sim2 = ngff_utils.read_sim_from_ome_zarr(str(zarr_path))
        #     msim = msi_utils.get_msim_from_sim(sim2)

        #     zarr_paths.append(str(zarr_path))
        #     msims.append(msim)
###################### OLD UP ########################################    
        for idx, tile in enumerate(tiles):
            xi, yi = tile["x_index"], tile["y_index"]

            if tile["filename"] is None:
                # This was a padded placeholder:
                print(f"   • Padding blank tile at ({xi},{yi}) in subgrid {subgrid_id}")
                # Create a black volume of shape (z, y, x)
                z, y, x = tile_shape["z"], tile_shape["y"], tile_shape["x"]
                arr = np.zeros((z, y, x), dtype=np.uint16)
                # Use a special stem so it never collides
                stem = f"blank_{xi}_{yi}"
            else:
                # A real TIFF → load and report
                tif_path = tif_folder / tile["filename"]
                print(f"   • Reading {tile['filename']} for subgrid {subgrid_id}")
                arr = tifffile.imread(tif_path)
                # remember to expand/trans­pose exactly as before
                if arr.ndim == 3:
                    arr = arr[None, ...]
                elif arr.ndim == 4:
                    arr = arr.transpose(1, 0, 2, 3)
                else:
                    raise ValueError(f"Unexpected arr shape {arr.shape} for {tif_path.name}")
                stem = Path(tile["filename"]).stem

            # GPU or CPU wrapping (unchanged)
            if use_gpu:
                import cupy
                darr = da.from_array(arr, chunks="auto", asarray=cupy.asarray)
            else:
                darr = da.from_array(arr, chunks="auto")

            sim = si_utils.get_sim_from_array(
                darr,
                dims=["c","z","y","x"],
                scale=scale,
                translation=translations[idx],
                transform_key=io.METADATA_TRANSFORM_KEY
            )

            # build a unique zarr name
            name = stem
            if subgrid_id:
                # this ensures no two subgrids overwrite each other
                name += f"_{subgrid_id}"
            zarr_path = zarr_folder / f"{name}.zarr"

            # write & read back as before
            ngff_utils.write_sim_to_ome_zarr(sim, str(zarr_path), overwrite=overwrite)
            sim2 = ngff_utils.read_sim_from_ome_zarr(str(zarr_path))
            msim = msi_utils.get_msim_from_sim(sim2)

            zarr_paths.append(str(zarr_path))
            msims.append(msim)
        
                    # Option A: persist (turns each D
        return zarr_paths, msims

    def compute_translations(self, meta, scale):
        tiles, overlap, ts = meta['tiles'], meta['overlap'], meta['tile_shape']
        transs = []
        for t in tiles:
            tx = t['x_index']*(1-overlap['x'])*ts['x']*scale['x']
            ty = t['y_index']*(1-overlap['y'])*ts['y']*scale['y']
            transs.append({'x':tx,'y':ty,'z':0.0})
        return transs
    
    def split_tiles_into_batches(self,
                                 meta: dict,
                                 batch_shape: tuple[int,int],
                                 overlap_tiles: int = 1) -> list[dict]:
        """
        Breaks the full meta['tiles'] (flattened, row-major) into
        sub-grid “batches” of size batch_shape, stepped so that
        each window overlaps its neighbors by overlap_tiles.

        Returns a list of dicts, each with:
          - 'x_start','y_start': the top-left tile coords of the window
          - 'tiles':       the list of tile-dicts in that window
        """
        cols = meta['grid_size']['cols']
        rows = meta['grid_size']['rows']
        tiles = meta['tiles']

        batch_w, batch_h = batch_shape
        step_x = batch_w - overlap_tiles
        step_y = batch_h - overlap_tiles

        batches = []
        seen = set()
        for y0 in range(0, rows, step_y):
            for x0 in range(0, cols, step_x):
                # clamp window to grid boundaries
                x_start = min(x0, cols - batch_w)
                y_start = min(y0, rows - batch_h)

                key = (x_start, y_start)
                if key in seen:
                    continue
                seen.add(key)

                # collect only tiles in this sub-grid
                sub = [
                    t for t in tiles
                    if (x_start <= t['x_index'] < x_start + batch_w)
                    and (y_start <= t['y_index'] < y_start + batch_h)
                ]
                sub.sort(key=lambda t: (t['y_index'], t['x_index']))
                batches.append({
                    'x_start': x_start,
                    'y_start': y_start,
                    'tiles': sub
                })
        return batches


    def perform_channel_alignment(self, msims, do_align, tile_ix=0):
        key = 'affine_metadata'
        if do_align:
            key = 'affine_metadata_ch_reg'
            # channel registration logic here
        return msims, key

    # def stitch_tiles(self, msims, transform_key):
    #     new_key = 'affine_registered'
    #     with dask.diagnostics.ProgressBar():
    #         params = registration.register(
    #             msims,
    #             registration_binning={'y':1,'x':1},
    #             reg_channel_index=0,
    #             transform_key=transform_key,
    #             new_transform_key=new_key,
    #             pre_registration_pruning_method="keep_axis_aligned"
    #         )
    #     return params, new_key
    
    def stitch_tiles(self, msims, transform_key, scheduler='threads'): # carrot changed for latest
        """
        Run tile-to-tile affine registration, store under 'affine_registered'.
        Use local threads scheduler by default to avoid sending large graphs.
        """
        new_key = 'affine_registered'
        with dask.diagnostics.ProgressBar():
            params = registration.register(
                msims,
                registration_binning={'y':1,'x':1},
                reg_channel_index=0,
                transform_key=transform_key,
                new_transform_key=new_key,
                pre_registration_pruning_method="keep_axis_aligned"
            )
        return params, new_key    

    def fuse_and_export(self, msims, out_folder, export_tiff=False):
        out = Path(out_folder)
        out.mkdir(exist_ok=True, parents=True)
        name = out.name
        zarr_out = out / f"{name}.zarr"
        fused = fusion.fuse([
            msi_utils.get_sim_from_msim(m) for m in msims
        ], transform_key='affine_registered', output_chunksize=256)
        with dask.diagnostics.ProgressBar():
            ngff_utils.write_sim_to_ome_zarr(fused, str(zarr_out), overwrite=True)
        if export_tiff:
            tif_out = out / f"{name}.tif"
            with dask.diagnostics.ProgressBar():
                io.save_sim_as_tif(str(tif_out), fused)


############### EXTRA ############################# Cilantro. Added just before carrot

    @staticmethod
    def _load_msim_from_zarr(zarr_path: str):
        """Read a SIM from disk and wrap it as a multiscale image."""
        sim = ngff_utils.read_sim_from_ome_zarr(zarr_path)
        return msi_utils.get_msim_from_sim(sim)

    def stitch_from_zarr(self, zarr_paths, transform_key):
        """
        Instead of taking MSIM objects, take the list of tile .zarr paths (small!)
        and do the registration entirely on the workers.
        """
        # 1) Build a list of *delayed* MSIMs so only the zarr_paths list crosses the wire
        msims_delayed = [
            delayed(self._load_msim_from_zarr)(str(p))
            for p in zarr_paths
        ]

        # 2) Perform registration on those delayed MSIMs
        new_key = 'affine_registered'
        with dask.diagnostics.ProgressBar():
            params = registration.register(
                msims_delayed,
                registration_binning={'y':1,'x':1},
                reg_channel_index=0,
                transform_key=transform_key,
                new_transform_key=new_key,
                pre_registration_pruning_method="keep_axis_aligned",
            )
        return params, new_key

################# CARROT ##############################

    def group_tiles_by_prefix(self, meta):
        """
        Returns a dict mapping each unique “A…_G…” prefix
        to the list of tile-dicts that share it.
        """
        groups = {}
        p = re.compile(r"(.+__A\d+_G\d+)_\d+")
        for t in meta["tiles"]:
            fn = t.get("filename") or ""
            m = p.match(Path(fn).stem)
            if m:
                groups.setdefault(m.group(1), []).append(t)
        return groups

    # def build_group_meta(self, meta, group_tiles, prefix):
    #     """
    #     Build a “mini-meta” for just those tiles in `group_tiles`,
    #     splitting *that* into subgrids and tagging each with an 'id'.
    #     """
    #     # --- 1) new grid_size based on x_index,y_index extents
    #     max_x = max(t["x_index"] for t in group_tiles) + 1
    #     max_y = max(t["y_index"] for t in group_tiles) + 1

    #     # --- 2) shallow copy and overwrite only tiles & grid_size
    #     gm = { **meta }
    #     gm["tiles"]     = group_tiles
    #     gm["grid_size"] = {"cols": max_x, "rows": max_y}

    #     # --- 3) split into batches
    #     batches = self.split_tiles_into_batches(
    #         gm,
    #         batch_shape=cfg.BATCH_GRID_SHAPE,
    #         overlap_tiles=cfg.BATCH_TILE_OVERLAP
    #     )

    #     # --- 4) build subgrids list, each with an 'id'
    #     gm["subgrids"] = []
    #     bw, bh = cfg.BATCH_GRID_SHAPE
    #     for i, b in enumerate(batches):
    #         x0, y0 = b["x_start"], b["y_start"]
    #         x1, y1 = min(max_x, x0 + bw), min(max_y, y0 + bh)
    #         gm["subgrids"].append({
    #             "id":      f"{prefix}_sg{i}",
    #             "tiles":   b["tiles"],
    #             "x_range": (x0, x1),
    #             "y_range": (y0, y1),
    #         })

    #     # --- 5) recalc num_subgrids
    #     gm["num_subgrids"] = {
    #         "cols": math.ceil(max_x / bw),
    #         "rows": math.ceil(max_y / bh)
    #     }

    #     return gm


    # def build_dynamic_group_meta(self,
    #                              meta,
    #                              group_tiles,
    #                              prefix,
    #                              tif_folder):
    #     """
    #     Build a mini‐meta for the tiles in this G‐group, splitting the
    #     rectangular grid into windows either:
    #       • of size cfg.BATCH_GRID_SHAPE (if set), or
    #       • of size S×S estimated by RAM (if BATCH_GRID_SHAPE is None)
    #     with cfg.BATCH_TILE_OVERLAP tiles of overlap.
    #     """
    #     # 1) compute raw grid dimensions
    #     grid_w = max(t["x_index"] for t in group_tiles) + 1
    #     grid_h = max(t["y_index"] for t in group_tiles) + 1

    #     # 2) decide your window size
    #     if cfg.BATCH_GRID_SHAPE is None:
    #         S = self._estimate_max_subgrid(Path(tif_folder),
    #                                        cfg.SAFETY_FRACTION,
    #                                        cfg.OVERHEAD_FACTOR)
    #         w_req = h_req = S
    #     else:
    #         w_req, h_req = cfg.BATCH_GRID_SHAPE

    #     overlap = cfg.BATCH_TILE_OVERLAP or 1
    #     step_x  = w_req - overlap
    #     step_y  = h_req - overlap

    #     # 3) how many windows needed?
    #     n_x = math.ceil((grid_w - overlap) / step_x)
    #     n_y = math.ceil((grid_h - overlap) / step_y)

    #     # 4) generate windows, nudging edges to stay full‐sized
    #     windows = []
    #     for ix in range(n_x):
    #         for iy in range(n_y):
    #             x0 = ix * step_x
    #             y0 = iy * step_y
    #             x1 = min(x0 + w_req, grid_w)
    #             y1 = min(y0 + h_req, grid_h)

    #             # if at right/bottom edge and window got clipped, shift back
    #             if (x1 - x0) < w_req:
    #                 x0 = max(0, grid_w - w_req)
    #             if (y1 - y0) < h_req:
    #                 y0 = max(0, grid_h - h_req)

    #             windows.append((x0, x1, y0, y1))

    #     # 5) build your subgrids list
    #     subgrids = []
    #     for i, (x0, x1, y0, y1) in enumerate(windows):
    #         tiles_in_batch = [
    #             t for t in group_tiles
    #             if x0 <= t["x_index"] < x1 and y0 <= t["y_index"] < y1
    #         ]
    #         subgrids.append({
    #             "id":      f"{prefix}_sg{i}",
    #             "tiles":   tiles_in_batch,
    #             "x_range": (x0, x1),
    #             "y_range": (y0, y1),
    #         })

    #     # 6) pack it all into a mini‐meta
    #     return {
    #         **meta,
    #         "tiles":        group_tiles,
    #         "grid_size":    {"cols": grid_w,   "rows": grid_h},
    #         "subgrids":     subgrids,
    #         "num_subgrids": {"cols": n_x,       "rows": n_y    }
    #     }

    def build_dynamic_group_meta(self,
                                 meta,
                                 group_tiles,
                                 prefix,
                                 tif_folder):
        """
        Split a rectangular grid (grid_w × grid_h in tiles) into
        windows roughly of size cfg.BATCH_GRID_SHAPE, with cfg.BATCH_TILE_OVERLAP
        overlap, shrinking only the edge‐windows to exactly fill the remainder.
        """
        # 1) grid dimensions in tiles
        grid_w = max(t["x_index"] for t in group_tiles) + 1
        grid_h = max(t["y_index"] for t in group_tiles) + 1

        # 2) read desired window size or fallback to S×S
        if cfg.BATCH_GRID_SHAPE is None:
            S = self._estimate_max_subgrid(Path(tif_folder),
                                           cfg.SAFETY_FRACTION,
                                           cfg.OVERHEAD_FACTOR)
            w_req = h_req = S
        else:
            w_req, h_req = cfg.BATCH_GRID_SHAPE

        overlap = cfg.BATCH_TILE_OVERLAP or 1
        step_x  = w_req - overlap
        step_y  = h_req - overlap

        # 3) build X origins: start at 0, then step_x, … until the last window touches the right edge
        origins_x = []
        x0 = 0
        while True:
            origins_x.append(x0)
            if x0 + w_req >= grid_w:
                break
            x0 += step_x

        # 4) same for Y
        origins_y = []
        y0 = 0
        while True:
            origins_y.append(y0)
            if y0 + h_req >= grid_h:
                break
            y0 += step_y

        # 5) assemble windows & shrink edges
        subgrids = []
        for i, x0 in enumerate(origins_x):
            for j, y0 in enumerate(origins_y):
                # window spans [x0, x0+w_req) but clipped to grid_w
                x1 = min(x0 + w_req, grid_w)
                y1 = min(y0 + h_req, grid_h)
                tiles_in_batch = [
                    t for t in group_tiles
                    if x0 <= t["x_index"] < x1
                    and y0 <= t["y_index"] < y1
                ]
                subgrids.append({
                    "id":      f"{prefix}_sg{i}_{j}",
                    "tiles":   tiles_in_batch,
                    "x_range": (x0, x1),
                    "y_range": (y0, y1),
                })

        # 6) return mini‐meta
        return {
            **meta,
            "tiles":        group_tiles,
            "grid_size":    {"cols": grid_w,   "rows": grid_h},
            "subgrids":     subgrids,
            "num_subgrids": {"cols": len(origins_x),
                             "rows": len(origins_y)}
        }

    

def plot_full_and_subgrid(meta, scale, subgrid, ax=None):
    """
    meta:    the full metadata dict returned by extract_metadata()
    scale:   same scale dict you pass to compute_translations()
    subgrid: one element of meta['subgrids']
    """
    # 1) compute translations for *all* tiles
    all_trans = st.compute_translations(meta, scale)
    # 2) compute translations for *this* subgrid
    sub_meta = { **meta, "tiles": subgrid["tiles"] }
    sub_trans = st.compute_translations(sub_meta, scale)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))

    # 3) plot all in light grey
    xs = [t['x'] for t in all_trans]
    ys = [t['y'] for t in all_trans]
    ax.scatter(xs, ys, marker='s', s=50, color='lightgrey', alpha=0.6, label='all tiles')

    # 4) overlay this subgrid in color
    xs2 = [t['x'] for t in sub_trans]
    ys2 = [t['y'] for t in sub_trans]
    ax.scatter(xs2, ys2, marker='s', s=80, color='C1', label=f"subgrid {subgrid['id']}")

    # 5) annotate each subgrid tile with its “frame number”
    for tile, t in zip(subgrid["tiles"], sub_trans):
        stem = Path(tile["filename"] or "").stem
        m = re.search(r'_(\d+)$', stem)
        num = m.group(1) if m else stem
        ax.text(t['x'], t['y'], num, ha='center', va='center', fontsize=6, color='black')

    # 6) draw a rectangle around the subgrid footprint
    #    using its x_range,y_range (tile‐indices) → corner positions:
    ox, oy = meta['overlap']['x'], meta['overlap']['y']
    ts = meta['tile_shape']
    w = (subgrid['x_range'][1] - subgrid['x_range'][0]) * (1-ox) * ts['x'] * scale['x']
    h = (subgrid['y_range'][1] - subgrid['y_range'][0]) * (1-oy) * ts['y'] * scale['y']
    x0 = subgrid['x_range'][0] * (1-ox) * ts['x'] * scale['x']
    y0 = subgrid['y_range'][0] * (1-oy) * ts['y'] * scale['y']
    # rect = plt.Rectangle((x0,y0), w, h, 
    #                      edgecolor='C1', facecolor='none', lw=2)
    # ax.add_patch(rect)

    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title(f"Full grid (grey) + subgrid {subgrid['id']}")
    plt.tight_layout()
    return ax
