

########################### CARROT #################################

import os
import zipfile
import json
import pprint
import xml.etree.ElementTree as ET
from pathlib import Path
import tifffile
import numpy as np
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

        first_tif = tif_folder / tiles[0]["filename"]
        arr = tifffile.imread(first_tif)
        if arr.ndim == 4:
            z, c, y, x = arr.shape
        else:
            z, y, x = arr.shape
        tile_shape = {"z": z, "y": y, "x": x}

        return {
            "overlap": overlap,
            "region_origin": region_origin,
            "region_size": region_size,
            "grid_size": {"cols": num_x, "rows": num_y},
            "tile_shape": tile_shape,
            "tiles": tiles
        }

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
    def convert_tiles_to_zarr(tiles, tif_folder, scale, translations,
                              overwrite=True, use_gpu=False):
        """
        Convert tile TIFFs to OME-Zarr using same basename.
        If use_gpu=True, wrap arrays in CuPy via Dask.
        """
        tif_folder = Path(tif_folder)
        zarr_folder = tif_folder
        zarr_paths, msims = [], []

        for idx, tile in enumerate(tiles):
            tif_path = tif_folder / tile['filename']
            arr = tifffile.imread(tif_path)
            if arr.ndim == 3:
                arr = arr[None, ...]
            elif arr.ndim == 4:
                arr = arr.transpose(1, 0, 2, 3)
            else:
                raise ValueError(f"Unexpected arr shape {arr.shape} for {tif_path.name}")

            if use_gpu:
                import cupy
                darr = da.from_array(arr, chunks="auto", asarray=cupy.asarray)
            else:
                darr = da.from_array(arr, chunks="auto")

            sim = si_utils.get_sim_from_array(
                darr, dims=["c","z","y","x"],
                scale=scale,
                translation=translations[idx],
                transform_key=io.METADATA_TRANSFORM_KEY
            )

            stem = Path(tile['filename']).stem
            zarr_path = zarr_folder / f"{stem}.zarr"
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

