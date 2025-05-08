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
        # find all condition folders prefixed with 'Condition_'
        self.conditions = sorted([
            p for p in self.master_folder.iterdir()
            if p.is_dir() and p.name.startswith("Condition_")
        ])

    def extract_metadata(self, condition):
        """
        Extract tiling metadata from the .omp2info file and TIFF folder for a given condition.

        Parameters:
          - condition: folder name (str) or Path under the master folder

        Returns:
          - meta: dict with keys 'overlap', 'region_origin', 'region_size',
                  'grid_size', 'tile_shape', 'tiles'
        """
        # resolve condition folder
        cond_folder = Path(condition)
        if not cond_folder.is_dir():
            cond_folder = self.master_folder / condition
        # find the .omp2info metadata file
        info_file = next(cond_folder.glob("*forVSIimages.omp2info"))
        tif_folder = cond_folder

        def strip_ns(tag):
            return tag.split("}")[-1]

        # parse XML
        tree = ET.parse(info_file)
        root = tree.getroot()

        # overlap fraction (same X & Y)
        overlap_pct = float(next(e for e in root.iter() if strip_ns(e.tag) == "overlap").text)
        overlap = {"x": overlap_pct/100, "y": overlap_pct/100}

        # region origin & size
        coords = next(e for e in root.iter() if strip_ns(e.tag) == "coordinates")
        region_origin = {"x": float(coords.attrib["x"]), "y": float(coords.attrib["y"]) }
        region_size   = {"width": float(coords.attrib["width"]), "height": float(coords.attrib["height"]) }

        # grid dimensions
        num_x = int(next(e for e in root.iter() if strip_ns(e.tag) == "numOfXAreas").text)
        num_y = int(next(e for e in root.iter() if strip_ns(e.tag) == "numOfYAreas").text)

        # collect available TIFF tiles
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

        # infer tile shape from first TIFF
        first_tif = tif_folder / tiles[0]["filename"]
        arr = tifffile.imread(first_tif)
        if arr.ndim == 4:
            z, c, y, x = arr.shape
        else:
            z, y, x = arr.shape
        tile_shape = {"z": z, "y": y, "x": x}

        # assemble metadata dict
        meta = {
            "overlap": overlap,
            "region_origin": region_origin,
            "region_size": region_size,
            "grid_size": {"cols": num_x, "rows": num_y},
            "tile_shape": tile_shape,
            "tiles": tiles
        }
        return meta

    @staticmethod
    def load_maxproj_images(folder):
        """
        Generator that yields (filename, max_proj) for each TIFF stack in `folder`.
        - If the stack is 4D, assumes shape (Z, C, Y, X) and takes C=0.
        - If the stack is 3D, assumes shape (Z, Y, X) and treats it as single‐channel.
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
                raise ValueError(f"Unexpected image dimensions {stack.shape} for {path.name}")
            max_proj = np.max(channel0, axis=0)
            yield path.name, max_proj

    @staticmethod
    def get_tile_grid_position_from_tile_index(tile_index, num_cols, num_rows=None):
        """
        Given a flat tile_index and grid shape, returns its (z, y, x) grid coordinates.
        """
        if num_rows is None:
            num_rows = (tile_index // num_cols) + 1  # not used explicitly here
        return {
            'z': 0,
            'y': tile_index // num_cols,
            'x': tile_index % num_cols,
        }

    # @staticmethod
    # def convert_tiles_to_zarr(tiles, tif_folder, zarr_folder, scale, translations, overwrite=True):
    #     """
    #     Convert a list of tile definitions to OME-Zarr, preserving full 3D and all channels.
    #     """
    #     tif_folder = Path(tif_folder)
    #     zarr_folder = Path(zarr_folder)
    #     zarr_folder.mkdir(exist_ok=True, parents=True)
    #     zarr_paths = []
    #     msims = []
    #     for idx, tile in enumerate(tiles):
    #         tif_path = tif_folder / tile['filename']
    #         arr = tifffile.imread(tif_path)
    #         if arr.ndim == 3:
    #             arr = arr[None, ...]
    #         elif arr.ndim == 4:
    #             arr = arr.transpose(1, 0, 2, 3)
    #         else:
    #             raise ValueError(f"Unexpected array shape {arr.shape} for {tif_path.name}")
    #         darr = da.from_array(arr, chunks="auto")
    #         sim = si_utils.get_sim_from_array(
    #             darr,
    #             dims=["c", "z", "y", "x"],
    #             scale=scale,
    #             translation=translations[idx],
    #             transform_key=io.METADATA_TRANSFORM_KEY,
    #         )
    #         zarr_path = zarr_folder / f"tile_{idx:03d}.zarr"
    #         ngff_utils.write_sim_to_ome_zarr(sim, str(zarr_path), overwrite=overwrite)
    #         sim = ngff_utils.read_sim_from_ome_zarr(str(zarr_path))
    #         msim = msi_utils.get_msim_from_sim(sim)
    #         zarr_paths.append(str(zarr_path))
    #         msims.append(msim)
    #     return zarr_paths, msims
    
    @staticmethod
    def convert_tiles_to_zarr(tiles, tif_folder, scale, translations, overwrite=True):
        """
        Convert a list of tile definitions to OME-Zarr, preserving full 3D and all channels.
        Saves each Zarr using the same basename as its source TIFF.
        """
        tif_folder = Path(tif_folder)
        zarr_folder = tif_folder

        zarr_paths = []
        msims      = []

        for idx, tile in enumerate(tiles):
            tif_path = tif_folder / tile['filename']
            arr = tifffile.imread(tif_path)
            if arr.ndim == 3:
                arr = arr[None, ...]
            elif arr.ndim == 4:
                arr = arr.transpose(1, 0, 2, 3)
            else:
                raise ValueError(f"Unexpected array shape {arr.shape} for {tif_path.name}")

            darr = da.from_array(arr, chunks="auto")

            sim = si_utils.get_sim_from_array(
                darr,
                dims=["c", "z", "y", "x"],
                scale=scale,
                translation=translations[idx],
                transform_key=io.METADATA_TRANSFORM_KEY,
            )

            # build the zarr name to match the TIFF basename
            stem = Path(tile['filename']).stem
            zarr_path = zarr_folder / f"{stem}.zarr"
            ngff_utils.write_sim_to_ome_zarr(sim, str(zarr_path), overwrite=overwrite)

            sim  = ngff_utils.read_sim_from_ome_zarr(str(zarr_path))
            msim = msi_utils.get_msim_from_sim(sim)

            zarr_paths.append(str(zarr_path))
            msims.append(msim)

        return zarr_paths, msims

    

 

    def compute_translations(self, meta, scale):
        """Compute µm translations from meta + scale."""
        tiles      = meta['tiles']
        overlap    = meta['overlap']
        ts         = meta['tile_shape']
        translations = []
        for tile in tiles:
            tx = tile['x_index'] * (1 - overlap['x']) * ts['x'] * scale['x']
            ty = tile['y_index'] * (1 - overlap['y']) * ts['y'] * scale['y']
            translations.append({'x': tx, 'y': ty, 'z': 0.0})
        return translations

    def perform_channel_alignment(self, msims, do_align, tile_ix=0):
        """Optionally register channels within one tile, update each msim’s transform key."""
        key = 'affine_metadata'
        if do_align:
            key = 'affine_metadata_ch_reg'
            # … your channel-registration block here, writing transforms to each msim using key …
        return msims, key

    def stitch_tiles(self, msims, transform_key):
        """Run tile-to-tile affine registration, store under 'affine_registered'."""
        new_key = 'affine_registered'
        with dask.diagnostics.ProgressBar():
            params = registration.register(
                msims,
                registration_binning={'y':1,'x':1},
                reg_channel_index=0,
                transform_key=transform_key,
                new_transform_key=new_key,
                pre_registration_pruning_method="keep_axis_aligned",
            )
        return params, new_key

    def fuse_and_export(self, msims, out_folder, export_tiff=False):
        """Fuse registered tiles into one Zarr (and optional TIFF)."""
        out_folder = Path(out_folder)
        out_folder.mkdir(exist_ok=True, parents=True)
        cond_name = out_folder.name  # or extract from path
        zarr_path = out_folder / f"{cond_name}.zarr"
        fused = fusion.fuse(
            [msi_utils.get_sim_from_msim(m) for m in msims],
            transform_key='affine_registered',
            output_chunksize=256,
        )
        with dask.diagnostics.ProgressBar():
            ngff_utils.write_sim_to_ome_zarr(fused, str(zarr_path), overwrite=True)

        if export_tiff:
            tif_path = out_folder / f"{cond_name}.tif"
            with dask.diagnostics.ProgressBar():
                io.save_sim_as_tif(str(tif_path), fused)

