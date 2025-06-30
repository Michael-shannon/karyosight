

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


############

# PyImageJ

# import imagej
# import scyjava as sj

# import scyjava

# Tell the JVM to open java.lang to unnamed modules (required on newer Java)
# scyjava.config.add_option("--add-opens=java.base/java.lang=ALL-UNNAMED") #disabled honestly

# Initialize once (in headless or interactive mode as you prefer)
# IJ = imagej.init('sc.fiji:fiji', mode='headless')
# BF = sj.jimport('loci.plugins.BF')
# ImporterOptions = sj.jimport('loci.plugins.in.ImporterOptions')

###############


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

        # tif_stems = {p.stem for p in tif_folder.glob("*.tif")} # REMOVED 5-29-2025

        # support TIFF, TIFFF, and OIR
        img_paths = (
            sorted(tif_folder.glob("*.tif")) +
            sorted(tif_folder.glob("*.tiff")) +
            sorted(tif_folder.glob("*.oir"))
        )
        if not img_paths:
            raise RuntimeError(f"No .tif/.tiff/.oir files found in {tif_folder}")
        # map stem → actual filename
        stem_to_file = {p.stem: p.name for p in img_paths}
        img_stems = set(stem_to_file)
 #carrot cilantro

        # tiles = []
        tiles = []
        for area in (e for e in root.iter() if strip_ns(e.tag) == "area"):
            img_name = next(e for e in area if strip_ns(e.tag) == "image").text
            stem     = Path(img_name).stem
            if stem in img_stems:
                xidx = int(next(e for e in area if strip_ns(e.tag) == "xIndex").text)
                yidx = int(next(e for e in area if strip_ns(e.tag) == "yIndex").text)
                tiles.append({
                    "filename": stem_to_file[stem],
                    "x_index":  xidx,
                    "y_index":  yidx
                })

        if not tiles:
            raise RuntimeError(
                f"No image files (tif/tiff/oir) matched the XML entries in {cond_folder}"
            )
        # for area in (e for e in root.iter() if strip_ns(e.tag) == "area"):
        #     img  = next(e for e in area if strip_ns(e.tag) == "image").text
        #     stem = Path(img).stem
        #     if stem in tif_stems:
        #         xidx = int(next(e for e in area if strip_ns(e.tag) == "xIndex").text)
        #         yidx = int(next(e for e in area if strip_ns(e.tag) == "yIndex").text)
        #         tiles.append({"filename": f"{stem}.tif", "x_index": xidx, "y_index": yidx})


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

        # first_tif = tif_folder / tiles[0]["filename"]
        # arr = tifffile.imread(first_tif)
        
        # if arr.ndim == 4:
        #     z, c, y, x = arr.shape
        # else:
        #     z, y, x = arr.shape

        # tile_shape = {"z": z, "y": y, "x": x}

        # pick your very first “tile” (which might be .tif or .oir)
        first_path = tif_folder / tiles[0]["filename"]
        suffix = first_path.suffix.lower()

        if suffix in (".tif", ".tiff"):
            arr = tifffile.imread(first_path)

        elif suffix == ".oir":
            # lazy‐load the Bio‐Formats JVM
            IJ, BF, ImporterOptions = _get_bioformats()
            opts = ImporterOptions()
            opts.setOpenAllSeries(True)
            opts.setVirtual(True)
            opts.setId(str(first_path))
            imps = BF.openImagePlus(opts)       # Java ImagePlus[]
            # grab the first series and convert to numpy
            jarr = IJ.py.from_java(imps[0])
            arr = jarr  # this is now a numpy array; may be 2D, 3D, or 4D

        else:
            raise RuntimeError(f"Unrecognized file extension {suffix!r} for {first_path}")

        # now infer tile_shape exactly as before
        if arr.ndim == 4:
            z, c, y, x = arr.shape
        elif arr.ndim == 3:
            z, y, x = arr.shape;  c = 1
        else:
            raise RuntimeError(f"Cannot parse shape {arr.shape} from first tile")
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

############## VERY IMPORTANT: THIS BELOW IS THE ONE THAT DEFINITELY  WORKS 5-29-2025 ################################
    # @staticmethod
    # def load_maxproj_images(folder):
    #     """
    #     Generator yielding (filename, max_projection) for each TIFF in `folder`.
    #     """
    #     folder = Path(folder)
    #     tif_paths = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
    #     for path in tif_paths:
    #         stack = tifffile.imread(path)
    #         if stack.ndim == 4:
    #             channel0 = stack[:, 0, ...]
    #         elif stack.ndim == 3:
    #             channel0 = stack
    #         else:
    #             raise ValueError(f"Unexpected image dims {stack.shape} for {path.name}")
    #         yield path.name, np.max(channel0, axis=0)

    ############## VERY IMPORTANT: THIS ABOVE IS THE ONE THAT DEFINITELY  WORKS 5-29-2025 ################################

    # @staticmethod
    # def load_maxproj_images(folder):
    #     """
    #     Generator yielding (name, max_projection) for each TIFF or OIR in `folder`,
    #     depending on cfg.IMAGE_LOADER.
    #     """
    #     folder = Path(folder)
    #     loader = cfg.IMAGE_LOADER.lower()

    #     # decide which extensions to scan
    #     tifs = []
    #     oirs = []
    #     if loader in ("tif", "both"):
    #         tifs = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
    #     if loader in ("oir", "both"):
    #         oirs = sorted(folder.glob("*.oir"))

    #     # 1) Yield TIFF max-projections
    #     for path in tifs:
    #         stack = tifffile.imread(path)
    #         # channel0 extraction as before
    #         if stack.ndim == 4:
    #             channel0 = stack[:, 0, ...]
    #         elif stack.ndim == 3:
    #             channel0 = stack
    #         else:
    #             raise ValueError(f"Unexpected dims {stack.shape} in {path.name}")
    #         name = path.name
    #         yield name, np.max(channel0, axis=0)

    #     # 2) If configured, open OIR series via Bio-Formats
    #     if oirs:
    #         IJ, BF, ImporterOptions = _get_bioformats()
    #         for path in oirs:
    #             opts = ImporterOptions()
    #             opts.setOpenAllSeries(True)
    #             opts.setVirtual(True)
    #             opts.setId(str(path))
    #             imps = BF.openImagePlus(opts)    # Java ImagePlus[]
    #             for idx, imp in enumerate(imps):
    #                 # convert to numpy
    #                 arr = IJ.py.from_java(imp)
    #                 # extract channel0 exactly as above
    #                 if arr.ndim == 4:
    #                     channel0 = arr[:, 0, ...]
    #                 elif arr.ndim == 3:
    #                     channel0 = arr
    #                 elif arr.ndim == 2:
    #                     channel0 = arr[None, ...]
    #                 else:
    #                     raise ValueError(f"Unexpected BF array shape {arr.shape}")
    #                 name = f"{path.stem}_series{idx}{path.suffix}"
    #                 yield name, np.max(channel0, axis=0)
#### THIS ABOVE IS FIRST TRY FOR OIR READING ############

    @staticmethod
    def load_maxproj_images(folder):
        """
        Generator yielding (name, max_projection) for each TIFF/.oir in `folder`,
        according to cfg.IMAGE_LOADER (“tif”, “oir”, or “both”).
        """
        folder = Path(folder)
        mode = cfg.IMAGE_LOADER.lower()

        tifs = []
        oirs = []
        if mode in ("tif", "both"):
            tifs = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
        if mode in ("oir", "both"):
            oirs = sorted(folder.glob("*.oir"))

        # TIFF pipeline
        for path in tifs:
            arr = tifffile.imread(path)
            if arr.ndim == 4:
                channel0 = arr[:, 0, ...]
            elif arr.ndim == 3:
                channel0 = arr
            else:
                raise ValueError(f"Unexpected TIFF dims {arr.shape} in {path.name}")
            yield path.name, np.max(channel0, axis=0)

        # Bio-Formats OIR pipeline
        if oirs:
            IJ, BF, ImporterOptions = _get_bioformats()
            for path in oirs:
                opts = ImporterOptions()
                opts.setOpenAllSeries(True)
                opts.setVirtual(True)
                opts.setId(str(path))
                imps = BF.openImagePlus(opts)
                for idx, imp in enumerate(imps):
                    jarr = IJ.py.from_java(imp)
                    if jarr.ndim == 4:
                        channel0 = jarr[:, 0, ...]
                    elif jarr.ndim == 3:
                        channel0 = jarr
                    elif jarr.ndim == 2:
                        channel0 = jarr[None, ...]
                    else:
                        raise ValueError(f"Unexpected BF array {jarr.shape}")
                    name = f"{path.stem}_series{idx}{path.suffix}"
                    yield name, np.max(channel0, axis=0)



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

    def fuse_and_export(self, msims, out_folder, export_tiff=False, crop_z_black_frames=False, z_crop_threshold=0.01):
        """
        Fuse multiple MSIMs and export to zarr/tiff.
        
        Parameters
        ----------
        msims : list
            List of multiscale image models
        out_folder : str or Path
            Output directory
        export_tiff : bool, optional
            Whether to also export as TIFF
        crop_z_black_frames : bool, optional
            Whether to crop black z-frames after fusion (default: False)
        z_crop_threshold : float, optional
            Threshold for detecting "empty" z-slices (default: 0.01)
            Z-slices with max intensity below this fraction of global max are considered empty
        """
        out = Path(out_folder)
        out.mkdir(exist_ok=True, parents=True)
        name = out.name
        zarr_out = out / f"{name}.zarr"
        
        # Perform fusion
        fused = fusion.fuse([
            msi_utils.get_sim_from_msim(m) for m in msims
        ], transform_key='affine_registered', output_chunksize=256)
        
        # Optionally crop black z-frames
        if crop_z_black_frames:
            print(f"🔍 Analyzing z-stack for black frames (threshold: {z_crop_threshold})...")
            fused = self._crop_black_z_frames(fused, threshold=z_crop_threshold)
        
        # Save to zarr
        with dask.diagnostics.ProgressBar():
            ngff_utils.write_sim_to_ome_zarr(fused, str(zarr_out), overwrite=True)
            
        # Optionally save to TIFF
        if export_tiff:
            tif_out = out / f"{name}.tif"
            with dask.diagnostics.ProgressBar():
                io.save_sim_as_tif(str(tif_out), fused)
    
    def _crop_black_z_frames(self, sim, threshold=0.01):
        """
        Remove z-slices that contain ANY black/padding regions from a fused SIM.
        
        This looks at each z-slice across the entire subgrid and removes any slice
        that has pixels below the threshold (indicating black padding regions).
        
        Parameters
        ----------
        sim : spatial_image
            Fused spatial image
        threshold : float
            Threshold for detecting black pixels as fraction of global max intensity
            
        Returns
        -------
        spatial_image
            Cropped spatial image with black z-frames removed
        """
        import dask.array as da
        import numpy as np
        
        # Get data array and find z-dimension
        data = sim.data
        print(f"   → Original shape: {data.shape}")
        print(f"   → Dimensions: {sim.dims}")
        
        # Find z-dimension index
        z_dim_idx = None
        for i, dim in enumerate(sim.dims):
            if dim == 'z':
                z_dim_idx = i
                break
        
        if z_dim_idx is None:
            print(f"   → No 'z' dimension found, skipping z-cropping")
            return sim
            
        z_slices = data.shape[z_dim_idx]
        print(f"   → Z-dimension at index {z_dim_idx} with {z_slices} slices")
        
        # Compute global max intensity to set threshold
        global_max = da.max(data).compute()
        intensity_threshold = global_max * threshold
        print(f"   → Global max intensity: {global_max:.2f}")
        print(f"   → Black pixel threshold: {intensity_threshold:.2f}")
        
        # Check for large black regions (400x400+ pixels) indicating padding tiles
        # Find spatial dimensions (y and x)
        y_dim_idx = None
        x_dim_idx = None
        for i, dim in enumerate(sim.dims):
            if dim == 'y':
                y_dim_idx = i
            elif dim == 'x':
                x_dim_idx = i
        
        if y_dim_idx is None or x_dim_idx is None:
            print(f"   → Could not find y/x dimensions, skipping z-cropping")
            return sim
        
        # Get spatial dimensions
        y_size = data.shape[y_dim_idx]
        x_size = data.shape[x_dim_idx]
        
        print(f"   → Spatial size: {y_size} x {x_size}")
        print(f"   → Looking for black regions ≥ 400x400 pixels")
        
        # Check each z-slice for large black regions
        min_black_size = 400  # 400x400 pixel minimum
        good_z_slices = []
        
        # Find channel dimension
        c_dim_idx = None
        for i, dim in enumerate(sim.dims):
            if dim == 'c':
                c_dim_idx = i
                break
        
        for z_idx in range(z_slices):
            # Extract channel 0 of this z-slice only
            slice_tuple = tuple(
                0 if i == c_dim_idx else  # Channel 0 only
                z_idx if i == z_dim_idx else  # This z-slice
                slice(None)  # All other dimensions
                for i in range(len(data.shape))
            )
            z_slice_ch0 = data[slice_tuple]
            
            # Compute this slice to check for black regions
            z_slice_np = z_slice_ch0.compute()
            
            # Create binary mask of black pixels
            black_mask = z_slice_np <= intensity_threshold
            
            # Use sliding window to find 400x400 black regions efficiently
            has_large_black_region = self._has_large_black_region(black_mask, min_black_size)
            
            good_z_slices.append(not has_large_black_region)
            
            if z_idx % 10 == 0 or z_idx == z_slices - 1:
                print(f"   → Processed z-slice {z_idx}/{z_slices-1}")
        
        good_z_slices = np.array(good_z_slices)
        print(f"   → Found {np.sum(good_z_slices)} good slices out of {len(good_z_slices)}")
        
        if not np.any(good_z_slices):
            print("   ⚠️ Warning: All z-slices contain black regions! Keeping original data.")
            return sim
        
        # Find the range of good z-slices (first good to last good)
        good_indices = np.where(good_z_slices)[0]
        z_start = good_indices[0]
        z_end = good_indices[-1] + 1  # +1 for inclusive end
        
        print(f"   → Good z-slices (no black pixels): {z_start} to {z_end-1} (total: {z_end - z_start})")
        print(f"   → Removing {z_start} slices from start, {z_slices - z_end} slices from end")
        
        if z_start == 0 and z_end == z_slices:
            print("   → No cropping needed - all slices are good!")
            return sim
        
        # Simple slicing approach - just slice the original SIM data
        # Create slice tuple for cropping
        crop_tuple = tuple(slice(None) if i != z_dim_idx else slice(z_start, z_end) 
                          for i in range(len(data.shape)))
        
        # Create a new SIM by slicing the original
        cropped_sim = sim.isel({sim.dims[z_dim_idx]: slice(z_start, z_end)})
        
        print(f"   → Cropped shape: {cropped_sim.data.shape}")
        
        return cropped_sim

    def _has_large_black_region(self, black_mask, min_size):
        """
        Efficiently check if a 2D binary mask has any contiguous black region >= min_size x min_size.
        
        Uses a sliding window approach for speed - much faster than connected components.
        """
        import numpy as np
        
        if black_mask.shape[0] < min_size or black_mask.shape[1] < min_size:
            return False
        
        # Use sliding window to check for min_size x min_size black squares
        # This is much faster than full connected component analysis
        for y in range(black_mask.shape[0] - min_size + 1):
            for x in range(black_mask.shape[1] - min_size + 1):
                # Check if this min_size x min_size window is all black
                window = black_mask[y:y+min_size, x:x+min_size]
                if np.all(window):
                    return True
        
        return False


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
        # p = re.compile(r"(.+__A\d+_G\d+)_\d+") # oh my god wahhh the issue was a double instead of a single underscore
        p = re.compile(r"(.+_A\d+_G\d+)_\d+") # fixed
        for t in meta["tiles"]:
            fn = t.get("filename") or ""
            m = p.match(Path(fn).stem)
            if m:
                groups.setdefault(m.group(1), []).append(t)
        return groups

   

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

_bioformats = None
def _get_bioformats():
    """
    Lazily initialize the JVM against your local Fiji.app,
    and return (IJ, BF, ImporterOptions).
    """
    global _bioformats
    if _bioformats is None:
        # use the path from config
        ij_dir = str(cfg.FIJI_APP)
        IJ = imagej.init(ij_dir, mode="headless", add_legacy=True)
        BF = scyjava.jimport("loci.plugins.BF")
        ImporterOptions = scyjava.jimport("loci.plugins.in.ImporterOptions")
        _bioformats = (IJ, BF, ImporterOptions)
    return _bioformats
