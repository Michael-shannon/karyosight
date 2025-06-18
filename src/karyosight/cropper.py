import zarr
import numpy as np
import dask
import dask.array as da
# from dask import delayed


import psutil


from dask import delayed
from skimage import exposure, filters, morphology, segmentation
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes
from skimage.measure import label, regionprops
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import napari
from pathlib import Path
from karyosight.config import CROPPED_DIR, CONDITION_PREF

# import karyosight.config as cfg

class Cropper:
    """
    Handles ROI extraction from low-res Zarr,
    upscaling, cropping from high-res Zarr, bundling,
    and visualization in notebook and Napari.
    """
    def __init__(self,
                 stitched_dir: Path,
                 cropped_dir: Path = CROPPED_DIR,
                 low_level: int = 1,
                 high_level: int = 0,
                 scheduler: str = 'threads'):
        self.stitched_dir = Path(stitched_dir)
        self.cropped_dir = Path(cropped_dir)
        self.low_level = low_level
        self.high_level = high_level
        self.scheduler = scheduler

    @staticmethod
    def pad_bbox(bbox, pad_pct, img_shape):
        minr, minc, maxr, maxc = bbox
        h, w = maxr - minr, maxc - minc
        pad_r = int(h * pad_pct / 2)
        pad_c = int(w * pad_pct / 2)
        H, W = img_shape
        return (
            max(0, minr - pad_r),
            max(0, minc - pad_c),
            min(H, maxr + pad_r),
            min(W, maxc + pad_c),
        )


    def extract_rois_from_level(self,
                                zarr_path: Path,
                                lower_threshold: float = 0.8,
                                min_area: int = 20000,
                                max_area: int = None,
                                hole_area: int = 1000,
                                gaussian_sigma: float = 10,
                                pad_pct: float = 0.6,
                                pad_step: float = 0.05,
                                visualize: bool = False,
                                verbose:   bool = False):
        """
        Segment level `self.low_level` to find centroids + padded bboxes,
        shrink to only the main organoid, overlay contours, and print diagnostics.
        """
        # 1) load and project
        root = zarr.open_group(str(zarr_path), mode='r')
        arr  = root[str(self.low_level)][:]  # (t,c,z,y,x)
        vol  = arr[0,0]                      # (z,y,x)
        proj = vol.max(axis=0)               # (y,x)

        # 2) preprocess + threshold
        eq     = exposure.equalize_adapthist(proj)
        smooth = filters.gaussian(eq, sigma=gaussian_sigma)
        th0    = threshold_otsu(smooth)
        bw     = smooth > (th0 * lower_threshold)
        bw     = morphology.closing(bw, morphology.square(3))
        bw     = remove_small_holes(bw, area_threshold=hole_area)
        bw     = segmentation.clear_border(bw)

        # 3) find organoid masks
        label_img = label(bw)
        props     = regionprops(label_img)

        big_props = [
            p for p in props
            if p.area >= min_area
            and (max_area is None or p.area <= max_area)
        ]
        if verbose:
            print(f"🔍 Found {len(big_props)} large organoids (min_area={min_area})")

        # map each organoid label → its raw low-res bbox
        lowres_bbox_map = {p.label: p.bbox for p in big_props}

        # 4) original padded ROIs
        orig_rois = []
        for p in big_props:
            obbox = self.pad_bbox(p.bbox, pad_pct, proj.shape)
            orig_rois.append({
                'label':         p.label,
                'centroid':      tuple(p.centroid),
                'bbox':          obbox,
                'orig_bbox_map': lowres_bbox_map
            })
            if verbose:
                print(f"  • Organoid {p.label}: raw bbox={p.bbox}, "
                    f"orig pad={pad_pct}, padded bbox={obbox}")

        # 5) shrink to avoid overlapping any other organoid mask bbox
        final_rois = []
        for p in big_props:
            pad = pad_pct
            if verbose:
                print(f"\n🛠 Shrinking organoid {p.label}, start pad={pad_pct}")

            while pad >= 0:
                r0, c0, r1, c1 = self.pad_bbox(p.bbox, pad, proj.shape)
                collisions = []
                for q in big_props:
                    if q.label == p.label:
                        continue
                    qr0, qc0, qr1, qc1 = q.bbox
                    # axis‐aligned rectangle intersection test
                    if not (qr1 <= r0 or qr0 >= r1 or qc1 <= c0 or qc0 >= c1):
                        collisions.append(q.label)

                if not collisions:
                    if verbose:
                        print(f"   ✓ pad={pad:.3f} OK, final bbox=({r0},{c0},{r1},{c1})")
                    final_rois.append({
                        'label':            p.label,
                        'centroid':         tuple(p.centroid),
                        'bbox':             (r0,c0,r1,c1),
                        'pad_used':         pad,
                        'collision_labels': [],
                        'orig_bbox_map':    lowres_bbox_map
                    })
                    break

                else:
                    if verbose:
                        print(f"   ✗ pad={pad:.3f} collisions with organoids {collisions}")
                pad -= pad_step

            else:
                # fallback: revert to original padded bbox, but only mask those touching the edge
                if verbose:
                    print(f"   ⚠️  pad < 0, reverting to pad={pad_pct} and marking overlaps")

                obox = self.pad_bbox(p.bbox, pad_pct, proj.shape)
                # all overlaps at pad=0
                coll0 = [
                    q.label for q in big_props
                    if not (q.bbox[2] <= obox[0] or q.bbox[0] >= obox[2]
                            or q.bbox[3] <= obox[1] or q.bbox[1] >= obox[3])
                ]
                # only keep those whose raw bbox actually touches the ROI border
                touching = []
                for lbl in coll0:
                    qr0, qc0, qr1, qc1 = lowres_bbox_map[lbl]
                    if (qr0 <= obox[0] or qr1 >= obox[2]
                        or qc0 <= obox[1] or qc1 >= obox[3]):
                        touching.append(lbl)

                if verbose:
                    print(f"   Zeroing only touching organoids: {touching}")

                final_rois.append({
                    'label':            p.label,
                    'centroid':         tuple(p.centroid),
                    'bbox':             obox,
                    'pad_used':         pad_pct,
                    'collision_labels': touching,
                    'orig_bbox_map':    lowres_bbox_map
                })

        # 6) visualize with contours, annotate ID, and save
        if visualize:
            fig, axes = plt.subplots(1,2, figsize=(12,6))
            titles = ["Original padding", "Shrunk to main organoid"]
            colors = ["yellow", "lime"]

            for ax, roi_list, title, color in zip(axes, [orig_rois, final_rois], titles, colors):
                ax.imshow(proj, cmap='gray')
                # full organoid mask outline
                ax.contour(bw, levels=[0.5], colors='red', linewidths=0.5)
                # each organoid mask in cyan
                for p in big_props:
                    mask = (label_img == p.label)
                    ax.contour(mask, levels=[0.5], colors='cyan')
                # overlay each ROI
                for roi in roi_list:
                    r0, c0, r1, c1 = roi['bbox']
                    cy, cx        = roi['centroid']
                    oid           = roi['label']
                    ax.add_patch(plt.Rectangle((c0,r0), c1-c0, r1-r0,
                                            edgecolor=color, facecolor='none', lw=2))
                    ax.plot(cx, cy, 'ro')
                    ax.text(c0, r0-5, str(oid),
                            color='white',
                            backgroundcolor=color,
                            fontsize=8, weight='bold')

                ax.set_title(title)
                ax.axis('off')

            plt.tight_layout()
            # save figure
            cond_name  = next(p.name for p in zarr_path.parents
                            if p.name.startswith(CONDITION_PREF))
            subgrid_id = zarr_path.parent.name
            prefix     = zarr_path.parent.parent.name
            overview_dir = Path(CROPPED_DIR)/"grid_overview"
            overview_dir.mkdir(exist_ok=True, parents=True)
            fname = f"{cond_name}__{prefix}__{subgrid_id}.png"
            fig.savefig(str(overview_dir/fname), dpi=150)
            plt.show()
            plt.close(fig)

        return final_rois


    @staticmethod
    def _crop_and_mask(data, roi, sy, sx):
        """
        Crop `data` at the high-res ROI bbox and zero out any
        smaller overlapping organoid regions.
        """
        # high-res crop
        r0,c0,r1,c1 = roi['bbox']
        hr0 = int(round(r0*sy)); hc0 = int(round(c0*sx))
        hr1 = int(round(r1*sy)); hc1 = int(round(c1*sx))
        sub = data[..., hr0:hr1, hc0:hc1]

        # for each overlapping organoid, zero out its scaled bbox
        for label in roi.get('collision_labels', []):
            # we need the original q.bbox from roi metadata
            # assume roi dict also has 'orig_bbox_map' mapping label->bbox
            qr0, qc0, qr1, qc1 = roi['orig_bbox_map'][label]
            qhr0 = int(round(qr0*sy)) - hr0
            qhc0 = int(round(qc0*sx)) - hc0
            qhr1 = int(round(qr1*sy)) - hr0
            qhc1 = int(round(qc1*sx)) - hc0
            # clamp to sub-shape
            qhr0, qhc0 = max(0,qhr0), max(0,qhc0)
            qhr1, qhc1 = min(sub.shape[-2],qhr1), min(sub.shape[-1],qhc1)
            sub[..., qhr0:qhr1, qhc0:qhc1] = 0
        return sub


    def bundle_and_save_rois(self,
                             zarr_path: Path,
                             rois: list,
                             memory_limit_bytes: int = None,
                             z_extraction_mode: str = 'all',
                             z_slices_around_peak: int = 20,
                             intensity_channel: int = 0,
                             remove_black_frames: bool = True,
                             black_frame_threshold: float = 0.001,
                             black_frame_method: str = 'intensity_threshold',
                             min_z_slices: int = 3):
        """
        Upscale ROIs, crop full-res, pad to uniform size, and bundle into one Zarr.
        If a bundled zarr already exists for this condition, append new ROIs to it.

        Parameters
        ----------
        zarr_path : Path
            Path to a stitched Zarr group containing levels low/high.
        rois : list of dict
            Each dict must have keys:
              - 'bbox': (r0, c0, r1, c1) in LOW-RES coords
              - 'centroid': (y, x) in LOW-RES coords   [only used for metadata]
              - optionally 'orig_bbox_map' and 'collision_labels' if masking is needed
        memory_limit_bytes : int, optional
            If provided, we will validate that no single high-res ROI
            exceeds this budget. If None, defaults to ~80% of available RAM.
        z_extraction_mode : str, optional
            'all' = keep all z-slices (with padding), 'peak' = extract slices around intensity peak
        z_slices_around_peak : int, optional
            Number of slices above and below peak intensity (total = 2*z_slices_around_peak + 1)
        intensity_channel : int, optional
            Channel to use for finding peak intensity (0-5 for CH1-CH6)
        remove_black_frames : bool, optional
            Whether to remove z-slices that are predominantly black/empty
        black_frame_threshold : float, optional
            Threshold for determining black frames (see remove_black_z_slices method)
        black_frame_method : str, optional
            Method for black frame detection: 'intensity_threshold', 'content_ratio', or 'mean_intensity'
        min_z_slices : int, optional
            Minimum number of z-slices to keep when removing black frames
        """
        # 1) Determine how much RAM we can use for a *single* ROI uncompressed
        if memory_limit_bytes is None:
            vm = psutil.virtual_memory()
            # leave at least 20% for OS/other processes:
            memory_limit_bytes = int(0.8 * vm.available)

        # 2) locate the Condition_ folder above zarr_path
        src = Path(zarr_path)
        cond_dir = next(
            (p for p in src.parents if p.name.startswith(CONDITION_PREF)),
            None
        )
        if cond_dir is None:
            raise RuntimeError(f"No Condition_ parent for {src!r}")
        cond = cond_dir.name

        # 3) Prepare the output Zarr group - check if it exists and load existing data
        out_grp = Path(self.cropped_dir) / cond / f"{cond}_bundled.zarr"
        
        # Check if we already have a bundled zarr for this condition
        existing_data = None
        existing_centroids = None
        existing_bboxes = None
        existing_n_rois = 0
        
        if out_grp.exists():
            print(f"📂 Found existing bundled zarr at {out_grp}")
            try:
                existing_bundle = zarr.open_group(str(out_grp), mode='r')
                existing_data = existing_bundle['data']
                existing_n_rois = existing_data.shape[0]
                existing_centroids = existing_bundle['centroids_lowres'][:]
                existing_bboxes = existing_bundle['bboxes_highres'][:]
                print(f"   → Found {existing_n_rois} existing ROIs")
            except Exception as e:
                print(f"   ⚠️ Error reading existing zarr: {e}")
                print(f"   → Removing corrupted zarr and starting fresh")
                shutil.rmtree(out_grp)
                existing_data = None
        
        out_grp.parent.mkdir(parents=True, exist_ok=True)

        # 4) Open the stitched Zarr levels (low/high)
        root     = zarr.open_group(str(zarr_path), mode='r')
        arr_low  = root[str(self.low_level)]
        arr_high = root[str(self.high_level)]
        dtype    = arr_high.dtype

        # 5) Compute the scale factors (low→high)
        sy = arr_high.shape[-2] / arr_low.shape[-2]
        sx = arr_high.shape[-1] / arr_low.shape[-1]

        # 6) Compute each ROI's high-res shape (c, z, H, W), check against memory_limit
        #    Also track (Hmax, Wmax) so we know how big to pad.
        Hmax = 0
        Wmax = 0
        roi_scaled_info = []

        for i, roi in enumerate(rois):
            r0, c0, r1, c1 = roi['bbox']
            low_h = (r1 - r0)
            low_w = (c1 - c0)

            # high-res dims (rounded to nearest int)
            high_h = int(np.round(low_h * sy))
            high_w = int(np.round(low_w * sx))
            

            # channels & z-slices in arr_high:
            #   arr_high has shape either (1, c, z, y, x) or (c, z, y, x)
            if arr_high.ndim == 5 and arr_high.shape[0] == 1:
                c_count = arr_high.shape[1]
                z_count = arr_high.shape[2]
            else:
                # either no singleton time or time >1 (in which case treat time as channel/day)
                # but typically it's (1, c, z, y, x)
                c_count = arr_high.shape[-4]
                z_count = arr_high.shape[-3]

            # uncompressed bytes for one ROI slice:
            # dtype.itemsize × c_count × z_count × high_h × high_w
            bytes_for_this_roi = dtype.itemsize * c_count * z_count * high_h * high_w
            print(f"→ Trying ROI #{i}: high‐res shape = (c={c_count}, z={z_count}, y={high_h}, x={high_w}) → {bytes_for_this_roi/1e9:.2f} GB")

            if bytes_for_this_roi > memory_limit_bytes:
                raise MemoryError(
                    f"ROI bbox low-res {roi['bbox']} → high-res shape "
                    f"(c={c_count},z={z_count},y={high_h},x={high_w}) "
                    f"requires {bytes_for_this_roi/1e9:.2f} GB > {memory_limit_bytes/1e9:.2f} GB"
                )

            Hmax = max(Hmax, high_h)
            Wmax = max(Wmax, high_w)

            roi_scaled_info.append({
                'low_bbox': roi['bbox'],
                'high_bbox': (
                    int(np.round(r0 * sy)),
                    int(np.round(c0 * sx)),
                    int(np.round(r1 * sy)),
                    int(np.round(c1 * sx))
                ),
                'collision_labels': roi.get('collision_labels', []),
                'orig_bbox_map': roi.get('orig_bbox_map', {}),
                'centroid_low': roi.get('centroid', None)
            })

        # 7) Determine final dimensions and create/extend Zarr dataset
        n_new_rois = len(rois)
        total_n_rois = existing_n_rois + n_new_rois
        
        # Handle z-dimension based on extraction mode
        if z_extraction_mode == 'peak':
            # Use fixed z-dimension based on slices around peak
            final_z_count = 2 * z_slices_around_peak + 1
            print(f"🎯 Using peak intensity mode: {final_z_count} z-slices ({z_slices_around_peak} above/below peak)")
        else:
            final_z_count = z_count
        
        # Handle case where existing data might have different dimensions
        if existing_data is not None:
            existing_c = existing_data.shape[-4]
            existing_z = existing_data.shape[-3] 
            existing_Hmax = existing_data.shape[-2]
            existing_Wmax = existing_data.shape[-1]
            
            # Use the maximum dimensions from both existing and new data
            final_c = max(c_count, existing_c)
            if z_extraction_mode == 'all':
                final_z = max(final_z_count, existing_z)
            else:
                final_z = final_z_count  # Use fixed z-dimension for peak mode
            final_Hmax = max(Hmax, existing_Hmax)
            final_Wmax = max(Wmax, existing_Wmax)
            
            print(f"   → Existing dims: (c={existing_c}, z={existing_z}, y={existing_Hmax}, x={existing_Wmax})")
            print(f"   → New dims: (c={c_count}, z={final_z_count}, y={Hmax}, x={Wmax})")
            print(f"   → Final dims: (c={final_c}, z={final_z}, y={final_Hmax}, x={final_Wmax})")
            
            c_count, z_count, Hmax, Wmax = final_c, final_z, final_Hmax, final_Wmax
        else:
            if z_extraction_mode == 'peak':
                z_count = final_z_count
        
        shape4d = (c_count, z_count, Hmax, Wmax)
        
        # Get chunking from arr_high
        if arr_high.ndim == 5 and arr_high.shape[0] == 1:
            chunk_tuple = arr_high.chunks[-4:]
        else:
            chunk_tuple = arr_high.chunks[-arr_high.ndim:]

        desired_chunks = (1,) + chunk_tuple  # one ROI at a time

        print(f"📊 Creating bundled zarr with {total_n_rois} total ROIs ({existing_n_rois} existing + {n_new_rois} new)")
        
        # 8) Create new zarr with combined size
        # First create a temporary path
        temp_grp = out_grp.with_suffix('.zarr.tmp')
        if temp_grp.exists():
            shutil.rmtree(temp_grp)
            
        outz = zarr.open_group(str(temp_grp), mode='w')
        outz.create_dataset(
            name='data',
            shape=(total_n_rois,) + shape4d,
            chunks=desired_chunks,
            dtype=dtype,
            compressor=arr_high.compressor
        )
        
        # 9) Copy existing data if any (with padding/cropping if needed)
        if existing_data is not None:
            print(f"   → Copying {existing_n_rois} existing ROIs...")
            for i in range(existing_n_rois):
                existing_roi = existing_data[i]  # Shape: (c, z, y, x)
                target_shape = (c_count, z_count, Hmax, Wmax)
                
                print(f"   → ROI {i}: existing shape {existing_roi.shape} → target shape {target_shape}")
                
                # Handle dimensions - special processing for peak mode z-dimension
                processed_roi = existing_roi
                
                # Special handling for z-dimension in peak mode
                if z_extraction_mode == 'peak' and processed_roi.shape[1] != target_shape[1]:
                    # Extract peak-centered z-slices from existing data
                    intensity_channel_data = processed_roi[intensity_channel]  # Shape: (z, y, x)
                    z_sums = np.sum(intensity_channel_data, axis=(1, 2))
                    peak_z = np.argmax(z_sums)
                    
                    z_start = max(0, peak_z - z_slices_around_peak)
                    z_end = min(processed_roi.shape[1], peak_z + z_slices_around_peak + 1)
                    
                    # Extract z-range for all channels
                    processed_roi = processed_roi[:, z_start:z_end, :, :]
                    print(f"     → Existing ROI peak at z={peak_z}, extracted z={z_start}:{z_end}")
                
                # Apply black frame removal to existing data if requested
                if remove_black_frames and processed_roi.shape[1] > min_z_slices:
                    original_z = processed_roi.shape[1]
                    processed_roi = self.remove_black_z_slices(
                        processed_roi,
                        threshold_pct=black_frame_threshold,
                        method=black_frame_method, 
                        min_slices=min_z_slices,
                        verbose=True
                    )
                    if processed_roi.shape[1] != original_z:
                        print(f"     → Existing ROI black frame removal: {original_z} → {processed_roi.shape[1]} z-slices")
                
                # Handle remaining dimensions - pad or crop as needed
                for dim in range(4):  # c, z, y, x
                    current_size = processed_roi.shape[dim]
                    target_size = target_shape[dim]
                    
                    if current_size != target_size:
                        if current_size < target_size:
                            # Need to pad this dimension
                            pad_amount = target_size - current_size
                            pad_spec = [(0, 0)] * processed_roi.ndim
                            pad_spec[dim] = (0, pad_amount)
                            processed_roi = np.pad(processed_roi, pad_spec, mode='constant', constant_values=0)
                            print(f"     → Padded dim {dim} from {current_size} to {target_size}")
                        else:
                            # Need to crop this dimension
                            slice_spec = [slice(None)] * processed_roi.ndim
                            slice_spec[dim] = slice(0, target_size)
                            processed_roi = processed_roi[tuple(slice_spec)]
                            print(f"     → Cropped dim {dim} from {current_size} to {target_size}")
                
                outz['data'][i, ...] = processed_roi

        # 10) Process and write new ROIs starting from index existing_n_rois
        for i, info in enumerate(roi_scaled_info):
            zarr_index = existing_n_rois + i
            hr0, hc0, hr1, hc1 = info['high_bbox']

            # Load exactly that slice from arr_high
            if arr_high.ndim == 5 and arr_high.shape[0] == 1:
                # arr_high shape: (1, c, z, y, x)
                subvol = arr_high[0, :, :, hr0:hr1, hc0:hc1]
            else:
                # arr_high shape: (c, z, y, x)
                subvol = arr_high[:, :, hr0:hr1, hc0:hc1]

            # For each overlapping organoid in collision_labels, zero out their high-res bbox
            for lbl in info['collision_labels']:
                # get that blob's low-res bbox → map it to high-res → subtract hr0,hc0
                lr0, lc0, lr1, lc1 = info['orig_bbox_map'][lbl]
                qhr0 = int(np.round(lr0 * sy)) - hr0
                qhr1 = int(np.round(lr1 * sy)) - hr0
                qhc0 = int(np.round(lc0 * sx)) - hc0
                qhc1 = int(np.round(lc1 * sx)) - hc0

                # clamp to subvol shape:
                qhr0 = max(0, qhr0)
                qhc0 = max(0, qhc0)
                qhr1 = min(subvol.shape[-2], qhr1)
                qhc1 = min(subvol.shape[-1], qhc1)

                # zero out voxels in that block:
                subvol[..., qhr0:qhr1, qhc0:qhc1] = 0

            # Handle z-dimension extraction
            if z_extraction_mode == 'peak':
                # Find peak intensity z-slice in the specified channel
                intensity_channel_data = subvol[intensity_channel]  # Shape: (z, y, x)
                z_sums = np.sum(intensity_channel_data, axis=(1, 2))  # Sum over y,x for each z
                peak_z = np.argmax(z_sums)
                
                # Extract slices around peak
                z_start = max(0, peak_z - z_slices_around_peak)
                z_end = min(subvol.shape[1], peak_z + z_slices_around_peak + 1)
                
                # Extract the z-range for all channels
                extracted_subvol = subvol[:, z_start:z_end, :, :]
                
                # Pad if we don't have enough slices (at edges of z-stack)
                target_z_slices = 2 * z_slices_around_peak + 1
                current_z_slices = extracted_subvol.shape[1]
                
                if current_z_slices < target_z_slices:
                    z_pad = target_z_slices - current_z_slices
                    # Pad at the end (or split between start/end if needed)
                    z_pad_spec = ((0, 0), (0, z_pad), (0, 0), (0, 0))
                    extracted_subvol = np.pad(extracted_subvol, z_pad_spec, mode='constant', constant_values=0)
                
                subvol = extracted_subvol
                print(f"     → Peak at z={peak_z}, extracted z={z_start}:{z_end} ({subvol.shape[1]} slices)")
            
            # Remove black frames if requested (after z-extraction but before spatial padding)
            if remove_black_frames and subvol.shape[1] > min_z_slices:
                original_z_count = subvol.shape[1]
                subvol = self.remove_black_z_slices(
                    subvol, 
                    threshold_pct=black_frame_threshold,
                    method=black_frame_method,
                    min_slices=min_z_slices,
                    verbose=True
                )
                if subvol.shape[1] != original_z_count:
                    print(f"     → Black frame removal: {original_z_count} → {subvol.shape[1]} z-slices")

            # Pad spatial dimensions to (c_count, z_count, Hmax, Wmax)
            h_sub = subvol.shape[-2]
            w_sub = subvol.shape[-1]
            z_sub = subvol.shape[-3]
            
            pad_y = Hmax - h_sub
            pad_x = Wmax - w_sub
            pad_z = z_count - z_sub
            
            # Build pad spec: (c_pad, z_pad, y_pad, x_pad)
            pad_spec = ((0, 0), (0, pad_z), (0, pad_y), (0, pad_x))
            padded = np.pad(subvol, pad_spec, mode='constant', constant_values=0)

            # Write padded ROI into the zarr_index-th slice of the Zarr dataset
            outz['data'][zarr_index, ...] = padded

            # Drop `subvol` and `padded` so Python can free them immediately
            del subvol, padded

        # 11) Combine metadata: existing + new
        # New centroids and bboxes
        new_cents_low = np.array(
            [roi.get('centroid', (np.nan, np.nan)) for roi in rois],
            dtype=float
        )
        new_bbs_high = np.array(
            [info['high_bbox'] for info in roi_scaled_info],
            dtype=int
        )
        
        # Combine with existing metadata
        if existing_centroids is not None:
            combined_centroids = np.vstack([existing_centroids, new_cents_low])
            combined_bboxes = np.vstack([existing_bboxes, new_bbs_high])
        else:
            combined_centroids = new_cents_low
            combined_bboxes = new_bbs_high
            
        outz.create_dataset('centroids_lowres', data=combined_centroids)
        outz.create_dataset('bboxes_highres', data=combined_bboxes)
        
        # 12) Atomically replace the old zarr with the new one
        if out_grp.exists():
            shutil.rmtree(out_grp)
        temp_grp.rename(out_grp)
        
        print(f"✅ Successfully bundled {total_n_rois} ROIs for {cond}")
        return out_grp



    def view_bundled_roi_notebook(self,
                                   bundled_path: Path,
                                   roi_index: int = 0,
                                   channel: int = 0,
                                   mode: str = 'max',
                                   z_index: int = None):
        """
        Load one ROI from bundled Zarr and show in notebook.
        """
        bundle = zarr.open_group(str(bundled_path), mode='r')
        roi = bundle['data'][roi_index]
        if roi.ndim == 5 and roi.shape[0] == 1: roi = roi[0]
        vol = roi[channel]
        if mode=='max': img = vol.max(axis=0)
        else:
            zi = z_index if z_index is not None else vol.shape[0]//2
            img = vol[zi]
        plt.imshow(img, cmap='gray'); plt.title(f"ROI {roi_index}"); plt.axis('off'); plt.show()

    def view_bundled_roi_napari(self,
                                 bundled_path: Path,
                                 roi_index: int = 0):
        """
        Open one ROI from bundled Zarr in Napari.
        """
        bundle = zarr.open_group(str(bundled_path), mode='r')
        data   = bundle['data'][roi_index]
        if data.ndim == 5 and data.shape[0] == 1: data = data[0]
        v = napari.Viewer(ndisplay=3)
        v.add_image(data, name=f"ROI_{roi_index}", channel_axis=0, blending='additive')
        napari.run()

    @staticmethod
    def remove_black_z_slices(volume: np.ndarray, 
                            threshold_pct: float = 0.001,
                            method: str = 'intensity_threshold',
                            min_slices: int = 3,
                            verbose: bool = False) -> np.ndarray:
        """
        Remove z-slices that are predominantly black/empty from a 4D volume.
        
        Parameters
        ----------
        volume : np.ndarray
            Input volume with shape (c, z, y, x)
        threshold_pct : float, optional
            Percentage of maximum intensity below which a slice is considered "black".
            For 'intensity_threshold': slices with max intensity < threshold_pct * global_max are removed
            For 'content_ratio': slices with >threshold_pct fraction of near-zero pixels are removed
        method : str, optional
            Method for determining black slices:
            - 'intensity_threshold': based on maximum intensity per slice
            - 'content_ratio': based on fraction of non-zero pixels
            - 'mean_intensity': based on mean intensity per slice
        min_slices : int, optional
            Minimum number of slices to keep (prevents removing all slices)
        verbose : bool, optional
            Print debugging information
            
        Returns
        -------
        np.ndarray
            Cropped volume with black slices removed, shape (c, z_new, y, x)
        """
        if volume.ndim != 4:
            raise ValueError(f"Expected 4D volume (c, z, y, x), got {volume.ndim}D")
        
        c, z, y, x = volume.shape
        
        if z <= min_slices:
            if verbose:
                print(f"   Volume has only {z} slices (<= min_slices={min_slices}), keeping all")
            return volume
        
        # Compute metrics for each z-slice across all channels
        if method == 'intensity_threshold':
            # Use maximum intensity per slice
            global_max = np.max(volume)
            if global_max == 0:
                if verbose:
                    print("   Volume is completely black, keeping original")
                return volume
            
            # Max intensity per slice across all channels
            slice_metrics = np.max(volume, axis=(0, 2, 3))  # Shape: (z,)
            threshold_value = threshold_pct * global_max
            keep_mask = slice_metrics >= threshold_value
            
        elif method == 'content_ratio':
            # Use fraction of non-zero pixels
            total_pixels_per_slice = c * y * x
            nonzero_counts = np.count_nonzero(volume, axis=(0, 2, 3))  # Shape: (z,)
            content_ratio = nonzero_counts / total_pixels_per_slice
            keep_mask = content_ratio >= threshold_pct
            
        elif method == 'mean_intensity':
            # Use mean intensity per slice
            global_mean = np.mean(volume[volume > 0])  # Mean of non-zero pixels
            if global_mean == 0:
                if verbose:
                    print("   Volume has no non-zero pixels, keeping original")
                return volume
            
            slice_means = np.mean(volume, axis=(0, 2, 3))  # Shape: (z,)
            threshold_value = threshold_pct * global_mean
            keep_mask = slice_means >= threshold_value
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Ensure we keep at least min_slices
        n_keep = np.sum(keep_mask)
        if n_keep < min_slices:
            if verbose:
                print(f"   Only {n_keep} slices pass threshold, keeping top {min_slices} by intensity")
            # Keep the top min_slices by intensity
            if method == 'intensity_threshold':
                top_indices = np.argsort(slice_metrics)[-min_slices:]
            elif method == 'content_ratio':
                top_indices = np.argsort(nonzero_counts)[-min_slices:]
            else:  # mean_intensity
                top_indices = np.argsort(slice_means)[-min_slices:]
            
            keep_mask = np.zeros(z, dtype=bool)
            keep_mask[top_indices] = True
            n_keep = min_slices
        
        if verbose:
            print(f"   Removing {z - n_keep}/{z} z-slices (keeping {n_keep} slices)")
            print(f"   Kept slice indices: {np.where(keep_mask)[0].tolist()}")
        
        # Return cropped volume
        return volume[:, keep_mask, :, :]
