import zarr
import numpy as np
import dask
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
                                visualize: bool = False):
        """
        Segment level `self.low_level` to find centroids + padded bboxes.
        """
        root = zarr.open_group(str(zarr_path), mode='r')
        arr = root[str(self.low_level)][:]        # (t, c, z, y, x)
        vol = arr[0, 0]                          # (z, y, x)
        proj = vol.max(axis=0)                   # (y, x)

        # CLAHE + smoothing
        proj_eq     = exposure.equalize_adapthist(proj)
        proj_smooth = filters.gaussian(proj_eq, sigma=gaussian_sigma)

        # threshold
        base_th = threshold_otsu(proj_smooth)
        bw = proj_smooth > (base_th * lower_threshold)

        # cleanup
        bw = morphology.closing(bw, morphology.square(3))
        bw = remove_small_holes(bw, area_threshold=hole_area)
        bw = segmentation.clear_border(bw)

        # label + filter
        lbl   = label(bw)
        props = regionprops(lbl)

        rois = []
        for p in props:
            A = p.area
            if (min_area is not None and A < min_area) or (max_area is not None and A > max_area):
                continue
            bbox = self.pad_bbox(p.bbox, pad_pct, proj.shape)
            rois.append({
                'centroid': tuple(p.centroid),
                'bbox': bbox
            })

        # visualization
        if visualize:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(proj, cmap='gray'); axes[0].axis('off'); axes[0].set_title('Max-proj')
            axes[1].imshow(proj, cmap='gray'); axes[1].contour(bw, colors='r'); axes[1].axis('off');
            for roi in rois:
                r0, c0, r1, c1 = roi['bbox']; cy, cx = roi['centroid']
                axes[1].add_patch(plt.Rectangle((c0, r0), c1-c0, r1-r0, edgecolor='yellow', facecolor='none', linewidth=2))
                axes[1].plot(cx, cy, 'ro')
            plt.tight_layout(); plt.show()

        return rois

    def bundle_and_save_rois(self,
                              zarr_path: Path,
                              rois: list):
        """
        Upscale low-res ROIs to high_res coords, crop full-res,
        pad to uniform, and bundle into one Zarr.
        """
        src = Path(zarr_path)
        cond = src.parent.name
        assert cond.startswith(CONDITION_PREF)

        # prepare output
        out_grp = self.cropped_dir / cond / f"{cond}_bundled.zarr"
        if out_grp.exists(): shutil.rmtree(out_grp)

        # open
        root      = zarr.open_group(str(zarr_path), mode='r')
        arr_low   = root[str(self.low_level)]; arr_high = root[str(self.high_level)]
        data      = arr_high[...]              # load full-res (t,c,z,y,x)
        # drop singleton time
        if data.ndim == 5 and data.shape[0] == 1:
            data = data.squeeze(0)            # now (c,z,y,x)

        # compute scale factors
        scale_y = arr_high.shape[-2] / arr_low.shape[-2]
        scale_x = arr_high.shape[-1] / arr_low.shape[-1]
        # scale rois
        scaled = []
        for roi in rois:
            r0, c0, r1, c1 = roi['bbox']
            scaled_bbox = (
                int(round(r0*scale_y)), int(round(c0*scale_x)),
                int(round(r1*scale_y)), int(round(c1*scale_x))
            )
            cy, cx = roi['centroid']
            scaled_centroid = (cy*scale_y, cx*scale_x)
            scaled.append({'bbox': scaled_bbox, 'centroid': scaled_centroid})
        rois = scaled

        # crop tasks
        compressor = arr_high.compressor; dtype = arr_high.dtype
        orig_chunks = arr_high.chunks
        core_chunks = orig_chunks[-data.ndim:]
        tasks = [delayed(lambda d,b: d[..., :, b[0]:b[2], b[1]:b[3]])(data, roi['bbox']) for roi in rois]
        subs = dask.compute(*tasks, scheduler=self.scheduler)

        # pad uniform
        hmax = max(s.shape[-2] for s in subs); wmax = max(s.shape[-1] for s in subs)
        padded = []
        for s in subs:
            ph = hmax - s.shape[-2]; pw = wmax - s.shape[-1]
            pad = ((0,0),)*(s.ndim-2) + ((0,ph),(0,pw))
            padded.append(np.pad(s, pad, mode='constant', constant_values=0))

        stacked = np.stack(padded, axis=0)   # (n_rois, c,z,y,x)

        # write bundled
        g = zarr.open_group(str(out_grp), mode='w')
        g.create_dataset('data', data=stacked, chunks=(1,)+core_chunks, dtype=dtype, compressor=compressor)
        cents = np.array([r['centroid'] for r in rois], dtype=float)
        bbs   = np.array([r['bbox']     for r in rois], dtype=int)
        g.create_dataset('centroids', data=cents)
        g.create_dataset('bboxes', data=bbs)

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
