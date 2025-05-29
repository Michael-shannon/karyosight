import zarr
import numpy as np
import dask
import dask.array as da
# from dask import delayed
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
############################## SAVED BELOW ######################################################
#     def extract_rois_from_level(self,
#                                 zarr_path: Path,
#                                 lower_threshold: float = 0.8, #shroom
#                                 min_area: int = 20000,
#                                 max_area: int = None,
#                                 hole_area: int = 1000,
#                                 gaussian_sigma: float = 10,
#                                 pad_pct: float = 0.6,
#                                 pad_step: float = 0.05,
#                                 visualize: bool = False):
#                                 # lower_threshold: float = 0.8,
#                                 # min_area: int = 20000,
#                                 # max_area: int = None,
#                                 # hole_area: int = 1000,
#                                 # gaussian_sigma: float = 10,
#                                 # pad_pct: float = 0.6,
#                                 # visualize: bool = False):
#         """
#         Segment level `self.low_level` to find centroids + padded bboxes.
#         """
#         root = zarr.open_group(str(zarr_path), mode='r')
#         arr = root[str(self.low_level)][:]        # (t, c, z, y, x)
#         vol = arr[0, 0]                          # (z, y, x)
#         proj = vol.max(axis=0)                   # (y, x)

#         # CLAHE + smoothing
#         proj_eq     = exposure.equalize_adapthist(proj)
#         proj_smooth = filters.gaussian(proj_eq, sigma=gaussian_sigma)

#         # threshold
#         base_th = threshold_otsu(proj_smooth)
#         bw = proj_smooth > (base_th * lower_threshold)

#         # cleanup
#         bw = morphology.closing(bw, morphology.square(3))
#         bw = remove_small_holes(bw, area_threshold=hole_area)
#         bw = segmentation.clear_border(bw)

#         # label + filter
#         lbl   = label(bw)
#         props = regionprops(lbl)
# #################### NEW #######################
#         rois = []
#         # only consider “large” blobs as organoids
#         big_props = [p for p in props if p.area >= min_area and (max_area is None or p.area <= max_area)]
#         for p in big_props:
#             # start with the user‐supplied pad_pct
#             pad = pad_pct

#             # shrink bounding box until only *this* centroid remains inside
#             while pad >= 0:
#                 # padded bbox around this blob
#                 r0, c0, r1, c1 = self.pad_bbox(p.bbox, pad, proj.shape)

#                 # check if any *other* big blob’s centroid is also inside
#                 others = [q for q in big_props if q.label != p.label]
#                 collision = False
#                 for q in others:
#                     cy, cx = q.centroid
#                     if r0 <= cy < r1 and c0 <= cx < c1:
#                         collision = True
#                         break

#                 if not collision:
#                     # safe: only our main blob within this padded box
#                     rois.append({
#                         'centroid': tuple(p.centroid),
#                         'bbox':     (r0, c0, r1, c1)
#                     })
#                     break

#                 # else shrink the pad and try again
#                 pad -= pad_step

#             # if pad dropped below 0, fallback to the original region bbox
#             else:
#                 rois.append({
#                     'centroid': tuple(p.centroid),
#                     'bbox':     p.bbox
#                 })
# ################## NEW ABOVE, OLD BELOW #############################
#         # rois = []
#         # for p in props:
#         #     A = p.area
#         #     if (min_area is not None and A < min_area) or (max_area is not None and A > max_area):
#         #         continue
#         #     bbox = self.pad_bbox(p.bbox, pad_pct, proj.shape)
#         #     rois.append({
#         #         'centroid': tuple(p.centroid),
#         #         'bbox': bbox
#         #     })

#         # visualization
#         if visualize:
#             fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#             axes[0].imshow(proj, cmap='gray'); axes[0].axis('off'); axes[0].set_title('Max-proj')
#             axes[1].imshow(proj, cmap='gray'); axes[1].contour(bw, colors='r'); axes[1].axis('off');
#             for roi in rois:
#                 r0, c0, r1, c1 = roi['bbox']; cy, cx = roi['centroid']
#                 axes[1].add_patch(plt.Rectangle((c0, r0), c1-c0, r1-r0, edgecolor='yellow', facecolor='none', linewidth=2))
#                 axes[1].plot(cx, cy, 'ro')
#             plt.tight_layout(); plt.show()

#         return rois

################################### SAVED ABOVE #################################################

    # def extract_rois_from_level(self,
    #                             zarr_path: Path,
    #                             lower_threshold: float = 0.8,
    #                             min_area: int = 20000,
    #                             max_area: int = None,
    #                             hole_area: int = 1000,
    #                             gaussian_sigma: float = 10,
    #                             pad_pct: float = 0.6,
    #                             pad_step: float = 0.05,
    #                             visualize: bool = False):
    #     """
    #     Segment level `self.low_level` to find centroids + shrunk bboxes.
    #     Also plots original vs final ROIs and saves to CROPPED_DIR/grid_overview/.
    #     """
    #     # 1) open low‐res group & get max‐proj
    #     root = zarr.open_group(str(zarr_path), mode='r')
    #     arr  = root[str(self.low_level)][:]
    #     vol  = arr[0,0]            # (z,y,x)
    #     proj = vol.max(axis=0)     # (y,x)

    #     # 2) preprocess & threshold
    #     eq     = exposure.equalize_adapthist(proj)
    #     smooth = filters.gaussian(eq, sigma=gaussian_sigma)
    #     th0    = threshold_otsu(smooth)
    #     bw     = smooth > (th0 * lower_threshold)
    #     bw     = morphology.closing(bw, morphology.square(3))
    #     bw     = remove_small_holes(bw, area_threshold=hole_area)
    #     bw     = segmentation.clear_border(bw)

    #     # 3) find large blobs
    #     lbl      = label(bw)
    #     props    = regionprops(lbl)
    #     big_props= [p for p in props
    #                 if p.area >= min_area 
    #                 and (max_area is None or p.area <= max_area)]

    #     # 4) original padded bboxes at pad_pct
    #     orig_rois = [{
    #         'centroid': tuple(p.centroid),
    #         'bbox':     self.pad_bbox(p.bbox, pad_pct, proj.shape)
    #     } for p in big_props]

    #     # 5) shrink to one‐blob
    #     final_rois = []
    #     for p in big_props:
    #         pad = pad_pct
    #         while pad >= 0:
    #             r0,c0,r1,c1 = self.pad_bbox(p.bbox, pad, proj.shape)
    #             # if any _other_ big centroid inside, shrink
    #             collision = any(
    #                 (r0 <= q.centroid[0] < r1 and c0 <= q.centroid[1] < c1)
    #                 for q in big_props if q.label != p.label
    #             )
    #             if not collision:
    #                 final_rois.append({'centroid': tuple(p.centroid),
    #                                    'bbox':     (r0,c0,r1,c1)})
    #                 break
    #             pad -= pad_step
    #         else:
    #             # fallback
    #             final_rois.append({'centroid': tuple(p.centroid),
    #                                'bbox':     p.bbox})

    #     # 6) visualization + save
    #     if visualize or True:
    #         fig, axes = plt.subplots(1,2, figsize=(12,6))
    #         for ax, roi_list, title, color in zip(
    #             axes,
    #             [orig_rois, final_rois],
    #             ["Original padding", "Shrunk to main blob"],
    #             ["yellow", "lime"]
    #         ):
    #             ax.imshow(proj, cmap='gray')
    #             for roi in roi_list:
    #                 r0,c0,r1,c1 = roi['bbox']
    #                 cy,cx     = roi['centroid']
    #                 ax.add_patch(plt.Rectangle(
    #                     (c0,r0), c1-c0, r1-r0,
    #                     edgecolor=color, facecolor='none', lw=2
    #                 ))
    #                 ax.plot(cx, cy, 'ro')
    #             ax.set_title(title)
    #             ax.axis('off')
    #         plt.tight_layout()

    #         # derive condition / prefix / subgrid from zarr_path
    #         cond_name  = next(p.name for p in zarr_path.parents
    #                           if p.name.startswith(CONDITION_PREF))
    #         subgrid_id = zarr_path.parent.name
    #         prefix     = zarr_path.parent.parent.name

    #         # save
    #         overview_dir = Path(CROPPED_DIR)/"grid_overview"
    #         overview_dir.mkdir(parents=True, exist_ok=True)
    #         fname = f"{cond_name}__{prefix}__{subgrid_id}.png"
    #         fig.savefig(str(overview_dir/fname), dpi=150)

    #         if visualize:
    #             plt.show()
    #         plt.close(fig)

    #     return final_rois

    ########## THIS ONE WORKS AMAZING< SAVE BELOW #########################

    # def extract_rois_from_level(self,
    #                             zarr_path: Path,
    #                             lower_threshold: float = 0.8,
    #                             min_area: int = 20000,
    #                             max_area: int = None,
    #                             hole_area: int = 1000,
    #                             gaussian_sigma: float = 10,
    #                             pad_pct: float = 0.6,
    #                             pad_step: float = 0.05,
    #                             visualize: bool = False):
    #     """
    #     Segment level `self.low_level` to find centroids + padded bboxes,
    #     shrink to only the main blob, overlay contours, and print diagnostics.
    #     """
    #     # 1) load and project
    #     root = zarr.open_group(str(zarr_path), mode='r')
    #     arr  = root[str(self.low_level)][:]
    #     vol  = arr[0,0]           
    #     proj = vol.max(axis=0)    

    #     # 2) preprocess + threshold
    #     eq     = exposure.equalize_adapthist(proj)
    #     smooth = filters.gaussian(eq, sigma=gaussian_sigma)
    #     th0    = threshold_otsu(smooth)
    #     bw     = smooth > (th0 * lower_threshold)
    #     bw     = morphology.closing(bw, morphology.square(3))
    #     bw     = remove_small_holes(bw, area_threshold=hole_area)
    #     bw     = segmentation.clear_border(bw)

    #     # 3) find blobs
    #     lbl    = label(bw)
    #     props  = regionprops(lbl)
    #     big_props = [
    #         p for p in props
    #         if p.area >= min_area and (max_area is None or p.area <= max_area)
    #     ]
    #     print(f"🔍 Found {len(big_props)} large blobs (min_area={min_area})")

    #     # 4) original bboxes
    #     orig_rois = []
    #     for p in big_props:
    #         obbox = self.pad_bbox(p.bbox, pad_pct, proj.shape)
    #         orig_rois.append({'centroid':tuple(p.centroid),'bbox':obbox})
    #         print(f"  • Blob {p.label}: raw bbox={p.bbox}, orig pad={pad_pct}, padded bbox={obbox}")

    #     # 5) shrink to one‐blob
    #     final_rois = []
        
        
        
        
    #     # for p in big_props:
    #     #     pad = pad_pct
    #     #     print(f"\n🛠 Shrinking blob {p.label}, start pad={pad_pct}")
    #     #     while pad >= 0:
    #     #         r0,c0,r1,c1 = self.pad_bbox(p.bbox, pad, proj.shape)
    #     #         collisions = []
    #     #         for q in big_props:
    #     #             if q.label == p.label: continue
    #     #             cy,cx = q.centroid
    #     #             if r0 <= cy < r1 and c0 <= cx < c1:
    #     #                 collisions.append(q.label)
    #     #         if not collisions:
    #     #             print(f"   ✓ pad={pad:.3f} OK, final bbox=({r0},{c0},{r1},{c1})")
    #     #             final_rois.append({
    #     #                 'centroid':tuple(p.centroid),
    #     #                 'bbox':     (r0,c0,r1,c1)
    #     #             })
    #     #             break
    #     #         else:
    #     #             print(f"   ✗ pad={pad:.3f} collision with blobs {collisions}")
    #     #         pad -= pad_step
    #     #     else:
    #     #         print(f"   ⚠️  pad fell below 0, fallback to raw bbox {p.bbox}")
    #     #         final_rois.append({
    #     #             'centroid':tuple(p.centroid),
    #     #             'bbox':     p.bbox
    #     #         })



    #     # shrink to avoid overlapping any other blob’s mask bbox
    #     for p in big_props:
    #         pad = pad_pct
    #         print(f"\n🛠 Shrinking blob {p.label} (raw bbox={p.bbox}), start pad={pad_pct}")
    #         while pad >= 0:
    #             r0, c0, r1, c1 = self.pad_bbox(p.bbox, pad, proj.shape)
    #             collisions = []
    #             for q in big_props:
    #                 if q.label == p.label:
    #                     continue
    #                 # check bbox overlap between ROI and other blob’s bbox
    #                 qr0, qc0, qr1, qc1 = q.bbox
    #                 # if the two rectangles intersect:
    #                 if not (qr1 <= r0 or qr0 >= r1 or qc1 <= c0 or qc0 >= c1):
    #                     collisions.append(q.label)
    #             if not collisions:
    #                 print(f"   ✓ pad={pad:.3f} OK, final bbox=({r0},{c0},{r1},{c1})")
    #                 final_rois.append({'centroid':tuple(p.centroid),
    #                                    'bbox':(r0,c0,r1,c1)})
    #                 break
    #             else:
    #                 print(f"   ✗ pad={pad:.3f} collisions with blobs {collisions}")
    #             pad -= pad_step
    #         else:
    #             print(f"   ⚠️  pad < 0, fallback to raw bbox {p.bbox}")
    #             final_rois.append({'centroid':tuple(p.centroid),
    #                                'bbox':p.bbox})                

    #     # 6) visualize with contours and save
    #     if visualize:
    #         fig, axes = plt.subplots(1,2, figsize=(12,6))
    #         titles = ["Original padding", "Shrunk to main blob"]
    #         colors = ["yellow", "lime"]
    #         for ax, roi_list, title, color in zip(axes, [orig_rois, final_rois], titles, colors):
    #             ax.imshow(proj, cmap='gray')
    #             # overlay full bw-contour
    #             ax.contour(bw, levels=[0.5], colors='red', linewidths=0.5)
    #             # overlay each blob's contour
    #             for p in big_props:
    #                 mask = (lbl == p.label)
    #                 ax.contour(mask, levels=[0.5], colors='cyan')
    #             # overlay the ROI rectangles+centroids
    #             for roi in roi_list:
    #                 r0,c0,r1,c1 = roi['bbox']
    #                 cy,cx     = roi['centroid']
    #                 ax.add_patch(plt.Rectangle(
    #                     (c0,r0), c1-c0, r1-r0,
    #                     edgecolor=color, facecolor='none', lw=2
    #                 ))
    #                 ax.plot(cx, cy, 'ro')
    #             ax.set_title(title)
    #             ax.axis('off')
    #         plt.tight_layout()

    #         # save
    #         cond_name  = next(p.name for p in zarr_path.parents
    #                           if p.name.startswith(CONDITION_PREF))
    #         subgrid_id = zarr_path.parent.name
    #         prefix     = zarr_path.parent.parent.name
    #         overview_dir = Path(CROPPED_DIR)/"grid_overview"
    #         overview_dir.mkdir(exist_ok=True, parents=True)
    #         fname = f"{cond_name}__{prefix}__{subgrid_id}.png"
    #         fig.savefig(str(overview_dir/fname), dpi=150)
    #         plt.show()
    #         plt.close(fig)

    #     return final_rois

########## ABOVE WORKS AMAZING< SAVE ABOVE #########################

    # def extract_rois_from_level(self,
    #                             zarr_path: Path,
    #                             lower_threshold: float = 0.8,
    #                             min_area: int = 20000,
    #                             max_area: int = None,
    #                             hole_area: int = 1000,
    #                             gaussian_sigma: float = 10,
    #                             pad_pct: float = 0.6,
    #                             pad_step: float = 0.05,
    #                             visualize: bool = False,
    #                             verbose: bool = False):
    #     """
    #     Segment level `self.low_level` to find centroids + padded bboxes,
    #     shrink to only the main blob, overlay contours, and print diagnostics.
    #     """
    #     # 1) load and project
    #     root = zarr.open_group(str(zarr_path), mode='r')
    #     arr  = root[str(self.low_level)][:]
    #     vol  = arr[0,0]           
    #     proj = vol.max(axis=0)    

    #     # 2) preprocess + threshold
    #     eq     = exposure.equalize_adapthist(proj)
    #     smooth = filters.gaussian(eq, sigma=gaussian_sigma)
    #     th0    = threshold_otsu(smooth)
    #     bw     = smooth > (th0 * lower_threshold)
    #     bw     = morphology.closing(bw, morphology.square(3))
    #     bw     = remove_small_holes(bw, area_threshold=hole_area)
    #     bw     = segmentation.clear_border(bw)

    #     # 3) find blobs
    #     # lbl    = label(bw) # thor
    #     # props  = regionprops(lbl) #thor
    #     label_img = label(bw)
    #     props = regionprops(label_img) 

    #     big_props = [
    #         p for p in props
    #         if p.area >= min_area and (max_area is None or p.area <= max_area)
    #     ]
    #     # print(f"🔍 Found {len(big_props)} large blobs (min_area={min_area})")
    #     if verbose:
    #         print(f"🔍 Found {len(big_props)} large organoids (min_area={min_area})")

    #     # map each organoid label → its raw low-res bbox
    #     lowres_bbox_map = {p.label: p.bbox for p in big_props} #added this to pass to bundle and save. #RASMUS

    #     # 4) original bboxes
    #     orig_rois = []
    #     # for p in big_props:
    #     #     # obbox = self.pad_bbox(p.bbox, pad_pct, proj.shape)
    #     #     # orig_rois.append({'centroid':tuple(p.centroid),'bbox':obbox})
    #     #     # print(f"  • Blob {p.label}: raw bbox={p.bbox}, orig pad={pad_pct}, padded bbox={obbox}")

    #     #     obbox = self.pad_bbox(p.bbox, pad_pct, proj.shape)
    #     #     orig_rois.append({'centroid':tuple(p.centroid),
    #     #                       'bbox':obbox,
    #     #                       'orig_bbox':p.bbox})
    #     #     if verbose:
    #     #        print(f"  • Organoid {p.label}: raw bbox={p.bbox}, orig pad={pad_pct}, padded bbox={obbox}")
    #     # orig_rois = []
    #     for p in big_props:
    #         obbox = self.pad_bbox(p.bbox, pad_pct, proj.shape)
    #         orig_rois.append({
    #             'label': p.label,
    #             'centroid': tuple(p.centroid),
    #             'bbox': obbox,
    #             'orig_bbox_map': lowres_bbox_map
    #         })
    #         if verbose:
    #             print(f"  • Organoid {p.label}: raw bbox={p.bbox}, orig pad={pad_pct}, padded bbox={obbox}")



    #     # 5) shrink to one‐blob
    #     final_rois = []
        
        
        
        
    #     # for p in big_props:
    #     #     pad = pad_pct
    #     #     print(f"\n🛠 Shrinking blob {p.label}, start pad={pad_pct}")
    #     #     while pad >= 0:
    #     #         r0,c0,r1,c1 = self.pad_bbox(p.bbox, pad, proj.shape)
    #     #         collisions = []
    #     #         for q in big_props:
    #     #             if q.label == p.label: continue
    #     #             cy,cx = q.centroid
    #     #             if r0 <= cy < r1 and c0 <= cx < c1:
    #     #                 collisions.append(q.label)
    #     #         if not collisions:
    #     #             print(f"   ✓ pad={pad:.3f} OK, final bbox=({r0},{c0},{r1},{c1})")
    #     #             final_rois.append({
    #     #                 'centroid':tuple(p.centroid),
    #     #                 'bbox':     (r0,c0,r1,c1)
    #     #             })
    #     #             break
    #     #         else:
    #     #             print(f"   ✗ pad={pad:.3f} collision with blobs {collisions}")
    #     #         pad -= pad_step
    #     #     else:
    #     #         print(f"   ⚠️  pad fell below 0, fallback to raw bbox {p.bbox}")
    #     #         final_rois.append({
    #     #             'centroid':tuple(p.centroid),
    #     #             'bbox':     p.bbox
    #     #         })



    #     # shrink to avoid overlapping any other blob’s mask bbox
    #     # for p in big_props:
    #     #     pad = pad_pct
    #     #     print(f"\n🛠 Shrinking blob {p.label} (raw bbox={p.bbox}), start pad={pad_pct}")
    #     for p in big_props:
    #         pad = pad_pct
    #         if verbose:
    #             print(f"\n🛠 Shrinking organoid {p.label}, start pad={pad_pct}")

    #         while pad >= 0:
    #             r0, c0, r1, c1 = self.pad_bbox(p.bbox, pad, proj.shape)
    #             collisions = []
    #             for q in big_props:
    #                 if q.label == p.label:
    #                     continue
    #                 # check bbox overlap between ROI and other blob’s bbox
    #                 qr0, qc0, qr1, qc1 = q.bbox
    #                 # if the two rectangles intersect:
    #                 if not (qr1 <= r0 or qr0 >= r1 or qc1 <= c0 or qc0 >= c1):
    #                     collisions.append(q.label)


    #             # if not collisions:
    #             #     print(f"   ✓ pad={pad:.3f} OK, final bbox=({r0},{c0},{r1},{c1})")
    #             #     final_rois.append({'centroid':tuple(p.centroid),
    #             #                        'bbox':(r0,c0,r1,c1)})
    #             if not collisions:
    #                 if verbose:
    #                     print(f"   ✓ pad={pad:.3f} OK, final bbox=({r0},{c0},{r1},{c1})")
    #                 # final_rois.append({
    #                 #     'centroid':       tuple(p.centroid),
    #                 #     'bbox':           (r0,c0,r1,c1),
    #                 #     'pad_used':       pad,
    #                 #     'collision_labels': []
    #                 # })                    
    #                 final_rois.append({
    #                     'label': p.label,
    #                     'centroid': tuple(p.centroid),
    #                     'bbox': (r0,c0,r1,c1),
    #                     'pad_used': pad,
    #                     'collision_labels': [],
    #                     'orig_bbox_map': lowres_bbox_map
    #                 })

    #                 break
    #             else:
    #                 if verbose:
    #                     print(f"   ✗ pad={pad:.3f} collisions with other organoids {collisions}")
    #                 # print(f"   ✗ pad={pad:.3f} collisions with blobs {collisions}")
    #             pad -= pad_step
    #         else:
    #             # print(f"   ⚠️  pad < 0, fallback to raw bbox {p.bbox}")
    #             # final_rois.append({'centroid':tuple(p.centroid),
    #             #                    'bbox':p.bbox})                
    #             if verbose:
    #                 print(f"   ⚠️  pad < 0, reverting to original padded bbox and marking overlaps")
    #             # revert to original padded bbox and record collisions at pad=0
    #             obox = self.pad_bbox(p.bbox, pad_pct, proj.shape)
    #             # collisions at pad=0
    #             coll0 = [q.label for q in big_props
    #                      if not (q.bbox[2] <= obox[0] or q.bbox[0] >= obox[2]
    #                              or q.bbox[3] <= obox[1] or q.bbox[1] >= obox[3])]
    #             # final_rois.append({
    #             #     'centroid':        tuple(p.centroid),
    #             #     'bbox':            obox,
    #             #     'pad_used':        pad_pct,
    #             #     'collision_labels': coll0
    #             # })

    #             final_rois.append({
    #                 'label': p.label,
    #                 'centroid': tuple(p.centroid),
    #                 'bbox': (r0,c0,r1,c1),
    #                 'pad_used': pad,
    #                 'collision_labels': [],
    #                 'orig_bbox_map': lowres_bbox_map
    #             })

    #     # 6) visualize with contours and save
    #     if visualize:
    #         fig, axes = plt.subplots(1,2, figsize=(12,6))
    #         titles = ["Original padding", "Shrunk to main organoid"]
    #         colors = ["yellow", "lime"]
    #         for ax, roi_list, title, color in zip(axes, [orig_rois, final_rois], titles, colors):
    #             ax.imshow(proj, cmap='gray')
    #             # overlay full bw-contour
    #             ax.contour(bw, levels=[0.5], colors='red', linewidths=0.5)
    #             # overlay each blob's contour
    #             for p in big_props:
    #                 # mask = (lbl == p.label) #thor
    #                 mask = (label_img == p.label)
    #                 # print(f"   • Organoid FART {p.label}: label img={label_img}")
    #                 ax.contour(mask, levels=[0.5], colors='cyan')
    #             # overlay the ROI rectangles+centroids
    #             for roi in roi_list:
    #                 r0,c0,r1,c1 = roi['bbox']
    #                 cy,cx     = roi['centroid']
    #                 oid = roi['label']
    #                 ax.add_patch(plt.Rectangle(
    #                     (c0,r0), c1-c0, r1-r0,
    #                     edgecolor=color, facecolor='none', lw=2
    #                 ))
    #                 ax.plot(cx, cy, 'ro')
    #                 # Annotate label at the top-left corner of the box
    #                 ax.text(c0, r0, str(oid),  #  - 5
    #                         color='white',
    #                         backgroundcolor=None,
    #                         fontsize=8,
    #                         weight='bold')
    #             ax.set_title(title)
    #             ax.axis('off')
    #         plt.tight_layout()

    #         # save
    #         cond_name  = next(p.name for p in zarr_path.parents
    #                           if p.name.startswith(CONDITION_PREF))
    #         subgrid_id = zarr_path.parent.name
    #         prefix     = zarr_path.parent.parent.name
    #         overview_dir = Path(CROPPED_DIR)/"grid_overview"
    #         overview_dir.mkdir(exist_ok=True, parents=True)
    #         fname = f"{cond_name}__{prefix}__{subgrid_id}.png"
    #         fig.savefig(str(overview_dir/fname), dpi=150)
    #         plt.show()
    #         plt.close(fig)

    #     return final_rois

########## ABOVE ODIN ##############################################################################_----------------------------------------

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




    # def bundle_and_save_rois(self,
    #                           zarr_path: Path,
    #                           rois: list):
    #     """
    #     Upscale low-res ROIs to high_res coords, crop full-res,
    #     pad to uniform, and bundle into one Zarr.
    #     """
    #     src = Path(zarr_path)
    #     cond = src.parent.name
    #     assert cond.startswith(CONDITION_PREF)

    #     # prepare output
    #     out_grp = self.cropped_dir / cond / f"{cond}_bundled.zarr"
    #     if out_grp.exists(): shutil.rmtree(out_grp)

    #     # open
    #     root      = zarr.open_group(str(zarr_path), mode='r')
    #     arr_low   = root[str(self.low_level)]; arr_high = root[str(self.high_level)]
    #     data      = arr_high[...]              # load full-res (t,c,z,y,x)
    #     # drop singleton time
    #     if data.ndim == 5 and data.shape[0] == 1:
    #         data = data.squeeze(0)            # now (c,z,y,x)

    #     # compute scale factors
    #     scale_y = arr_high.shape[-2] / arr_low.shape[-2]
    #     scale_x = arr_high.shape[-1] / arr_low.shape[-1]
    #     # scale rois
    #     scaled = []
    #     for roi in rois:
    #         r0, c0, r1, c1 = roi['bbox']
    #         scaled_bbox = (
    #             int(round(r0*scale_y)), int(round(c0*scale_x)),
    #             int(round(r1*scale_y)), int(round(c1*scale_x))
    #         )
    #         cy, cx = roi['centroid']
    #         scaled_centroid = (cy*scale_y, cx*scale_x)
    #         scaled.append({'bbox': scaled_bbox, 'centroid': scaled_centroid})
    #     rois = scaled

    #     # crop tasks
    #     compressor = arr_high.compressor; dtype = arr_high.dtype
    #     orig_chunks = arr_high.chunks
    #     core_chunks = orig_chunks[-data.ndim:]
    #     tasks = [delayed(lambda d,b: d[..., :, b[0]:b[2], b[1]:b[3]])(data, roi['bbox']) for roi in rois]
    #     subs = dask.compute(*tasks, scheduler=self.scheduler)

    #     # pad uniform
    #     hmax = max(s.shape[-2] for s in subs); wmax = max(s.shape[-1] for s in subs)
    #     padded = []
    #     for s in subs:
    #         ph = hmax - s.shape[-2]; pw = wmax - s.shape[-1]
    #         pad = ((0,0),)*(s.ndim-2) + ((0,ph),(0,pw))
    #         padded.append(np.pad(s, pad, mode='constant', constant_values=0))

    #     stacked = np.stack(padded, axis=0)   # (n_rois, c,z,y,x)

    #     # write bundled
    #     g = zarr.open_group(str(out_grp), mode='w')
    #     g.create_dataset('data', data=stacked, chunks=(1,)+core_chunks, dtype=dtype, compressor=compressor)
    #     cents = np.array([r['centroid'] for r in rois], dtype=float)
    #     bbs   = np.array([r['bbox']     for r in rois], dtype=int)
    #     g.create_dataset('centroids', data=cents)
    #     g.create_dataset('bboxes', data=bbs)
    ###################### BELOW WORKED WELL ######################

    # def bundle_and_save_rois(self,
    #                           zarr_path: Path,
    #                           rois: list):
    #     """
    #     Upscale low-res ROIs to high_res coords, crop full-res,
    #     pad to uniform, and bundle into one Zarr.
    #     """
    #     # src = Path(zarr_path)
    #     # cond = src.parent.name
    #     # assert cond.startswith(CONDITION_PREF)

    #     src = Path(zarr_path)
    #     # climb up until we hit a folder whose name starts with CONDITION_PREF
    #     cond_dir = next(
    #         (p for p in src.parents if p.name.startswith(CONDITION_PREF)),
    #         None
    #     )
    #     if cond_dir is None:
    #         raise RuntimeError(f"Could not find a parent Condition_ folder for {src!r}")
    #     cond = cond_dir.name

    #     # prepare output
    #     # out_grp = self.cropped_dir / cond / f"{cond}_bundled.zarr"
    #     out_grp = self.cropped_dir / cond / f"{cond}_bundled.zarr"
    #     out_grp.parent.mkdir(exist_ok=True, parents=True)
    #     # if out_grp.exists(): shutil.rmtree(out_grp)

    #     # open
    #     root      = zarr.open_group(str(zarr_path), mode='r')
    #     arr_low   = root[str(self.low_level)]; arr_high = root[str(self.high_level)]
    #     data      = arr_high[...]              # load full-res (t,c,z,y,x)
    #     # drop singleton time
    #     if data.ndim == 5 and data.shape[0] == 1:
    #         data = data.squeeze(0)            # now (c,z,y,x)

    #     # compute scale factors
    #     scale_y = arr_high.shape[-2] / arr_low.shape[-2]
    #     scale_x = arr_high.shape[-1] / arr_low.shape[-1]
    #     # scale rois
    #     scaled = []
    #     for roi in rois:
    #         r0, c0, r1, c1 = roi['bbox']
    #         scaled_bbox = (
    #             int(round(r0*scale_y)), int(round(c0*scale_x)),
    #             int(round(r1*scale_y)), int(round(c1*scale_x))
    #         )
    #         cy, cx = roi['centroid']
    #         scaled_centroid = (cy*scale_y, cx*scale_x)
    #         scaled.append({'bbox': scaled_bbox, 'centroid': scaled_centroid})
    #     rois = scaled

    #     # crop tasks
    #     compressor = arr_high.compressor; dtype = arr_high.dtype
    #     orig_chunks = arr_high.chunks
    #     core_chunks = orig_chunks[-data.ndim:]
    #     tasks = [delayed(lambda d,b: d[..., :, b[0]:b[2], b[1]:b[3]])(data, roi['bbox']) for roi in rois]
    #     subs = dask.compute(*tasks, scheduler=self.scheduler)

    #     # pad uniform
    #     hmax = max(s.shape[-2] for s in subs); wmax = max(s.shape[-1] for s in subs)
    #     padded = []
    #     for s in subs:
    #         ph = hmax - s.shape[-2]; pw = wmax - s.shape[-1]
    #         pad = ((0,0),)*(s.ndim-2) + ((0,ph),(0,pw))
    #         padded.append(np.pad(s, pad, mode='constant', constant_values=0))

    #     stacked = np.stack(padded, axis=0)   # (n_rois, c,z,y,x)

    #     # write bundled
    #     g = zarr.open_group(str(out_grp), mode='w')
    #     g.create_dataset('data', data=stacked, chunks=(1,)+core_chunks, dtype=dtype, compressor=compressor)
    #     cents = np.array([r['centroid'] for r in rois], dtype=float)
    #     bbs   = np.array([r['bbox']     for r in rois], dtype=int)
    #     g.create_dataset('centroids', data=cents)
    #     g.create_dataset('bboxes', data=bbs)    

    ################### ABOVE WORKED WELL ######################
########### BELOW IS FINAL D #########################
    
    # def bundle_and_save_rois(self,
    #                          zarr_path: Path,
    #                          rois: list):
    #     """
    #     Upscale ROIs, crop full-res, pad to uniform size, and bundle into one Zarr
    #     via Dask, streaming chunk-by-chunk to avoid RAM spikes.
    #     """
    #     # 1) locate condition name
    #     src = Path(zarr_path)
    #     cond_dir = next(
    #         (p for p in src.parents if p.name.startswith(CONDITION_PREF)),
    #         None
    #     )
    #     if cond_dir is None:
    #         raise RuntimeError(f"No Condition_ parent for {src!r}")
    #     cond = cond_dir.name

    #     # 2) prepare output group
    #     out_grp = self.cropped_dir / cond / f"{cond}_bundled.zarr"
    #     if out_grp.exists():
    #         shutil.rmtree(out_grp)
    #     out_grp.parent.mkdir(parents=True, exist_ok=True)

    #     # 3) open stitched Zarr, pull low/high arrays
    #     root     = zarr.open_group(str(zarr_path), mode='r')
    #     arr_low  = root[str(self.low_level)]
    #     arr_high = root[str(self.high_level)]
    #     data     = arr_high[...]           # (t,c,z,y,x)
    #     if data.ndim == 5 and data.shape[0] == 1:
    #         data = data.squeeze(0)         # → (c,z,y,x)

    #     # 4) scale and upscale ROI bboxes
    #     sy = arr_high.shape[-2] / arr_low.shape[-2]
    #     sx = arr_high.shape[-1] / arr_low.shape[-1]
    #     scaled = []
    #     for roi in rois:
    #         r0, c0, r1, c1 = roi['bbox']
    #         sb = (
    #             int(round(r0*sy)), int(round(c0*sx)),
    #             int(round(r1*sy)), int(round(c1*sx))
    #         )
    #         cy, cx = roi['centroid']
    #         sc = (cy*sy, cx*sx)
    #         scaled.append({'bbox': sb, 'centroid': sc})
    #     rois = scaled

    #     # 5) crop each ROI via delayed → list of NumPy slices
    #     tasks = [
    #         delayed(lambda arr, bb: arr[..., :, bb[0]:bb[2], bb[1]:bb[3]])(data, roi['bbox'])
    #         for roi in rois
    #     ]
    #     subs = da.compute(*tasks, scheduler=self.scheduler)

    #     # 6) find the maximum Y/X so we know how much padding to apply
    #     hmax = max(s.shape[-2] for s in subs)
    #     wmax = max(s.shape[-1] for s in subs)

    #     # 7) grab the 4D chunk shape from arr_high (last dims of its .chunks)
    #     #    Zarr stores its chunks as a tuple of ints per axis.
    #     #    After squeeze, data.ndim == 4, so:
    #     chunk_shape_4d = arr_high.chunks[-data.ndim:]

    #     # 8) build a Dask array for each ROI, padded to (c,z,hmax,wmax)
    #     darrs = []
    #     for s in subs:
    #         ph = hmax - s.shape[-2]
    #         pw = wmax - s.shape[-1]
    #         pad = ((0,0),)*(s.ndim-2) + ((0,ph),(0,pw))
    #         padded = np.pad(s, pad, mode='constant', constant_values=0)
    #         # wrap in Dask with exactly the 4D chunk shape we extracted
    #         darr = da.from_array(padded, chunks=chunk_shape_4d)
    #         darrs.append(darr)

    #     # 9) stack into one Dask array of shape (n_rois, c, z, y, x)
    #     stacked = da.stack(darrs, axis=0)
    #     # we want to chunk that first axis by 1 ROI at a time:
    #     # stacked_chunks = (1,) + chunk_shape_4d

    #     # 10) write out via Dask → Zarr (this streams each chunk to disk)
    #     # compressor = arr_high.compressor
    #     # stacked.to_zarr(
    #     #     store=str(out_grp),
    #     #     component='data',
    #     #     overwrite=True,
    #     #     compute=True,
    #     #     chunks=stacked_chunks,
    #     #     compressor=compressor
    #     # )

    #     # compressor = arr_high.compressor
    #     # stacked.to_zarr(
    #     #     str(out_grp),          # <— first positional argument is the Zarr path
    #     #     component='data',
    #     #     overwrite=True,
    #     #     compute=True,
    #     #     chunks=stacked_chunks,
    #     #     compressor=compressor
    #     # )        
    #     # 10) rechunk so we write one ROI at a time (1 along axis-0),
    #     #     and use arr_high's 4D chunk shape on the remaining axes:
    #     desired_chunks = (1,) + chunk_shape_4d
    #     stacked = stacked.rechunk(desired_chunks)
    
    #     # 11) write out via Dask → Zarr (streams each chunk through RAM one at a time)
    #     compressor = arr_high.compressor
    #     stacked.to_zarr(
    #         str(out_grp),          # first positional arg = path
    #         component='data',
    #         overwrite=True,
    #         compute=True,
    #         compressor=compressor
    #     )

    #     # 11) write your metadata (centroids + bboxes)
    #     g = zarr.open_group(str(out_grp), mode='a')
    #     cents = np.array([r['centroid'] for r in rois], dtype=float)
    #     bbs   = np.array([r['bbox']     for r in rois], dtype=int)
    #     g.create_dataset('centroids', data=cents)
    #     g.create_dataset('bboxes',    data=bbs)

    #     return out_grp
    
############ ABOVE IS FINAL D #########################

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
                             rois: list):
        """
        Upscale ROIs, crop full-res, pad to uniform size, and bundle into one Zarr
        via Dask, streaming chunk-by-chunk to avoid RAM spikes.
        """
        # 1) locate condition name
        src = Path(zarr_path)
        cond_dir = next(
            (p for p in src.parents if p.name.startswith(CONDITION_PREF)),
            None
        )
        if cond_dir is None:
            raise RuntimeError(f"No Condition_ parent for {src!r}")
        cond = cond_dir.name

        # 2) prepare output group
        out_grp = self.cropped_dir / cond / f"{cond}_bundled.zarr"
        if out_grp.exists():
            shutil.rmtree(out_grp)
        out_grp.parent.mkdir(parents=True, exist_ok=True)

        # 3) open stitched Zarr, pull low/high arrays
        root     = zarr.open_group(str(zarr_path), mode='r')
        arr_low  = root[str(self.low_level)]
        arr_high = root[str(self.high_level)]
        data     = arr_high[...]           # (t,c,z,y,x)
        if data.ndim == 5 and data.shape[0] == 1:
            data = data.squeeze(0)         # → (c,z,y,x)

        # 4) scale and upscale ROI bboxes
        sy = arr_high.shape[-2] / arr_low.shape[-2]
        sx = arr_high.shape[-1] / arr_low.shape[-1]
        scaled = []
        for roi in rois:
            r0, c0, r1, c1 = roi['bbox']
            sb = (
                int(round(r0*sy)), int(round(c0*sx)),
                int(round(r1*sy)), int(round(c1*sx))
            )
            cy, cx = roi['centroid']
            sc = (cy*sy, cx*sx)
            scaled.append({'bbox': sb, 'centroid': sc})
        rois = scaled

        # 5) crop each ROI via delayed → list of NumPy slices
        # tasks = [
        #     delayed(lambda arr, bb: arr[..., :, bb[0]:bb[2], bb[1]:bb[3]])(data, roi['bbox'])
        #     for roi in rois
        # ]
        # compute scale factors
        sy = arr_high.shape[-2] / arr_low.shape[-2]
        sx = arr_high.shape[-1] / arr_low.shape[-1]
        # pack q.bbox for each collision label
        # for roi in rois: # DROPPED RASMUS. Each ROI has its own dict with orig_bbox_map
        #     # map label -> raw low-res bbox
        #     roi['orig_bbox_map'] = {p.label:p.bbox for p in big_props}

        tasks = [
            delayed(self._crop_and_mask)(data, roi, sy, sx)
            for roi in rois
        ]


        subs = da.compute(*tasks, scheduler=self.scheduler)

        # 6) find the maximum Y/X so we know how much padding to apply
        hmax = max(s.shape[-2] for s in subs)
        wmax = max(s.shape[-1] for s in subs)

        # 7) grab the 4D chunk shape from arr_high (last dims of its .chunks)
        #    Zarr stores its chunks as a tuple of ints per axis.
        #    After squeeze, data.ndim == 4, so:
        chunk_shape_4d = arr_high.chunks[-data.ndim:]

        # 8) build a Dask array for each ROI, padded to (c,z,hmax,wmax)
        darrs = []
        for s in subs:
            ph = hmax - s.shape[-2]
            pw = wmax - s.shape[-1]
            pad = ((0,0),)*(s.ndim-2) + ((0,ph),(0,pw))
            padded = np.pad(s, pad, mode='constant', constant_values=0)
            # wrap in Dask with exactly the 4D chunk shape we extracted
            darr = da.from_array(padded, chunks=chunk_shape_4d)
            darrs.append(darr)

        # 9) stack into one Dask array of shape (n_rois, c, z, y, x)
        stacked = da.stack(darrs, axis=0)
        # we want to chunk that first axis by 1 ROI at a time:
        # stacked_chunks = (1,) + chunk_shape_4d

        # 10) write out via Dask → Zarr (this streams each chunk to disk)
        # compressor = arr_high.compressor
        # stacked.to_zarr(
        #     store=str(out_grp),
        #     component='data',
        #     overwrite=True,
        #     compute=True,
        #     chunks=stacked_chunks,
        #     compressor=compressor
        # )

        # compressor = arr_high.compressor
        # stacked.to_zarr(
        #     str(out_grp),          # <— first positional argument is the Zarr path
        #     component='data',
        #     overwrite=True,
        #     compute=True,
        #     chunks=stacked_chunks,
        #     compressor=compressor
        # )        
        # 10) rechunk so we write one ROI at a time (1 along axis-0),
        #     and use arr_high's 4D chunk shape on the remaining axes:
        desired_chunks = (1,) + chunk_shape_4d
        stacked = stacked.rechunk(desired_chunks)
    
        # 11) write out via Dask → Zarr (streams each chunk through RAM one at a time)
        compressor = arr_high.compressor
        stacked.to_zarr(
            str(out_grp),          # first positional arg = path
            component='data',
            overwrite=True,
            compute=True,
            compressor=compressor
        )

        # 11) write your metadata (centroids + bboxes)
        g = zarr.open_group(str(out_grp), mode='a')
        cents = np.array([r['centroid'] for r in rois], dtype=float)
        bbs   = np.array([r['bbox']     for r in rois], dtype=int)
        g.create_dataset('centroids', data=cents)
        g.create_dataset('bboxes',    data=bbs)

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
