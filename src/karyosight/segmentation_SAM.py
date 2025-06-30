"""
Enhanced nuclei segmentation using Cellpose-SAM (Cellpose 4.x)

This module provides enhanced nuclei segmentation capabilities using the latest
Cellpose with integrated SAM (Segment Anything Model) for improved edge detection
and segmentation accuracy.

Key Features:
- Cellpose 4.x API with cpsam model (SAM integration built-in)
- GPU acceleration with memory optimization
- 3D segmentation with automatic anisotropy correction
- Enhanced visualization and analysis tools
- Compatible with existing karyosight workflows
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass
import zarr
import time

# Add current directory to path for config access
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

try:
    from config import *
    CONFIG_AVAILABLE = True
except ImportError:
    print("Warning: Could not import config. Some parameters may need manual setting.")
    CONFIG_AVAILABLE = False
    # Default values if config not available
    PIXEL_SIZE_X = 0.325
    PIXEL_SIZE_Z = 1.0
    OPTIMAL_DIAMETER = 50.0
    OPTIMAL_FLOW_THRESHOLD = 0.4
    OPTIMAL_CELLPROB_THRESHOLD = 0.0
    USE_3D_SEGMENTATION = True
    OPTIMAL_BATCH_SIZE = 16
    SEGMENTATION_ANISOTROPY = PIXEL_SIZE_Z / PIXEL_SIZE_X
    MIN_NUCLEI_SIZE = 100
    CONSERVATIVE_MIN_SIZE = 100
    AGGRESSIVE_MIN_SIZE = 200

@dataclass
class SegmentationParams:
    """Parameters for Cellpose-SAM segmentation"""
    diameter: float = OPTIMAL_DIAMETER
    do_3D: bool = USE_3D_SEGMENTATION
    anisotropy: Optional[float] = None
    batch_size: int = OPTIMAL_BATCH_SIZE
    use_gpu: bool = True
    
    def __post_init__(self):
        """Calculate anisotropy if not provided"""
        if self.anisotropy is None:
            try:
                if CONFIG_AVAILABLE:
                    self.anisotropy = SEGMENTATION_ANISOTROPY
                else:
                    self.anisotropy = PIXEL_SIZE_Z / PIXEL_SIZE_X
            except NameError:
                print("Warning: Could not calculate anisotropy from config. Using default 3.0")
                self.anisotropy = 3.0

class NucleiSegmenterSAM:
    """
    Enhanced nuclei segmenter using Cellpose 4.x with SAM integration
    
    Features:
    - Built-in SAM enhancement for better edge detection
    - GPU memory optimization
    - 3D segmentation with proper anisotropy
    - Lazy model loading for memory efficiency
    - Zarr saving compatible with original segmentation structure
    """
    
    def __init__(self, use_gpu: bool = True, cropped_dir: str = None):
        """
        Initialize the nuclei segmenter
        
        Args:
            use_gpu: Whether to use GPU acceleration
            cropped_dir: Directory containing cropped zarr files
        """
        self.use_gpu = use_gpu
        self._model = None
        self._gpu_available = self._check_gpu()
        
        # Set up directories
        if cropped_dir is None:
            try:
                self.cropped_dir = Path(CROPPED_DIR)
            except NameError:
                self.cropped_dir = Path("D:/LB_TEST/cropped")
        else:
            self.cropped_dir = Path(cropped_dir)
        
        print(f"🧬 NucleiSegmenterSAM initialized")
        print(f"   GPU requested: {use_gpu}")
        print(f"   GPU available: {self._gpu_available}")
        print(f"   Will use: {'GPU' if self.use_gpu and self._gpu_available else 'CPU'}")
        print(f"   Cropped dir: {self.cropped_dir}")
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available for Cellpose"""
        try:
            from cellpose import core
            gpu_available = core.use_gpu()
            if gpu_available:
                import torch
                print(f"   🚀 GPU detected: {torch.cuda.get_device_name(0)}")
            return gpu_available
        except Exception as e:
            print(f"   ⚠️  GPU check failed: {e}")
            return False
    
    @property
    def model(self):
        """Lazy loading of Cellpose model"""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self):
        """Load Cellpose-SAM model (cpsam is default in v4.x)"""
        try:
            from cellpose import models, core
            
            # Use GPU if available and requested
            use_gpu = self.use_gpu and self._gpu_available
            
            print(f"🔄 Loading Cellpose-SAM model...")
            print(f"   Using: {'GPU' if use_gpu else 'CPU'}")
            
            # Cellpose 4.x API - cpsam (SAM integration) is the default model
            self._model = models.CellposeModel(gpu=use_gpu)
            
            print(f"✅ Cellpose-SAM model loaded successfully")
            print(f"   Model type: cpsam (Cellpose-SAM)")
            print(f"   GPU enabled: {use_gpu}")
            
        except Exception as e:
            print(f"❌ Failed to load Cellpose-SAM model: {e}")
            print("   Please ensure Cellpose 4.x is properly installed")
            raise
    
    def filter_objects_by_size(
        self,
        masks: np.ndarray,
        min_size: int,
        verbose: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Filter segmented objects by minimum size (post-processing)
        
        Args:
            masks: 3D segmentation masks [Z, Y, X] with integer labels
            min_size: Minimum number of pixels for an object to be kept
            verbose: Print filtering statistics
            
        Returns:
            Tuple of (filtered_masks, filtering_stats)
        """
        if verbose:
            print(f"   🔍 Filtering objects smaller than {min_size} pixels...")
        
        # Get unique labels (excluding background = 0)
        unique_labels = np.unique(masks)
        unique_labels = unique_labels[unique_labels > 0]
        
        original_count = len(unique_labels)
        
        if original_count == 0:
            if verbose:
                print(f"   ⚠️  No objects found to filter")
            return masks, {'original_count': 0, 'filtered_count': 0, 'removed_count': 0}
        
        # Count pixels for each object
        object_sizes = {}
        for label in unique_labels:
            object_sizes[label] = np.sum(masks == label)
        
        # Find objects to keep (above minimum size)
        labels_to_keep = [label for label, size in object_sizes.items() if size >= min_size]
        labels_to_remove = [label for label, size in object_sizes.items() if size < min_size]
        
        # Create filtered masks
        filtered_masks = np.zeros_like(masks)
        
        # Relabel kept objects with consecutive IDs starting from 1
        for new_id, old_label in enumerate(labels_to_keep, 1):
            filtered_masks[masks == old_label] = new_id
        
        filtered_count = len(labels_to_keep)
        removed_count = len(labels_to_remove)
        
        if verbose:
            print(f"   ✅ Filtering complete:")
            print(f"      → Original objects: {original_count}")
            print(f"      → Kept objects: {filtered_count}")
            print(f"      → Removed objects: {removed_count}")
            print(f"      → Removal rate: {(removed_count/original_count)*100:.1f}%")
        
        # Statistics about removed objects
        removed_sizes = [object_sizes[label] for label in labels_to_remove] if labels_to_remove else []
        kept_sizes = [object_sizes[label] for label in labels_to_keep] if labels_to_keep else []
        
        stats = {
            'original_count': original_count,
            'filtered_count': filtered_count,
            'removed_count': removed_count,
            'removal_rate': (removed_count/original_count)*100 if original_count > 0 else 0,
            'min_size_threshold': min_size,
            'removed_sizes': removed_sizes,
            'kept_sizes': kept_sizes,
            'mean_removed_size': np.mean(removed_sizes) if removed_sizes else 0,
            'mean_kept_size': np.mean(kept_sizes) if kept_sizes else 0
        }
        
        return filtered_masks, stats
    
    def segment_organoid(
        self,
        image: np.ndarray,
        params: Optional[SegmentationParams] = None,
        min_nuclei_size: Optional[int] = None,
        verbose: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Segment nuclei in a single organoid using Cellpose-SAM
        
        Args:
            image: 3D numpy array [Z, Y, X]
            params: Segmentation parameters
            min_nuclei_size: Minimum nuclei size in pixels (post-processing filter)
            verbose: Print progress information
            
        Returns:
            Tuple of (masks, metadata)
        """
        if params is None:
            params = SegmentationParams()
        
        if verbose:
            print(f"🔍 Segmenting organoid with Cellpose-SAM")
            print(f"   Image shape: {image.shape}")
            print(f"   Diameter: {params.diameter}")
            print(f"   3D: {params.do_3D}")
            print(f"   Anisotropy: {params.anisotropy:.2f}")
        
        start_time = time.time()
        
        try:
            # Run segmentation - Cellpose 4.x requires z_axis for 3D
            if params.do_3D and image.ndim == 3:
                # For 3D data in (Z, Y, X) format, z_axis=0
                masks, flows, styles = self.model.eval(
                    image,
                    diameter=params.diameter,
                    do_3D=params.do_3D,
                    anisotropy=params.anisotropy,
                    batch_size=params.batch_size,
                    z_axis=0,  # Z-axis is the first dimension
                    progress=verbose
                )
            else:
                # 2D segmentation
                masks, flows, styles = self.model.eval(
                    image,
                    diameter=params.diameter,
                    do_3D=False,
                    batch_size=params.batch_size,
                    progress=verbose
                )
            
            # Apply size filtering if requested
            if min_nuclei_size is not None and min_nuclei_size > 0:
                if verbose:
                    print(f"🔍 Applying size filtering (min_size = {min_nuclei_size} pixels)")
                
                # Apply size filter
                masks, filter_stats = self.filter_objects_by_size(
                    masks, min_nuclei_size, verbose=verbose
                )
                
                num_cells = filter_stats['filtered_count']
                filter_metadata = {
                    'size_filtering_applied': True,
                    'min_nuclei_size': min_nuclei_size,
                    'original_nuclei_count': filter_stats['original_count'],
                    'filtered_nuclei_count': filter_stats['filtered_count'],
                    'removed_nuclei_count': filter_stats['removed_count'],
                    'removal_rate_percent': filter_stats['removal_rate'],
                    'mean_removed_size': filter_stats['mean_removed_size'],
                    'mean_kept_size': filter_stats['mean_kept_size']
                }
            else:
                # Calculate stats without filtering
                num_cells = len(np.unique(masks)) - 1  # Subtract background
                filter_metadata = {
                    'size_filtering_applied': False,
                    'min_nuclei_size': None
                }
            
            processing_time = time.time() - start_time
            
            metadata = {
                'num_cells': num_cells,
                'processing_time': processing_time,
                'diameter': params.diameter,
                'anisotropy': params.anisotropy,
                'do_3D': params.do_3D,
                'model_type': 'cpsam',
                'gpu_used': self.use_gpu and self._gpu_available,
                'image_shape': image.shape,
                **filter_metadata  # Include size filtering metadata
            }
            
            if verbose:
                print(f"✅ Segmentation complete!")
                print(f"   Cells detected: {num_cells}")
                print(f"   Processing time: {processing_time:.1f}s")
            
            return masks, metadata
            
        except Exception as e:
            print(f"❌ Segmentation failed: {e}")
            raise
    
    def segment_zarr_file(
        self,
        zarr_path: str,
        organoid_idx: int,
        channel_idx: int = 0,
        params: Optional[SegmentationParams] = None,
        save_results: bool = True,
        output_dir: str = "segmented_sam"
    ) -> Tuple[np.ndarray, Dict]:
        """
        Segment nuclei from a zarr file using the same loading method as the original segmentation.py
        
        Args:
            zarr_path: Path to zarr file
            organoid_idx: Index of organoid to segment
            channel_idx: Channel index (default 0 for nuclei)
            params: Segmentation parameters
            save_results: Whether to save segmentation results
            output_dir: Directory to save results
            
        Returns:
            Tuple of (masks, metadata)
        """
        print(f"📂 Loading organoid {organoid_idx} from {zarr_path}")
        
        # Use the same zarr loading approach as the original segmentation.py
        input_bundle = zarr.open_group(str(zarr_path), mode='r')
        organoid_key = f"organoid_{organoid_idx:04d}"
        
        # Load organoid data using the same method as segmentation.py
        if hasattr(input_bundle, 'keys') and organoid_key in input_bundle:
            # New structure with individual organoid groups
            organoid_data = input_bundle[organoid_key]['data'][:]
            print(f"   Using new zarr structure: {organoid_key}")
        else:
            # Old structure with single 'data' array
            if 'data' in input_bundle:
                organoid_data = input_bundle['data'][organoid_idx]
                print(f"   Using old zarr structure: data[{organoid_idx}]")
            else:
                raise ValueError(f"Cannot find organoid data in zarr file. Available keys: {list(input_bundle.keys())}")
        
        # Extract channel for segmentation (c, z, y, x) -> (z, y, x)
        if organoid_data.ndim == 4:
            image = organoid_data[channel_idx]  # Extract channel
        elif organoid_data.ndim == 3:
            image = organoid_data  # Already single channel
        else:
            raise ValueError(f"Unexpected data shape: {organoid_data.shape}")
        
        # Convert to numpy array if needed
        if hasattr(image, 'compute'):
            image = image.compute()
        else:
            image = np.array(image)
        
        print(f"   Loaded image shape: {image.shape}")
        print(f"   Image dtype: {image.dtype}")
        print(f"   Image range: {image.min():.1f} - {image.max():.1f}")
        
        # Run segmentation
        masks, metadata = self.segment_organoid(image, params)
        
        # Add file information to metadata
        metadata.update({
            'zarr_path': zarr_path,
            'organoid_idx': organoid_idx,
            'channel_idx': channel_idx,
            'organoid_key': organoid_key,
            'data_structure': 'new' if organoid_key in input_bundle else 'old'
        })
        
        # Save results if requested
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            
            zarr_name = Path(zarr_path).stem
            output_file = f"{output_dir}/{zarr_name}_org{organoid_idx}_masks.zarr"
            
            # Save masks as zarr
            zarr.save(output_file, masks)
            print(f"💾 Saved segmentation to: {output_file}")
            
            # Save metadata
            metadata_file = f"{output_dir}/{zarr_name}_org{organoid_idx}_metadata.txt"
            with open(metadata_file, 'w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
            print(f"💾 Saved metadata to: {metadata_file}")
        
        return masks, metadata
    
    def find_bundled_zarrs(self) -> List[str]:
        """
        Find all bundled zarr files in the cropped directory.
        
        Returns:
            List of paths to bundled zarr files
        """
        zarr_files = []
        if self.cropped_dir.exists():
            for condition_dir in self.cropped_dir.iterdir():
                if condition_dir.is_dir():
                    bundled_zarr = condition_dir / f"{condition_dir.name}_bundled.zarr"
                    if bundled_zarr.exists():
                        zarr_files.append(str(bundled_zarr))
        
        print(f"📂 Found {len(zarr_files)} bundled zarr files")
        return sorted(zarr_files)
    
    def segment_zarr_file_and_save(
        self,
        zarr_path: str,
        channel: int = 0,
        params: Optional[SegmentationParams] = None,
        organoid_indices: Optional[List[int]] = None,
        overwrite: bool = False,
        progress_bar: bool = True
    ) -> str:
        """
        Segment nuclei in a bundled zarr file and save to segmented_SAM directory.
        
        Args:
            zarr_path: Path to the bundled zarr file
            channel: Channel to use for segmentation (0-based, typically 0 for DAPI/nuclei)
            params: Segmentation parameters
            organoid_indices: Specific organoid indices to segment. If None, segments all
            overwrite: Whether to overwrite existing segmentations
            progress_bar: Whether to show progress bar
            
        Returns:
            Path to the output segmented zarr file
        """
        if params is None:
            params = SegmentationParams()
        
        # Determine condition and setup output path
        zarr_path_obj = Path(zarr_path)
        condition_name = zarr_path_obj.parent.name
        output_zarr = self.segmented_dir / condition_name / f"{condition_name}_segmented_SAM.zarr"
        output_zarr.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🧬 Segmenting {condition_name} with Cellpose-SAM")
        print(f"   Input:  {zarr_path}")
        print(f"   Output: {output_zarr}")
        
        # Open input zarr
        input_bundle = zarr.open_group(str(zarr_path), mode='r')
        
        # Determine which organoids to process
        if hasattr(input_bundle, 'keys'):
            # New structure with individual organoid groups
            organoid_keys = [k for k in input_bundle.keys() if k.startswith('organoid_')]
            if organoid_indices is not None:
                organoid_keys = [f"organoid_{i:04d}" for i in organoid_indices if f"organoid_{i:04d}" in organoid_keys]
        else:
            # Old structure with single 'data' array
            data_shape = input_bundle['data'].shape
            n_organoids = data_shape[0]
            if organoid_indices is not None:
                organoid_indices = [i for i in organoid_indices if i < n_organoids]
            else:
                organoid_indices = list(range(n_organoids))
            organoid_keys = [f"organoid_{i:04d}" for i in organoid_indices]
        
        print(f"   → Processing {len(organoid_keys)} organoids")
        
        # Create or open output zarr
        if output_zarr.exists() and not overwrite:
            output_bundle = zarr.open_group(str(output_zarr), mode='r+')
            print(f"   → Appending to existing segmentation zarr")
        else:
            if output_zarr.exists():
                import shutil
                shutil.rmtree(output_zarr)
            output_bundle = zarr.open_group(str(output_zarr), mode='w')
            output_bundle.attrs['condition'] = condition_name
            output_bundle.attrs['segmentation_params'] = {
                'model_type': 'cpsam',
                'diameter': params.diameter,
                'do_3D': params.do_3D,
                'anisotropy': params.anisotropy,
                'batch_size': params.batch_size,
                'use_gpu': params.use_gpu,
                'channel': channel,
                'timestamp': str(time.time())
            }
        
        # Process each organoid
        try:
            from tqdm import tqdm
            has_tqdm = True
        except ImportError:
            has_tqdm = False
        
        if progress_bar and has_tqdm:
            organoid_iterator = tqdm(organoid_keys, desc=f"Segmenting {condition_name}")
        else:
            organoid_iterator = organoid_keys
        
        total_nuclei = 0
        for organoid_key in organoid_iterator:
            # Skip if already exists and not overwriting
            if organoid_key in output_bundle and not overwrite:
                if progress_bar and has_tqdm and hasattr(organoid_iterator, 'set_postfix_str'):
                    organoid_iterator.set_postfix_str("(skipped)")
                continue
            
            try:
                # Load organoid data using same method as segment_zarr_file
                if hasattr(input_bundle, 'keys') and organoid_key in input_bundle:
                    # New structure
                    organoid_data = input_bundle[organoid_key]['data'][:]
                else:
                    # Old structure
                    organoid_idx = int(organoid_key.split('_')[1])
                    organoid_data = input_bundle['data'][organoid_idx]
                
                # Extract channel for segmentation (c, z, y, x) -> (z, y, x)
                if organoid_data.ndim == 4:
                    img = organoid_data[channel]  # Extract channel
                elif organoid_data.ndim == 3:
                    img = organoid_data  # Already single channel
                else:
                    print(f"   ⚠️  Unexpected data shape: {organoid_data.shape}")
                    continue
                
                # Convert to numpy array if needed
                if hasattr(img, 'compute'):
                    img = img.compute()
                else:
                    img = np.array(img)
                
                # Run cellpose-SAM segmentation
                masks, metadata = self.segment_organoid(img, params, verbose=False)
                
                # Count nuclei
                n_nuclei = metadata['num_cells']
                total_nuclei += n_nuclei
                
                # Create organoid group in output
                if organoid_key in output_bundle:
                    del output_bundle[organoid_key]
                
                organoid_group = output_bundle.create_group(organoid_key)
                
                # Save masks with compression
                masks_uint16 = masks.astype(np.uint16)  # Save space with uint16
                organoid_group.create_dataset(
                    'masks',
                    shape=masks_uint16.shape,
                    data=masks_uint16,
                    chunks=(min(masks_uint16.shape[0], 32), 
                           min(masks_uint16.shape[1], 256), 
                           min(masks_uint16.shape[2], 256))
                )
                
                # Save metadata (compatible with original format but with SAM info)
                organoid_group.attrs.update({
                    'n_nuclei': int(n_nuclei),
                    'num_cells': int(n_nuclei),  # Alternative naming
                    'original_shape': list(organoid_data.shape),
                    'masks_shape': list(masks.shape),
                    'channel_used': channel,
                    'processing_time': metadata['processing_time'],
                    'model_type': 'cpsam',
                    'diameter': metadata['diameter'],
                    'anisotropy': metadata['anisotropy'],
                    'gpu_used': metadata['gpu_used']
                })
                
                if progress_bar and has_tqdm and hasattr(organoid_iterator, 'set_postfix_str'):
                    organoid_iterator.set_postfix_str(f"{n_nuclei} nuclei")
                
            except Exception as e:
                print(f"   ❌ Error processing {organoid_key}: {e}")
                continue
        
        # Update bundle metadata
        output_bundle.attrs['total_nuclei'] = int(total_nuclei)
        output_bundle.attrs['n_organoids'] = len(organoid_keys)
        output_bundle.attrs['last_updated'] = str(time.time())
        
        print(f"   ✅ Segmentation complete: {total_nuclei} nuclei across {len(organoid_keys)} organoids")
        return str(output_zarr)
    
    def segment_all_zarrs(
        self,
        channel: int = 0,
        params: Optional[SegmentationParams] = None,
        overwrite: bool = False,
        progress_bar: bool = True
    ) -> List[str]:
        """
        Segment all bundled zarr files in the cropped directory.
        
        Args:
            channel: Channel to use for segmentation
            params: Segmentation parameters
            overwrite: Whether to overwrite existing segmentations
            progress_bar: Whether to show progress bars
            
        Returns:
            List of paths to output segmented zarr files
        """
        zarr_files = self.find_bundled_zarrs()
        output_paths = []
        
        for zarr_path in zarr_files:
            try:
                output_path = self.segment_zarr_file_and_save(
                    zarr_path, 
                    channel=channel,
                    params=params,
                    overwrite=overwrite,
                    progress_bar=progress_bar
                )
                output_paths.append(output_path)
            except Exception as e:
                print(f"❌ Failed to segment {zarr_path}: {e}")
                continue
        
        print(f"\n🏁 Batch segmentation complete!")
        print(f"   → Processed {len(output_paths)}/{len(zarr_files)} zarr files")
        return output_paths
    
    def load_existing_segmentations(
        self,
        condition_name: str,
        organoid_idx: int,
        channel_idx: int = 0
    ) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
        """
        Load existing segmentations and raw image from cropped zarr files.
        
        Args:
            condition_name: Name of the condition
            organoid_idx: Index of organoid to load
            channel_idx: Channel index for raw image
            
        Returns:
            Tuple of (raw_image, segmentations_dict)
        """
        raw_image = None
        segmentations_dict = {}
        
        # Load both raw data and segmentation from cropped directory
        cropped_zarr = self.cropped_dir / condition_name / f"{condition_name}_bundled.zarr"
        
        if cropped_zarr.exists():
            print(f"📂 Loading data from: {cropped_zarr}")
            
            try:
                bundle = zarr.open_group(str(cropped_zarr), mode='r')
                organoid_key = f"organoid_{organoid_idx:04d}"
                
                if organoid_key in bundle:
                    organoid_group = bundle[organoid_key]
                    
                    # Load raw image
                    organoid_data = organoid_group['data'][:]
                    raw_image = organoid_data[channel_idx]
                    print(f"   ✅ Loaded raw image: {raw_image.shape}")
                    
                    # Load segmentation masks if available
                    if 'masks' in organoid_group:
                        masks = organoid_group['masks'][:]
                        metadata = dict(organoid_group.attrs)
                        n_nuclei = metadata.get('n_nuclei', len(np.unique(masks)) - 1)
                        
                        segmentation_name = f"Segmentation ({n_nuclei} nuclei)"
                        segmentations_dict[segmentation_name] = masks
                        
                        print(f"   ✅ Loaded segmentation: {n_nuclei} nuclei")
                    else:
                        print(f"   ⚠️  No segmentation masks found")
                else:
                    print(f"   ❌ Organoid {organoid_key} not found in {cropped_zarr}")
            
            except Exception as e:
                print(f"   ❌ Error loading data: {e}")
        
        return raw_image, segmentations_dict

# Napari visualization functions

def open_sam_segmentations_in_napari(
    image: np.ndarray,
    segmentations_dict: Dict[str, np.ndarray],
    image_name: str = "Original DAPI"
):
    """
    Open image and multiple SAM segmentations in napari for comparison.
    
    Args:
        image: 3D image data (z, y, x)
        segmentations_dict: Dictionary of segmentation results {name: masks}
        image_name: Name for the original image layer
    """
    try:
        import napari
    except ImportError:
        print("❌ Napari not available. Install with: pip install napari")
        return None
    
    print("🔍 Opening Cellpose-SAM results in napari...")
    
    # Create napari viewer
    viewer = napari.Viewer()
    
    # Add original image
    viewer.add_image(
        image, 
        name=image_name,
        colormap='gray',
        contrast_limits=[image.min(), np.percentile(image, 99)]
    )
    
    # Add each segmentation as a labels layer
    colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan']
    
    for i, (name, masks) in enumerate(segmentations_dict.items()):
        if masks.max() > 0:  # Only add if segmentation exists
            n_nuclei = len(np.unique(masks)) - 1
            viewer.add_labels(
                masks,
                name=f'{name} ({n_nuclei} nuclei)',
                opacity=0.7
            )
            print(f"   → Added: {name} ({n_nuclei} nuclei)")
        else:
            print(f"   → Skipped: {name} (failed segmentation)")
    
    print(f"\n🎯 Napari viewer opened!")
    print(f"💡 Tips for inspection:")
    print(f"   • Use the layer visibility toggles to compare segmentations")
    print(f"   • Scroll through z-slices with the slider or mouse wheel")
    print(f"   • Toggle between 2D and 3D view with the 2D/3D button")
    print(f"   • Adjust opacity of labels layers for better overlay visualization")
    print(f"   • Use Ctrl+E to toggle between different label colors")
    
    return viewer

def open_existing_segmentations_in_napari(
    segmenter: 'NucleiSegmenterSAM',
    condition_name: Optional[str] = None,
    organoid_idx: int = 0,
    channel_idx: int = 0
):
    """
    Open existing segmentations from disk in napari for inspection.
    
    Args:
        segmenter: NucleiSegmenterSAM instance
        condition_name: Name of condition to load (if None, uses first available)
        organoid_idx: Index of organoid to visualize
        channel_idx: Channel index for raw image
        
    Returns:
        napari.Viewer instance or None
    """
    try:
        import napari
    except ImportError:
        print("❌ Napari not available. Install with: pip install napari")
        return None
    
    # Find available conditions if not specified
    if condition_name is None:
        available_conditions = []
        if segmenter.segmented_dir.exists():
            for condition_dir in segmenter.segmented_dir.iterdir():
                if condition_dir.is_dir():
                    seg_zarr = condition_dir / f"{condition_dir.name}_segmented_SAM.zarr"
                    if seg_zarr.exists():
                        available_conditions.append(condition_dir.name)
        
        if not available_conditions:
            print("❌ No existing segmentations found")
            print(f"   → Check directory: {segmenter.segmented_dir}")
            return None
        
        condition_name = available_conditions[0]
        print(f"📋 Available conditions: {available_conditions}")
        print(f"🎯 Using: {condition_name}")
    
    # Load data
    raw_image, segmentations_dict = segmenter.load_existing_segmentations(
        condition_name, organoid_idx, channel_idx
    )
    
    if raw_image is None:
        print("❌ Could not load raw image")
        return None
    
    if not segmentations_dict:
        print("❌ No segmentations found to display")
        return None
    
    # Open in napari
    viewer = open_sam_segmentations_in_napari(
        image=raw_image,
        segmentations_dict=segmentations_dict,
        image_name=f"{condition_name} - Organoid {organoid_idx} - DAPI"
    )
    
    return viewer

def test_cellpose_sam_installation():
    """Test if Cellpose-SAM is properly installed"""
    print("🧪 TESTING CELLPOSE-SAM INSTALLATION")
    print("=" * 50)
    
    try:
        # Test imports
        import torch
        from cellpose import models, core
        
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        
        # Test Cellpose
        gpu_available = core.use_gpu()
        print(f"✅ Cellpose GPU: {gpu_available}")
        
        # Test model loading
        model = models.CellposeModel(gpu=gpu_available)
        print("✅ Cellpose-SAM model loaded successfully")
        print("✅ Model type: cpsam (Cellpose-SAM)")
        
        # Test with dummy data
        test_image = np.random.randint(0, 255, (10, 64, 64), dtype=np.uint8)
        masks, _, _ = model.eval(test_image, diameter=20, do_3D=True, z_axis=0)
        print(f"✅ Test segmentation successful: {masks.shape}")
        
        print("\n🎯 Cellpose-SAM installation is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        print("\nPlease run: pip install cellpose[gui]")
        return False

def segment_example_organoid_sam(
    zarr_path: Optional[str] = None,
    organoid_idx: int = 0,
    channel_idx: int = 0,
    diameter: float = 30.0,
    cropped_dir: str = "D:/LB_TEST/cropped"
) -> Tuple[np.ndarray, Dict]:
    """
    Example function to segment a single organoid with Cellpose-SAM
    
    Args:
        zarr_path: Path to zarr file (if None, finds first available bundled zarr)
        organoid_idx: Organoid index to segment
        channel_idx: Channel index (0 for nuclei)
        diameter: Expected cell diameter
        cropped_dir: Directory containing cropped zarr files
        
    Returns:
        Tuple of (masks, metadata)
    """
    # Initialize segmenter
    segmenter = NucleiSegmenterSAM(use_gpu=True, cropped_dir=cropped_dir)
    
    # Find zarr file if not provided
    if zarr_path is None:
        zarr_files = segmenter.find_bundled_zarrs()
        if not zarr_files:
            raise ValueError(f"No bundled zarr files found in {cropped_dir}")
        zarr_path = zarr_files[0]
        print(f"📂 Using first available zarr: {Path(zarr_path).name}")
    
    print(f"🔬 CELLPOSE-SAM ORGANOID SEGMENTATION")
    print(f"   File: {zarr_path}")
    print(f"   Organoid: {organoid_idx}")
    print(f"   Channel: {channel_idx}")
    print(f"   Diameter: {diameter}")
    
    try:
        # Set parameters
        params = SegmentationParams(
            diameter=diameter,
            do_3D=True,
            batch_size=8
        )
        
        # Run segmentation
        masks, metadata = segmenter.segment_zarr_file(
            zarr_path=zarr_path,
            organoid_idx=organoid_idx,
            channel_idx=channel_idx,
            params=params,
            save_results=False  # Don't save for examples
        )
        
        print(f"\n🎯 Segmentation Results:")
        print(f"   Cells detected: {metadata['num_cells']}")
        print(f"   Processing time: {metadata['processing_time']:.1f}s")
        print(f"   GPU used: {metadata['gpu_used']}")
        print(f"   Data structure: {metadata['data_structure']}")
        
        return masks, metadata
        
    except Exception as e:
        print(f"❌ Segmentation failed: {e}")
        raise

def visualize_segmentation_sam(
    image: np.ndarray,
    masks: np.ndarray,
    metadata: Dict,
    slice_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Visualize Cellpose-SAM segmentation results
    
    Args:
        image: Original 3D image [Z, Y, X]
        masks: Segmentation masks [Z, Y, X]
        metadata: Segmentation metadata
        slice_idx: Z-slice to visualize (middle slice if None)
        figsize: Figure size
    """
    if slice_idx is None:
        slice_idx = image.shape[0] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original image
    axes[0].imshow(image[slice_idx], cmap='gray')
    axes[0].set_title(f'Original (Z={slice_idx})')
    axes[0].axis('off')
    
    # Masks
    axes[1].imshow(masks[slice_idx], cmap='tab20')
    axes[1].set_title(f'Cellpose-SAM Masks\n{metadata["num_cells"]} cells')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(image[slice_idx], cmap='gray', alpha=0.7)
    axes[2].imshow(masks[slice_idx], cmap='tab20', alpha=0.3)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    # Add metadata
    info_text = f"""Model: {metadata.get('model_type', 'cpsam')}
Diameter: {metadata.get('diameter', 'N/A')}
3D: {metadata.get('do_3D', 'N/A')}
GPU: {metadata.get('gpu_used', 'N/A')}
Time: {metadata.get('processing_time', 0):.1f}s
Anisotropy: {metadata.get('anisotropy', 'N/A'):.2f}"""
    
    plt.figtext(0.02, 0.02, info_text, fontsize=9, family='monospace',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def load_parameter_testing_results(
    segmenter: 'NucleiSegmenterSAM',
    test_image: Optional[np.ndarray] = None
) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
    """
    Load parameter testing results for comparison in napari.
    
    Args:
        segmenter: NucleiSegmenterSAM instance
        test_image: Original test image (if None, tries to load from metadata)
        
    Returns:
        Tuple of (test_image, segmentations_dict)
    """
    param_test_dir = segmenter.segmented_dir / "parameter_testing"
    
    if not param_test_dir.exists():
        print("❌ No parameter testing results found")
        print(f"   → Expected directory: {param_test_dir}")
        return None, {}
    
    print(f"📂 Loading parameter testing results from: {param_test_dir}")
    
    # Find all zarr mask files
    mask_files = list(param_test_dir.glob("*_masks.zarr"))
    
    if not mask_files:
        print("❌ No mask zarr files found in parameter testing directory")
        return None, {}
    
    print(f"   Found {len(mask_files)} parameter configurations")
    
    segmentations_dict = {}
    
    # Load summary file if available
    summary_file = param_test_dir / "comprehensive_results_summary.json"
    results_summary = {}
    if summary_file.exists():
        try:
            import json
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
            # Convert to dict keyed by config_name
            for result in summary_data:
                if 'config_name' in result:
                    results_summary[result['config_name']] = result
        except Exception as e:
            print(f"   ⚠️  Could not load summary file: {e}")
    
    # Load each mask file
    for mask_file in mask_files:
        config_name = mask_file.stem.replace('_masks', '')
        
        try:
            # Load masks
            masks = zarr.open(str(mask_file), mode='r')[:]
            
            # Get metadata from summary or metadata file
            if config_name in results_summary:
                metadata = results_summary[config_name]
                num_cells = metadata.get('num_cells', len(np.unique(masks)) - 1)
                params = metadata.get('parameters', {})
                
                # Create descriptive name
                diameter = params.get('diameter', 'unknown')
                flow_thresh = params.get('flow_threshold', 'unknown')
                cellprob_thresh = params.get('cellprob_threshold', 'unknown')
                
                display_name = f"{config_name} ({num_cells} nuclei)"
            else:
                # Fallback to basic name
                num_cells = len(np.unique(masks)) - 1
                display_name = f"{config_name} ({num_cells} nuclei)"
            
            segmentations_dict[display_name] = masks
            print(f"   ✅ Loaded: {display_name}")
            
        except Exception as e:
            print(f"   ❌ Failed to load {mask_file.name}: {e}")
    
    # Try to load test image if not provided
    if test_image is None and results_summary:
        # Get image info from first result
        first_result = next(iter(results_summary.values()))
        if 'condition' in first_result and 'organoid_idx' in first_result and 'channel' in first_result:
            try:
                condition = first_result['condition']
                organoid_idx = first_result['organoid_idx']
                channel = first_result['channel']
                
                print(f"   📥 Loading original test image...")
                print(f"      → Condition: {condition}")
                print(f"      → Organoid: {organoid_idx}")
                print(f"      → Channel: {channel}")
                
                cropped_zarr = segmenter.cropped_dir / condition / f"{condition}_bundled.zarr"
                if cropped_zarr.exists():
                    input_bundle = zarr.open_group(str(cropped_zarr), mode='r')
                    organoid_key = f"organoid_{organoid_idx:04d}"
                    
                    if hasattr(input_bundle, 'keys') and organoid_key in input_bundle:
                        organoid_data = input_bundle[organoid_key]['data'][:]
                        test_image = organoid_data[channel]
                    else:
                        organoid_data = input_bundle['data'][organoid_idx]
                        test_image = organoid_data[channel]
                    
                    if hasattr(test_image, 'compute'):
                        test_image = test_image.compute()
                    else:
                        test_image = np.array(test_image)
                    
                    print(f"   ✅ Loaded test image: {test_image.shape}")
                
            except Exception as e:
                print(f"   ⚠️  Could not load test image: {e}")
    
    print(f"📊 Parameter testing results loaded:")
    print(f"   → Configurations: {len(segmentations_dict)}")
    print(f"   → Test image: {'Available' if test_image is not None else 'Not available'}")
    
    return test_image, segmentations_dict

def open_parameter_testing_in_napari(segmenter: 'NucleiSegmenterSAM'):
    """
    Open parameter testing results in napari for comparison.
    
    Args:
        segmenter: NucleiSegmenterSAM instance
        
    Returns:
        napari.Viewer instance or None
    """
    try:
        import napari
    except ImportError:
        print("❌ Napari not available. Install with: pip install napari")
        return None
    
    # Load parameter testing results
    test_image, segmentations_dict = load_parameter_testing_results(segmenter)
    
    if test_image is None or not segmentations_dict:
        print("❌ Cannot open napari - missing data")
        print("   → Run comprehensive parameter testing first")
        return None
    
    # Open in napari
    viewer = open_sam_segmentations_in_napari(
        image=test_image,
        segmentations_dict=segmentations_dict,
        image_name="Parameter Testing - Original DAPI"
    )
    
    if viewer is not None:
        print(f"\n🎯 PARAMETER COMPARISON GUIDE:")
        print(f"   • Compare nuclei counts across configurations")
        print(f"   • Look for over/under-segmentation patterns")
        print(f"   • Check edge quality and boundary accuracy")
        print(f"   • Assess 3D continuity across z-slices")
        print(f"   • Identify optimal parameter combinations")
    
    return viewer

def check_existing_segmentation(zarr_path: Path) -> Dict[str, Any]:
    """
    Check which organoids already have segmentation masks for resume capability
    
    Args:
        zarr_path: Path to bundled zarr file
        
    Returns:
        Dictionary with existing segmentation status
    """
    condition_name = zarr_path.parent.name
    
    try:
        bundle = zarr.open_group(str(zarr_path), mode='r')
        
        # Find all organoids
        if hasattr(bundle, 'keys'):
            organoid_keys = [k for k in bundle.keys() if k.startswith('organoid_')]
            organoid_indices = [int(k.split('_')[1]) for k in organoid_keys]
        else:
            data_shape = bundle['data'].shape
            n_organoids = data_shape[0]
            organoid_indices = list(range(n_organoids))
            organoid_keys = [f"organoid_{i:04d}" for i in organoid_indices]
        
        # Check which have segmentation masks
        existing_segmentation = {}
        for org_idx, org_key in zip(organoid_indices, organoid_keys):
            has_masks = False
            n_nuclei = 0
            
            if hasattr(bundle, 'keys') and org_key in bundle:
                organoid_group = bundle[org_key]
                if 'masks' in organoid_group:
                    has_masks = True
                    n_nuclei = organoid_group.attrs.get('n_nuclei', 0)
            
            existing_segmentation[org_idx] = {
                'has_masks': has_masks,
                'n_nuclei': n_nuclei,
                'organoid_key': org_key
            }
        
        existing_count = sum(1 for v in existing_segmentation.values() if v['has_masks'])
        
        return {
            'condition': condition_name,
            'total_organoids': len(organoid_indices),
            'existing_segmentation_count': existing_count,
            'existing_segmentation': existing_segmentation,
            'completion_percentage': (existing_count / len(organoid_indices)) * 100 if organoid_indices else 0
        }
        
    except Exception as e:
        print(f"   ❌ Error checking existing segmentation: {e}")
        return {
            'condition': condition_name,
            'total_organoids': 0,
            'existing_segmentation_count': 0,
            'existing_segmentation': {},
            'completion_percentage': 0,
            'error': str(e)
        }

def add_segmentation_to_zarr_optimized(
    zarr_path: Path,
    segmentation_results: Dict[str, Any],
    dry_run: bool = False,
    save_full_resolution_only: bool = True,
    verbose: bool = True
) -> bool:
    """
    Add segmentation masks to the original zarr file structure with optimized storage
    
    Args:
        zarr_path: Path to bundled zarr file
        segmentation_results: Results from segmentation
        dry_run: If True, don't actually modify files
        save_full_resolution_only: If True, save only level 0 (full resolution)
        verbose: Print detailed information
        
    Returns:
        True if successful, False otherwise
    """
    import datetime
    
    condition_name = zarr_path.parent.name
    if verbose:
        print(f"\n💾 Adding segmentation masks to: {condition_name}")
    
    if dry_run:
        successful_count = sum(1 for r in segmentation_results.values() if r['success'])
        if verbose:
            print(f"   🔍 DRY RUN: Would add segmentation masks to {successful_count} organoids")
        return True
    
    save_start_time = time.time()
    
    try:
        # Open zarr file in read-write mode
        bundle = zarr.open_group(str(zarr_path), mode='r+')
        
        # Add segmentation metadata to bundle
        bundle.attrs['segmentation_applied'] = True
        bundle.attrs['segmentation_timestamp'] = str(datetime.datetime.now())
        bundle.attrs['segmentation_params'] = {
            'model_type': 'cpsam',
            'full_resolution_only': save_full_resolution_only
        }
        
        # Add masks to each organoid
        masks_saved = 0
        total_nuclei_saved = 0
        
        for org_idx, result in segmentation_results.items():
            if not result['success']:
                continue
                
            org_key = result['organoid_key']
            masks = result['masks']
            metadata = result['metadata']
            
            # Access organoid group
            if org_key in bundle:
                organoid_group = bundle[org_key]
            else:
                if verbose:
                    print(f"      ⚠️  Organoid {org_key} not found in bundle")
                continue
            
            # Add masks dataset - FULL RESOLUTION ONLY (Level 0)
            if 'masks' in organoid_group:
                del organoid_group['masks']  # Remove existing if present
            
            # Save masks with optimized settings for level 0 only
            masks_uint16 = masks.astype(np.uint16)
            
            # Create dataset with specific compression settings for level 0
            organoid_group.create_dataset(
                'masks',
                data=masks_uint16,
                chunks=(min(masks_uint16.shape[0], 32), 
                       min(masks_uint16.shape[1], 256), 
                       min(masks_uint16.shape[2], 256)),
                dtype=np.uint16
            )
            
            # Add segmentation metadata to organoid
            organoid_group.attrs['segmentation_applied'] = True
            organoid_group.attrs['n_nuclei'] = metadata['num_cells']
            organoid_group.attrs['segmentation_metadata'] = {
                'processing_time': metadata['processing_time'],
                'diameter': metadata['diameter'],
                'anisotropy': metadata['anisotropy'],
                'gpu_used': metadata['gpu_used'],
                'size_filtering_applied': metadata.get('size_filtering_applied', False),
                'min_nuclei_size': metadata.get('min_nuclei_size', None),
                'full_resolution_only': save_full_resolution_only,
                'zarr_level': 0  # Confirm this is level 0
            }
            
            masks_saved += 1
            total_nuclei_saved += metadata['num_cells']
            
            if verbose:
                print(f"      ✅ Saved masks for organoid {org_idx} ({metadata['num_cells']} nuclei)")
        
        save_time = time.time() - save_start_time
        
        if verbose:
            print(f"   ✅ Successfully added segmentation data to zarr file")
            print(f"   → Masks saved: {masks_saved} organoids")
            print(f"   → Total nuclei: {total_nuclei_saved}")
            print(f"   → Save time: {save_time:.1f}s")
            print(f"   → Storage: Full resolution only (Level 0)")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"   ❌ Error adding segmentation to zarr: {e}")
        return False

def segment_condition(
    zarr_path: Path,
    optimal_diameter: float = 50.0,
    optimal_flow_threshold: float = 0.4,
    optimal_cellprob_threshold: float = 0.0,
    optimal_batch_size: int = 16,
    min_nuclei_size: int = 1000,
    channel: int = 0,
    dry_run: bool = False,
    resume: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run segmentation on a single condition with timing, interruption safety, and resume capability
    
    Args:
        zarr_path: Path to bundled zarr file
        optimal_diameter: Cellpose diameter parameter
        optimal_flow_threshold: Cellpose flow threshold
        optimal_cellprob_threshold: Cellpose cellprob threshold
        optimal_batch_size: Batch size for processing
        min_nuclei_size: Minimum nuclei size for filtering
        channel: Channel to segment (0 for DAPI)
        dry_run: If True, don't actually modify files
        resume: If True, skip organoids that already have segmentation
        verbose: Print detailed information
        
    Returns:
        Dictionary with segmentation results and timing information
    """
    condition_name = zarr_path.parent.name
    if verbose:
        print(f"\n🧬 Segmentation: {condition_name}")
    
    # ⏱️ Start timing
    pipeline_start_time = time.time()
    
    if dry_run:
        if verbose:
            print(f"   🔍 DRY RUN: Segmentation simulation")
        return {
            'condition': condition_name,
            'zarr_path': str(zarr_path),
            'dry_run': True,
            'total_organoids': 0,
            'successful_segments': 0,
            'failed_segments': 0,
            'total_nuclei': 0,
            'segmentation_results': {},
            'processing_times': {'total': 0}
        }
    
    # 🔍 Check existing segmentation for resume capability
    if verbose:
        print(f"   🔍 Checking existing segmentation...")
    existing_check_start = time.time()
    existing_status = check_existing_segmentation(zarr_path)
    existing_check_time = time.time() - existing_check_start
    
    if existing_status['existing_segmentation_count'] > 0:
        if verbose:
            print(f"   ✅ Found existing: {existing_status['existing_segmentation_count']}/{existing_status['total_organoids']} organoids ({existing_status['completion_percentage']:.1f}%)")
            if resume:
                print(f"   🔄 Resume mode: Will skip organoids with existing segmentation")
            else:
                print(f"   ⚠️  Resume disabled: Will overwrite existing segmentation")
    else:
        if verbose:
            print(f"   📋 No existing segmentation - starting fresh")
    
    # Set up optimized parameters
    segmentation_params = SegmentationParams(
        diameter=optimal_diameter,
        do_3D=True,
        batch_size=optimal_batch_size,
        use_gpu=True
    )
    
    if verbose:
        print(f"   ⚙️  Using optimized parameters:")
        print(f"      • Diameter: {segmentation_params.diameter}")
        print(f"      • Batch size: {segmentation_params.batch_size}")
        print(f"      • Min nuclei size: {min_nuclei_size}")
    
    # Initialize segmenter
    if verbose:
        print(f"   🔄 Initializing Cellpose-SAM segmenter...")
    segmenter_init_start = time.time()
    segmenter = NucleiSegmenterSAM(use_gpu=segmentation_params.use_gpu)
    segmenter_init_time = time.time() - segmenter_init_start
    if verbose:
        print(f"   ✅ Segmenter initialized in {segmenter_init_time:.1f}s")
    
    # Load zarr data
    if verbose:
        print(f"   📂 Loading zarr data...")
    zarr_load_start = time.time()
    bundle = zarr.open_group(str(zarr_path), mode='r')
    zarr_load_time = time.time() - zarr_load_start
    if verbose:
        print(f"   ✅ Zarr loaded in {zarr_load_time:.1f}s")
    
    # Determine organoids to segment
    if hasattr(bundle, 'keys'):
        # New structure with individual organoid groups
        organoid_keys = [k for k in bundle.keys() if k.startswith('organoid_')]
        organoid_indices = [int(k.split('_')[1]) for k in organoid_keys]
    else:
        # Old structure with single 'data' array
        data_shape = bundle['data'].shape
        n_organoids = data_shape[0]
        organoid_indices = list(range(n_organoids))
        organoid_keys = [f"organoid_{i:04d}" for i in organoid_indices]
    
    # Filter organoids based on resume setting
    organoids_to_process = []
    organoids_skipped = []
    
    if resume and existing_status['existing_segmentation_count'] > 0:
        for org_idx in organoid_indices:
            if existing_status['existing_segmentation'].get(org_idx, {}).get('has_masks', False):
                organoids_skipped.append(org_idx)
            else:
                organoids_to_process.append(org_idx)
    else:
        organoids_to_process = organoid_indices
    
    if verbose:
        print(f"   → Total organoids: {len(organoid_indices)}")
        print(f"   → To process: {len(organoids_to_process)}")
        print(f"   → Skipping (existing): {len(organoids_skipped)}")
    
    # Segment each organoid
    segmentation_results = {}
    total_nuclei = 0
    processing_times = {
        'existing_check': existing_check_time,
        'segmenter_init': segmenter_init_time,
        'zarr_load': zarr_load_time,
        'segmentation': 0,
        'per_organoid': {}
    }
    
    segmentation_start_time = time.time()
    
    for org_idx in organoids_to_process:
        org_key = f"organoid_{org_idx:04d}"
        organoid_start_time = time.time()
        
        if verbose:
            print(f"\n      🔍 Processing organoid {org_idx}...")
        
        try:
            # Load organoid image
            data_load_start = time.time()
            if hasattr(bundle, 'keys') and org_key in bundle:
                # New structure
                organoid_data = bundle[org_key]['data'][:]
                img_3d = organoid_data[channel]  # Extract channel
            else:
                # Old structure
                organoid_data = bundle['data'][org_idx]
                img_3d = organoid_data[channel]  # Extract channel
            
            # Convert to numpy if needed
            if hasattr(img_3d, 'compute'):
                img_3d = img_3d.compute()
            else:
                img_3d = np.array(img_3d)
            
            data_load_time = time.time() - data_load_start
            
            # Run segmentation
            seg_start = time.time()
            masks, metadata = segmenter.segment_organoid(
                img_3d, 
                params=segmentation_params,
                min_nuclei_size=min_nuclei_size,
                verbose=False
            )
            seg_time = time.time() - seg_start
            
            # Store results
            segmentation_results[org_idx] = {
                'masks': masks,
                'metadata': metadata,
                'organoid_key': org_key,
                'success': True
            }
            
            nuclei_count = metadata['num_cells']
            total_nuclei += nuclei_count
            
            organoid_total_time = time.time() - organoid_start_time
            processing_times['per_organoid'][org_idx] = {
                'data_load': data_load_time,
                'segmentation': seg_time,
                'total': organoid_total_time
            }
            
            if verbose:
                print(f"         ✅ Success: {nuclei_count} nuclei detected")
                print(f"         ⏱️ Times: Load {data_load_time:.1f}s | Segment {seg_time:.1f}s | Total {organoid_total_time:.1f}s")
            
            if metadata.get('size_filtering_applied', False):
                removed_count = metadata.get('removed_nuclei_count', 0)
                if verbose:
                    print(f"         🔍 Size filtering: {removed_count} small objects removed")
            
        except Exception as e:
            organoid_total_time = time.time() - organoid_start_time
            processing_times['per_organoid'][org_idx] = {
                'error': str(e),
                'total': organoid_total_time
            }
            
            if verbose:
                print(f"         ❌ Error: {e}")
                print(f"         ⏱️ Failed after {organoid_total_time:.1f}s")
            
            segmentation_results[org_idx] = {
                'error': str(e),
                'organoid_key': org_key,
                'success': False
            }
            continue
    
    segmentation_time = time.time() - segmentation_start_time
    processing_times['segmentation'] = segmentation_time
    processing_times['total'] = time.time() - pipeline_start_time
    
    # Summary
    successful_segments = sum(1 for r in segmentation_results.values() if r['success'])
    failed_segments = len(segmentation_results) - successful_segments
    skipped_segments = len(organoids_skipped)
    
    if verbose:
        print(f"\n   📊 SEGMENTATION SUMMARY:")
        print(f"      → Total organoids: {len(organoid_indices)}")
        print(f"      → Processed: {len(organoids_to_process)}")
        print(f"      → Successful: {successful_segments}")
        print(f"      → Failed: {failed_segments}")
        print(f"      → Skipped (existing): {skipped_segments}")
        print(f"      → Total nuclei detected: {total_nuclei}")
        
        avg_nuclei = total_nuclei/successful_segments if successful_segments > 0 else 0
        print(f"      → Average nuclei per organoid: {avg_nuclei:.1f}" if successful_segments > 0 else "      → Average nuclei per organoid: N/A")
        
        print(f"\n   ⏱️ PERFORMANCE SUMMARY:")
        print(f"      → Total processing time: {processing_times['total']:.1f}s")
        print(f"      → Segmentation time: {processing_times['segmentation']:.1f}s")
        avg_per_organoid = segmentation_time/len(organoids_to_process) if organoids_to_process else 0
        print(f"      → Average per organoid: {avg_per_organoid:.1f}s" if organoids_to_process else "      → Average per organoid: N/A")
        print(f"      → Setup time: {processing_times['existing_check'] + processing_times['segmenter_init'] + processing_times['zarr_load']:.1f}s")
    
    return {
        'condition': condition_name,
        'zarr_path': str(zarr_path),
        'segmentation_params': segmentation_params,
        'total_organoids': len(organoid_indices),
        'processed_organoids': len(organoids_to_process),
        'successful_segments': successful_segments,
        'failed_segments': failed_segments,
        'skipped_segments': skipped_segments,
        'total_nuclei': total_nuclei,
        'segmentation_results': segmentation_results,
        'processing_times': processing_times,
        'existing_status': existing_status,
        'resume_used': resume,
        'dry_run': dry_run
    }

def segment_all_conditions(
    cropped_dir: Path,
    optimal_diameter: float = 50.0,
    optimal_flow_threshold: float = 0.4,
    optimal_cellprob_threshold: float = 0.0,
    optimal_batch_size: int = 16,
    min_nuclei_size: int = 1000,
    channel: int = 0,
    dry_run: bool = False,
    resume: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run segmentation on ALL conditions in the directory
    
    Args:
        cropped_dir: Directory containing condition folders
        optimal_diameter: Cellpose diameter parameter
        optimal_flow_threshold: Cellpose flow threshold
        optimal_cellprob_threshold: Cellpose cellprob threshold
        optimal_batch_size: Batch size for processing
        min_nuclei_size: Minimum nuclei size for filtering
        channel: Channel to segment (0 for DAPI)
        dry_run: If True, don't actually modify files
        resume: If True, skip organoids that already have segmentation
        verbose: Print detailed information
        
    Returns:
        Dictionary with results for all conditions
    """
    if verbose:
        print(f"🚀 SEGMENTATION - ALL CONDITIONS")
        print(f"=" * 60)
        print(f"Directory: {cropped_dir}")
    
    # Find all zarr files
    zarr_files = []
    for condition_dir in cropped_dir.iterdir():
        if condition_dir.is_dir():
            bundled_zarr = condition_dir / f"{condition_dir.name}_bundled.zarr"
            if bundled_zarr.exists():
                zarr_files.append(bundled_zarr)
    
    if verbose:
        print(f"→ Found {len(zarr_files)} conditions to process")
    
    # Process all conditions
    all_results = {}
    overall_start_time = time.time()
    
    for i, zarr_path in enumerate(zarr_files, 1):
        condition_name = zarr_path.parent.name
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"CONDITION {i}/{len(zarr_files)}: {condition_name}")
            print(f"{'='*50}")
        
        try:
            # Run segmentation
            seg_results = segment_condition(
                zarr_path,
                optimal_diameter=optimal_diameter,
                optimal_flow_threshold=optimal_flow_threshold,
                optimal_cellprob_threshold=optimal_cellprob_threshold,
                optimal_batch_size=optimal_batch_size,
                min_nuclei_size=min_nuclei_size,
                channel=channel,
                dry_run=dry_run,
                resume=resume,
                verbose=verbose
            )
            
            # Add segmentation to zarr file if successful
            if seg_results['successful_segments'] > 0:
                zarr_success = add_segmentation_to_zarr_optimized(
                    zarr_path,
                    seg_results['segmentation_results'],
                    dry_run=dry_run,
                    save_full_resolution_only=True,
                    verbose=verbose
                )
                seg_results['zarr_update_success'] = zarr_success
            
            all_results[condition_name] = seg_results
            
        except Exception as e:
            if verbose:
                print(f"❌ Error processing {condition_name}: {e}")
            all_results[condition_name] = {'error': str(e), 'success': False}
            continue
    
    overall_time = time.time() - overall_start_time
    
    # Overall summary
    if verbose:
        print(f"\n🎯 OVERALL SUMMARY")
        print(f"{'='*60}")
        
        total_conditions = len(all_results)
        successful_conditions = sum(1 for r in all_results.values() if r.get('successful_segments', 0) > 0)
        total_organoids = sum(r.get('total_organoids', 0) for r in all_results.values())
        total_successful = sum(r.get('successful_segments', 0) for r in all_results.values())
        total_nuclei = sum(r.get('total_nuclei', 0) for r in all_results.values())
        
        print(f"   → Conditions processed: {total_conditions}")
        print(f"   → Conditions successful: {successful_conditions}")
        print(f"   → Total organoids: {total_organoids}")
        print(f"   → Successful segmentations: {total_successful}")
        print(f"   → Total nuclei detected: {total_nuclei}")
        print(f"   → Overall processing time: {overall_time/60:.1f} minutes")
        
        if total_successful > 0:
            print(f"   → Average nuclei per organoid: {total_nuclei/total_successful:.1f}")
            print(f"   → Average time per organoid: {overall_time/total_successful:.1f}s")
    
    return {
        'all_results': all_results,
        'summary': {
            'total_conditions': len(all_results),
            'successful_conditions': sum(1 for r in all_results.values() if r.get('successful_segments', 0) > 0),
            'total_organoids': sum(r.get('total_organoids', 0) for r in all_results.values()),
            'total_successful': sum(r.get('successful_segments', 0) for r in all_results.values()),
            'total_nuclei': sum(r.get('total_nuclei', 0) for r in all_results.values()),
            'overall_time': overall_time
        }
    }

def open_segmentation_napari(
    condition_name: Optional[str] = None,
    organoid_idx: int = 0,
    cropped_dir: Optional[Path] = None
):
    """
    Open napari visualization for segmentation results
    
    Args:
        condition_name: Name of condition to visualize (uses first if None)
        organoid_idx: Index of organoid to visualize
        cropped_dir: Directory containing the cropped zarr files
        
    Returns:
        napari.Viewer instance or None
    """
    try:
        import napari
    except ImportError:
        print("❌ Napari not available. Install with: pip install napari")
        return None
    
    if cropped_dir is None:
        try:
            from karyosight.config import CROPPED_DIR
            cropped_dir = Path(CROPPED_DIR)
        except:
            print("❌ Could not determine cropped directory")
            return None
    
    if not cropped_dir.exists():
        print(f"❌ Cropped directory not found: {cropped_dir}")
        return None
    
    # Find available conditions
    condition_dirs = [d for d in cropped_dir.iterdir() if d.is_dir()]
    available_conditions = []
    
    for condition_dir in condition_dirs:
        bundled_zarr = condition_dir / f"{condition_dir.name}_bundled.zarr"
        if bundled_zarr.exists():
            available_conditions.append(condition_dir.name)
    
    if not available_conditions:
        print("❌ No conditions found")
        return None
    
    # Use specified condition or first available
    if condition_name is None:
        condition_name = available_conditions[0]
    
    if condition_name not in available_conditions:
        print(f"❌ Condition '{condition_name}' not found")
        print(f"   Available conditions: {available_conditions}")
        return None
    
    print(f"🎯 Loading condition: {condition_name}, organoid: {organoid_idx}")
    
    # Load data
    bundled_zarr = cropped_dir / condition_name / f"{condition_name}_bundled.zarr"
    
    try:
        bundle = zarr.open_group(str(bundled_zarr), mode='r')
        
        # Find organoid
        organoid_key = f"organoid_{organoid_idx:04d}"
        if organoid_key not in bundle:
            available_organoids = [k for k in bundle.keys() if k.startswith('organoid_')]
            print(f"❌ Organoid {organoid_key} not found")
            print(f"   Available organoids: {len(available_organoids)}")
            return None
        
        organoid_group = bundle[organoid_key]
        
        # Load raw data
        raw_data = organoid_group['data'][:]
        if hasattr(raw_data, 'compute'):
            raw_data = raw_data.compute()
        else:
            raw_data = np.array(raw_data)
        
        print(f"✅ Raw data loaded: {raw_data.shape}")
        
        # Load segmentation masks if available
        masks = None
        n_nuclei = 0
        if 'masks' in organoid_group:
            masks = organoid_group['masks'][:]
            if hasattr(masks, 'compute'):
                masks = masks.compute()
            else:
                masks = np.array(masks)
            
            n_nuclei = organoid_group.attrs.get('n_nuclei', len(np.unique(masks)) - 1)
            print(f"✅ Segmentation masks loaded: {masks.shape} ({n_nuclei} nuclei)")
            print(f"📍 Storage location: {bundled_zarr} → {organoid_key}/masks")
        else:
            print(f"⚠️  No segmentation masks found")
        
        # Create napari viewer
        print(f"🔍 Opening in napari...")
        viewer = napari.Viewer()
        
        # Add raw data channels
        if raw_data.ndim == 4:  # [C, Z, Y, X]
            # Get channel names from config
            try:
                if CONFIG_AVAILABLE:
                    from karyosight.config import CHANNELS
                    channel_names = CHANNELS
                else:
                    channel_names = ['nucleus', 'tetraploid', 'diploid', 'sox9', 'sox2', 'brightfield']
            except:
                channel_names = ['DAPI', 'GFP', 'RFP', 'Cy5', 'Channel_4', 'Channel_5']
            
            for c in range(raw_data.shape[0]):
                channel_name = channel_names[c] if c < len(channel_names) else f'Channel_{c}'
                viewer.add_image(
                    raw_data[c],
                    name=f'{condition_name} - {channel_name}',
                    colormap='gray' if c == 0 else 'green' if c == 1 else 'red' if c == 2 else 'cyan',
                    contrast_limits=[raw_data[c].min(), np.percentile(raw_data[c], 99)]
                )
        else:  # [Z, Y, X] - single channel
            viewer.add_image(
                raw_data,
                name=f'{condition_name} - DAPI',
                colormap='gray',
                contrast_limits=[raw_data.min(), np.percentile(raw_data, 99)]
            )
        
        # Add segmentation masks
        if masks is not None:
            viewer.add_labels(
                masks,
                name=f'Production Masks ({n_nuclei} nuclei)',
                opacity=0.7
            )
        
        print(f"✅ Napari opened successfully!")
        return viewer
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

if __name__ == "__main__":
    # Test installation when run directly
    test_cellpose_sam_installation() 