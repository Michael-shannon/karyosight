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
                
            # Apply post-segmentation z-slice optimization if enabled
            z_optimization_metadata = {}
            if CONFIG_AVAILABLE and ENABLE_Z_SLICE_OPTIMIZATION:
                if verbose:
                    print(f"✂️ Applying post-segmentation z-slice optimization...")
                
                optimized_image, optimized_masks, z_opt_meta = optimize_z_slices_after_segmentation(
                    image, masks,
                    strategy=Z_OPTIMIZATION_STRATEGY,
                    padding=Z_OPTIMIZATION_PADDING,
                    verbose=verbose
                )
                
                # Update image and masks if optimization was applied
                if z_opt_meta['z_optimization_applied']:
                    image = optimized_image  # For consistency in metadata
                    masks = optimized_masks
                    
                    if verbose:
                        compression = z_opt_meta['compression_ratio_percent']
                        print(f"      ✅ Z-slice optimization: {compression:.1f}% reduction")
                
                z_optimization_metadata = z_opt_meta
            
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
                **filter_metadata,  # Include size filtering metadata
                **z_optimization_metadata  # Include z-slice optimization metadata
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
                
                # Store results with original raw data for incremental saving
                segmentation_results[org_idx] = {
                    'masks': masks,
                    'metadata': metadata,
                    'organoid_key': org_key,
                    'original_raw_data': organoid_data,  # Include for incremental saving
                    'success': True
                }
                
                # 💾 INCREMENTAL SAVE: Save this organoid immediately to prevent data loss
                try:
                    from karyosight.config import SEGMENTED_DIR
                    segmented_dir = Path(SEGMENTED_DIR)
                except:
                    segmented_dir = Path("D:/LB_TEST/segmented")
                
                save_success = save_single_organoid_segmented(
                    segmentation_results[org_idx],
                    condition_name,
                    segmented_dir,
                    verbose=verbose
                )
                
                if save_success:
                    segmentation_results[org_idx]['saved_incrementally'] = True
                else:
                    segmentation_results[org_idx]['saved_incrementally'] = False
                    if verbose:
                        print(f"         ⚠️  Failed to save organoid {org_idx} incrementally")
                
                nuclei_count = metadata['num_cells']
                total_nuclei += nuclei_count
                
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
    Checks both cropped directory (legacy) and segmented directory (incremental saves)
    
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
        
        # Check which have segmentation masks (legacy cropped directory)
        existing_segmentation = {}
        for org_idx, org_key in zip(organoid_indices, organoid_keys):
            has_masks = False
            n_nuclei = 0
            save_location = 'none'
            
            # Check legacy location first (cropped directory)
            if hasattr(bundle, 'keys') and org_key in bundle:
                organoid_group = bundle[org_key]
                if 'masks' in organoid_group:
                    has_masks = True
                    n_nuclei = organoid_group.attrs.get('n_nuclei', 0)
                    save_location = 'cropped'
            
            # Check new incremental save location (segmented directory)
            if not has_masks:
                try:
                    from karyosight.config import SEGMENTED_DIR
                    segmented_dir = Path(SEGMENTED_DIR)
                    segmented_zarr = segmented_dir / condition_name / f"{condition_name}_segmented.zarr"
                    
                    if segmented_zarr.exists():
                        seg_bundle = zarr.open_group(str(segmented_zarr), mode='r')
                        if org_key in seg_bundle:
                            seg_organoid_group = seg_bundle[org_key]
                            if 'masks' in seg_organoid_group:
                                has_masks = True
                                n_nuclei = seg_organoid_group.attrs.get('n_nuclei', 0)
                                save_location = 'segmented'
                except:
                    pass  # Failed to check segmented directory, continue
            
            existing_segmentation[org_idx] = {
                'has_masks': has_masks,
                'n_nuclei': n_nuclei,
                'organoid_key': org_key,
                'save_location': save_location
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

def save_single_organoid_segmented(
    organoid_result: Dict[str, Any],
    condition_name: str,
    segmented_dir: Path,
    verbose: bool = False
) -> bool:
    """
    Save a single organoid's segmentation results incrementally to segmented directory
    
    Args:
        organoid_result: Single organoid segmentation result
        condition_name: Name of the condition
        segmented_dir: Directory to save segmented results
        verbose: Print detailed information
        
    Returns:
        True if successful, False otherwise
    """
    if not organoid_result['success']:
        return False
    
    try:
        # Create segmented directory structure
        condition_segmented_dir = segmented_dir / condition_name
        condition_segmented_dir.mkdir(parents=True, exist_ok=True)
        
        # Create or open output zarr file
        output_zarr_path = condition_segmented_dir / f"{condition_name}_segmented.zarr"
        
        # Open or create zarr bundle
        if output_zarr_path.exists():
            output_bundle = zarr.open_group(str(output_zarr_path), mode='r+')
        else:
            output_bundle = zarr.open_group(str(output_zarr_path), mode='w')
            # Initialize bundle metadata
            output_bundle.attrs['condition'] = condition_name
            output_bundle.attrs['creation_timestamp'] = str(time.time())
            output_bundle.attrs['segmentation_params'] = {
                'model_type': 'cpsam',
                'z_optimization_applied': True,
                'incremental_save': True
            }
        
        # Get organoid data
        org_key = organoid_result['organoid_key']
        masks = organoid_result['masks']
        metadata = organoid_result['metadata']
        
        # Load original raw data to apply z-optimization
        if 'original_raw_data' in organoid_result:
            # Raw data was included in result
            original_raw_data = organoid_result['original_raw_data']
        else:
            # Need to load from source - this is a fallback, should be avoided
            if verbose:
                print(f"      ⚠️  Loading original data for z-optimization (slower)")
            return False  # Skip for now - we'll include raw data in results
        
        # Apply z-optimization to raw data if it was applied to masks
        if metadata.get('z_optimization_applied', False):
            z_slices_kept = metadata.get('z_slices_kept', None)
            if z_slices_kept is not None:
                optimized_raw_data = original_raw_data[:, z_slices_kept, :, :]  # [C, Z, Y, X]
                optimized_masks = masks  # Already optimized
            else:
                optimized_raw_data = original_raw_data
                optimized_masks = masks
        else:
            optimized_raw_data = original_raw_data
            optimized_masks = masks
        
        # Create organoid group in output (overwrite if exists)
        if org_key in output_bundle:
            del output_bundle[org_key]
        
        organoid_group = output_bundle.create_group(org_key)
        
        # Save optimized raw data with compression
        organoid_group.create_dataset(
            'data',
            data=optimized_raw_data.astype(np.uint16),
            chunks=(1, min(optimized_raw_data.shape[1], 32), 
                   min(optimized_raw_data.shape[2], 256), 
                   min(optimized_raw_data.shape[3], 256)),
            dtype=np.uint16
        )
        
        # Save optimized masks with compression
        masks_uint16 = optimized_masks.astype(np.uint16)
        organoid_group.create_dataset(
            'masks',
            data=masks_uint16,
            chunks=(min(masks_uint16.shape[0], 32), 
                   min(masks_uint16.shape[1], 256), 
                   min(masks_uint16.shape[2], 256)),
            dtype=np.uint16
        )
        
        # Save comprehensive metadata
        organoid_group.attrs.update({
            'n_nuclei': metadata['num_cells'],
            'num_cells': metadata['num_cells'],
            'original_raw_shape': list(original_raw_data.shape),
            'optimized_raw_shape': list(optimized_raw_data.shape),
            'masks_shape': list(optimized_masks.shape),
            'processing_time': metadata['processing_time'],
            'model_type': 'cpsam',
            'diameter': metadata['diameter'],
            'anisotropy': metadata['anisotropy'],
            'gpu_used': metadata['gpu_used'],
            'z_optimization_applied': metadata.get('z_optimization_applied', False),
            'compression_ratio_percent': metadata.get('compression_ratio_percent', 0),
            'size_filtering_applied': metadata.get('size_filtering_applied', False),
            'min_nuclei_size': metadata.get('min_nuclei_size', None),
            'save_timestamp': str(time.time())
        })
        
        if verbose:
            org_idx = int(org_key.split('_')[1])
            compression = metadata.get('compression_ratio_percent', 0)
            print(f"      💾 Saved organoid {org_idx}: {metadata['num_cells']} nuclei, {compression:.1f}% compression")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"      ❌ Error saving organoid: {e}")
        return False

def save_segmented_zarr_optimized(
    zarr_path: Path,
    segmentation_results: Dict[str, Any],
    segmented_dir: Path,
    dry_run: bool = False,
    verbose: bool = True
) -> bool:
    """
    Save optimized segmentation results to new segmented directory
    Saves both z-optimized raw data and z-optimized masks
    
    Args:
        zarr_path: Path to original bundled zarr file
        segmentation_results: Results from segmentation with z-optimization
        segmented_dir: Directory to save segmented results
        dry_run: If True, don't actually save files
        verbose: Print detailed information
        
    Returns:
        True if successful, False otherwise
    """
    import datetime
    
    condition_name = zarr_path.parent.name
    if verbose:
        print(f"\n💾 Saving optimized segmentation to: segmented/{condition_name}")
    
    if dry_run:
        successful_count = sum(1 for r in segmentation_results.values() if r['success'])
        if verbose:
            print(f"   🔍 DRY RUN: Would save {successful_count} optimized organoids")
        return True
    
    save_start_time = time.time()
    
    try:
        # Create segmented directory structure
        condition_segmented_dir = segmented_dir / condition_name
        condition_segmented_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output zarr file
        output_zarr_path = condition_segmented_dir / f"{condition_name}_segmented.zarr"
        if output_zarr_path.exists():
            import shutil
            shutil.rmtree(output_zarr_path)
        
        output_bundle = zarr.open_group(str(output_zarr_path), mode='w')
        
        # Add metadata to bundle
        output_bundle.attrs['condition'] = condition_name
        output_bundle.attrs['creation_timestamp'] = str(datetime.datetime.now())
        output_bundle.attrs['segmentation_params'] = {
            'model_type': 'cpsam',
            'z_optimization_applied': True,
            'source_zarr': str(zarr_path)
        }
        
        # Load original zarr for raw data
        input_bundle = zarr.open_group(str(zarr_path), mode='r')
        
        # Process each successful segmentation
        organoids_saved = 0
        total_nuclei_saved = 0
        total_compression = 0
        
        for org_idx, result in segmentation_results.items():
            if not result['success']:
                continue
                
            org_key = result['organoid_key']
            masks = result['masks']
            metadata = result['metadata']
            
            # Load original raw data for this organoid
            if org_key in input_bundle:
                original_organoid_data = input_bundle[org_key]['data'][:]
                if hasattr(original_organoid_data, 'compute'):
                    original_organoid_data = original_organoid_data.compute()
                else:
                    original_organoid_data = np.array(original_organoid_data)
            else:
                if verbose:
                    print(f"      ⚠️  Could not load raw data for {org_key}")
                continue
            
            # Apply z-optimization to both raw data and masks if it was applied
            if metadata.get('z_optimization_applied', False):
                # Extract the z-slices that were kept
                z_slices_kept = metadata.get('z_slices_kept', None)
                
                if z_slices_kept is not None:
                    # Apply same z-optimization to raw data
                    optimized_raw_data = original_organoid_data[:, z_slices_kept, :, :]  # [C, Z, Y, X]
                    optimized_masks = masks  # Already optimized during segmentation
                    
                    compression_ratio = metadata.get('compression_ratio_percent', 0)
                    total_compression += compression_ratio
                    
                    if verbose:
                        print(f"      ✅ Organoid {org_idx}: {original_organoid_data.shape[1]} → {optimized_raw_data.shape[1]} z-slices ({compression_ratio:.1f}% compression)")
                else:
                    # No z-optimization metadata, use original
                    optimized_raw_data = original_organoid_data
                    optimized_masks = masks
            else:
                # No z-optimization applied
                optimized_raw_data = original_organoid_data
                optimized_masks = masks
            
            # Create organoid group in output
            organoid_group = output_bundle.create_group(org_key)
            
            # Save optimized raw data with compression
            organoid_group.create_dataset(
                'data',
                data=optimized_raw_data.astype(np.uint16),
                chunks=(1, min(optimized_raw_data.shape[1], 32), 
                       min(optimized_raw_data.shape[2], 256), 
                       min(optimized_raw_data.shape[3], 256)),
                dtype=np.uint16
            )
            
            # Save optimized masks with compression
            masks_uint16 = optimized_masks.astype(np.uint16)
            organoid_group.create_dataset(
                'masks',
                data=masks_uint16,
                chunks=(min(masks_uint16.shape[0], 32), 
                       min(masks_uint16.shape[1], 256), 
                       min(masks_uint16.shape[2], 256)),
                dtype=np.uint16
            )
            
            # Save comprehensive metadata
            organoid_group.attrs.update({
                'n_nuclei': metadata['num_cells'],
                'num_cells': metadata['num_cells'],
                'original_raw_shape': list(original_organoid_data.shape),
                'optimized_raw_shape': list(optimized_raw_data.shape),
                'masks_shape': list(optimized_masks.shape),
                'processing_time': metadata['processing_time'],
                'model_type': 'cpsam',
                'diameter': metadata['diameter'],
                'anisotropy': metadata['anisotropy'],
                'gpu_used': metadata['gpu_used'],
                'z_optimization_applied': metadata.get('z_optimization_applied', False),
                'compression_ratio_percent': metadata.get('compression_ratio_percent', 0),
                'size_filtering_applied': metadata.get('size_filtering_applied', False),
                'min_nuclei_size': metadata.get('min_nuclei_size', None)
            })
            
            organoids_saved += 1
            total_nuclei_saved += metadata['num_cells']
            
            if verbose:
                print(f"      💾 Saved: raw data {optimized_raw_data.shape} + masks {optimized_masks.shape}")
        
        save_time = time.time() - save_start_time
        avg_compression = total_compression / organoids_saved if organoids_saved > 0 else 0
        
        # Update bundle metadata
        output_bundle.attrs['organoids_saved'] = organoids_saved
        output_bundle.attrs['total_nuclei'] = total_nuclei_saved
        output_bundle.attrs['average_compression_percent'] = avg_compression
        output_bundle.attrs['save_time_seconds'] = save_time
        
        if verbose:
            print(f"   ✅ Successfully saved segmented data:")
            print(f"      → Output: {output_zarr_path}")
            print(f"      → Organoids saved: {organoids_saved}")
            print(f"      → Total nuclei: {total_nuclei_saved}")
            print(f"      → Average compression: {avg_compression:.1f}%")
            print(f"      → Save time: {save_time:.1f}s")
            print(f"      → Contains: z-optimized raw data + z-optimized masks")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"   ❌ Error saving segmented data: {e}")
        return False

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
            # Count save locations
            cropped_count = sum(1 for v in existing_status['existing_segmentation'].values() 
                              if v['save_location'] == 'cropped')
            segmented_count = sum(1 for v in existing_status['existing_segmentation'].values() 
                                if v['save_location'] == 'segmented')
            
            print(f"   ✅ Found existing: {existing_status['existing_segmentation_count']}/{existing_status['total_organoids']} organoids ({existing_status['completion_percentage']:.1f}%)")
            if cropped_count > 0:
                print(f"      → In cropped directory: {cropped_count} organoids")
            if segmented_count > 0:
                print(f"      → In segmented directory: {segmented_count} organoids")
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
                organoid_data = organoid_data.compute()  # Also convert full data for saving
            else:
                img_3d = np.array(img_3d)
                organoid_data = np.array(organoid_data)
            
            data_load_time = time.time() - data_load_start
            
            # Apply focus filtering if enabled
            focus_metadata = {}
            if CONFIG_AVAILABLE and ENABLE_FOCUS_FILTERING:
                focus_score = calculate_organoid_focus_score(img_3d, method=FOCUS_DETECTION_METHOD)
                is_in_focus = focus_score >= MIN_FOCUS_SCORE
                
                focus_metadata = {
                    'focus_filtering_applied': True,
                    'focus_score': focus_score,
                    'is_in_focus': is_in_focus,
                    'focus_method': FOCUS_DETECTION_METHOD,
                    'focus_threshold': MIN_FOCUS_SCORE
                }
                
                if not is_in_focus:
                    if verbose:
                        print(f"         ⚠️  Skipping out-of-focus organoid (score={focus_score:.1f} < {MIN_FOCUS_SCORE})")
                    
                    # Store failed result
                    segmentation_results[org_idx] = {
                        'error': f'Out of focus (score={focus_score:.1f})',
                        'organoid_key': org_key,
                        'success': False,
                        'focus_metadata': focus_metadata
                    }
                    continue
            
            # Apply z-slice filtering if enabled
            z_filter_metadata = {}
            if CONFIG_AVAILABLE and REMOVE_BLACK_FRAMES:
                img_3d, z_filter_metadata = remove_black_z_slices(
                    img_3d,
                    threshold=BLACK_FRAME_THRESHOLD,
                    method=BLACK_FRAME_METHOD,
                    min_z_slices=MIN_Z_SLICES,
                    verbose=False
                )
            
            # Run segmentation
            seg_start = time.time()
            masks, metadata = segmenter.segment_organoid(
                img_3d, 
                params=segmentation_params,
                min_nuclei_size=min_nuclei_size,
                verbose=False
            )
            seg_time = time.time() - seg_start
            
            # Add filtering metadata to segmentation results
            metadata.update(focus_metadata)
            metadata.update(z_filter_metadata)
            
            # Store results with original raw data for incremental saving
            segmentation_results[org_idx] = {
                'masks': masks,
                'metadata': metadata,
                'organoid_key': org_key,
                'original_raw_data': organoid_data,  # Include for incremental saving
                'success': True
            }
            
            # 💾 INCREMENTAL SAVE: Save this organoid immediately to prevent data loss
            try:
                from karyosight.config import SEGMENTED_DIR
                segmented_dir = Path(SEGMENTED_DIR)
            except:
                segmented_dir = Path("D:/LB_TEST/segmented")
            
            save_success = save_single_organoid_segmented(
                segmentation_results[org_idx],
                condition_name,
                segmented_dir,
                verbose=verbose
            )
            
            if save_success:
                segmentation_results[org_idx]['saved_incrementally'] = True
            else:
                segmentation_results[org_idx]['saved_incrementally'] = False
                if verbose:
                    print(f"         ⚠️  Failed to save organoid {org_idx} incrementally")
            
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
    
    # Count focus filtering results
    focus_filtered_count = sum(1 for r in segmentation_results.values() 
                              if not r['success'] and 'Out of focus' in r.get('error', ''))
    other_failed_count = failed_segments - focus_filtered_count
    
    if verbose:
        print(f"\n   📊 SEGMENTATION SUMMARY:")
        print(f"      → Total organoids: {len(organoid_indices)}")
        print(f"      → Processed: {len(organoids_to_process)}")
        print(f"      → Successful: {successful_segments}")
        print(f"      → Failed: {failed_segments}")
        if CONFIG_AVAILABLE and ENABLE_FOCUS_FILTERING:
            print(f"         • Out of focus: {focus_filtered_count}")
            print(f"         • Other errors: {other_failed_count}")
        print(f"      → Skipped (existing): {skipped_segments}")
        print(f"      → Total nuclei detected: {total_nuclei}")
        
        avg_nuclei = total_nuclei/successful_segments if successful_segments > 0 else 0
        print(f"      → Average nuclei per organoid: {avg_nuclei:.1f}" if successful_segments > 0 else "      → Average nuclei per organoid: N/A")
        
        print(f"\n   🔍 FILTERING SUMMARY:")
        if CONFIG_AVAILABLE:
            print(f"      → Focus filtering: {'ENABLED' if ENABLE_FOCUS_FILTERING else 'DISABLED'}")
            if ENABLE_FOCUS_FILTERING:
                print(f"         • Method: {FOCUS_DETECTION_METHOD}")
                print(f"         • Threshold: {MIN_FOCUS_SCORE}")
                print(f"         • Filtered out: {focus_filtered_count} organoids")
            print(f"      → Z-slice filtering: {'ENABLED' if REMOVE_BLACK_FRAMES else 'DISABLED'}")
            if REMOVE_BLACK_FRAMES:
                print(f"         • Method: {BLACK_FRAME_METHOD}")
                print(f"         • Threshold: {BLACK_FRAME_THRESHOLD*100:.1f}% of max intensity")
                print(f"         • Min z-slices: {MIN_Z_SLICES}")
        else:
            print(f"      → Config not available - filtering disabled")
        
        print(f"\n   ⏱️ PERFORMANCE SUMMARY:")
        print(f"      → Total processing time: {processing_times['total']:.1f}s")
        print(f"      → Segmentation time: {processing_times['segmentation']:.1f}s")
        avg_per_organoid = segmentation_time/len(organoids_to_process) if organoids_to_process else 0
        print(f"      → Average per organoid: {avg_per_organoid:.1f}s" if organoids_to_process else "      → Average per organoid: N/A")
        print(f"      → Setup time: {processing_times['existing_check'] + processing_times['segmenter_init'] + processing_times['zarr_load']:.1f}s")
    
    # Incremental saving summary
    if successful_segments > 0:
        saved_incrementally = sum(1 for r in segmentation_results.values() 
                                if r.get('saved_incrementally', False))
        failed_saves = successful_segments - saved_incrementally
        
        print(f"\n   💾 INCREMENTAL SAVING SUMMARY:")
        print(f"      → Successfully saved: {saved_incrementally}/{successful_segments} organoids")
        if failed_saves > 0:
            print(f"      → Failed to save: {failed_saves} organoids")
            print(f"      → ⚠️  Some organoids processed but not saved - may need manual save")
        else:
            print(f"      → ✅ All successful segmentations saved immediately")
        print(f"      → Save location: segmented/{condition_name}/")
        print(f"      → Resume-safe: ✅ Process can be interrupted and resumed")
    
    return {
        'condition': condition_name,
        'zarr_path': str(zarr_path),
        'segmentation_params': segmentation_params,
        'total_organoids': len(organoid_indices),
        'processed_organoids': len(organoids_to_process),
        'successful_segments': successful_segments,
        'failed_segments': failed_segments,
        'focus_filtered_segments': focus_filtered_count,
        'other_failed_segments': other_failed_count,
        'skipped_segments': skipped_segments,
        'total_nuclei': total_nuclei,
        'segmentation_results': segmentation_results,
        'processing_times': processing_times,
        'existing_status': existing_status,
        'resume_used': resume,
        'dry_run': dry_run,
        'filtering_applied': {
            'focus_filtering': CONFIG_AVAILABLE and ENABLE_FOCUS_FILTERING,
            'z_slice_filtering': CONFIG_AVAILABLE and REMOVE_BLACK_FRAMES,
            'focus_method': FOCUS_DETECTION_METHOD if CONFIG_AVAILABLE else None,
            'focus_threshold': MIN_FOCUS_SCORE if CONFIG_AVAILABLE else None,
            'z_filter_method': BLACK_FRAME_METHOD if CONFIG_AVAILABLE else None,
            'z_filter_threshold': BLACK_FRAME_THRESHOLD if CONFIG_AVAILABLE else None
        }
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
            
            # Save segmented data to new segmented directory if successful
            if seg_results['successful_segments'] > 0:
                # With incremental saving, organoids are already saved
                seg_results['segmented_save_success'] = True
                if verbose:
                    saved_count = sum(1 for r in seg_results['segmentation_results'].values() 
                                    if r.get('saved_incrementally', False))
                    print(f"   💾 Incremental saves: {saved_count}/{seg_results['successful_segments']} organoids saved")
            else:
                seg_results['segmented_save_success'] = False
            
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
        
        # Focus filtering statistics
        total_focus_filtered = sum(r.get('focus_filtered_segments', 0) for r in all_results.values())
        total_other_failed = sum(r.get('other_failed_segments', 0) for r in all_results.values())
        
        if total_focus_filtered > 0 or total_other_failed > 0:
            print(f"   → Failed segmentations: {total_focus_filtered + total_other_failed}")
            if total_focus_filtered > 0:
                print(f"      • Out of focus: {total_focus_filtered}")
            if total_other_failed > 0:
                print(f"      • Other errors: {total_other_failed}")
        
        if total_successful > 0:
            print(f"   → Average nuclei per organoid: {total_nuclei/total_successful:.1f}")
            print(f"   → Average time per organoid: {overall_time/total_successful:.1f}s")
        
        # Overall filtering summary
        any_focus_filtering = any(r.get('filtering_applied', {}).get('focus_filtering', False) 
                                for r in all_results.values())
        any_z_filtering = any(r.get('filtering_applied', {}).get('z_slice_filtering', False) 
                            for r in all_results.values())
        
        if any_focus_filtering or any_z_filtering:
            print(f"\n   🔍 FILTERING SUMMARY:")
            if any_focus_filtering:
                print(f"      → Focus filtering: ✅ APPLIED")
                print(f"         • Organoids filtered: {total_focus_filtered}")
                focus_rate = (total_focus_filtered / total_organoids * 100) if total_organoids > 0 else 0
                print(f"         • Filtering rate: {focus_rate:.1f}%")
            if any_z_filtering:
                print(f"      → Z-slice filtering: ✅ APPLIED")
                print(f"         • Black frames removed from all organoids")
        else:
            print(f"\n   ⚠️  NO FILTERING APPLIED - All organoids processed")
    
    return {
        'all_results': all_results,
        'summary': {
            'total_conditions': len(all_results),
            'successful_conditions': sum(1 for r in all_results.values() if r.get('successful_segments', 0) > 0),
            'total_organoids': sum(r.get('total_organoids', 0) for r in all_results.values()),
            'total_successful': sum(r.get('successful_segments', 0) for r in all_results.values()),
            'total_nuclei': sum(r.get('total_nuclei', 0) for r in all_results.values()),
            'total_focus_filtered': sum(r.get('focus_filtered_segments', 0) for r in all_results.values()),
            'total_other_failed': sum(r.get('other_failed_segments', 0) for r in all_results.values()),
            'focus_filtering_applied': any(r.get('filtering_applied', {}).get('focus_filtering', False) for r in all_results.values()),
            'z_slice_filtering_applied': any(r.get('filtering_applied', {}).get('z_slice_filtering', False) for r in all_results.values()),
            'overall_time': overall_time
        }
    }

def open_segmented_data_in_napari(
    condition_name: Optional[str] = None,
    organoid_idx: int = 0,
    segmented_dir: Optional[Path] = None
):
    """
    Open napari visualization for segmented data (z-optimized raw data + masks)
    
    Args:
        condition_name: Name of condition to visualize (uses first if None)
        organoid_idx: Index of organoid to visualize
        segmented_dir: Directory containing the segmented zarr files
        
    Returns:
        napari.Viewer instance or None
    """
    try:
        import napari
    except ImportError:
        print("❌ Napari not available. Install with: pip install napari")
        return None
    
    # Determine segmented directory
    if segmented_dir is None:
        try:
            from karyosight.config import SEGMENTED_DIR
            segmented_dir = Path(SEGMENTED_DIR)
        except:
            print("❌ Could not determine segmented directory")
            return None
    
    if not segmented_dir.exists():
        print(f"❌ Segmented directory not found: {segmented_dir}")
        print(f"   → Make sure to run segmentation first to create segmented data")
        return None
    
    # Find available conditions in segmented directory
    condition_dirs = [d for d in segmented_dir.iterdir() if d.is_dir()]
    available_conditions = []
    
    for condition_dir in condition_dirs:
        segmented_zarr = condition_dir / f"{condition_dir.name}_segmented.zarr"
        if segmented_zarr.exists():
            available_conditions.append(condition_dir.name)
    
    if not available_conditions:
        print("❌ No segmented conditions found")
        print(f"   → Check directory: {segmented_dir}")
        print(f"   → Run segmentation first to generate segmented data")
        return None
    
    # Use specified condition or first available
    if condition_name is None:
        condition_name = available_conditions[0]
    
    if condition_name not in available_conditions:
        print(f"❌ Condition '{condition_name}' not found in segmented data")
        print(f"   Available conditions: {available_conditions}")
        return None
    
    print(f"🎯 Loading segmented data: {condition_name}, organoid: {organoid_idx}")
    
    # Load segmented data
    segmented_zarr = segmented_dir / condition_name / f"{condition_name}_segmented.zarr"
    
    try:
        bundle = zarr.open_group(str(segmented_zarr), mode='r')
        
        # Find organoid
        organoid_key = f"organoid_{organoid_idx:04d}"
        if organoid_key not in bundle:
            available_organoids = [k for k in bundle.keys() if k.startswith('organoid_')]
            print(f"❌ Organoid {organoid_key} not found")
            print(f"   Available organoids: {len(available_organoids)}")
            return None
        
        organoid_group = bundle[organoid_key]
        
        # Load z-optimized raw data
        raw_data = organoid_group['data'][:]
        if hasattr(raw_data, 'compute'):
            raw_data = raw_data.compute()
        else:
            raw_data = np.array(raw_data)
        
        # Load z-optimized masks
        masks = organoid_group['masks'][:]
        if hasattr(masks, 'compute'):
            masks = masks.compute()
        else:
            masks = np.array(masks)
        
        # Get metadata
        metadata = dict(organoid_group.attrs)
        n_nuclei = metadata.get('n_nuclei', len(np.unique(masks)) - 1)
        original_shape = metadata.get('original_raw_shape', raw_data.shape)
        compression = metadata.get('compression_ratio_percent', 0)
        
        print(f"✅ Segmented data loaded:")
        print(f"   → Raw data: {raw_data.shape} (z-optimized)")
        print(f"   → Masks: {masks.shape} (z-optimized)")
        print(f"   → Nuclei count: {n_nuclei}")
        print(f"   → Original shape: {original_shape}")
        print(f"   → Z-compression: {compression:.1f}%")
        print(f"   → Source: {segmented_zarr}")
        
        # Create napari viewer
        print(f"🔍 Opening in napari...")
        viewer = napari.Viewer()
        
        # Add z-optimized raw data channels
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
                    name=f'{condition_name} - {channel_name} (Z-opt)',
                    colormap='gray' if c == 0 else 'green' if c == 1 else 'red' if c == 2 else 'cyan',
                    contrast_limits=[raw_data[c].min(), np.percentile(raw_data[c], 99)]
                )
        else:  # [Z, Y, X] - single channel
            viewer.add_image(
                raw_data,
                name=f'{condition_name} - DAPI (Z-opt)',
                colormap='gray',
                contrast_limits=[raw_data.min(), np.percentile(raw_data, 99)]
            )
        
        # Add z-optimized segmentation masks
        viewer.add_labels(
            masks,
            name=f'Z-Optimized Masks ({n_nuclei} nuclei)',
            opacity=0.7
        )
        
        print(f"✅ Napari opened successfully!")
        print(f"💡 Viewing z-optimized data ({compression:.1f}% compression)")
        return viewer
        
    except Exception as e:
        print(f"❌ Error loading segmented data: {e}")
        return None

def open_segmentation_napari(
    condition_name: Optional[str] = None,
    organoid_idx: int = 0,
    cropped_dir: Optional[Path] = None
):
    """
    Open napari visualization for segmentation results from cropped directory (legacy)
    
    Args:
        condition_name: Name of condition to visualize (uses first if None)
        organoid_idx: Index of organoid to visualize
        cropped_dir: Directory containing the cropped zarr files
        
    Returns:
        napari.Viewer instance or None
    """
    print("⚠️  Using legacy cropped directory visualization")
    print("   → For new z-optimized results, use open_segmented_data_in_napari()")
    
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

# Focus detection and filtering functions
def calculate_organoid_focus_score(image_3d: np.ndarray, method: str = 'variance') -> float:
    """
    Calculate focus score for an organoid using specified method
    
    Args:
        image_3d: 3D image array [Z, Y, X]
        method: Focus detection method ('variance', 'gradient', 'laplacian')
        
    Returns:
        Focus score (higher = better focus)
    """
    if method == 'variance':
        return np.var(image_3d.astype(np.float64))
    elif method == 'gradient':
        # Gradient-based focus measure
        grad_x = np.gradient(image_3d, axis=2)
        grad_y = np.gradient(image_3d, axis=1)
        grad_z = np.gradient(image_3d, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        return np.mean(gradient_magnitude)
    elif method == 'laplacian':
        # Laplacian-based focus measure
        from scipy import ndimage
        laplacian = ndimage.laplace(image_3d.astype(np.float64))
        return np.var(laplacian)
    else:
        raise ValueError(f"Unknown focus detection method: {method}")

def remove_black_z_slices(
    image_3d: np.ndarray, 
    threshold: float = 0.001,
    method: str = 'intensity_threshold',
    min_z_slices: int = 20,
    verbose: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Remove black/empty z-slices from 3D image
    
    Args:
        image_3d: 3D image array [Z, Y, X]
        threshold: Threshold for black frame detection (fraction of max intensity)
        method: Detection method ('intensity_threshold', 'content_ratio', 'mean_intensity')
        min_z_slices: Minimum number of z-slices to keep
        verbose: Print processing information
        
    Returns:
        Tuple of (filtered_image, metadata)
    """
    if verbose:
        print(f"🧹 Removing black z-slices (threshold={threshold*100:.1f}%, method={method})")
    
    original_shape = image_3d.shape
    nz, ny, nx = original_shape
    
    # Calculate threshold based on image max
    max_intensity = image_3d.max()
    intensity_threshold = max_intensity * threshold
    
    # Identify non-black slices based on method
    if method == 'intensity_threshold':
        # Keep slices with max intensity above threshold
        slice_max = np.max(image_3d, axis=(1, 2))
        keep_slices = slice_max > intensity_threshold
    elif method == 'content_ratio':
        # Keep slices with enough non-zero pixels
        non_zero_ratio = np.sum(image_3d > intensity_threshold, axis=(1, 2)) / (ny * nx)
        keep_slices = non_zero_ratio > 0.1  # At least 10% non-zero pixels
    elif method == 'mean_intensity':
        # Keep slices with mean intensity above threshold
        slice_mean = np.mean(image_3d, axis=(1, 2))
        keep_slices = slice_mean > intensity_threshold
    else:
        raise ValueError(f"Unknown black frame detection method: {method}")
    
    # Find continuous regions of non-black slices
    keep_indices = np.where(keep_slices)[0]
    
    if len(keep_indices) == 0:
        if verbose:
            print(f"   ⚠️  No slices pass threshold - keeping middle {min_z_slices} slices")
        # Fallback: keep middle slices
        start_idx = max(0, (nz - min_z_slices) // 2)
        end_idx = min(nz, start_idx + min_z_slices)
        filtered_image = image_3d[start_idx:end_idx]
        kept_slices = list(range(start_idx, end_idx))
    else:
        # Keep the largest continuous region
        if len(keep_indices) < min_z_slices:
            if verbose:
                print(f"   ⚠️  Only {len(keep_indices)} slices pass threshold - expanding to {min_z_slices}")
            # Expand around the center of good slices
            center_idx = keep_indices[len(keep_indices) // 2]
            start_idx = max(0, center_idx - min_z_slices // 2)
            end_idx = min(nz, start_idx + min_z_slices)
            filtered_image = image_3d[start_idx:end_idx]
            kept_slices = list(range(start_idx, end_idx))
        else:
            # Use the continuous good slices
            start_idx = keep_indices[0]
            end_idx = keep_indices[-1] + 1
            filtered_image = image_3d[start_idx:end_idx]
            kept_slices = list(keep_indices)
    
    removed_count = nz - len(kept_slices)
    
    if verbose:
        print(f"   ✅ Z-slice filtering complete:")
        print(f"      → Original slices: {nz}")
        print(f"      → Kept slices: {len(kept_slices)} (z={kept_slices[0]}-{kept_slices[-1]})")
        print(f"      → Removed slices: {removed_count}")
        print(f"      → New shape: {filtered_image.shape}")
    
    metadata = {
        'z_filtering_applied': True,
        'original_z_slices': nz,
        'kept_z_slices': len(kept_slices),
        'removed_z_slices': removed_count,
        'kept_slice_indices': kept_slices,
        'filtering_method': method,
        'threshold_used': threshold,
        'max_intensity': max_intensity,
        'intensity_threshold': intensity_threshold
    }
    
    return filtered_image, metadata

def optimize_z_slices_after_segmentation(
    image_3d: np.ndarray,
    masks_3d: np.ndarray,
    strategy: str = 'z_range',
    padding: int = 2,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Remove z-slices that contain no segmented nuclei (post-segmentation optimization)
    
    Args:
        image_3d: Original 3D image [Z, Y, X]
        masks_3d: Segmentation masks [Z, Y, X] 
        strategy: Optimization strategy ('covered_only', 'z_range', 'padded_range')
        padding: Number of extra z-slices to keep on each side (for 'padded_range')
        verbose: Print processing information
        
    Returns:
        Tuple of (optimized_image, optimized_masks, metadata)
    """
    if verbose:
        print(f"✂️ Post-segmentation z-slice optimization (strategy={strategy})")
    
    original_z, ny, nx = masks_3d.shape
    
    # Find z-slices that contain segmented nuclei
    z_has_nuclei = np.array([np.any(masks_3d[z] > 0) for z in range(original_z)])
    z_indices_with_nuclei = np.where(z_has_nuclei)[0]
    
    if len(z_indices_with_nuclei) == 0:
        if verbose:
            print(f"   ⚠️  No nuclei found in any z-slice - keeping original data")
        return image_3d, masks_3d, {
            'z_optimization_applied': False,
            'reason': 'no_nuclei_found',
            'original_z_count': original_z,
            'optimized_z_count': original_z
        }
    
    # Determine z-slices to keep based on strategy
    if strategy == "covered_only":
        # Keep only z-slices with segmented nuclei
        z_slices_to_keep = z_indices_with_nuclei.tolist()
        
    elif strategy == "z_range":
        # Keep z-range from first to last slice with nuclei
        z_min, z_max = z_indices_with_nuclei[0], z_indices_with_nuclei[-1]
        z_slices_to_keep = list(range(z_min, z_max + 1))
        
    elif strategy == "padded_range":
        # Keep z-range with padding around nuclei-containing slices
        z_min, z_max = z_indices_with_nuclei[0], z_indices_with_nuclei[-1]
        z_min_padded = max(0, z_min - padding)
        z_max_padded = min(original_z - 1, z_max + padding)
        z_slices_to_keep = list(range(z_min_padded, z_max_padded + 1))
        
    else:
        # Unknown strategy - keep all
        z_slices_to_keep = list(range(original_z))
    
    # Apply optimization if beneficial
    z_slices_removed = [z for z in range(original_z) if z not in z_slices_to_keep]
    
    if len(z_slices_removed) == 0:
        if verbose:
            print(f"   💡 No z-slices can be removed with strategy '{strategy}'")
        return image_3d, masks_3d, {
            'z_optimization_applied': False,
            'reason': 'no_slices_to_remove',
            'original_z_count': original_z,
            'optimized_z_count': original_z,
            'strategy': strategy
        }
    
    # Create optimized data
    optimized_image = image_3d[z_slices_to_keep]
    optimized_masks = masks_3d[z_slices_to_keep]
    
    compression_ratio = len(z_slices_removed) / original_z * 100
    
    if verbose:
        print(f"   ✅ Z-slice optimization complete:")
        print(f"      → Original z-slices: {original_z}")
        print(f"      → Optimized z-slices: {len(z_slices_to_keep)}")
        print(f"      → Removed z-slices: {len(z_slices_removed)}")
        print(f"      → Compression: {compression_ratio:.1f}%")
        print(f"      → Nuclei z-range: {z_indices_with_nuclei[0]}-{z_indices_with_nuclei[-1]}")
        print(f"      → Kept z-range: {z_slices_to_keep[0]}-{z_slices_to_keep[-1]}")
    
    metadata = {
        'z_optimization_applied': True,
        'strategy': strategy,
        'padding': padding if strategy == 'padded_range' else None,
        'original_z_count': original_z,
        'optimized_z_count': len(z_slices_to_keep),
        'removed_z_count': len(z_slices_removed),
        'compression_ratio_percent': compression_ratio,
        'nuclei_z_range': (int(z_indices_with_nuclei[0]), int(z_indices_with_nuclei[-1])),
        'kept_z_range': (z_slices_to_keep[0], z_slices_to_keep[-1]),
        'z_slices_kept': z_slices_to_keep,
        'z_slices_removed': z_slices_removed,
        'nuclei_containing_slices': len(z_indices_with_nuclei)
    }
    
    return optimized_image, optimized_masks, metadata

def test_filtering_setup(
    cropped_dir: Optional[Path] = None,
    condition_name: Optional[str] = None,
    organoid_idx: int = 0,
    channel: int = 0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Test focus filtering and z-slice processing setup on a single organoid
    
    Args:
        cropped_dir: Directory containing cropped zarr files
        condition_name: Condition to test (uses first available if None)
        organoid_idx: Organoid index to test
        channel: Channel to test
        verbose: Print detailed information
        
    Returns:
        Dictionary with test results
    """
    print("🧪 TESTING FILTERING SETUP")
    print("=" * 50)
    
    # Set up directories
    if cropped_dir is None:
        try:
            from karyosight.config import CROPPED_DIR
            cropped_dir = Path(CROPPED_DIR)
        except:
            cropped_dir = Path("D:/LB_TEST/cropped")
    
    # Find available conditions
    zarr_files = []
    for condition_dir in cropped_dir.iterdir():
        if condition_dir.is_dir():
            bundled_zarr = condition_dir / f"{condition_dir.name}_bundled.zarr"
            if bundled_zarr.exists():
                zarr_files.append(bundled_zarr)
    
    if not zarr_files:
        print(f"❌ No zarr files found in {cropped_dir}")
        return {'error': 'No zarr files found'}
    
    # Select zarr file
    if condition_name is None:
        zarr_path = zarr_files[0]
        condition_name = zarr_path.parent.name
    else:
        zarr_path = None
        for zf in zarr_files:
            if zf.parent.name == condition_name:
                zarr_path = zf
                break
        if zarr_path is None:
            available = [zf.parent.name for zf in zarr_files]
            print(f"❌ Condition '{condition_name}' not found. Available: {available}")
            return {'error': f'Condition not found: {condition_name}'}
    
    print(f"📂 Testing with: {condition_name}, organoid {organoid_idx}")
    print(f"   File: {zarr_path}")
    
    # Load organoid data
    try:
        bundle = zarr.open_group(str(zarr_path), mode='r')
        organoid_key = f"organoid_{organoid_idx:04d}"
        
        if hasattr(bundle, 'keys') and organoid_key in bundle:
            organoid_data = bundle[organoid_key]['data'][:]
            img_3d = organoid_data[channel]
        else:
            organoid_data = bundle['data'][organoid_idx]
            img_3d = organoid_data[channel]
        
        if hasattr(img_3d, 'compute'):
            img_3d = img_3d.compute()
        else:
            img_3d = np.array(img_3d)
        
        print(f"✅ Loaded image: {img_3d.shape}")
        print(f"   Data type: {img_3d.dtype}")
        print(f"   Intensity range: {img_3d.min():.1f} - {img_3d.max():.1f}")
        
    except Exception as e:
        print(f"❌ Error loading organoid: {e}")
        return {'error': f'Failed to load organoid: {e}'}
    
    test_results = {
        'condition': condition_name,
        'organoid_idx': organoid_idx,
        'original_shape': img_3d.shape,
        'original_intensity_range': (float(img_3d.min()), float(img_3d.max()))
    }
    
    # Test focus filtering
    print(f"\n🔍 TESTING FOCUS FILTERING:")
    if CONFIG_AVAILABLE and ENABLE_FOCUS_FILTERING:
        print(f"   ✅ Focus filtering ENABLED")
        print(f"      → Method: {FOCUS_DETECTION_METHOD}")
        print(f"      → Threshold: {MIN_FOCUS_SCORE}")
        
        focus_score = calculate_organoid_focus_score(img_3d, method=FOCUS_DETECTION_METHOD)
        is_in_focus = focus_score >= MIN_FOCUS_SCORE
        
        print(f"   📊 Test Results:")
        print(f"      → Focus score: {focus_score:.1f}")
        print(f"      → In focus: {'YES' if is_in_focus else 'NO'}")
        
        if is_in_focus:
            print(f"      ✅ Organoid would be SEGMENTED")
        else:
            print(f"      ❌ Organoid would be SKIPPED (out of focus)")
        
        test_results['focus_filtering'] = {
            'enabled': True,
            'method': FOCUS_DETECTION_METHOD,
            'threshold': MIN_FOCUS_SCORE,
            'focus_score': focus_score,
            'is_in_focus': is_in_focus,
            'would_be_processed': is_in_focus
        }
    else:
        print(f"   ❌ Focus filtering DISABLED")
        print(f"      → All organoids will be processed regardless of focus")
        
        test_results['focus_filtering'] = {
            'enabled': False,
            'would_be_processed': True
        }
    
    # Test z-slice filtering
    print(f"\n🧹 TESTING Z-SLICE FILTERING:")
    if CONFIG_AVAILABLE and REMOVE_BLACK_FRAMES:
        print(f"   ✅ Z-slice filtering ENABLED")
        print(f"      → Method: {BLACK_FRAME_METHOD}")
        print(f"      → Threshold: {BLACK_FRAME_THRESHOLD*100:.1f}% of max intensity")
        print(f"      → Min z-slices: {MIN_Z_SLICES}")
        
        filtered_img, z_metadata = remove_black_z_slices(
            img_3d,
            threshold=BLACK_FRAME_THRESHOLD,
            method=BLACK_FRAME_METHOD,
            min_z_slices=MIN_Z_SLICES,
            verbose=True
        )
        
        test_results['z_slice_filtering'] = {
            'enabled': True,
            'method': BLACK_FRAME_METHOD,
            'threshold': BLACK_FRAME_THRESHOLD,
            'min_z_slices': MIN_Z_SLICES,
            'original_z_slices': z_metadata['original_z_slices'],
            'kept_z_slices': z_metadata['kept_z_slices'],
            'removed_z_slices': z_metadata['removed_z_slices'],
            'filtered_shape': filtered_img.shape,
            'kept_slice_indices': z_metadata['kept_slice_indices']
        }
        
        img_3d = filtered_img  # Use filtered image for any further processing
        
    else:
        print(f"   ❌ Z-slice filtering DISABLED")
        print(f"      → All z-slices will be used")
        
        test_results['z_slice_filtering'] = {
            'enabled': False,
            'filtered_shape': img_3d.shape
        }
    
    # Test post-segmentation z-slice optimization
    print(f"\n✂️ TESTING POST-SEGMENTATION Z-SLICE OPTIMIZATION:")
    if CONFIG_AVAILABLE and ENABLE_Z_SLICE_OPTIMIZATION:
        print(f"   ✅ Z-slice optimization ENABLED")
        print(f"      → Strategy: {Z_OPTIMIZATION_STRATEGY}")
        if Z_OPTIMIZATION_STRATEGY == 'padded_range':
            print(f"      → Padding: {Z_OPTIMIZATION_PADDING} slices")
        
        # Simulate masks with nuclei in middle z-slices for demonstration
        demo_shape = test_results['z_slice_filtering']['filtered_shape']
        demo_masks = np.zeros(demo_shape, dtype=np.uint16)
        # Put fake nuclei in middle third of z-stack
        z_start = demo_shape[0] // 3
        z_end = 2 * demo_shape[0] // 3
        demo_masks[z_start:z_end, 100:200, 100:200] = 1  # Fake nucleus
        
        # Test optimization
        _, _, z_opt_test = optimize_z_slices_after_segmentation(
            img_3d, demo_masks,
            strategy=Z_OPTIMIZATION_STRATEGY,
            padding=Z_OPTIMIZATION_PADDING,
            verbose=True
        )
        
        test_results['z_slice_optimization'] = {
            'enabled': True,
            'strategy': Z_OPTIMIZATION_STRATEGY,
            'padding': Z_OPTIMIZATION_PADDING,
            'demo_results': z_opt_test,
            'potential_compression': z_opt_test.get('compression_ratio_percent', 0)
        }
        
    else:
        print(f"   ❌ Post-segmentation z-slice optimization DISABLED")
        print(f"      → Final masks will keep all z-slices")
        
        test_results['z_slice_optimization'] = {
            'enabled': False
        }
    
    # Test overall processing status
    print(f"\n🎯 OVERALL TEST RESULTS:")
    would_process = (
        test_results['focus_filtering']['would_be_processed'] 
        if test_results['focus_filtering']['enabled'] 
        else True
    )
    
    if would_process:
        final_shape = test_results['z_slice_filtering']['filtered_shape']
        print(f"   ✅ This organoid WOULD BE SEGMENTED")
        print(f"      → Image shape before segmentation: {final_shape}")
        if test_results['z_slice_filtering']['enabled']:
            print(f"      → Z-slices used for segmentation: {test_results['z_slice_filtering']['kept_z_slices']}/{test_results['z_slice_filtering']['original_z_slices']}")
        if test_results['z_slice_optimization']['enabled']:
            potential_compression = test_results['z_slice_optimization']['potential_compression']
            print(f"      → Z-slices will be further optimized after segmentation (~{potential_compression:.1f}% reduction)")
    else:
        print(f"   ❌ This organoid WOULD BE SKIPPED")
        print(f"      → Reason: Out of focus")
    
    test_results['final_processing_decision'] = {
        'would_be_processed': would_process,
        'final_shape': test_results['z_slice_filtering']['filtered_shape']
    }
    
    print(f"\n💡 CONFIG STATUS:")
    print(f"   → Config available: {CONFIG_AVAILABLE}")
    if CONFIG_AVAILABLE:
        print(f"   → Focus filtering: {ENABLE_FOCUS_FILTERING}")
        print(f"   → Pre-segmentation z-slice filtering: {REMOVE_BLACK_FRAMES}")
        print(f"   → Post-segmentation z-slice optimization: {ENABLE_Z_SLICE_OPTIMIZATION}")
    else:
        print(f"   → ⚠️  No filtering will be applied (config not available)")
    
    return test_results

# ======================================================================================
# SEGMENTED DATA UTILITIES
# ======================================================================================

def list_segmented_conditions(segmented_dir: Optional[Path] = None) -> List[str]:
    """
    List available segmented conditions
    
    Args:
        segmented_dir: Directory containing segmented zarr files
        
    Returns:
        List of available condition names
    """
    if segmented_dir is None:
        try:
            from karyosight.config import SEGMENTED_DIR
            segmented_dir = Path(SEGMENTED_DIR)
        except:
            segmented_dir = Path("D:/LB_TEST/segmented")
    
    if not segmented_dir.exists():
        print(f"❌ Segmented directory not found: {segmented_dir}")
        return []
    
    available_conditions = []
    for condition_dir in segmented_dir.iterdir():
        if condition_dir.is_dir():
            segmented_zarr = condition_dir / f"{condition_dir.name}_segmented.zarr"
            if segmented_zarr.exists():
                available_conditions.append(condition_dir.name)
    
    return sorted(available_conditions)

def get_segmented_organoid_count(condition_name: str, segmented_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get information about organoids in a segmented condition
    
    Args:
        condition_name: Name of condition
        segmented_dir: Directory containing segmented zarr files
        
    Returns:
        Dictionary with organoid information
    """
    if segmented_dir is None:
        try:
            from karyosight.config import SEGMENTED_DIR
            segmented_dir = Path(SEGMENTED_DIR)
        except:
            segmented_dir = Path("D:/LB_TEST/segmented")
    
    segmented_zarr = segmented_dir / condition_name / f"{condition_name}_segmented.zarr"
    
    if not segmented_zarr.exists():
        return {'error': f'Segmented zarr not found: {segmented_zarr}'}
    
    try:
        bundle = zarr.open_group(str(segmented_zarr), mode='r')
        
        organoid_keys = [k for k in bundle.keys() if k.startswith('organoid_')]
        organoid_indices = sorted([int(k.split('_')[1]) for k in organoid_keys])
        
        total_nuclei = 0
        organoid_info = {}
        
        for org_key in organoid_keys:
            org_idx = int(org_key.split('_')[1])
            org_group = bundle[org_key]
            metadata = dict(org_group.attrs)
            
            n_nuclei = metadata.get('n_nuclei', 0)
            compression = metadata.get('compression_ratio_percent', 0)
            original_shape = metadata.get('original_raw_shape', [])
            optimized_shape = metadata.get('optimized_raw_shape', [])
            
            organoid_info[org_idx] = {
                'n_nuclei': n_nuclei,
                'compression_percent': compression,
                'original_shape': original_shape,
                'optimized_shape': optimized_shape
            }
            
            total_nuclei += n_nuclei
        
        bundle_metadata = dict(bundle.attrs)
        
        return {
            'condition': condition_name,
            'n_organoids': len(organoid_indices),
            'organoid_indices': organoid_indices,
            'total_nuclei': total_nuclei,
            'average_nuclei_per_organoid': total_nuclei / len(organoid_indices) if organoid_indices else 0,
            'average_compression_percent': bundle_metadata.get('average_compression_percent', 0),
            'organoid_details': organoid_info,
            'zarr_path': str(segmented_zarr)
        }
        
    except Exception as e:
        return {'error': f'Error reading segmented data: {e}'}

def quick_view_segmented(
    condition_name: Optional[str] = None,
    organoid_idx: int = 0,
    segmented_dir: Optional[Path] = None
):
    """
    Quick function to view segmented data in napari
    
    Args:
        condition_name: Condition to view (first available if None)
        organoid_idx: Organoid index to view
        segmented_dir: Directory containing segmented zarr files
        
    Returns:
        napari.Viewer instance or None
    """
    # List available conditions if none specified
    if condition_name is None:
        available = list_segmented_conditions(segmented_dir)
        if not available:
            print("❌ No segmented conditions found")
            return None
        condition_name = available[0]
        print(f"📋 Available conditions: {available}")
        print(f"🎯 Using: {condition_name}")
    
    # Get organoid info
    info = get_segmented_organoid_count(condition_name, segmented_dir)
    if 'error' in info:
        print(f"❌ {info['error']}")
        return None
    
    print(f"📊 Condition info:")
    print(f"   → Organoids: {info['n_organoids']}")
    print(f"   → Total nuclei: {info['total_nuclei']}")
    print(f"   → Average compression: {info['average_compression_percent']:.1f}%")
    print(f"   → Available organoid indices: {info['organoid_indices']}")
    
    if organoid_idx not in info['organoid_indices']:
        print(f"❌ Organoid {organoid_idx} not available")
        print(f"   → Use one of: {info['organoid_indices']}")
        return None
    
    # Open in napari
    return open_segmented_data_in_napari(condition_name, organoid_idx, segmented_dir)

def compare_segmented_conditions(
    condition_names: Optional[List[str]] = None,
    organoid_idx: int = 0,
    segmented_dir: Optional[Path] = None
):
    """
    Compare multiple segmented conditions side by side in napari
    
    Args:
        condition_names: List of conditions to compare (all available if None)
        organoid_idx: Organoid index to compare
        segmented_dir: Directory containing segmented zarr files
        
    Returns:
        napari.Viewer instance or None
    """
    try:
        import napari
    except ImportError:
        print("❌ Napari not available. Install with: pip install napari")
        return None
    
    # Get available conditions
    available = list_segmented_conditions(segmented_dir)
    if not available:
        print("❌ No segmented conditions found")
        return None
    
    if condition_names is None:
        condition_names = available
    
    # Validate condition names
    missing = [c for c in condition_names if c not in available]
    if missing:
        print(f"❌ Conditions not found: {missing}")
        print(f"   Available: {available}")
        return None
    
    print(f"🔀 Comparing {len(condition_names)} conditions, organoid {organoid_idx}")
    
    # Determine segmented directory
    if segmented_dir is None:
        try:
            from karyosight.config import SEGMENTED_DIR
            segmented_dir = Path(SEGMENTED_DIR)
        except:
            segmented_dir = Path("D:/LB_TEST/segmented")
    
    # Create napari viewer
    print(f"🔍 Opening comparison in napari...")
    viewer = napari.Viewer()
    
    # Load data for each condition
    for i, condition_name in enumerate(condition_names):
        try:
            segmented_zarr = segmented_dir / condition_name / f"{condition_name}_segmented.zarr"
            bundle = zarr.open_group(str(segmented_zarr), mode='r')
            organoid_key = f"organoid_{organoid_idx:04d}"
            
            if organoid_key not in bundle:
                print(f"   ⚠️  Organoid {organoid_idx} not found in {condition_name}")
                continue
            
            organoid_group = bundle[organoid_key]
            
            # Load data
            raw_data = organoid_group['data'][:]
            if hasattr(raw_data, 'compute'):
                raw_data = raw_data.compute()
            masks = organoid_group['masks'][:]
            if hasattr(masks, 'compute'):
                masks = masks.compute()
            
            # Get metadata
            metadata = dict(organoid_group.attrs)
            n_nuclei = metadata.get('n_nuclei', 0)
            compression = metadata.get('compression_ratio_percent', 0)
            
            # Add to viewer with unique names
            if raw_data.ndim == 4:  # [C, Z, Y, X]
                # Add nucleus channel only for comparison
                viewer.add_image(
                    raw_data[0],
                    name=f'{condition_name} - DAPI (Z-opt)',
                    colormap='gray',
                    visible=(i == 0),  # Only show first condition initially
                    contrast_limits=[raw_data[0].min(), np.percentile(raw_data[0], 99)]
                )
            else:  # [Z, Y, X]
                viewer.add_image(
                    raw_data,
                    name=f'{condition_name} - DAPI (Z-opt)',
                    colormap='gray',
                    visible=(i == 0),
                    contrast_limits=[raw_data.min(), np.percentile(raw_data, 99)]
                )
            
            # Add masks
            viewer.add_labels(
                masks,
                name=f'{condition_name} - Masks ({n_nuclei} nuclei, {compression:.1f}% comp)',
                opacity=0.7,
                visible=(i == 0)  # Only show first condition initially
            )
            
            print(f"   ✅ Loaded {condition_name}: {n_nuclei} nuclei, {compression:.1f}% compression")
            
        except Exception as e:
            print(f"   ❌ Error loading {condition_name}: {e}")
            continue
    
    print(f"✅ Comparison viewer opened!")
    print(f"💡 Tips:")
    print(f"   • Toggle layer visibility to compare conditions")
    print(f"   • All data is z-optimized for consistent comparison")
    print(f"   • Use opacity sliders to overlay different masks")
    
    return viewer

def compare_raw_vs_optimized(
    condition_name: Optional[str] = None,
    organoid_idx: int = 0,
    cropped_dir: Optional[Path] = None,
    segmented_dir: Optional[Path] = None
):
    """
    Compare original raw data vs z-optimized data side-by-side in napari
    
    Args:
        condition_name: Condition to compare (first available if None)
        organoid_idx: Organoid index to compare
        cropped_dir: Directory containing original cropped zarr files
        segmented_dir: Directory containing z-optimized segmented zarr files
        
    Returns:
        napari.Viewer instance or None
    """
    try:
        import napari
    except ImportError:
        print("❌ Napari not available. Install with: pip install napari")
        return None
    
    # Determine directories
    if cropped_dir is None:
        try:
            from karyosight.config import CROPPED_DIR
            cropped_dir = Path(CROPPED_DIR)
        except:
            cropped_dir = Path("D:/LB_TEST/cropped")
    
    if segmented_dir is None:
        try:
            from karyosight.config import SEGMENTED_DIR
            segmented_dir = Path(SEGMENTED_DIR)
        except:
            segmented_dir = Path("D:/LB_TEST/segmented")
    
    # Find available conditions
    if condition_name is None:
        # Look for conditions that exist in both directories
        cropped_conditions = []
        for condition_dir in cropped_dir.iterdir():
            if condition_dir.is_dir():
                bundled_zarr = condition_dir / f"{condition_dir.name}_bundled.zarr"
                if bundled_zarr.exists():
                    cropped_conditions.append(condition_dir.name)
        
        segmented_conditions = list_segmented_conditions(segmented_dir)
        
        # Find conditions that exist in both
        common_conditions = list(set(cropped_conditions) & set(segmented_conditions))
        
        if not common_conditions:
            print("❌ No conditions found in both cropped and segmented directories")
            print(f"   Cropped: {cropped_conditions}")
            print(f"   Segmented: {segmented_conditions}")
            return None
        
        condition_name = common_conditions[0]
        print(f"📋 Available conditions: {common_conditions}")
        print(f"🎯 Using: {condition_name}")
    
    print(f"🔀 COMPARING RAW vs Z-OPTIMIZED DATA")
    print(f"=" * 60)
    print(f"Condition: {condition_name}, Organoid: {organoid_idx}")
    
    # Load original raw data from cropped directory
    cropped_zarr = cropped_dir / condition_name / f"{condition_name}_bundled.zarr"
    original_data = None
    original_shape = None
    
    if cropped_zarr.exists():
        try:
            bundle = zarr.open_group(str(cropped_zarr), mode='r')
            organoid_key = f"organoid_{organoid_idx:04d}"
            
            if organoid_key in bundle:
                organoid_group = bundle[organoid_key]
                original_data = organoid_group['data'][:]
                if hasattr(original_data, 'compute'):
                    original_data = original_data.compute()
                else:
                    original_data = np.array(original_data)
                
                original_shape = original_data.shape
                print(f"✅ Original raw data loaded: {original_shape}")
            else:
                print(f"❌ Organoid {organoid_key} not found in cropped data")
                return None
                
        except Exception as e:
            print(f"❌ Error loading original data: {e}")
            return None
    else:
        print(f"❌ Cropped zarr not found: {cropped_zarr}")
        return None
    
    # Load z-optimized data from segmented directory
    segmented_zarr = segmented_dir / condition_name / f"{condition_name}_segmented.zarr"
    optimized_data = None
    optimized_masks = None
    optimized_shape = None
    compression_ratio = 0
    
    if segmented_zarr.exists():
        try:
            bundle = zarr.open_group(str(segmented_zarr), mode='r')
            organoid_key = f"organoid_{organoid_idx:04d}"
            
            if organoid_key in bundle:
                organoid_group = bundle[organoid_key]
                
                # Load z-optimized raw data
                optimized_data = organoid_group['data'][:]
                if hasattr(optimized_data, 'compute'):
                    optimized_data = optimized_data.compute()
                else:
                    optimized_data = np.array(optimized_data)
                
                # Load z-optimized masks
                optimized_masks = organoid_group['masks'][:]
                if hasattr(optimized_masks, 'compute'):
                    optimized_masks = optimized_masks.compute()
                else:
                    optimized_masks = np.array(optimized_masks)
                
                # Get metadata
                metadata = dict(organoid_group.attrs)
                optimized_shape = optimized_data.shape
                compression_ratio = metadata.get('compression_ratio_percent', 0)
                n_nuclei = metadata.get('n_nuclei', 0)
                
                print(f"✅ Z-optimized data loaded: {optimized_shape}")
                print(f"✅ Z-optimized masks loaded: {optimized_masks.shape}")
                print(f"📊 Compression: {compression_ratio:.1f}%")
                print(f"🧬 Nuclei count: {n_nuclei}")
            else:
                print(f"❌ Organoid {organoid_key} not found in segmented data")
                return None
                
        except Exception as e:
            print(f"❌ Error loading optimized data: {e}")
            return None
    else:
        print(f"❌ Segmented zarr not found: {segmented_zarr}")
        return None
    
    # Calculate z-slice comparison
    if original_shape and optimized_shape:
        original_z = original_shape[1]
        optimized_z = optimized_shape[1]
        z_reduction = original_z - optimized_z
        z_reduction_percent = (z_reduction / original_z) * 100
        
        print(f"\n📏 Z-SLICE COMPARISON:")
        print(f"   → Original z-slices: {original_z}")
        print(f"   → Optimized z-slices: {optimized_z}")
        print(f"   → Slices removed: {z_reduction}")
        print(f"   → Z-reduction: {z_reduction_percent:.1f}%")
    
    # Create napari viewer
    print(f"\n🔍 Opening comparison in napari...")
    viewer = napari.Viewer()
    
    # Get channel names
    try:
        if CONFIG_AVAILABLE:
            from karyosight.config import CHANNELS
            channel_names = CHANNELS
        else:
            channel_names = ['nucleus', 'tetraploid', 'diploid', 'sox9', 'sox2', 'brightfield']
    except:
        channel_names = ['DAPI', 'GFP', 'RFP', 'Cy5', 'Channel_4', 'Channel_5']
    
    # Add original raw data channels
    if original_data.ndim == 4:  # [C, Z, Y, X]
        for c in range(original_data.shape[0]):
            channel_name = channel_names[c] if c < len(channel_names) else f'Channel_{c}'
            viewer.add_image(
                original_data[c],
                name=f'ORIGINAL - {channel_name} ({original_z} z-slices)',
                colormap='gray' if c == 0 else 'green' if c == 1 else 'red' if c == 2 else 'cyan',
                visible=(c == 0),  # Only show DAPI initially
                contrast_limits=[original_data[c].min(), np.percentile(original_data[c], 99)]
            )
    
    # Add z-optimized raw data channels
    if optimized_data.ndim == 4:  # [C, Z, Y, X]
        for c in range(optimized_data.shape[0]):
            channel_name = channel_names[c] if c < len(channel_names) else f'Channel_{c}'
            viewer.add_image(
                optimized_data[c],
                name=f'Z-OPTIMIZED - {channel_name} ({optimized_z} z-slices)',
                colormap='gray' if c == 0 else 'green' if c == 1 else 'red' if c == 2 else 'cyan',
                visible=False,  # Start hidden for comparison
                contrast_limits=[optimized_data[c].min(), np.percentile(optimized_data[c], 99)]
            )
    
    # Add z-optimized masks
    if optimized_masks is not None:
        viewer.add_labels(
            optimized_masks,
            name=f'Z-Optimized Masks ({n_nuclei} nuclei)',
            opacity=0.7,
            visible=False  # Start hidden
        )
    
    print(f"✅ Comparison viewer opened!")
    print(f"\n💡 COMPARISON TIPS:")
    print(f"   • Toggle between 'ORIGINAL' and 'Z-OPTIMIZED' layers")
    print(f"   • Notice the different z-slice counts in layer names")
    print(f"   • Original: {original_z} slices → Optimized: {optimized_z} slices")
    print(f"   • File size reduction: {compression_ratio:.1f}%")
    print(f"   • All nuclei preserved in optimized version")
    print(f"   • Use layer visibility toggles to compare")
    
    return viewer

def quick_compare_raw_vs_optimized(
    condition_name: Optional[str] = None,
    organoid_idx: int = 0
):
    """
    Quick function to compare raw vs optimized data
    
    Args:
        condition_name: Condition to compare (first available if None)
        organoid_idx: Organoid index to compare
        
    Returns:
        napari.Viewer instance or None
    """
    return compare_raw_vs_optimized(condition_name, organoid_idx)

# ======================================================================================
# SEGMENTED ORGANOID VISUALIZATION
# ======================================================================================

class SegmentedOrganoidVisualizer:
    """
    A class for visualizing segmented organoids from the new segmented directory structure.
    Creates grid layouts of max projections with segmentation mask overlays.
    """
    
    def __init__(self, segmented_dir: Optional[Path] = None):
        """
        Initialize the visualizer with the segmented data directory.
        
        Args:
            segmented_dir: Path to the segmented directory (uses config default if None)
        """
        if segmented_dir is None:
            try:
                from karyosight.config import SEGMENTED_DIR
                self.segmented_dir = Path(SEGMENTED_DIR)
            except:
                self.segmented_dir = Path("D:/LB_TEST/segmented")
        else:
            self.segmented_dir = Path(segmented_dir)
        
        # Create visualization subdirectory
        self.vis_dir = self.segmented_dir / "visualization"
        self.vis_dir.mkdir(exist_ok=True)
        
        print(f"📊 SegmentedOrganoidVisualizer initialized")
        print(f"   → Segmented dir: {self.segmented_dir}")
        print(f"   → Visualization dir: {self.vis_dir}")
    
    def count_organoids_per_condition(self) -> Dict[str, int]:
        """
        Count the number of organoids in each condition's segmented zarr file.
        
        Returns:
            Dictionary mapping condition names to organoid counts
        """
        counts = {}
        
        # Find all segmented zarr files
        segmented_zarrs = list(self.segmented_dir.rglob("*_segmented.zarr"))
        
        for zarr_path in segmented_zarrs:
            condition = zarr_path.parent.name
            
            try:
                bundle = zarr.open_group(str(zarr_path), mode='r')
                organoid_keys = [k for k in bundle.keys() if k.startswith('organoid_')]
                n_organoids = len(organoid_keys)
                counts[condition] = n_organoids
                
            except Exception as e:
                print(f"Warning: Could not read {zarr_path}: {e}")
                counts[condition] = 0
        
        return counts
    
    def _load_segmented_data(self, 
                            zarr_path: Path, 
                            channel: int = 0,
                            sample_count: Optional[int] = None) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        """
        Load segmented organoid data (raw data + masks) from a zarr file.
        
        Args:
            zarr_path: Path to the segmented zarr file
            channel: Channel index to load for raw data
            sample_count: Number of organoids to sample (None for all)
            
        Returns:
            Tuple of (max_projections, mask_projections, indices)
        """
        bundle = zarr.open_group(str(zarr_path), mode='r')
        
        # Get all organoid keys
        organoid_keys = sorted([k for k in bundle.keys() if k.startswith('organoid_')])
        total_organoids = len(organoid_keys)
        
        # Determine which organoids to load
        if sample_count is None or sample_count >= total_organoids:
            indices = list(range(total_organoids))
            selected_keys = organoid_keys
        else:
            # Randomly sample organoids
            indices = np.random.choice(total_organoids, sample_count, replace=False).tolist()
            indices.sort()
            selected_keys = [organoid_keys[i] for i in indices]
        
        # Load data for selected organoids
        max_projections = []
        mask_projections = []
        
        for key in selected_keys:
            organoid_group = bundle[key]
            
            # Load z-optimized raw data
            raw_data = organoid_group['data'][:]
            if hasattr(raw_data, 'compute'):
                raw_data = raw_data.compute()
            else:
                raw_data = np.array(raw_data)
            
            # Load z-optimized masks
            masks = organoid_group['masks'][:]
            if hasattr(masks, 'compute'):
                masks = masks.compute()
            else:
                masks = np.array(masks)
            
            # Create max projections
            if raw_data.ndim == 4:  # [C, Z, Y, X]
                raw_max_proj = np.max(raw_data[channel], axis=0)  # Max along Z
            else:  # [Z, Y, X] - single channel
                raw_max_proj = np.max(raw_data, axis=0)
            
            # Create mask max projection (any nuclei present across z)
            mask_max_proj = np.max(masks, axis=0)
            
            max_projections.append(raw_max_proj)
            mask_projections.append(mask_max_proj)
        
        return max_projections, mask_projections, indices
    
    def _calculate_grid_size(self, n_items: int) -> Tuple[int, int]:
        """Calculate optimal grid size for displaying n_items."""
        if n_items == 1:
            return 1, 1
        
        # Try to make a roughly square grid
        cols = int(np.ceil(np.sqrt(n_items)))
        rows = int(np.ceil(n_items / cols))
        return rows, cols
    
    def create_segmented_organoid_grid(self,
                                      condition: str,
                                      channel: int = 0,
                                      sample_count: Optional[int] = None,
                                      mask_overlay: bool = True,
                                      mask_alpha: float = 0.3,
                                      auto_scale_method: str = 'individual',
                                      figsize_per_organoid: Tuple[float, float] = (3, 3),
                                      save_svg: bool = True,
                                      save_png: bool = False,
                                      show_plot: bool = True) -> Optional[Path]:
        """
        Create a grid visualization of segmented organoids with mask overlays.
        
        Args:
            condition: Name of the condition to visualize
            channel: Channel to visualize for raw data
            sample_count: Number of organoids to sample (None for all)
            mask_overlay: Whether to overlay segmentation masks
            mask_alpha: Transparency of mask overlay (0-1)
            auto_scale_method: 'individual' or 'global' scaling
            figsize_per_organoid: Size of each subplot in inches
            save_svg: Whether to save as SVG (vector format)
            save_png: Whether to save as PNG (raster format)
            show_plot: Whether to display the plot
            
        Returns:
            Path to saved file if save_svg/save_png=True, else None
        """
        # Find the segmented zarr file for this condition
        zarr_path = self.segmented_dir / condition / f"{condition}_segmented.zarr"
        
        if not zarr_path.exists():
            print(f"Warning: No segmented zarr found for condition {condition}")
            print(f"Expected path: {zarr_path}")
            return None
        
        # Load segmented data
        projections, mask_projections, indices = self._load_segmented_data(
            zarr_path, channel=channel, sample_count=sample_count
        )
        
        if len(projections) == 0:
            print(f"No organoids found for condition {condition}")
            return None
        
        # Calculate grid layout
        rows, cols = self._calculate_grid_size(len(projections))
        
        # Calculate figure size
        fig_width = cols * figsize_per_organoid[0]
        fig_height = rows * figsize_per_organoid[1]
        
        # Compute scaling
        vmin_vals = []
        vmax_vals = []
        
        if auto_scale_method == 'individual':
            for proj in projections:
                vmin = np.percentile(proj[proj > 0], 1) if np.any(proj > 0) else 0
                vmax = np.percentile(proj, 99.5)
                vmin_vals.append(vmin)
                vmax_vals.append(vmax)
        else:  # global
            all_nonzero = np.concatenate([p[p > 0] for p in projections if np.any(p > 0)])
            if len(all_nonzero) > 0:
                global_vmin = np.percentile(all_nonzero, 1)
                global_vmax = np.percentile(np.concatenate([p.flatten() for p in projections]), 99.5)
            else:
                global_vmin = 0
                global_vmax = max(p.max() for p in projections)
            
            vmin_vals = [global_vmin] * len(projections)
            vmax_vals = [global_vmax] * len(projections)
        
        # Create the figure
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        if rows * cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if rows > 1 or cols > 1 else [axes]
        
        # Plot each organoid
        for i, (proj, mask_proj, idx) in enumerate(zip(projections, mask_projections, indices)):
            ax = axes[i]
            
            vmin, vmax = vmin_vals[i], vmax_vals[i]
            
            # Show raw data
            ax.imshow(proj, cmap='gray', vmin=vmin, vmax=vmax)
            
            # Overlay masks if requested
            if mask_overlay and np.any(mask_proj > 0):
                # Create colored mask overlay
                mask_colored = np.zeros((*mask_proj.shape, 4))  # RGBA
                mask_colored[..., 0] = 1.0  # Red channel
                mask_colored[..., 3] = (mask_proj > 0).astype(float) * mask_alpha  # Alpha
                ax.imshow(mask_colored)
            
            # Get nuclei count from mask
            n_nuclei = len(np.unique(mask_proj)) - 1  # Exclude background
            
            title = f'Organoid {idx}'
            if mask_overlay:
                title += f' ({n_nuclei} nuclei)'
            
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(len(projections), len(axes)):
            axes[i].axis('off')
        
        # Set overall title
        try:
            if CONFIG_AVAILABLE:
                from karyosight.config import CHANNELS
                channel_names = CHANNELS
                channel_name = channel_names[channel] if channel < len(channel_names) else f'Channel_{channel}'
            else:
                channel_name = f'Channel_{channel}'
        except:
            channel_name = f'Channel_{channel}'
        
        title = f"{condition} - {channel_name} (Z-Optimized)"
        if mask_overlay:
            title += " + Segmentation Masks"
        if sample_count is not None:
            title += f" - {len(projections)} sampled"
        else:
            title += f" - All {len(projections)} organoids"
        
        fig.suptitle(title, fontsize=16, y=0.98)
        plt.tight_layout()
        
        # Save if requested
        saved_path = None
        if save_svg or save_png:
            # Create filename
            filename = f"{condition}_{channel_name}"
            if mask_overlay:
                filename += "_with_masks"
            if sample_count is not None:
                filename += f"_n{len(projections)}"
            filename += f"_{auto_scale_method}"
            
            if save_svg:
                svg_path = self.vis_dir / f"{filename}.svg"
                plt.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300)
                print(f"💾 Saved SVG: {svg_path}")
                saved_path = svg_path
            
            if save_png:
                png_path = self.vis_dir / f"{filename}.png"
                plt.savefig(png_path, format='png', dpi=150, bbox_inches='tight')
                print(f"💾 Saved PNG: {png_path}")
                if not save_svg:  # Only set as saved_path if SVG wasn't saved
                    saved_path = png_path
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return saved_path
    
    def create_all_conditions_grid(self,
                                  channel: int = 0,
                                  sample_count: Optional[int] = 12,
                                  mask_overlay: bool = True,
                                  mask_alpha: float = 0.3,
                                  auto_scale_method: str = 'individual',
                                  save_svg: bool = True,
                                  save_png: bool = False,
                                  show_plot: bool = True) -> List[Path]:
        """
        Create grid visualizations for all segmented conditions.
        
        Args:
            channel: Channel to visualize
            sample_count: Number of organoids to sample per condition
            mask_overlay: Whether to overlay segmentation masks
            mask_alpha: Transparency of mask overlay
            auto_scale_method: Scaling method ('individual' or 'global')
            save_svg: Whether to save as SVG
            save_png: Whether to save as PNG
            show_plot: Whether to display plots
            
        Returns:
            List of paths to saved files
        """
        saved_paths = []
        
        # Find all segmented conditions
        segmented_zarrs = list(self.segmented_dir.rglob("*_segmented.zarr"))
        conditions = [zarr.parent.name for zarr in segmented_zarrs]
        
        print(f"🎨 Creating visualizations for {len(conditions)} conditions...")
        
        for condition in sorted(conditions):
            print(f"\n📊 Visualizing {condition}...")
            saved_path = self.create_segmented_organoid_grid(
                condition=condition,
                channel=channel,
                sample_count=sample_count,
                mask_overlay=mask_overlay,
                mask_alpha=mask_alpha,
                auto_scale_method=auto_scale_method,
                save_svg=save_svg,
                save_png=save_png,
                show_plot=show_plot
            )
            if saved_path:
                saved_paths.append(saved_path)
        
        print(f"\n✅ Visualization complete! Saved {len(saved_paths)} files to:")
        print(f"   → {self.vis_dir}")
        
        return saved_paths
    
    def create_channel_comparison(self,
                                 condition: str,
                                 organoid_idx: int = 0,
                                 channels: Optional[List[int]] = None,
                                 mask_overlay: bool = True,
                                 mask_alpha: float = 0.3,
                                 auto_scale_method: str = 'individual',
                                 save_svg: bool = True,
                                 save_png: bool = False,
                                 show_plot: bool = True) -> Optional[Path]:
        """
        Create a comparison showing the same organoid across different channels.
        
        Args:
            condition: Condition name
            organoid_idx: Index of organoid to show
            channels: List of channels to show (default: [0, 1, 2])
            mask_overlay: Whether to overlay segmentation masks
            mask_alpha: Transparency of mask overlay
            auto_scale_method: Scaling method
            save_svg: Whether to save as SVG
            save_png: Whether to save as PNG
            show_plot: Whether to display the plot
            
        Returns:
            Path to saved file if save_svg/save_png=True, else None
        """
        if channels is None:
            channels = [0, 1, 2]  # Default to first 3 channels
        
        # Find the segmented zarr file
        zarr_path = self.segmented_dir / condition / f"{condition}_segmented.zarr"
        
        if not zarr_path.exists():
            print(f"Warning: No segmented zarr found for condition {condition}")
            return None
        
        try:
            bundle = zarr.open_group(str(zarr_path), mode='r')
            organoid_key = f"organoid_{organoid_idx:04d}"
            
            if organoid_key not in bundle:
                available_organoids = [k for k in bundle.keys() if k.startswith('organoid_')]
                print(f"Warning: Organoid {organoid_idx} not found in {condition}")
                print(f"Available organoids: {len(available_organoids)}")
                return None
            
            organoid_group = bundle[organoid_key]
            
            # Load raw data and masks
            raw_data = organoid_group['data'][:]
            if hasattr(raw_data, 'compute'):
                raw_data = raw_data.compute()
            
            masks = organoid_group['masks'][:]
            if hasattr(masks, 'compute'):
                masks = masks.compute()
            
            # Get metadata
            metadata = dict(organoid_group.attrs)
            n_nuclei = metadata.get('n_nuclei', len(np.unique(masks)) - 1)
            compression = metadata.get('compression_ratio_percent', 0)
            
        except Exception as e:
            print(f"Error loading data for {condition}, organoid {organoid_idx}: {e}")
            return None
        
        # Create max projections for each channel
        projections = []
        for channel in channels:
            if raw_data.ndim == 4 and channel < raw_data.shape[0]:  # [C, Z, Y, X]
                proj = np.max(raw_data[channel], axis=0)
                projections.append(proj)
            elif raw_data.ndim == 3 and channel == 0:  # [Z, Y, X] - single channel
                proj = np.max(raw_data, axis=0)
                projections.append(proj)
            else:
                print(f"Warning: Channel {channel} not available")
                channels.remove(channel)
        
        if not projections:
            print("No valid channels found")
            return None
        
        # Create mask projection
        mask_proj = np.max(masks, axis=0)
        
        # Create figure
        n_channels = len(projections)
        fig, axes = plt.subplots(1, n_channels, figsize=(4 * n_channels, 4))
        if n_channels == 1:
            axes = [axes]
        
        # Plot each channel
        for i, (proj, channel) in enumerate(zip(projections, channels)):
            ax = axes[i]
            
            # Compute scaling
            if auto_scale_method == 'individual':
                vmin = np.percentile(proj[proj > 0], 1) if np.any(proj > 0) else 0
                vmax = np.percentile(proj, 99.5)
            else:  # global across all channels
                all_nonzero = np.concatenate([p[p > 0] for p in projections if np.any(p > 0)])
                if len(all_nonzero) > 0:
                    vmin = np.percentile(all_nonzero, 1)
                    vmax = np.percentile(np.concatenate([p.flatten() for p in projections]), 99.5)
                else:
                    vmin = 0
                    vmax = max(p.max() for p in projections)
            
            # Show raw data
            ax.imshow(proj, cmap='gray', vmin=vmin, vmax=vmax)
            
            # Overlay masks if requested
            if mask_overlay and np.any(mask_proj > 0):
                mask_colored = np.zeros((*mask_proj.shape, 4))  # RGBA
                mask_colored[..., 0] = 1.0  # Red
                mask_colored[..., 3] = (mask_proj > 0).astype(float) * mask_alpha
                ax.imshow(mask_colored)
            
            # Get channel name
            try:
                if CONFIG_AVAILABLE:
                    from karyosight.config import CHANNELS
                    channel_names = CHANNELS
                    channel_name = channel_names[channel] if channel < len(channel_names) else f'Channel_{channel}'
                else:
                    channel_name = f'Channel_{channel}'
            except:
                channel_name = f'Channel_{channel}'
            
            ax.set_title(channel_name, fontsize=12)
            ax.axis('off')
        
        # Set overall title
        title = f"{condition} - Organoid {organoid_idx}"
        if mask_overlay:
            title += f" ({n_nuclei} nuclei, {compression:.1f}% compression)"
        
        fig.suptitle(title, fontsize=14, y=0.95)
        plt.tight_layout()
        
        # Save if requested
        saved_path = None
        if save_svg or save_png:
            filename = f"{condition}_organoid{organoid_idx}_channels"
            if mask_overlay:
                filename += "_with_masks"
            
            if save_svg:
                svg_path = self.vis_dir / f"{filename}.svg"
                plt.savefig(svg_path, format='svg', bbox_inches='tight', dpi=300)
                print(f"💾 Saved SVG: {svg_path}")
                saved_path = svg_path
            
            if save_png:
                png_path = self.vis_dir / f"{filename}.png"
                plt.savefig(png_path, format='png', dpi=150, bbox_inches='tight')
                print(f"💾 Saved PNG: {png_path}")
                if not save_svg:
                    saved_path = png_path
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return saved_path

if __name__ == "__main__":
    # Test installation when run directly
    test_cellpose_sam_installation() 