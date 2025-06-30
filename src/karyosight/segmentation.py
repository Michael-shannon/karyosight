import zarr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import datetime
from typing import Optional, List, Tuple, Dict, Any
import json

try:
    from cellpose import models, io
    from cellpose.core import use_gpu
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
    print("⚠️  Cellpose not installed. Run: pip install cellpose")

from karyosight.config import (
    CROPPED_DIR, CONDITION_PREF, PIXEL_SIZE_X, PIXEL_SIZE_Y, PIXEL_SIZE_Z,
    OPTIMAL_DIAMETER, OPTIMAL_FLOW_THRESHOLD, OPTIMAL_CELLPROB_THRESHOLD,
    USE_3D_SEGMENTATION, SEGMENTATION_CHANNEL, OPTIMAL_BATCH_SIZE
)


class NucleiSegmenter:
    """
    Handles 3D nuclei segmentation using Cellpose on zarr files from the cropped directory.
    
    Saves segmentations in a structure parallel to the cropped data:
    - segmented_dir/condition/condition_segmented.zarr
    
    Each segmentation zarr contains:
    - Individual organoid groups (organoid_0000, organoid_0001, etc.)
    - Each group has 'masks' dataset with integer labels for nuclei
    - Metadata including segmentation parameters and cell counts
    """
    
    def __init__(self,
                 cropped_dir: Path = CROPPED_DIR,
                 segmented_dir: Path = None,
                 model_type: str = 'nuclei',
                 use_gpu: bool = True,
                 diameter: Optional[float] = OPTIMAL_DIAMETER,
                 flow_threshold: float = OPTIMAL_FLOW_THRESHOLD,
                 cellprob_threshold: float = OPTIMAL_CELLPROB_THRESHOLD,
                 do_3D: bool = USE_3D_SEGMENTATION,
                 anisotropy: Optional[float] = None):
        """
        Initialize the nuclei segmenter.
        
        Parameters
        ----------
        cropped_dir : Path
            Directory containing cropped zarr files
        segmented_dir : Path, optional
            Directory to save segmentations. If None, uses cropped_dir/segmented
        model_type : str
            Cellpose model type ('nuclei', 'cyto', 'cyto2', etc.)
        use_gpu : bool
            Whether to use GPU acceleration
        diameter : float, optional
            Average nuclei diameter in pixels. If None, cellpose will estimate
        flow_threshold : float
            Flow error threshold for mask generation
        cellprob_threshold : float
            Cell probability threshold
        do_3D : bool
            Whether to run 3D segmentation
        anisotropy : float, optional
            Z vs XY pixel size ratio for 3D segmentation. If None and do_3D=True,
            automatically calculated from config pixel sizes (Z_spacing/XY_pixel_size)
        """
        if not CELLPOSE_AVAILABLE:
            raise ImportError("Cellpose is required but not installed. Run: pip install cellpose")
        
        self.cropped_dir = Path(cropped_dir)
        self.segmented_dir = Path(segmented_dir) if segmented_dir else self.cropped_dir.parent / "segmented"
        self.segmented_dir.mkdir(parents=True, exist_ok=True)
        
        # Cellpose parameters
        self.model_type = model_type
        from cellpose.core import use_gpu as cellpose_use_gpu
        self.use_gpu = use_gpu and cellpose_use_gpu()
        self.diameter = diameter
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.do_3D = do_3D
        
        # Calculate anisotropy automatically if not provided
        if anisotropy is None and self.do_3D:
            # Anisotropy = Z_spacing / XY_pixel_size
            self.anisotropy = PIXEL_SIZE_Z / PIXEL_SIZE_X
            print(f"   → Auto-calculated anisotropy: {self.anisotropy:.3f} (Z={PIXEL_SIZE_Z}µm / XY={PIXEL_SIZE_X:.3f}µm)")
        else:
            self.anisotropy = anisotropy
        
        # Initialize cellpose model
        self.model = models.Cellpose(gpu=self.use_gpu, model_type=self.model_type)
        
        print(f"🧠 Nuclei Segmenter initialized")
        print(f"   → Model: {self.model_type}")
        print(f"   → GPU: {self.use_gpu}")
        print(f"   → 3D: {self.do_3D}")
        if self.do_3D and self.anisotropy:
            print(f"   → Anisotropy: {self.anisotropy:.3f}")
        if self.diameter:
            print(f"   → Diameter: {self.diameter} pixels")
        else:
            print(f"   → Diameter: Auto-estimate")
    
    def find_bundled_zarrs(self) -> List[Path]:
        """
        Find all bundled zarr files in the cropped directory.
        
        Returns
        -------
        List[Path]
            List of paths to bundled zarr files
        """
        zarr_files = []
        for condition_dir in self.cropped_dir.iterdir():
            if condition_dir.is_dir() and condition_dir.name.startswith(CONDITION_PREF):
                bundled_zarr = condition_dir / f"{condition_dir.name}_bundled.zarr"
                if bundled_zarr.exists():
                    zarr_files.append(bundled_zarr)
        
        print(f"📂 Found {len(zarr_files)} bundled zarr files")
        return sorted(zarr_files)
    
    def segment_zarr_file(self,
                         zarr_path: Path,
                         channel: int = 0,
                         organoid_indices: Optional[List[int]] = None,
                         overwrite: bool = False,
                         progress_bar: bool = True) -> Path:
        """
        Segment nuclei in a bundled zarr file.
        
        Parameters
        ----------
        zarr_path : Path
            Path to the bundled zarr file
        channel : int
            Channel to use for segmentation (0-based, typically 0 for DAPI/nuclei)
        organoid_indices : List[int], optional
            Specific organoid indices to segment. If None, segments all
        overwrite : bool
            Whether to overwrite existing segmentations
        progress_bar : bool
            Whether to show progress bar
            
        Returns
        -------
        Path
            Path to the output segmented zarr file
        """
        # Determine condition and setup output path
        condition_name = zarr_path.parent.name
        output_zarr = self.segmented_dir / condition_name / f"{condition_name}_segmented.zarr"
        output_zarr.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🧬 Segmenting {condition_name}")
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
                'model_type': self.model_type,
                'diameter': self.diameter,
                'flow_threshold': self.flow_threshold,
                'cellprob_threshold': self.cellprob_threshold,
                'do_3D': self.do_3D,
                'anisotropy': self.anisotropy,
                'channel': channel,
                'timestamp': str(datetime.datetime.now())
            }
        
        # Process each organoid
        organoid_iterator = tqdm(organoid_keys, desc=f"Segmenting {condition_name}", 
                                disable=not progress_bar) if progress_bar else organoid_keys
        
        total_nuclei = 0
        for organoid_key in organoid_iterator:
            # Skip if already exists and not overwriting
            if organoid_key in output_bundle and not overwrite:
                if progress_bar:
                    organoid_iterator.set_postfix_str("(skipped)")
                continue
            
            try:
                # Load organoid data
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
                
                # Run cellpose segmentation
                masks, flows, styles, probs = self.model.eval(
                    img,
                    diameter=self.diameter,
                    flow_threshold=self.flow_threshold,
                    cellprob_threshold=self.cellprob_threshold,
                    do_3D=self.do_3D,
                    anisotropy=self.anisotropy,
                    channels=[0, 0]  # Use single channel
                )
                
                # Count nuclei
                n_nuclei = len(np.unique(masks)) - 1  # Subtract background
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
                    # Note: Removed compressor due to zarr v3 API changes
                )
                
                # Save metadata
                organoid_group.attrs.update({
                    'n_nuclei': int(n_nuclei),
                    'original_shape': list(organoid_data.shape),
                    'masks_shape': list(masks.shape),
                    'channel_used': channel
                })
                
                # Optionally save flows and probs (comment out to save space)
                # organoid_group.create_dataset('flows', data=flows[0].astype(np.float32))
                # organoid_group.create_dataset('cellprob', data=probs.astype(np.float32))
                
                if progress_bar:
                    organoid_iterator.set_postfix_str(f"{n_nuclei} nuclei")
                
            except Exception as e:
                print(f"   ❌ Error processing {organoid_key}: {e}")
                continue
        
        # Update bundle metadata
        output_bundle.attrs['total_nuclei'] = int(total_nuclei)
        output_bundle.attrs['n_organoids'] = len(organoid_keys)
        output_bundle.attrs['last_updated'] = str(datetime.datetime.now())
        
        print(f"   ✅ Segmentation complete: {total_nuclei} nuclei across {len(organoid_keys)} organoids")
        return output_zarr
    
    def segment_all_zarrs(self,
                         channel: int = 0,
                         overwrite: bool = False,
                         progress_bar: bool = True) -> List[Path]:
        """
        Segment all bundled zarr files in the cropped directory.
        
        Parameters
        ----------
        channel : int
            Channel to use for segmentation
        overwrite : bool
            Whether to overwrite existing segmentations
        progress_bar : bool
            Whether to show progress bars
            
        Returns
        -------
        List[Path]
            List of paths to output segmented zarr files
        """
        zarr_files = self.find_bundled_zarrs()
        output_paths = []
        
        for zarr_path in zarr_files:
            try:
                output_path = self.segment_zarr_file(
                    zarr_path, 
                    channel=channel, 
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
    
    def get_segmentation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all segmentations.
        
        Returns
        -------
        Dict[str, Any]
            Summary statistics for all conditions
        """
        summary = {}
        
        for condition_dir in self.segmented_dir.iterdir():
            if condition_dir.is_dir() and condition_dir.name.startswith(CONDITION_PREF):
                segmented_zarr = condition_dir / f"{condition_dir.name}_segmented.zarr"
                if segmented_zarr.exists():
                    try:
                        bundle = zarr.open_group(str(segmented_zarr), mode='r')
                        
                        condition_info = {
                            'total_nuclei': bundle.attrs.get('total_nuclei', 0),
                            'n_organoids': bundle.attrs.get('n_organoids', 0),
                            'last_updated': bundle.attrs.get('last_updated', 'Unknown'),
                            'segmentation_params': bundle.attrs.get('segmentation_params', {}),
                            'organoids': {}
                        }
                        
                        # Get per-organoid statistics
                        for key in bundle.keys():
                            if key.startswith('organoid_'):
                                organoid_group = bundle[key]
                                condition_info['organoids'][key] = {
                                    'n_nuclei': organoid_group.attrs.get('n_nuclei', 0),
                                    'shape': organoid_group.attrs.get('masks_shape', [])
                                }
                        
                        summary[condition_dir.name] = condition_info
                        
                    except Exception as e:
                        print(f"⚠️  Error reading {segmented_zarr}: {e}")
        
        return summary
    
    def visualize_segmentation(self,
                              zarr_path: Path,
                              organoid_idx: int = 0,
                              channel: int = 0,
                              z_slices: Optional[List[int]] = None,
                              figsize: Tuple[int, int] = (20, 15),
                              alpha: float = 0.5) -> plt.Figure:
        """
        Visualize segmentation results for a single organoid showing all z-slices with overlay only.
        
        Parameters
        ----------
        zarr_path : Path
            Path to the segmented zarr file
        organoid_idx : int
            Index of organoid to visualize
        channel : int
            Channel to show as background
        z_slices : List[int], optional
            Specific z-slices to show. If None, shows ALL z-slices
        figsize : Tuple[int, int]
            Figure size
        alpha : float
            Transparency of mask overlay
            
        Returns
        -------
        plt.Figure
            The matplotlib figure
        """
        # Load data
        bundle = zarr.open_group(str(zarr_path), mode='r')
        organoid_key = f"organoid_{organoid_idx:04d}"
        
        if organoid_key not in bundle:
            raise ValueError(f"Organoid {organoid_key} not found in {zarr_path}")
        
        organoid_group = bundle[organoid_key]
        masks = organoid_group['masks'][:]
        
        # Load original image data for background
        condition_name = zarr_path.parent.name
        cropped_zarr = self.cropped_dir / condition_name / f"{condition_name}_bundled.zarr"
        cropped_bundle = zarr.open_group(str(cropped_zarr), mode='r')
        
        if organoid_key in cropped_bundle:
            img_data = cropped_bundle[organoid_key]['data'][channel]
        else:
            img_data = cropped_bundle['data'][organoid_idx, channel]
        
        # Determine z-slices to show - show ALL z-slices by default
        nz = masks.shape[0]
        if z_slices is None:
            z_slices = list(range(nz))  # Show ALL z-slices
        
        # Create figure with appropriate grid layout
        ncols = min(8, len(z_slices))  # Max 8 columns
        nrows = int(np.ceil(len(z_slices) / ncols))
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows == 1 and ncols == 1:
            axes = [axes]
        elif nrows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        n_nuclei = organoid_group.attrs.get('n_nuclei', 0)
        fig.suptitle(f'{condition_name} - {organoid_key} - {n_nuclei} nuclei - All Z-slices', fontsize=16)
        
        for i, z in enumerate(z_slices):
            ax = axes[i]
            
            # Show overlay only (background + segmentation)
            ax.imshow(img_data[z], cmap='gray')
            if masks[z].max() > 0:
                # Create colored mask overlay
                mask_colored = plt.cm.jet(masks[z] / masks[z].max())
                mask_colored[masks[z] == 0] = [0, 0, 0, 0]  # Transparent background
                ax.imshow(mask_colored, alpha=alpha)
            
            ax.set_title(f'Z={z}', fontsize=10)
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(len(z_slices), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig


def test_cellpose_installation() -> bool:
    """
    Test if cellpose is properly installed and can detect GPU.
    
    Returns
    -------
    bool
        True if cellpose is working correctly
    """
    if not CELLPOSE_AVAILABLE:
        print("❌ Cellpose not installed")
        return False
    
    print("✅ Cellpose imported successfully")
    
    # Test GPU availability
    gpu_available = use_gpu()
    print(f"🖥️  GPU available: {gpu_available}")
    
    try:
        # Test model loading
        model = models.Cellpose(gpu=gpu_available, model_type='nuclei')
        print("✅ Cellpose model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading cellpose model: {e}")
        return False


# Convenience function for notebook use
def segment_example_zarr(zarr_path: Optional[Path] = None,
                        organoid_idx: int = 0,
                        channel: int = 0,
                        diameter: Optional[float] = None,
                        visualize: bool = True) -> Tuple[Path, Optional[plt.Figure]]:
    """
    Segment a single example zarr file for testing parameters.
    
    Parameters
    ----------
    zarr_path : Path, optional
        Path to zarr file. If None, uses first available
    organoid_idx : int
        Index of organoid to segment
    channel : int
        Channel to use for segmentation
    diameter : float, optional
        Diameter parameter for cellpose
    visualize : bool
        Whether to create visualization
        
    Returns
    -------
    Tuple[Path, Optional[plt.Figure]]
        Path to segmented zarr and optional figure
    """
    segmenter = NucleiSegmenter(diameter=diameter)
    
    if zarr_path is None:
        zarr_files = segmenter.find_bundled_zarrs()
        if not zarr_files:
            raise ValueError("No bundled zarr files found")
        zarr_path = zarr_files[0]
        print(f"Using example zarr: {zarr_path}")
    
    # Segment just one organoid for testing
    output_path = segmenter.segment_zarr_file(
        zarr_path, 
        channel=channel, 
        organoid_indices=[organoid_idx],
        overwrite=True
    )
    
    # Visualize results
    fig = None
    if visualize:
        fig = segmenter.visualize_segmentation(
            output_path, 
            organoid_idx=organoid_idx, 
            channel=channel
        )
        plt.show()
    
    return output_path, fig