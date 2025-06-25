# import matplotlib.patches as patches


# import re
# import matplotlib.pyplot as plt
# from karyosight.stitcher import Stitcher as st

# def plot_subgrid_layout(gm, scale, ax=None):
#     """
#     Draws all subgrid footprints (light grey) and full
#     grid tile‐centers so you can check that everything is covered.
#     """
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(6,6))

#     # 1) plot ALL tile centers
#     trans_all = st.compute_translations(gm, scale)
#     xs = [t['x'] for t in trans_all]
#     ys = [t['y'] for t in trans_all]
#     ax.scatter(xs, ys, marker='o', s=10, color='lightgrey', alpha=0.5)

#     # 2) overlay each subgrid rect
#     ox, oy = gm["overlap"]["x"], gm["overlap"]["y"]
#     ts = gm["tile_shape"]
#     for sg in gm["subgrids"]:
#         x0_i, x1_i = sg["x_range"]
#         y0_i, y1_i = sg["y_range"]
#         px0 = x0_i*(1-ox)*ts["x"]*scale["x"]
#         py0 = y0_i*(1-oy)*ts["y"]*scale["y"]
#         w   = (x1_i-x0_i)*(1-ox)*ts["x"]*scale["x"]
#         h   = (y1_i-y0_i)*(1-oy)*ts["y"]*scale["y"]
#         rect = patches.Rectangle((px0,py0), w, h,
#                                  linewidth=1, edgecolor='C0', facecolor='none', alpha=0.7)
#         ax.add_patch(rect)

#     ax.set_aspect('equal')
#     ax.set_title(f"Subgrid layout ({len(gm['subgrids'])} windows)")
#     return ax



# def plot_full_and_subgrid(meta, scale, subgrid, ax=None):
#     """
#     meta:    the full metadata dict returned by extract_metadata()
#     scale:   same scale dict you pass to compute_translations()
#     subgrid: one element of meta['subgrids']
#     """
#     # 1) compute translations for *all* tiles
#     all_trans = st.compute_translations(meta, scale)
#     # 2) compute translations for *this* subgrid
#     sub_meta = { **meta, "tiles": subgrid["tiles"] }
#     sub_trans = st.compute_translations(sub_meta, scale)

#     if ax is None:
#         fig, ax = plt.subplots(figsize=(6,6))

#     # 3) plot all in light grey
#     xs = [t['x'] for t in all_trans]
#     ys = [t['y'] for t in all_trans]
#     ax.scatter(xs, ys, marker='s', s=50, color='lightgrey', alpha=0.6, label='all tiles')

#     # 4) overlay this subgrid in color
#     xs2 = [t['x'] for t in sub_trans]
#     ys2 = [t['y'] for t in sub_trans]
#     ax.scatter(xs2, ys2, marker='s', s=80, color='C1', label=f"subgrid {subgrid['id']}")

#     # 5) annotate each subgrid tile with its "frame number"
#     for tile, t in zip(subgrid["tiles"], sub_trans):
#         stem = Path(tile["filename"] or "").stem
#         m = re.search(r'_(\d+)$', stem)
#         num = m.group(1) if m else stem
#         ax.text(t['x'], t['y'], num, ha='center', va='center', fontsize=6, color='black')

#     # 6) draw a rectangle around the subgrid footprint
#     #    using its x_range,y_range (tile-indices) → corner positions:
#     ox, oy = meta['overlap']['x'], meta['overlap']['y']
#     ts = meta['tile_shape']
#     w = (subgrid['x_range'][1] - subgrid['x_range'][0]) * (1-ox) * ts['x'] * scale['x']
#     h = (subgrid['y_range'][1] - subgrid['y_range'][0]) * (1-oy) * ts['y'] * scale['y']
#     x0 = subgrid['x_range'][0] * (1-ox) * ts['x'] * scale['x']
#     y0 = subgrid['y_range'][0] * (1-oy) * ts['y'] * scale['y']
#     # rect = plt.Rectangle((x0,y0), w, h, 
#     #                      edgecolor='C1', facecolor='none', lw=2)
#     # ax.add_patch(rect)

#     ax.set_aspect('equal')
#     ax.legend(loc='upper right')
#     ax.set_title(f"Full grid (grey) + subgrid {subgrid['id']}")
#     plt.tight_layout()
#     return ax
# karyosight/visualization.py

import re
import zarr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, Union, List, Dict, Tuple
import warnings
import tifffile
from karyosight.config import CHANNELS

def plot_subgrid_layout(stitcher, gm, scale, ax=None):
    """
    Draws all subgrid footprints (light grey) and full grid tile‐centers.
    
    Params:
      - stitcher: your Stitcher instance
      - gm:       the mini-meta returned by build_dynamic_group_meta
      - scale:    dict with x/y/z pixel sizes
      - ax:       optional matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))

    # 1) plot ALL tile centers
    trans_all = stitcher.compute_translations(gm, scale)
    xs = [t['x'] for t in trans_all]
    ys = [t['y'] for t in trans_all]
    ax.scatter(xs, ys, marker='o', s=10, color='lightgrey', alpha=0.5)

    # 2) overlay each subgrid rectangle
    ox, oy = gm["overlap"]["x"], gm["overlap"]["y"]
    ts    = gm["tile_shape"]
    for sg in gm["subgrids"]:
        x0_i, x1_i = sg["x_range"]
        y0_i, y1_i = sg["y_range"]
        px0 = x0_i*(1-ox)*ts["x"]*scale["x"]
        py0 = y0_i*(1-oy)*ts["y"]*scale["y"]
        w   = (x1_i-x0_i)*(1-ox)*ts["x"]*scale["x"]
        h   = (y1_i-y0_i)*(1-oy)*ts["y"]*scale["y"]
        rect = patches.Rectangle((px0,py0), w, h,
                                 linewidth=1, edgecolor='C0',
                                 facecolor='none', alpha=0.7)
        ax.add_patch(rect)

    ax.set_aspect('equal')
    ax.set_title(f"Subgrid layout ({len(gm['subgrids'])} windows)")
    return ax


def plot_full_and_subgrid(stitcher, meta, scale, subgrid, ax=None):
    """
    Overlays one subgrid on the full-grid tile centers.

    Params:
      - stitcher: your Stitcher instance
      - meta:     the full meta dict from extract_metadata()
      - scale:    dict with x/y/z pixel sizes
      - subgrid:  one element of meta['subgrids']
      - ax:       optional matplotlib Axes
    """
    # 1) full‐grid translations
    all_trans = stitcher.compute_translations(meta, scale)
    # 2) subgrid translations
    sub_meta  = { **meta, "tiles": subgrid["tiles"] }
    sub_trans = stitcher.compute_translations(sub_meta, scale)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))

    # 3) plot all in light grey
    xs = [t['x'] for t in all_trans]
    ys = [t['y'] for t in all_trans]
    ax.scatter(xs, ys, marker='s', s=50,
               color='lightgrey', alpha=0.6,
               label='all tiles')

    # 4) overlay this subgrid in bold
    xs2 = [t['x'] for t in sub_trans]
    ys2 = [t['y'] for t in sub_trans]
    ax.scatter(xs2, ys2, marker='s', s=80,
               color='C1', label=f"subgrid {subgrid['id']}")

    # 5) annotate each subgrid tile with its frame number
    for tile, t in zip(subgrid["tiles"], sub_trans):
        stem = Path(tile.get("filename","")).stem
        m = re.search(r'_(\d+)$', stem)
        num = m.group(1) if m else stem
        ax.text(t['x'], t['y'], num,
                ha='center', va='center',
                fontsize=6, color='black')

    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title(f"Full grid + subgrid {subgrid['id']}")
    plt.tight_layout()
    return ax


class OrganoidVisualizer:
    """
    A class for visualizing cropped organoids from bundled zarr files.
    Creates grid layouts of max projections for quality control and analysis.
    """
    
    def __init__(self, cropped_dir: Union[str, Path]):
        """
        Initialize the visualizer with the cropped data directory.
        
        Parameters
        ----------
        cropped_dir : str or Path
            Path to the directory containing cropped organoid zarr files
        """
        self.cropped_dir = Path(cropped_dir)
        self.vis_dir = self.cropped_dir / "visualization"
        self.vis_dir.mkdir(exist_ok=True)
        
    def count_organoids_per_condition(self) -> Dict[str, int]:
        """
        Count the number of organoids in each condition's bundled zarr file.
        
        Returns
        -------
        dict
            Dictionary mapping condition names to organoid counts
        """
        counts = {}
        
        # Find all bundled zarr files
        bundled_zarrs = list(self.cropped_dir.rglob("*_bundled.zarr"))
        
        for zarr_path in bundled_zarrs:
            # Extract condition name from path
            condition = zarr_path.parent.name
            
            try:
                # Open zarr and get organoid count
                bundle = zarr.open_group(str(zarr_path), mode='r')
                
                # Check storage type
                storage_type = bundle.attrs.get('storage_type', 'uniform_padding')
                
                if storage_type == 'variable_size_groups':
                    # NEW OPTIMIZED FORMAT: Count organoid groups
                    organoid_keys = [k for k in bundle.keys() if k.startswith('organoid_')]
                    n_organoids = len(organoid_keys)
                else:
                    # LEGACY FORMAT: Count from data array
                    n_organoids = bundle['data'].shape[0]
                
                counts[condition] = n_organoids
                print(f"{condition}: {n_organoids} organoids")
                
            except Exception as e:
                print(f"Warning: Could not read {zarr_path}: {e}")
                counts[condition] = 0
                
        return counts
    
    def _load_organoid_data(self, 
                           zarr_path: Path, 
                           channel: int = 0, 
                           level: int = 1,
                           sample_count: Optional[int] = None) -> Tuple[np.ndarray, List[int]]:
        """
        Load organoid data from a bundled zarr file.
        
        Parameters
        ----------
        zarr_path : Path
            Path to the bundled zarr file
        channel : int
            Channel index to load (0-5 for CH1-CH6)
        level : int
            Zarr level to use (default 1 for lower resolution)
        sample_count : int, optional
            Number of organoids to sample. If None, load all.
            
        Returns
        -------
        np.ndarray
            Array of max projections, shape (n_samples, height, width)
        list
            List of organoid indices that were loaded
        """
        bundle = zarr.open_group(str(zarr_path), mode='r')
        
        # Check storage type to handle different formats
        storage_type = bundle.attrs.get('storage_type', 'uniform_padding')
        
        if storage_type == 'variable_size_groups':
            # NEW OPTIMIZED FORMAT: Variable-size groups
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
            for key in selected_keys:
                organoid_group = bundle[key]
                organoid_data = organoid_group['data'][channel]  # Shape: (z, y, x)
                max_proj = np.max(organoid_data, axis=0)  # Max projection along z
                max_projections.append(max_proj)
            
            # For variable-size format, return as list instead of numpy array
            return max_projections, indices
                
        else:
            # LEGACY FORMAT: Uniform padding in single array
            # Check if the specified level exists
            if str(level) in bundle:
                data = bundle[str(level)]
            else:
                # Fall back to main data array
                data = bundle['data']
                
            total_organoids = data.shape[0]
            
            # Determine which organoids to load
            if sample_count is None or sample_count >= total_organoids:
                indices = list(range(total_organoids))
            else:
                # Randomly sample organoids
                indices = np.random.choice(total_organoids, sample_count, replace=False).tolist()
                indices.sort()
            
            # Load data for selected organoids
            max_projections = []
            for idx in indices:
                organoid_data = data[idx, channel]  # Shape: (z, y, x)
                max_proj = np.max(organoid_data, axis=0)  # Max projection along z
                max_projections.append(max_proj)
            
        return np.array(max_projections), indices
    
    def _calculate_grid_size(self, n_items: int) -> Tuple[int, int]:
        """Calculate optimal grid size for displaying n_items."""
        if n_items == 1:
            return 1, 1
        
        # Try to make a roughly square grid
        cols = int(np.ceil(np.sqrt(n_items)))
        rows = int(np.ceil(n_items / cols))
        return rows, cols
    
    def _compute_scaling(self, 
                        projections: np.ndarray, 
                        auto_scale_method: str = 'individual') -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute vmin and vmax for scaling.
        
        Parameters
        ----------
        projections : np.ndarray
            Array of max projections
        auto_scale_method : str
            'individual' for per-organoid scaling, 'global' for shared scaling
            
        Returns
        -------
        tuple
            (vmin_array, vmax_array) - either per-organoid or global values
        """
        if auto_scale_method == 'individual':
            # Per-organoid scaling
            vmin_vals = np.array([np.percentile(proj[proj > 0], 1) if np.any(proj > 0) else 0 
                                 for proj in projections])
            vmax_vals = np.array([np.percentile(proj, 99.5) for proj in projections])
        else:  # global
            # Global scaling across all organoids
            all_nonzero = projections[projections > 0]
            if len(all_nonzero) > 0:
                global_vmin = np.percentile(all_nonzero, 1)
                global_vmax = np.percentile(projections, 99.5)
            else:
                global_vmin = 0
                global_vmax = projections.max()
            
            vmin_vals = np.full(len(projections), global_vmin)
            vmax_vals = np.full(len(projections), global_vmax)
            
        return vmin_vals, vmax_vals
    
    def create_organoid_grid(self,
                           condition: str,
                           channel: int = 0,
                           level: int = 1,
                           sample_count: Optional[int] = None,
                           auto_scale_method: str = 'individual',
                           figsize_per_organoid: Tuple[float, float] = (2, 2),
                           save_png: bool = True,
                           show_plot: bool = True) -> Optional[Path]:
        """
        Create a grid visualization of organoid max projections for a condition.
        
        Parameters
        ----------
        condition : str
            Name of the condition to visualize
        channel : int
            Channel to visualize (0-5 for CH1-CH6)
        level : int
            Zarr pyramid level to use (default 1)
        sample_count : int, optional
            Number of organoids to sample. If None, show all.
        auto_scale_method : str
            'individual' for per-organoid scaling, 'global' for shared scaling
        figsize_per_organoid : tuple
            Size of each subplot in inches
        save_png : bool
            Whether to save the figure as PNG
        show_plot : bool
            Whether to display the plot
            
        Returns
        -------
        Path or None
            Path to saved PNG file if save_png=True, else None
        """
        # Find the bundled zarr file for this condition
        zarr_path = self.cropped_dir / condition / f"{condition}_bundled.zarr"
        
        if not zarr_path.exists():
            print(f"Warning: No bundled zarr found for condition {condition}")
            print(f"Expected path: {zarr_path}")
            return None
        
        # Load organoid data
        projections, indices = self._load_organoid_data(
            zarr_path, channel=channel, level=level, sample_count=sample_count
        )
        
        if len(projections) == 0:
            print(f"No organoids found for condition {condition}")
            return None
        
        # Calculate grid layout
        rows, cols = self._calculate_grid_size(len(projections))
        
        # Calculate figure size
        fig_width = cols * figsize_per_organoid[0]
        fig_height = rows * figsize_per_organoid[1]
        
        # Handle both list (variable-size) and numpy array (uniform-size) formats
        if isinstance(projections, list):
            # Variable-size projections - compute scaling for each individually
            vmin_vals = []
            vmax_vals = []
            for proj in projections:
                if auto_scale_method == 'individual':
                    vmin = np.percentile(proj[proj > 0], 1) if np.any(proj > 0) else 0
                    vmax = np.percentile(proj, 99.5)
                else:  # global - compute from all projections
                    all_nonzero = np.concatenate([p[p > 0] for p in projections if np.any(p > 0)])
                    if len(all_nonzero) > 0:
                        vmin = np.percentile(all_nonzero, 1)
                        vmax = np.percentile(np.concatenate([p.flatten() for p in projections]), 99.5)
                    else:
                        vmin = 0
                        vmax = max(p.max() for p in projections)
                vmin_vals.append(vmin)
                vmax_vals.append(vmax)
        else:
            # Uniform-size projections - use original method
            vmin_vals, vmax_vals = self._compute_scaling(projections, auto_scale_method)
        
        # Create the figure
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        if rows * cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if rows > 1 or cols > 1 else [axes]
        
        # Plot each organoid
        for i, (proj, idx) in enumerate(zip(projections, indices)):
            ax = axes[i]
            
            # Handle both list and array indexing for vmin/vmax
            if isinstance(vmin_vals, list):
                vmin, vmax = vmin_vals[i], vmax_vals[i]
            else:
                vmin, vmax = vmin_vals[i], vmax_vals[i]
            
            im = ax.imshow(proj, cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(f'Organoid {idx}', fontsize=8)
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(len(projections), len(axes)):
            axes[i].axis('off')
        
        # Set overall title
        channel_name = f"CH{channel + 1}"
        title = f"{condition} - {channel_name} (Level {level})"
        if sample_count is not None:
            title += f" - {len(projections)} sampled"
        else:
            title += f" - All {len(projections)} organoids"
        
        fig.suptitle(title, fontsize=14, y=0.98)
        plt.tight_layout()
        
        # Save if requested
        saved_path = None
        if save_png:
            filename = f"{condition}_{channel_name}_level{level}"
            if sample_count is not None:
                filename += f"_n{len(projections)}"
            filename += f"_{auto_scale_method}.png"
            
            saved_path = self.vis_dir / filename
            plt.savefig(saved_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization: {saved_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return saved_path
    
    def create_all_conditions_grid(self,
                                 channel: int = 0,
                                 level: int = 1,
                                 sample_count: Optional[int] = 12,
                                 auto_scale_method: str = 'individual',
                                 save_png: bool = True,
                                 show_plot: bool = True) -> List[Path]:
        """
        Create grid visualizations for all conditions.
        
        Parameters
        ----------
        channel : int
            Channel to visualize (0-5 for CH1-CH6)
        level : int
            Zarr pyramid level to use
        sample_count : int, optional
            Number of organoids to sample per condition
        auto_scale_method : str
            Scaling method ('individual' or 'global')
        save_png : bool
            Whether to save figures
        show_plot : bool
            Whether to display plots
            
        Returns
        -------
        list
            List of paths to saved PNG files
        """
        saved_paths = []
        
        # Find all conditions
        bundled_zarrs = list(self.cropped_dir.rglob("*_bundled.zarr"))
        conditions = [zarr.parent.name for zarr in bundled_zarrs]
        
        for condition in sorted(conditions):
            print(f"\nCreating visualization for {condition}...")
            saved_path = self.create_organoid_grid(
                condition=condition,
                channel=channel,
                level=level,
                sample_count=sample_count,
                auto_scale_method=auto_scale_method,
                save_png=save_png,
                show_plot=show_plot
            )
            if saved_path:
                saved_paths.append(saved_path)
        
        return saved_paths
    
    def create_channel_comparison(self,
                                condition: str,
                                organoid_idx: int = 0,
                                level: int = 1,
                                channels: Optional[List[int]] = None,
                                auto_scale_method: str = 'individual',
                                save_png: bool = True,
                                show_plot: bool = True) -> Optional[Path]:
        """
        Create a comparison plot showing the same organoid across different channels.
        
        Parameters
        ----------
        condition : str
            Condition name (e.g., 'Condition_DMSO')
        organoid_idx : int, optional
            Index of the organoid to show (default: 0)
        level : int, optional
            Resolution level to use (default: 1)
        channels : List[int], optional
            List of channels to show. If None, shows channels 0-5
        auto_scale_method : str, optional
            How to scale intensities: 'individual' or 'global'
        save_png : bool, optional
            Whether to save as PNG file
        show_plot : bool, optional
            Whether to display the plot
            
        Returns
        -------
        Optional[Path]
            Path to saved PNG file if save_png=True, otherwise None
        """
        if channels is None:
            channels = list(range(6))  # CH1-CH6 (0-5)
            
        # Find the bundled zarr file
        condition_dir = self.cropped_dir / condition
        zarr_files = list(condition_dir.glob("*bundled.zarr"))
        
        if not zarr_files:
            print(f"❌ No bundled zarr found in {condition_dir}")
            return None
            
        zarr_path = zarr_files[0]
        
        try:
            # Load organoid data directly from zarr for channel comparison
            bundle = zarr.open_group(str(zarr_path), mode='r')
            storage_type = bundle.attrs.get('storage_type', 'uniform_padding')
            
            if storage_type == 'variable_size_groups':
                # NEW OPTIMIZED FORMAT: Load specific organoid directly
                organoid_key = f"organoid_{organoid_idx:04d}"
                if organoid_key not in bundle:
                    available_organoids = len([k for k in bundle.keys() if k.startswith('organoid_')])
                    print(f"❌ Organoid index {organoid_idx} not found (only {available_organoids} organoids available)")
                    return None
                
                organoid_group = bundle[organoid_key]
                organoid_data = organoid_group['data'][:]  # Shape: (n_channels, z, y, x)
                
            else:
                # LEGACY FORMAT: Load from uniform array
                if 'data' not in bundle:
                    print(f"❌ No data found in {zarr_path}")
                    return None
                    
                data = bundle['data']
                if organoid_idx >= data.shape[0]:
                    print(f"❌ Organoid index {organoid_idx} not found (only {data.shape[0]} organoids available)")
                    return None
                
                organoid_data = data[organoid_idx]  # Shape: (n_channels, z, y, x)
            
            # Create projection for each channel
            projections = []
            valid_channels = []
            
            for ch in channels:
                if ch < organoid_data.shape[0]:
                    proj = np.max(organoid_data[ch], axis=0)  # Max projection
                    projections.append(proj)
                    valid_channels.append(ch)
                else:
                    print(f"⚠️ Channel {ch} not available (only {organoid_data.shape[0]} channels)")
            
            if not projections:
                print("❌ No valid channels found")
                return None
                
            projections = np.array(projections)
            
            # Calculate scaling
            vmin_vals, vmax_vals = self._compute_scaling(projections, auto_scale_method)
            
            # Create the plot
            n_channels = len(valid_channels)
            cols = min(3, n_channels)
            rows = (n_channels + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
            if n_channels == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.reshape(1, -1)
            
            # Plot each channel
            for i, (ch, proj) in enumerate(zip(valid_channels, projections)):
                row = i // cols
                col = i % cols
                ax = axes[row, col] if rows > 1 else axes[col]
                
                im = ax.imshow(proj, cmap='gray', vmin=vmin_vals[i], vmax=vmax_vals[i])
                ax.set_title(f'CH{ch+1}', fontsize=10, fontweight='bold')
                ax.axis('off')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Hide unused subplots
            for i in range(n_channels, rows * cols):
                row = i // cols
                col = i % cols
                if rows > 1:
                    axes[row, col].axis('off')
                else:
                    axes[col].axis('off')
            
            fig.suptitle(f'{condition} - Organoid {organoid_idx} - Channel Comparison', 
                        fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            # Save if requested
            saved_path = None
            if save_png:
                filename = f"{condition}_organoid{organoid_idx}_channels.png"
                saved_path = condition_dir / filename
                fig.savefig(saved_path, dpi=150, bbox_inches='tight')
                print(f"💾 Saved: {saved_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close(fig)
                
        except Exception as e:
            print(f"❌ Error creating channel comparison: {e}")
            return None
            
        return saved_path

    def view_organoid_z_slices(self,
                              condition: str,
                              organoid_idx: int = 0,
                              channel: int = 0,
                              level: int = 1,
                              z_slices: Optional[int] = None,
                              save_png: bool = True,
                              show_plot: bool = True) -> Optional[Path]:
        """
        Visualize specific z-slices of an organoid with flexible slice selection.
        
        Parameters
        ----------
        condition : str
            Condition name (e.g., 'Condition_DMSO')
        organoid_idx : int, optional
            Index of the organoid to show (default: 0)
        channel : int, optional
            Channel to visualize (0-5 for CH1-CH6, default: 0)
        level : int, optional
            Resolution level to use (default: 1)
        z_slices : int or None, optional
            Number of z-slices to show:
            - None: show all z-slices
            - 3: show top, middle, and bottom slices
            - 9 (or other number): show evenly spaced slices including top and bottom
        save_png : bool, optional
            Whether to save as PNG file
        show_plot : bool, optional
            Whether to display the plot
            
        Returns
        -------
        Optional[Path]
            Path to saved PNG file if save_png=True, otherwise None
        """
        # Find the bundled zarr file
        condition_dir = self.cropped_dir / condition
        zarr_files = list(condition_dir.glob("*bundled.zarr"))
        
        if not zarr_files:
            print(f"❌ No bundled zarr found in {condition_dir}")
            return None
            
        zarr_path = zarr_files[0]
        
        try:
            # Load the bundled zarr file - handle both old and new storage formats
            bundle = zarr.open_group(str(zarr_path), mode='r')
            
            # Check storage type
            storage_type = bundle.attrs.get('storage_type', 'uniform_padding')
            
            if storage_type == 'individual_zarrs':
                # COMPLEX OPTIMIZED FORMAT: Individual zarr files
                import json
                metadata = json.loads(bundle.attrs['metadata'])
                
                if organoid_idx >= len(metadata):
                    print(f"❌ Organoid index {organoid_idx} not found (only {len(metadata)} organoids available)")
                    return None
                
                # Load individual organoid zarr
                organoid_info = metadata[organoid_idx]
                organoid_zarr_name = organoid_info['zarr_file']
                organoid_zarr_path = zarr_path.parent / organoid_zarr_name
                
                organoid_group = zarr.open_group(str(organoid_zarr_path), mode='r')
                organoid_data = organoid_group['data'][:]  # Shape: (c, z, y, x)
                organoid_volume = organoid_data[channel]  # Shape: (z, y, x)
                
                print(f"📊 Organoid volume shape: {organoid_volume.shape} (complex optimized format)")
                print(f"📊 Organoid {organoid_idx} has {organoid_volume.shape[0]} z-slices")
                
            elif storage_type == 'variable_size_groups':
                # SIMPLE OPTIMIZED FORMAT: Variable-size groups in single zarr
                organoid_key = f"organoid_{organoid_idx:04d}"
                
                if organoid_key not in bundle:
                    available_organoids = len([k for k in bundle.keys() if k.startswith('organoid_')])
                    print(f"❌ Organoid index {organoid_idx} not found (only {available_organoids} organoids available)")
                    return None
                
                organoid_group = bundle[organoid_key]
                organoid_data = organoid_group['data'][:]  # Shape: (c, z, y, x)
                organoid_volume = organoid_data[channel]  # Shape: (z, y, x)
                
                print(f"📊 Organoid volume shape: {organoid_volume.shape} (simple optimized format)")
                print(f"📊 Organoid {organoid_idx} has {organoid_volume.shape[0]} z-slices")
                
            else:
                # OLD FORMAT: Uniform padding in single zarr
                # Check if the specified level exists
                if str(level) in bundle:
                    data = bundle[str(level)]
                else:
                    # Fall back to main data array
                    data = bundle['data']
                
                if organoid_idx >= data.shape[0]:
                    print(f"❌ Organoid index {organoid_idx} not found (only {data.shape[0]} organoids available)")
                    return None
                
                # Get the specific organoid's 3D volume for the specified channel
                # Data shape: (n_organoids, n_channels, z, y, x)
                organoid_volume = data[organoid_idx, channel]  # Shape: (z, y, x)
                print(f"📊 Organoid volume shape: {organoid_volume.shape} (legacy format)")
                print(f"📊 Organoid {organoid_idx} has {organoid_volume.shape[0]} z-slices")
            
            n_z = organoid_volume.shape[0]
            
            print(f"📊 Organoid {organoid_idx} has {n_z} z-slices")
            
            # Determine which z-slices to show
            if z_slices is None:
                # Show all z-slices
                z_indices = list(range(n_z))
                title_suffix = f"All {n_z} Z-slices"
            elif z_slices == 3:
                # Show top, middle, bottom
                z_indices = [0, n_z // 2, n_z - 1]
                title_suffix = "Top, Middle, Bottom Z-slices"
            else:
                # Show evenly spaced slices including top and bottom
                if z_slices >= n_z:
                    z_indices = list(range(n_z))
                    title_suffix = f"All {n_z} Z-slices"
                else:
                    z_indices = np.linspace(0, n_z - 1, z_slices, dtype=int).tolist()
                    title_suffix = f"{z_slices} Evenly Spaced Z-slices"
            
            # Remove duplicates and sort
            z_indices = sorted(list(set(z_indices)))
            n_slices_to_show = len(z_indices)
            
            print(f"🎯 Showing z-slices: {z_indices}")
            
            # Calculate grid size
            cols = min(4, n_slices_to_show)  # Max 4 columns
            rows = (n_slices_to_show + cols - 1) // cols
            
            # Create the plot
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
            
            # Handle different subplot configurations
            if n_slices_to_show == 1:
                axes = [axes]  # Single subplot
            elif rows == 1 and cols > 1:
                # Single row, multiple columns - axes is already 1D
                pass
            elif rows > 1 and cols == 1:
                # Multiple rows, single column - axes is already 1D
                pass
            else:
                # Multiple rows and columns - axes is 2D, flatten it for easier indexing
                axes = axes.flatten()
            
            # Plot each z-slice
            for i, z_idx in enumerate(z_indices):
                if n_slices_to_show == 1:
                    ax = axes[0]
                elif rows == 1 or cols == 1:
                    ax = axes[i]
                else:
                    ax = axes[i]
                
                slice_img = organoid_volume[z_idx]
                im = ax.imshow(slice_img, cmap='gray')
                ax.set_title(f'Z={z_idx}', fontsize=10, fontweight='bold')
                ax.axis('off')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Hide unused subplots
            total_subplots = rows * cols
            for i in range(n_slices_to_show, total_subplots):
                if n_slices_to_show == 1:
                    break  # No unused subplots
                elif rows == 1 or cols == 1:
                    axes[i].axis('off')
                else:
                    axes[i].axis('off')
            
            fig.suptitle(f'{condition} - Organoid {organoid_idx} - CH{channel+1} - {title_suffix}', 
                        fontsize=12, fontweight='bold')
            plt.tight_layout()
            
            # Save if requested
            saved_path = None
            if save_png:
                slice_str = "all" if z_slices is None else str(z_slices)
                filename = f"{condition}_organoid{organoid_idx}_ch{channel+1}_z{slice_str}.png"
                saved_path = condition_dir / filename
                fig.savefig(saved_path, dpi=150, bbox_inches='tight')
                print(f"💾 Saved: {saved_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close(fig)
                
        except Exception as e:
            print(f"❌ Error creating z-slice visualization: {e}")
            return None
            
        return saved_path

class ZarrAnalyzer:
    """
    Utility class to analyze zarr files and save them as TIFF stacks for debugging.
    """
    
    def __init__(self):
        pass
    
    def analyze_single_zarr(self, 
                           zarr_path: Union[str, Path],
                           level: int = 0,
                           max_z_to_show: int = 5,
                           save_tiff: bool = True,
                           output_dir: Optional[Union[str, Path]] = None) -> Dict:
        """
        Analyze a single zarr file (from analyze folder or stitched subgrid).
        
        Parameters
        ----------
        zarr_path : Path
            Path to the zarr file/group
        level : int
            Pyramid level to analyze (0 = highest resolution)
        max_z_to_show : int
            Maximum number of z-slices to show in visualization
        save_tiff : bool
            Whether to save each channel as a TIFF stack
        output_dir : Path, optional
            Directory to save TIFF files (defaults to same dir as zarr)
            
        Returns
        -------
        dict : Analysis results including shape, dtype, etc.
        """
        zarr_path = Path(zarr_path)
        print(f"🔍 Analyzing zarr: {zarr_path}")
        
        # Open zarr group
        try:
            root = zarr.open_group(str(zarr_path), mode='r')
            
            # Get the specified level
            if str(level) in root:
                data = root[str(level)]
            else:
                available_levels = [k for k in root.keys() if k.isdigit()]
                print(f"❌ Level {level} not found. Available levels: {available_levels}")
                if available_levels:
                    level = int(available_levels[0])
                    data = root[str(level)]
                    print(f"   Using level {level} instead")
                else:
                    raise ValueError(f"No numeric levels found in zarr")
                    
        except Exception as e:
            print(f"❌ Error opening zarr: {e}")
            return {}
        
        # Analyze data
        print(f"📊 Data shape: {data.shape}")
        print(f"📊 Data dtype: {data.dtype}")
        print(f"📊 Data chunks: {data.chunks}")
        
        # Handle different zarr structures
        if data.ndim == 5:  # (t, c, z, y, x)
            t_count, c_count, z_count, y_size, x_size = data.shape
            print(f"   → Time points: {t_count}")
            print(f"   → Channels: {c_count}")
            print(f"   → Z-slices: {z_count}")
            print(f"   → Y size: {y_size}")
            print(f"   → X size: {x_size}")
            
            # Use first timepoint for analysis
            data_to_analyze = data[0]  # Shape: (c, z, y, x)
            
        elif data.ndim == 4:  # (c, z, y, x)
            c_count, z_count, y_size, x_size = data.shape
            print(f"   → Channels: {c_count}")
            print(f"   → Z-slices: {z_count}")
            print(f"   → Y size: {y_size}")
            print(f"   → X size: {x_size}")
            
            data_to_analyze = data
            
        else:
            print(f"❌ Unexpected data dimensionality: {data.ndim}")
            return {}
        
        # Create visualization
        fig, axes = plt.subplots(c_count, min(max_z_to_show, z_count), 
                                figsize=(min(max_z_to_show, z_count) * 3, c_count * 3))
        
        if c_count == 1:
            axes = axes.reshape(1, -1)
        if min(max_z_to_show, z_count) == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f"Zarr Analysis: {zarr_path.name}\nLevel {level} - Shape: {data_to_analyze.shape}")
        
        z_slices_to_show = min(max_z_to_show, z_count)
        
        for c in range(c_count):
            channel_name = CHANNELS[c] if c < len(CHANNELS) else f"CH{c+1}"
            
            for z_idx in range(z_slices_to_show):
                ax = axes[c, z_idx] if z_slices_to_show > 1 else axes[c, 0]
                
                # Get slice data
                slice_data = data_to_analyze[c, z_idx, :, :]
                
                # Display
                im = ax.imshow(slice_data, cmap='gray')
                ax.set_title(f"{channel_name}\nZ={z_idx}")
                ax.axis('off')
                
                # Add colorbar for first slice of each channel
                if z_idx == 0:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
        
        # Save TIFF stacks if requested
        if save_tiff:
            if output_dir is None:
                output_dir = zarr_path.parent
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"💾 Saving TIFF stacks to: {output_dir}")
            
            for c in range(c_count):
                channel_name = CHANNELS[c] if c < len(CHANNELS) else f"CH{c+1}"
                
                # Extract channel data: (z, y, x)
                channel_data = data_to_analyze[c, :, :, :]
                
                # Save as TIFF stack
                output_file = output_dir / f"{zarr_path.stem}_level{level}_{channel_name}.tif"
                
                print(f"   → Saving {channel_name}: {channel_data.shape} → {output_file}")
                tifffile.imwrite(str(output_file), channel_data, 
                               metadata={'axes': 'ZYX'})
        
        # Return analysis results
        results = {
            'zarr_path': str(zarr_path),
            'level': level,
            'shape': data_to_analyze.shape,
            'dtype': str(data_to_analyze.dtype),
            'chunks': data.chunks,
            'channels': c_count,
            'z_slices': z_count,
            'spatial_size': (y_size, x_size)
        }
        
        return results
    
    def analyze_stitched_subgrid(self,
                                stitched_dir: Union[str, Path],
                                condition: str,
                                subgrid_pattern: str = "*sg*",
                                level: int = 0,
                                max_z_to_show: int = 5,
                                save_tiff: bool = True) -> List[Dict]:
        """
        Analyze all stitched subgrids for a given condition.
        
        Parameters
        ----------
        stitched_dir : Path
            Path to stitched directory
        condition : str
            Condition name (e.g., "Condition_D50T50")
        subgrid_pattern : str
            Pattern to match subgrid zarr files (default "*sg*" for subgrid)
        level : int
            Pyramid level to analyze
        max_z_to_show : int
            Max z-slices to show in visualization
        save_tiff : bool
            Whether to save TIFF stacks
            
        Returns
        -------
        list : Analysis results for each subgrid
        """
        stitched_dir = Path(stitched_dir)
        condition_dir = stitched_dir / condition
        
        if not condition_dir.exists():
            print(f"❌ Condition directory not found: {condition_dir}")
            return []
        
        # Find subgrid zarr files using recursive search
        # Pattern: condition_dir/*/*sg*/*.zarr (sg is in the directory name, not zarr name)
        subgrid_zarrs = []
        
        # Search pattern based on your directory structure:
        # Condition_D50T50/LB_diplarge_tetlarge_true_A01_G003/LB_diplarge_tetlarge_true_A01_G003_sg0_0/LB_diplarge_tetlarge_true_A01_G003_sg0_0.zarr
        # The 'sg' is in the directory name, not the zarr filename
        search_patterns = [
            f"*/*{subgrid_pattern}*/*.zarr",      # sg in directory name: */*/sg*/*.zarr
            f"**/*{subgrid_pattern}*/*.zarr",     # Recursive: **/sg*/*.zarr  
            f"*/*/*{subgrid_pattern}*.zarr",      # sg in zarr filename: */*/*sg*.zarr
            f"**/*{subgrid_pattern}*.zarr",       # Recursive sg in filename
        ]
        
        for pattern in search_patterns:
            found_zarrs = list(condition_dir.glob(pattern))
            subgrid_zarrs.extend(found_zarrs)
            if found_zarrs:
                print(f"✅ Found zarr files with pattern: {pattern}")
                break
        
        # Remove duplicates
        subgrid_zarrs = list(set(subgrid_zarrs))
        
        if not subgrid_zarrs:
            print(f"❌ No subgrid zarr files found in {condition_dir}")
            print(f"   Tried patterns: {search_patterns}")
            print(f"   Directory contents:")
            try:
                for item in condition_dir.iterdir():
                    if item.is_dir():
                        print(f"     📁 {item.name}/")
                        # Show one level deeper
                        for subitem in item.iterdir():
                            if subitem.is_dir():
                                print(f"       📁 {subitem.name}/")
                            elif subitem.name.endswith('.zarr'):
                                print(f"       📦 {subitem.name}")
            except Exception as e:
                print(f"     Error listing directory: {e}")
            return []
        
        print(f"🔍 Found {len(subgrid_zarrs)} subgrid zarr files:")
        for zarr_path in subgrid_zarrs:
            # Show relative path from condition directory
            rel_path = zarr_path.relative_to(condition_dir)
            print(f"   → {rel_path}")
        
        results = []
        for zarr_path in sorted(subgrid_zarrs):
            print(f"\n{'='*60}")
            result = self.analyze_single_zarr(
                zarr_path=zarr_path,
                level=level,
                max_z_to_show=max_z_to_show,
                save_tiff=save_tiff
            )
            results.append(result)
        
        return results
