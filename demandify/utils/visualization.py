"""
Visualization utilities for demandify.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from pathlib import Path
from typing import Collection, Mapping, Optional
import logging
from demandify.sumo.network import SUMONetwork

logger = logging.getLogger(__name__)

def plot_network_geometry(
    network_file: Path,
    output_file: Path,
    observed_edge_ids: Optional[Collection[str]] = None,
):
    """
    Plot the geometry of the SUMO network and save to file.
    
    Args:
        network_file: Path to .net.xml file
        output_file: Path to save .png image
        observed_edge_ids: Optional edge-id collection to overlay in red
    """
    try:
        net = SUMONetwork(network_file)
        observed_set = {str(edge_id) for edge_id in (observed_edge_ids or [])}
        
        # Setup plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot all edges in gray first
        for edge_id, geometry in net.edge_geometries.items():
            if geometry:
                x, y = geometry.xy
                ax.plot(x, y, color='#333333', linewidth=0.8, alpha=0.6)

        # Overlay observed edges in red so they stand out clearly.
        observed_plotted = 0
        for edge_id in observed_set:
            geometry = net.edge_geometries.get(edge_id)
            if geometry:
                x, y = geometry.xy
                ax.plot(x, y, color='#e53935', linewidth=1.6, alpha=0.95)
                observed_plotted += 1

        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add simple metadata
        meta_lines = [f"Edges: {len(net.edges)}"]
        if observed_set:
            meta_lines.append(f"Observed: {observed_plotted}/{len(observed_set)}")
        ax.text(0.02, 0.02, "\n".join(meta_lines), transform=ax.transAxes, fontsize=8)
        
        # Save
        fig.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
        logger.debug(f"Saved network plot to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to plot network: {e}")


def plot_edge_speed_heatmap(
    network_file: Path,
    output_file: Path,
    edge_speeds: Optional[Mapping[str, float]],
    title: str,
    colorbar_label: str = "Average speed (km/h)",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """
    Plot a network heatmap where colored edges represent average speed.

    Args:
        network_file: Path to .net.xml file
        output_file: Path to save .png image
        edge_speeds: Mapping of edge_id -> average speed in km/h
        title: Plot title
        colorbar_label: Label for the right-side color bar
        vmin: Optional fixed minimum for color normalization
        vmax: Optional fixed maximum for color normalization
    """
    try:
        net = SUMONetwork(network_file)

        # Keep only finite numeric speeds and normalize ids to strings.
        speed_lookup = {}
        for edge_id, speed in (edge_speeds or {}).items():
            try:
                speed_value = float(speed)
            except Exception:
                continue
            if np.isfinite(speed_value):
                speed_lookup[str(edge_id)] = speed_value

        background_segments = []
        colored_segments = []
        color_values = []

        for edge_id, geometry in net.edge_geometries.items():
            if geometry is None:
                continue
            coords = np.asarray(geometry.coords)
            if len(coords) < 2:
                continue

            background_segments.append(coords)
            if edge_id in speed_lookup:
                colored_segments.append(coords)
                color_values.append(speed_lookup[edge_id])

        fig, ax = plt.subplots(figsize=(10, 10))

        if background_segments:
            background = LineCollection(
                background_segments,
                colors="#d1d5db",
                linewidths=0.7,
                alpha=0.6,
                zorder=1,
            )
            ax.add_collection(background)

        if color_values:
            color_values_arr = np.asarray(color_values, dtype=float)

            speed_min = float(np.min(color_values_arr)) if vmin is None else float(vmin)
            speed_max = float(np.max(color_values_arr)) if vmax is None else float(vmax)
            if speed_max <= speed_min:
                speed_max = speed_min + 1.0

            heat = LineCollection(
                colored_segments,
                cmap="viridis",
                norm=Normalize(vmin=speed_min, vmax=speed_max),
                linewidths=1.8,
                zorder=2,
            )
            heat.set_array(color_values_arr)
            ax.add_collection(heat)
            fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
        else:
            ax.text(
                0.5,
                0.5,
                "No speed data available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="#6b7280",
            )

        ax.set_aspect("equal")
        ax.autoscale()
        ax.axis("off")
        ax.set_title(title, fontsize=12)

        meta_lines = [f"Edges: {len(net.edges)}", f"Colored: {len(color_values)}"]
        if color_values:
            meta_lines.append(
                f"Range: {min(color_values):.1f}-{max(color_values):.1f} km/h"
            )
        ax.text(0.02, 0.02, "\n".join(meta_lines), transform=ax.transAxes, fontsize=8)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=150, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)

        logger.debug(f"Saved edge-speed heatmap to {output_file}")

    except Exception as e:
        logger.error(f"Failed to plot edge-speed heatmap: {e}")
