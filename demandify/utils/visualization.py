"""
Visualization utilities for demandify.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Collection, Optional
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
