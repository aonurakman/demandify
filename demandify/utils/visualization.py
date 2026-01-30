"""
Visualization utilities for demandify.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from demandify.sumo.network import SUMONetwork

logger = logging.getLogger(__name__)

def plot_network_geometry(network_file: Path, output_file: Path):
    """
    Plot the geometry of the SUMO network and save to file.
    
    Args:
        network_file: Path to .net.xml file
        output_file: Path to save .png image
    """
    try:
        net = SUMONetwork(network_file)
        
        # Setup plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot edges
        count = 0
        for edge_id, geometry in net.edge_geometries.items():
            if geometry:
                x, y = geometry.xy
                ax.plot(x, y, color='#333333', linewidth=0.8, alpha=0.6)
                count += 1
                
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add simple metadata
        ax.text(0.02, 0.02, f"Edges: {len(net.edges)}", transform=ax.transAxes, fontsize=8)
        
        # Save
        fig.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
        logger.debug(f"Saved network plot to {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to plot network: {e}")
