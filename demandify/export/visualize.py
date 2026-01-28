"""
Network visualization utilities.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from demandify.sumo.network import SUMONetwork

logger = logging.getLogger(__name__)


def visualize_network(network_file: Path, output_file: Path):
    """
    Create a PNG visualization of the SUMO network.
    
    Args:
        network_file: Path to .net.xml file
        output_file: Path to save PNG
    """
    logger.info(f"Visualizing network: {network_file}")
    
    network = SUMONetwork(network_file)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot all edges
    for edge_id in network.get_all_edges():
        geom = network.get_edge_geometry(edge_id)
        if geom:
            x, y = geom.xy
            ax.plot(x, y, 'b-', linewidth=0.5, alpha=0.6)
    
    ax.set_aspect('equal')
    ax.set_title(f'SUMO Network ({len(network.get_all_edges())} edges)', fontsize=14, fontweight='bold')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.grid(True, alpha=0.3)
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Network visualization saved: {output_file}")
