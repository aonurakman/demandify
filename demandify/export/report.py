"""
HTML report generation for calibration results.
"""
from pathlib import Path
from typing import Dict, List
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate HTML report for calibration results."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save report
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(
        self,
        observed_edges: pd.DataFrame,
        simulated_speeds: Dict[str, float],
        loss_history: List[float],
        metadata: Dict
    ) -> Path:
        """
        Generate HTML report.
        
        Args:
            observed_edges: DataFrame with observed edge data
            simulated_speeds: Dict of simulated speeds
            loss_history: List of loss values per generation
            metadata: Run metadata
        
        Returns:
            Path to report.html
        """
        logger.info("Generating calibration report")
        
         # Create visualizations
        loss_plot = self._create_loss_plot(loss_history)
        speed_plot = self._create_speed_comparison(observed_edges, simulated_speeds)
        
        # Calculate mismatches
        mismatches = self._find_top_mismatches(observed_edges, simulated_speeds, top_n=10)
        
        # Build HTML
        html = self._build_html(loss_plot, speed_plot, mismatches, metadata, observed_edges)
        
        # Save
        report_file = self.output_dir / "report.html"
        with open(report_file, 'w') as f:
            f.write(html)
        
        logger.info(f"Report generated: {report_file}")
        
        return report_file
    
    def _create_loss_plot(self, loss_history: List[float]) -> str:
        """Create loss convergence plot."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(loss_history, marker='o', linewidth=2)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Loss (MAE)')
        ax.set_title('Calibration Convergence')
        ax.grid(True, alpha=0.3)
        
        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        plot_path = plot_dir / "loss_plot.png"
        
        fig.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return "plots/loss_plot.png"
    
    def _create_speed_comparison(
        self,
        observed_edges: pd.DataFrame,
        simulated_speeds: Dict[str, float]
    ) -> str:
        """Create observed vs simulated speed scatter plot."""
        # Match speeds
        obs_speeds = []
        sim_speeds = []
        plot_data = []
        
        for _, row in observed_edges.iterrows():
            edge_id = row['edge_id']
            obs_speed = row['current_speed']
            
            sim_speed = simulated_speeds.get(edge_id)
            
            # Add to CSV data regardless of match (use None for missing)
            plot_data.append({
                "edge_id": edge_id,
                "observed_speed": obs_speed,
                "simulated_speed": sim_speed if sim_speed is not None else None,
                "status": "matched" if sim_speed is not None else "missing_in_sim"
            })
            
            # Add to plot only if matched
            if sim_speed is not None:
                obs_speeds.append(obs_speed)
                sim_speeds.append(sim_speed)
        
        # Save scatter plot data to CSV for user analysis
        if plot_data:
            df_comp = pd.DataFrame(plot_data)
            data_dir = self.output_dir / "data"
            data_dir.mkdir(exist_ok=True)
            df_comp.to_csv(data_dir / "speed_comparison.csv", index=False)
            logger.debug(f"Saved speed comparison data ({len(df_comp)} rows) to CSV")

        # Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(obs_speeds, sim_speeds, alpha=0.5)
        
        # Diagonal line
        max_speed = max(max(obs_speeds, default=0), max(sim_speeds, default=0))
        ax.plot([0, max_speed], [0, max_speed], 'r--', label='Perfect match')
        
        ax.set_xlabel('Observed Speed (km/h)')
        ax.set_ylabel('Simulated Speed (km/h)')
        ax.set_title('Speed Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add descriptive annotations
        # Top Left: Sim > Obs (Too fast)
        ax.text(0.05, 0.95, 'Sim > Obs\n(Too Fast / Empty)', 
                transform=ax.transAxes, verticalalignment='top', 
                color='red', fontsize=9, alpha=0.7)
                
        # Bottom Right: Sim < Obs (Too slow)
        ax.text(0.95, 0.05, 'Sim < Obs\n(Too Slow / Congested)', 
                transform=ax.transAxes, horizontalalignment='right', 
                verticalalignment='bottom', 
                color='orange', fontsize=9, alpha=0.7)
        
        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        plot_path = plot_dir / "speed_comparison.png"
        
        fig.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return "plots/speed_comparison.png"
    
    def _find_top_mismatches(
        self,
        observed_edges: pd.DataFrame,
        simulated_speeds: Dict[str, float],
        top_n: int = 10
    ) -> pd.DataFrame:
        """Find edges with largest speed mismatches."""
        mismatches = []
        
        for _, row in observed_edges.iterrows():
            edge_id = row['edge_id']
            obs_speed = row['current_speed']
            
            if edge_id in simulated_speeds:
                sim_speed = simulated_speeds[edge_id]
                error = abs(sim_speed - obs_speed)
                note = ""
            else:
                sim_speed = 0.0
                error = obs_speed  # penalty for missing traffic
                note = "(No Traffic)"
                
            mismatches.append({
                'edge_id': edge_id,
                'observed': round(obs_speed, 1),
                'simulated': round(sim_speed, 1),
                'error': round(error, 1),
                'note': note
            })
        
        df = pd.DataFrame(mismatches)
        if len(df) > 0:
            df = df.nlargest(top_n, 'error')
        
        return df
    
    def _build_html(
        self,
        loss_plot: str,
        speed_plot: str,
        mismatches: pd.DataFrame,
        metadata: Dict,
        observed_edges: pd.DataFrame
    ) -> str:
        """Build HTML report."""
        
        # Safely extract metrics
        results = metadata.get('results', {})
        final_loss = results.get('final_loss_mae_kmh')
        final_loss_str = f"{final_loss:.2f}" if final_loss is not None else "N/A"
        
        quality = results.get('quality_metrics', {})
        matched_edges = quality.get('matched_edges', 0)
        total_edges = quality.get('total_observed_edges', len(observed_edges))
        
        run_info = metadata.get('run_info', {})
        timestamp = run_info.get('timestamp', 'N/A')
        
        sim_config = metadata.get('simulation_config', {})
        window = sim_config.get('window_minutes', 'N/A')
        
        calib_config = metadata.get('calibration_config', {})
        ga_pop = calib_config.get('ga_population', 'N/A')
        ga_gen = calib_config.get('ga_generations', 'N/A')
        
        bbox = run_info.get('bbox_coordinates', {})
        seed = run_info.get('seed', 'N/A')
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>demandify Calibration Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            max-width: 1200px;
            margin: 40px auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .header {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .section {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #2563eb; margin: 0; }}
        h2 {{ color: #374151; margin-top: 0; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }}
        th {{ background-color: #f3f4f6; font-weight: 600; }}
        .metric {{ font-size: 1.5em; color: #2563eb; font-weight: bold; }}
        img {{ max-width: 100%; height: auto; }}
        .plots {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .plot {{ flex: 1; min-width: 400px; }}
        .plots {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .plot {{ flex: 1; min-width: 400px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));  gap: 15px; }}
        .help-text {{ font-size: 0.9em; color: #666; margin-top: 10px; background: #f0f9ff; padding: 10px; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚗 demandify Calibration Report</h1>
        <p>Generated on {timestamp}</p>
    </div>
    
    <div class="section">
        <h2>📊 Results Summary</h2>
        <div class="metrics">
            <p>Final Loss (MAE): <span class="metric">{final_loss_str} km/h</span></p>
            <p>Observed Edges: <span class="metric">{total_edges}</span></p>
            <p>Matched Edges: <span class="metric">{matched_edges}</span></p>
        </div>
    </div>
    
    <div class="section">
        <h2>⚙️ Run Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Bounding Box</td><td>{bbox}</td></tr>
            <tr><td>Window</td><td>{window} minutes</td></tr>
            <tr><td>Seed</td><td>{seed}</td></tr>
            <tr><td>GA Population</td><td>{ga_pop}</td></tr>
            <tr><td>GA Generations</td><td>{ga_gen}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>📈 Calibration Progress</h2>
        <div class="plots">
            <div class="plot">
                <h3>Loss Convergence</h3>
                <img src="{loss_plot}" alt="Loss Plot">
            </div>
            <div class="plot">
                <h3>Speed Comparison</h3>
                <img src="{speed_plot}" alt="Speed Comparison">
                <p class="help-text">
                    <strong>How to read:</strong> Points on the diagonal dashed line are perfect matches.
                    <ul>
                        <li><strong>Above diagonal:</strong> Simulation is faster than real world (needs more traffic).</li>
                        <li><strong>Below diagonal:</strong> Simulation is slower/more congested than real world.</li>
                    </ul>
                </p>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>🔍 Top Mismatched Edges</h2>
        {mismatches.to_html(index=False) if len(mismatches) > 0 else '<p>No data available</p>'}
    </div>
</body>
</html>
"""
        return html
