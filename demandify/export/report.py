"""
HTML report generation for calibration results.
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
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
        metadata: Dict,
        generation_stats: Optional[List[dict]] = None,
    ) -> Path:
        """
        Generate HTML report.

        Args:
            observed_edges: DataFrame with observed edge data
            simulated_speeds: Dict of simulated speeds
            loss_history: List of loss values per generation
            metadata: Run metadata
            generation_stats: Optional list of per-generation statistics dicts

        Returns:
            Path to report.html
        """
        logger.info("Generating calibration report")

        # Create visualizations
        loss_plot = self._create_loss_plot(loss_history, generation_stats)
        speed_plot = self._create_speed_comparison(observed_edges, simulated_speeds)

        # Create additional plots if generation_stats available
        extra_plots = []
        if generation_stats:
            failures_plot = self._create_failures_plot(generation_stats)
            if failures_plot:
                extra_plots.append(("Failures & Zero-Flow Edges", failures_plot))

            magnitude_plot = self._create_magnitude_plot(generation_stats)
            if magnitude_plot:
                extra_plots.append(("Genome Magnitude (Total Vehicles)", magnitude_plot))

            diversity_plot = self._create_diversity_plot(generation_stats)
            if diversity_plot:
                extra_plots.append(("Population Diversity", diversity_plot))

        # Calculate mismatches
        mismatches = self._find_top_mismatches(observed_edges, simulated_speeds, top_n=10)

        # Build HTML
        html = self._build_html(
            loss_plot, speed_plot, mismatches, metadata, observed_edges, extra_plots
        )

        # Save
        report_file = self.output_dir / "report.html"
        with open(report_file, "w") as f:
            f.write(html)

        logger.info(f"Report generated: {report_file}")

        return report_file

    def _create_loss_plot(
        self, loss_history: List[float], generation_stats: Optional[List[dict]] = None
    ) -> str:
        """Create loss convergence plot with optional mean±stddev band."""
        fig, ax = plt.subplots(figsize=(8, 4))

        generations = list(range(1, len(loss_history) + 1))

        # Best loss line
        ax.plot(
            generations,
            loss_history,
            marker="o",
            linewidth=2,
            color="#2563eb",
            markersize=5,
            label="Best Loss",
            zorder=3,
        )

        # Mean ± stddev band if generation_stats available
        if generation_stats and len(generation_stats) == len(loss_history):
            mean_losses = [s["mean_loss"] for s in generation_stats]
            std_losses = [s["std_loss"] for s in generation_stats]

            ax.plot(
                generations,
                mean_losses,
                linewidth=1.5,
                linestyle="--",
                color="#f59e0b",
                label="Mean Loss",
                zorder=2,
            )

            lower = [m - s for m, s in zip(mean_losses, std_losses)]
            upper = [m + s for m, s in zip(mean_losses, std_losses)]
            ax.fill_between(
                generations, lower, upper, alpha=0.15, color="#f59e0b", label="±1 Std Dev"
            )

        ax.set_xlabel("Generation")
        ax.set_ylabel("Loss (MAE, km/h)")
        ax.set_title("Calibration Convergence")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        plot_path = plot_dir / "loss_plot.png"

        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        return "plots/loss_plot.png"

    def _create_speed_comparison(
        self, observed_edges: pd.DataFrame, simulated_speeds: Dict[str, float]
    ) -> str:
        """Create observed vs simulated speed scatter plot with statistics."""
        # Match speeds
        obs_speeds = []
        sim_speeds = []
        plot_data = []

        for _, row in observed_edges.iterrows():
            edge_id = row["edge_id"]
            obs_speed = row["current_speed"]

            sim_speed = simulated_speeds.get(edge_id)

            # Add to CSV data regardless of match (use None for missing)
            plot_data.append(
                {
                    "edge_id": edge_id,
                    "observed_speed": obs_speed,
                    "simulated_speed": sim_speed if sim_speed is not None else None,
                    "status": "matched" if sim_speed is not None else "missing_in_sim",
                }
            )

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

        n_matched = len(obs_speeds)
        n_missing = len(plot_data) - n_matched

        ax.scatter(
            obs_speeds,
            sim_speeds,
            alpha=0.5,
            s=30,
            color="#2563eb",
            edgecolors="white",
            linewidths=0.3,
            label=f"Matched edges (n={n_matched})",
        )

        # Diagonal line
        max_speed = max(max(obs_speeds, default=0), max(sim_speeds, default=0))
        ax.plot([0, max_speed], [0, max_speed], "r--", linewidth=1.5, label="Perfect match (y=x)")

        # Compute R² and RMSE for matched edges
        stats_text = f"n = {n_matched} matched"
        if n_missing > 0:
            stats_text += f", {n_missing} missing"
        if n_matched >= 2:
            obs_arr = np.array(obs_speeds)
            sim_arr = np.array(sim_speeds)
            ss_res = np.sum((sim_arr - obs_arr) ** 2)
            ss_tot = np.sum((obs_arr - np.mean(obs_arr)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            rmse = np.sqrt(np.mean((sim_arr - obs_arr) ** 2))
            mae = np.mean(np.abs(sim_arr - obs_arr))
            stats_text += f"\nR² = {r_squared:.3f}, RMSE = {rmse:.1f} km/h\nMAE = {mae:.1f} km/h"

        # Statistics box (bottom-right area, above the region label)
        ax.text(
            0.95,
            0.22,
            stats_text,
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="#ccc"),
        )

        ax.set_xlabel("Observed Speed (km/h)")
        ax.set_ylabel("Simulated Speed (km/h)")
        ax.set_title("Speed Comparison: Observed vs Simulated")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add descriptive annotations
        # Top Left: Sim > Obs (Too fast)
        ax.text(
            0.05,
            0.95,
            "Sim > Obs\n(Too Fast / Empty)",
            transform=ax.transAxes,
            verticalalignment="top",
            color="red",
            fontsize=9,
            alpha=0.7,
        )

        # Bottom Right: Sim < Obs (Too slow)
        ax.text(
            0.95,
            0.05,
            "Sim < Obs\n(Too Slow / Congested)",
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="bottom",
            color="orange",
            fontsize=9,
            alpha=0.7,
        )

        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        plot_path = plot_dir / "speed_comparison.png"

        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        return "plots/speed_comparison.png"

    def _create_failures_plot(self, generation_stats: List[dict]) -> Optional[str]:
        """Create plot showing zero-flow edges and routing failures over generations."""
        generations = [s["generation"] for s in generation_stats]

        has_zero_flow = any(s.get("best_zero_flow") is not None for s in generation_stats)
        has_failures = any(s.get("best_routing_failures") is not None for s in generation_stats)

        if not has_zero_flow and not has_failures:
            return None

        fig, ax1 = plt.subplots(figsize=(8, 4))

        if has_zero_flow:
            best_zf = [s.get("best_zero_flow", 0) or 0 for s in generation_stats]
            mean_zf = [s.get("mean_zero_flow", 0) or 0 for s in generation_stats]
            ax1.plot(
                generations,
                best_zf,
                marker="s",
                markersize=4,
                linewidth=2,
                color="#8b5cf6",
                label="Best: Zero-Flow Edges",
            )
            ax1.plot(
                generations,
                mean_zf,
                linestyle="--",
                linewidth=1.5,
                color="#8b5cf6",
                alpha=0.6,
                label="Mean: Zero-Flow Edges",
            )

        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Zero-Flow Edges", color="#8b5cf6")
        ax1.tick_params(axis="y", labelcolor="#8b5cf6")
        ax1.grid(True, alpha=0.3)

        if has_failures:
            ax2 = ax1.twinx()
            best_rf = [s.get("best_routing_failures", 0) or 0 for s in generation_stats]
            mean_rf = [s.get("mean_routing_failures", 0) or 0 for s in generation_stats]
            ax2.plot(
                generations,
                best_rf,
                marker="^",
                markersize=4,
                linewidth=2,
                color="#ef4444",
                label="Best: Routing Failures",
            )
            ax2.plot(
                generations,
                mean_rf,
                linestyle="--",
                linewidth=1.5,
                color="#ef4444",
                alpha=0.6,
                label="Mean: Routing Failures",
            )
            ax2.set_ylabel("Routing Failures", color="#ef4444")
            ax2.tick_params(axis="y", labelcolor="#ef4444")

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        if has_failures:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
        else:
            ax1.legend(loc="upper right", fontsize=8)

        ax1.set_title("Zero-Flow Edges & Routing Failures Over Generations")
        fig.tight_layout()

        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        plot_path = plot_dir / "failures_plot.png"

        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        return "plots/failures_plot.png"

    def _create_magnitude_plot(self, generation_stats: List[dict]) -> Optional[str]:
        """Create plot showing best and mean genome magnitude over generations."""
        generations = [s["generation"] for s in generation_stats]

        has_magnitude = any(s.get("best_magnitude") is not None for s in generation_stats)
        if not has_magnitude:
            return None

        best_mag = [s.get("best_magnitude", 0) for s in generation_stats]
        mean_mag = [s.get("mean_magnitude", 0) for s in generation_stats]

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(
            generations,
            best_mag,
            marker="o",
            markersize=4,
            linewidth=2,
            color="#059669",
            label="Best Individual",
        )
        ax.plot(
            generations,
            mean_mag,
            linestyle="--",
            linewidth=1.5,
            color="#059669",
            alpha=0.6,
            label="Population Mean",
        )

        ax.set_xlabel("Generation")
        ax.set_ylabel("Total Vehicles (Genome Sum)")
        ax.set_title("Genome Magnitude Over Generations")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        plot_path = plot_dir / "magnitude_plot.png"

        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        return "plots/magnitude_plot.png"

    def _create_diversity_plot(self, generation_stats: List[dict]) -> Optional[str]:
        """Create plot showing genotypic and phenotypic diversity over generations."""
        generations = [s["generation"] for s in generation_stats]

        has_genotypic = any(s.get("genotypic_diversity") is not None for s in generation_stats)
        has_phenotypic = any(s.get("phenotypic_diversity") is not None for s in generation_stats)

        if not has_genotypic and not has_phenotypic:
            return None

        fig, ax1 = plt.subplots(figsize=(8, 4))

        if has_genotypic:
            gen_div = [s.get("genotypic_diversity", 0) or 0 for s in generation_stats]
            ax1.plot(
                generations,
                gen_div,
                marker="o",
                markersize=4,
                linewidth=2,
                color="#2563eb",
                label="Genotypic Diversity (L2)",
            )

        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Genotypic Diversity (L2)", color="#2563eb")
        ax1.tick_params(axis="y", labelcolor="#2563eb")
        ax1.grid(True, alpha=0.3)

        if has_phenotypic:
            ax2 = ax1.twinx()
            phen_div = [s.get("phenotypic_diversity", 0) or 0 for s in generation_stats]
            ax2.plot(
                generations,
                phen_div,
                marker="s",
                markersize=4,
                linewidth=2,
                color="#dc2626",
                label="Phenotypic Diversity (σ fitness)",
            )
            ax2.set_ylabel("Phenotypic Diversity (σ fitness)", color="#dc2626")
            ax2.tick_params(axis="y", labelcolor="#dc2626")

        # Mark stagnation boost generations if available
        boosted_gens = [s["generation"] for s in generation_stats if s.get("mutation_boosted")]
        if boosted_gens:
            for i, bg in enumerate(boosted_gens):
                ax1.axvline(
                    x=bg,
                    color="#f59e0b",
                    alpha=0.3,
                    linestyle="--",
                    label="Mutation Boost Active" if i == 0 else None,
                )

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        if has_phenotypic:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
        else:
            ax1.legend(loc="upper right", fontsize=8)

        ax1.set_title("Population Diversity Over Generations")
        fig.tight_layout()

        plot_dir = self.output_dir / "plots"
        plot_dir.mkdir(exist_ok=True)
        plot_path = plot_dir / "diversity_plot.png"

        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        return "plots/diversity_plot.png"

    def _find_top_mismatches(
        self, observed_edges: pd.DataFrame, simulated_speeds: Dict[str, float], top_n: int = 10
    ) -> pd.DataFrame:
        """Find edges with largest speed mismatches."""
        mismatches = []

        for _, row in observed_edges.iterrows():
            edge_id = row["edge_id"]
            obs_speed = row["current_speed"]

            if edge_id in simulated_speeds:
                sim_speed = simulated_speeds[edge_id]
                error = abs(sim_speed - obs_speed)
                note = ""
            else:
                sim_speed = 0.0
                error = obs_speed  # penalty for missing traffic
                note = "(No Traffic)"

            mismatches.append(
                {
                    "edge_id": edge_id,
                    "observed": round(obs_speed, 1),
                    "simulated": round(sim_speed, 1),
                    "error": round(error, 1),
                    "note": note,
                }
            )

        df = pd.DataFrame(mismatches)
        if len(df) > 0:
            df = df.nlargest(top_n, "error")

        return df

    def _build_html(
        self,
        loss_plot: str,
        speed_plot: str,
        mismatches: pd.DataFrame,
        metadata: Dict,
        observed_edges: pd.DataFrame,
        extra_plots: Optional[List[tuple]] = None,
    ) -> str:
        """Build HTML report."""

        # Safely extract metrics
        results = metadata.get("results", {})
        final_loss = results.get("final_loss_mae_kmh")
        final_loss_str = f"{final_loss:.2f}" if final_loss is not None else "N/A"

        quality = results.get("quality_metrics", {})
        matched_edges = quality.get("matched_edges", 0)
        total_edges = quality.get("total_observed_edges", len(observed_edges))

        run_info = metadata.get("run_info", {})
        timestamp = run_info.get("timestamp", "N/A")

        sim_config = metadata.get("simulation_config", {})
        window = sim_config.get("window_minutes", "N/A")

        calib_config = metadata.get("calibration_config", {})
        ga_pop = calib_config.get("ga_population", "N/A")
        ga_gen = calib_config.get("ga_generations", "N/A")

        bbox = run_info.get("bbox_coordinates", {})
        seed = run_info.get("seed", "N/A")

        # Build extra plots HTML
        extra_plots_html = ""
        if extra_plots:
            for title, plot_path in extra_plots:
                extra_plots_html += f"""
            <div class="plot">
                <h3>{title}</h3>
                <img src="{plot_path}" alt="{title}">
            </div>"""

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
        <h2>📉 GA Population Statistics</h2>
        <div class="plots">{extra_plots_html if extra_plots_html else '<p>No additional GA population plots available.</p>'}
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
