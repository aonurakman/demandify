
"""
Experiment 1: The Phantom Trip
Verifies if a trip completing during warmup is invisible to edge_data.xml output.
"""
import sys
from pathlib import Path
import logging
import xml.etree.ElementTree as ET

# Add project root to path
sys.path.append(str(Path.cwd()))

from demandify.sumo.simulation import SUMOSimulation
from demandify.sumo.network import SUMONetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("exp1")

def run_experiment():
    print("🧪 Running Experiment 1: The Phantom Trip")
    
    # Setup paths
    output_dir = Path("demandify_runs/exp1_phantom")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Use existing network (Nowa Huta or similar if available, else standard)
    # We'll use the one from a previous run or cache if possible
    # For simplicity, we assume we can generate a small synthetic one or use an existing one.
    # Let's try to locate a network file.
    network_file = None
    for f in Path("demandify_runs").rglob("*.net.xml"):
        network_file = f
        break
    
    if not network_file:
        print("❌ No network file found in demandify_runs. Please run a calibration first to generate one.")
        return

    print(f"   Using network: {network_file}")
    network_file = network_file.resolve()
    
    # 2. Create a specific long trip and a short trip
    # We need valid edges.
    net = SUMONetwork(network_file)
    edges = net.get_all_edges()
    if len(edges) < 2:
        print("❌ Network has too few edges.")
        return
        
    origin = edges[0]
    dest = edges[-1]
    
    print(f"   Trip: {origin} -> {dest}")
    
    # Create trips.xml
    trips_file = output_dir / "trips.xml"
    root = ET.Element('routes')
    
    # Trip 1: Departs at t=0, Expected duration < 300s
    ET.SubElement(root, 'trip', {
        'id': 'phantom_trip',
        'depart': '0.00',
        'from': origin,
        'to': dest
    })
    
    tree = ET.ElementTree(root)
    tree.write(trips_file)
    
    # 3. Run Simulation with Warmup
    # Warmup = 300s. If trip takes 100s, it finishes at t=100.
    # Measurement starts at t=300.
    # Result: Phantom Trip should NOT appear in edge_data.xml
    
    sim = SUMOSimulation(
        network_file=network_file,
        vehicle_file=trips_file,
        warmup_time=300,
        simulation_time=600,
        debug=True # Keep artifacts!
    )
    
    print("   Running simulation (Warmup=300s, Window=300s)...")
    edge_stats, failures = sim.run(output_dir=output_dir)
    
    # 4. Analyze Results
    print(f"   Routing Failures: {failures}")
    print(f"   Edges with recorded speed: {len(edge_stats)}")
    
    # Check tripinfo for actual duration
    tripinfo_file = output_dir / "tripinfo.xml"
    if tripinfo_file.exists():
        tree = ET.parse(tripinfo_file)
        root = tree.getroot()
        trip = root.find("tripinfo[@id='phantom_trip']")
        if trip is not None:
            duration = float(trip.get('duration'))
            arrival = float(trip.get('arrival'))
            print(f"   ✅ Trip completed! Duration: {duration:.1f}s, Arrival: t={arrival:.1f}s")
            
            if arrival < 300:
                print("   ℹ️  Trip finished BEFORE warmup ended (t=300).")
                if len(edge_stats) == 0:
                    print("   🔴 RESULT: CONFIRMED. Trip is invisible in edge data.")
                    print("      This explains the 'constant loss' issue if demand separates too early.")
                else:
                    print(f"   🟡 RESULT: UNEXPECTED. Trip finished early but {len(edge_stats)} edges recorded data??")
            else:
                print("   ℹ️  Trip finished AFTER warmup ended.")
                if len(edge_stats) > 0:
                    print("   🟢 RESULT: Trip visible (Normal behavior).")
                else:
                    print("   🔴 RESULT: Trip missing despite finishing in window (Something else is wrong).")
        else:
            print("   ❌ Trip did not complete (Routing failure?).")
    else:
        print("   ❌ No tripinfo generated.")

if __name__ == "__main__":
    run_experiment()
