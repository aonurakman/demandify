#!/usr/bin/env python3
"""
demandify CLI entry point.
"""
import argparse
import subprocess
from pathlib import Path
from demandify import __version__

ASCII_ART = r"""
          ▒▒▒                                                                ▒▒▒▓ ▒▒▒▒   ▒▒▒▒▒▒▒                               
          ▒▓▓                                                                ▒▓▓▓      ▒▓▓▓    ▒▓▓                             
    ▒▓▓▓▓▓▓▓▓   ▒▓▓▓▓▓▓    ▒▓▓▓▓▓▓ ▓▓▓▓▓     ▒▓▓▓▓▓▓     ▒▓▓▓▓▓▓▓▓      ▓▓▓▓▓▓▓▓▓ ▒▓▓▓ ▒▓▓▓       ▓▓▓▓    ▒▓▓                  
  ▒▓▓     ▓▓▓ ▒▓▓▓    ▒▓▓  ▒▓▓   ▒▓▓   ▓▓▓ ▒▓▓▓    ▒▓▓   ▒▓▓▓    ▒▓▓  ▓▓▓    ▒▓▓▓ ▒▓▓▓ ▒▓▓▓▓▓▓▓   ▓▓▓▓    ▒▓▓                  
  ▒▓▓     ▓▓▓ ▒▓▓▓▓▓▓      ▒▓▓   ▒▓▓   ▓▓▓ ▒▓▓▓    ▒▓▓   ▒▓▓▓    ▒▓▓  ▓▓▓    ▒▓▓▓ ▒▓▓▓ ▒▓▓▓       ▓▓▓▓    ▒▓▓                  
  ▒▓▓     ▓▓▓ ▒▓▓▓    ▒▓▓  ▒▓▓   ▒▓▓   ▓▓▓ ▒▓▓▓    ▒▓▓   ▒▓▓▓    ▒▓▓  ▓▓▓    ▒▓▓▓ ▒▓▓▓ ▒▓▓▓         ▒▓▓▓▓▓▓▓▓                  
    ▒▒▒▒▒▒▒     ▒▒▒▒▒▒▒    ▒▒▒   ▒▒▒   ▒▒▒   ▒▒▒▒▒▒▒ ▒▒▒ ▒▒▒▓    ▒▒▒    ▒▒▒▒▒▒▒   ▒▒▒▒ ▒▒▒▒               ▒▒▒                  
                                                                                                  ▓▒▒▒    ▒▒▒                  
                                                                                                    ▒▒▒▒▒▒▒    
                                                                                                                                                                                                                                            
"""


def cmd_cache_clear(args):
    """Clear the cache."""
    from demandify.config import get_config
    import shutil
    
    config = get_config()
    cache_dir = config.cache_dir
    
    if not cache_dir.exists():
        print("Cache is already empty")
        return
    
    # Count items
    count = sum(1 for _ in cache_dir.rglob('*') if _.is_file())
    
    # Clear
    shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Cleared {count} cached items")


def cmd_set_key(args):
    """Update TomTom API key."""
    from demandify.config import save_config, get_config
    
    config = get_config()
    config.tomtom_api_key = args.api_key
    save_config(config)
    
    print(f"✅ TomTom API key updated: {args.api_key[:8]}...")
    print(f"   Key saved to {config.config_dir / 'config.json'}")


def cmd_doctor(args):
    """Run system diagnostics."""
    from demandify.config import get_config
    
    print("🏥 Running demandify diagnostics...\n")
    
    # Check SUMO tools
    print("Checking SUMO installation:")
    try:
        result = subprocess.run(['netconvert', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✅ SUMO tools found")
        else:
            print("  ❌ SUMO tools not working")
    except FileNotFoundError:
        print("  ❌ SUMO not found in PATH")
    
    # Check config
    print("\nChecking configuration:")
    config = get_config()
    if config.tomtom_api_key:
        print(f"  ✅ TomTom API key configured ({config.tomtom_api_key[:8]}...)")
    else:
        print("  ⚠️  TomTom API key not configured")
    
    print(f"  ✅ Cache directory: {config.cache_dir}")
    
    print("\n✅ Diagnostics complete")


async def cmd_run(args):
    """Run calibration in headless mode."""
    from demandify.pipeline import CalibrationPipeline
    import time
    
    # Parse bbox
    try:
        bbox_parts = [float(x.strip()) for x in args.bbox.split(',')]
        if len(bbox_parts) != 4:
            raise ValueError
        bbox = tuple(bbox_parts)
    except ValueError:
        print("❌ Invalid bbox format. Expected: west,south,east,north")
        print("   Example: 2.29,48.84,2.31,48.86")
        return
        
    print(ASCII_ART)
    print(f"demandify v{__version__}")
    print("🚀 Starting headless calibration run")
    print(f"   BBox: {bbox}")
    print(f"   Window: {args.window} min")
    print(f"   Seed: {args.seed}")
    print(f"   Run ID: {args.name if args.name else 'Auto-generated'}")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        pipeline = CalibrationPipeline(
            bbox=bbox,
            window_minutes=args.window,
            seed=args.seed,
            ga_population=args.pop,
            ga_generations=args.gen,
            ga_mutation_rate=args.mutation,
            ga_crossover_rate=args.crossover,
            ga_elitism=args.elitism,
            ga_mutation_sigma=args.sigma,
            ga_mutation_indpb=args.indpb,
            num_origins=args.origins,
            num_destinations=args.destinations,
            max_od_pairs=args.max_ods,
            bin_minutes=args.bin_size,
            run_id=args.name
        )
        
        def confirm_stats(stats):
            print("\n📊 Traffic Data Check:")
            print(f"   - Fetched Segments: {stats['fetched_segments']}")
            print(f"   - Matched Edges:    {stats['matched_edges']}")
            print(f"   - Total Graph Edges: {stats.get('total_network_edges', '-')}")
            
            if stats['matched_edges'] == 0:
                print("\n❌ CRITICAL: No edges matched! Run will fail.")
            elif stats['matched_edges'] < 5:
                print("\n⚠️  WARNING: Very few edges matched (<5). Results may be poor.")
            
            print(f"\n   Logs: {pipeline.output_dir}/logs/pipeline.log")
                
            try:
                response = input("\nProceed with calibration? [y/N] ").strip().lower()
                return response == 'y'
            except EOFError:
                return False
        
        result = await pipeline.run(confirm_callback=confirm_stats)
        
        if result is None:
            return

        elapsed = time.time() - start_time
        run_dir_name = Path(result['run_dir']).name if 'run_dir' in result else pipeline.output_dir.name
        run_id = pipeline.run_id
        
        print("\n" + "="*60)
        print(f"✅ CALIBRATION SUCCESSFUL")
        print("="*60)
        print(f"⏱️  Time: {elapsed:.1f}s")
        print(f"📂 Folder: {pipeline.output_dir}")
        
        # High visibility box for Run ID
        print("\n" + "╔" + "═"*58 + "╗")
        print(f"║{'RUN COMPLETE'.center(58)}║")
        print("║" + " "*58 + "║")
        print(f"║  Run ID: {run_id:<48}║")
        print(f"║  Path:   {run_dir_name:<48}║")
        print("║" + " "*58 + "║")
        print("╚" + "═"*58 + "╝")
        print(f"\n📄 Report available at: {pipeline.output_dir}/report.html\n")
        
    except Exception as e:
        if "No traffic sensors matches" in str(e):
            print("\n⚠️  WARNING: No traffic sensors in this area.")
            return

        print(f"\n❌ Error: {e}")
        return


def cmd_serve(args):
    """Start the web server."""
    import uvicorn
    from demandify.config import get_config
    
    config = get_config()
    
    print(ASCII_ART)
    print(f"demandify v{__version__}")
    print(f"Cache directory: {config.cache_dir}")
    print(f"\n🌐 Starting server at http://{args.host}:{args.port}")
    print("   Press Ctrl+C to stop\n")
    
    uvicorn.run(
        "demandify.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


def cli():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="demandify - SUMO traffic calibration tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", action="version", version=f"demandify {__version__}")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # cache command
    cache_parser = subparsers.add_parser("cache", help="Cache management")
    cache_subparsers = cache_parser.add_subparsers(dest="cache_command")
    cache_subparsers.add_parser("clear", help="Clear all cached data")
    
    # doctor command
    subparsers.add_parser("doctor", help="Check system requirements")
    
    # set-key command
    key_parser = subparsers.add_parser("set-key", help="Update TomTom API key")
    key_parser.add_argument("api_key", help="TomTom API key")
    
    # run command (headless)
    run_parser = subparsers.add_parser("run", help="Run calibration (headless mode)")
    run_parser.add_argument("bbox", help="Bounding box (west,south,east,north)")
    run_parser.add_argument("--name", help="Custom run ID/name")
    run_parser.add_argument("--window", type=int, default=15, help="Simulation window minutes (default: 15)")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    run_parser.add_argument("--pop", type=int, default=50, help="GA population size (default: 50)")
    run_parser.add_argument("--gen", type=int, default=20, help="GA generations (default: 20)")
    run_parser.add_argument("--mutation", type=float, default=0.5, help="Mutation rate (default: 0.5)")
    run_parser.add_argument("--crossover", type=float, default=0.7, help="Crossover rate (default: 0.7)")
    run_parser.add_argument("--elitism", type=int, default=2, help="Elitism count (default: 2)")
    run_parser.add_argument("--sigma", type=int, default=20, help="Mutation sigma (default: 20)")
    run_parser.add_argument("--indpb", type=float, default=0.3, help="Mutation gene probability (default: 0.3)")
    run_parser.add_argument("--origins", type=int, default=10, help="Number of origin candidates (default: 10)")
    run_parser.add_argument("--destinations", type=int, default=10, help="Number of destination candidates (default: 10)")
    run_parser.add_argument("--max-ods", type=int, default=1000, help="Max OD pairs to generate (default: 1000)")
    run_parser.add_argument("--bin-size", type=float, default=1.0, help="Time bin size in minutes (default: 1.0)")
    
    # serve command (default)
    serve_parser = subparsers.add_parser("serve", help="Start web server (default)")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host address")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Default to serve if no command specified
    if args.command is None:
        args.command = "serve"
        args.host = "127.0.0.1"
        args.port = 8000
        args.reload = False
    
    # Route to appropriate handler
    if args.command == "cache":
        if args.cache_command == "clear":
            cmd_cache_clear(args)
        else:
            cache_parser.print_help()
    elif args.command == "doctor":
        cmd_doctor(args)
    elif args.command == "set-key":
        cmd_set_key(args)
    elif args.command == "run":
        try:
            import asyncio
            asyncio.run(cmd_run(args))
        except KeyboardInterrupt:
            print("\n🛑 Run aborted by user")
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
