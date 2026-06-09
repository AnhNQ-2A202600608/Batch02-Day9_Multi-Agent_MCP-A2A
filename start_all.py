import subprocess
import time
import sys

services = [
    {"name": "Registry", "module": "registry", "port": 10000, "delay": 2},
    {"name": "Tax Agent", "module": "tax_agent", "port": 10102, "delay": 0},
    {"name": "Compliance Agent", "module": "compliance_agent", "port": 10103, "delay": 3},
    {"name": "Law Agent", "module": "law_agent", "port": 10101, "delay": 3},
    {"name": "Customer Agent", "module": "customer_agent", "port": 10100, "delay": 0}
]

processes = []

def main():
    print("=" * 60)
    print("Starting Legal Multi-Agent System Services")
    print("=" * 60)
    
    try:
        for service in services:
            print(f"Starting {service['name']} on port {service['port']}...")
            # Run module as python -m <module>
            p = subprocess.Popen([sys.executable, "-m", service["module"]])
            processes.append(p)
            if service["delay"] > 0:
                time.sleep(service["delay"])
        
        print("\n" + "=" * 60)
        print("All services successfully started in the background:")
        print("  Registry:         http://localhost:10000")
        print("  Customer Agent:   http://localhost:10100")
        print("  Law Agent:        http://localhost:10101")
        print("  Tax Agent:        http://localhost:10102")
        print("  Compliance Agent: http://localhost:10103")
        print("=" * 60)
        print("\nPress Ctrl+C to stop all services.")
        
        while True:
            # Check if any process has terminated unexpectedly
            for p in processes:
                if p.poll() is not None:
                    raise RuntimeError("One of the background services terminated unexpectedly.")
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Stopping all background services...")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            p.wait()
        print("All background services stopped.")

if __name__ == "__main__":
    main()
