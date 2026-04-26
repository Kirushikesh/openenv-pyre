"""
Thin wrapper for PPO training via HTTP.

This script is now a convenience wrapper around train_torch_ppo.py.
You can achieve the same result by running:
    python training/ppo/train_torch_ppo.py --server http://localhost:8000
"""

import sys
import subprocess
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent.parent
    script = root / "training" / "ppo" / "train_torch_ppo.py"
    
    cmd = [sys.executable, str(script), "--server", "http://localhost:8000"]
    # Pass along any additional arguments
    cmd.extend(sys.argv[1:])
    
    print(f"[wrapper] Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        sys.exit(1)

if __name__ == "__main__":
    main()
