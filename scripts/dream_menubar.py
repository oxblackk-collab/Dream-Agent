"""Launch the Dream menubar app.

Usage:
    python scripts/dream_menubar.py
    python scripts/dream_menubar.py --api http://localhost:8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dream Menubar App")
    parser.add_argument(
        "--api", type=str, default="http://localhost:8000",
        help="Dream Agent API base URL",
    )
    args = parser.parse_args()

    try:
        from dream.menubar import DreamMenubar
    except ImportError:
        print("Error: rumps not installed.")
        print("Install with: pip install 'dream-agent[menubar]'")
        print("Or: pip install rumps")
        sys.exit(1)

    print("Starting Dream menubar...")
    print(f"  API: {args.api}")
    print("  Look for the mushroom icon in your menu bar")

    app = DreamMenubar(api_base=args.api)
    app.run()


if __name__ == "__main__":
    main()
