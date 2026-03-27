"""Launch the Dream menubar app.

Usage:
    uv run python scripts/dream_menubar.py
    uv run python scripts/dream_menubar.py --api http://localhost:8000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dream Menubar App")
    parser.add_argument(
        "--api",
        type=str,
        default="http://localhost:8000",
        help="Mycelium API base URL",
    )
    args = parser.parse_args()

    try:
        from dream.menubar import DreamMenubar
    except ImportError:
        logger.error(
            "rumps not installed. Install with: pip install 'mycelium[dream]' or: pip install rumps"
        )
        sys.exit(1)

    logger.info("Starting Dream menubar...")
    logger.info("  API: %s", args.api)
    logger.info("  Look for the mushroom icon in your menu bar")

    app = DreamMenubar(api_base=args.api)
    app.run()


if __name__ == "__main__":
    main()
