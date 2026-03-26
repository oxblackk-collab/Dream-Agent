"""Dream Inbox Daemon — watches ~/dream/inbox/ for new files.

Uses watchdog for filesystem events with a polling fallback.
Processes files through the inbox processor and optionally triggers dreaming.

Usage:
    uv run python scripts/dream_inbox_daemon.py
    uv run python scripts/dream_inbox_daemon.py --api http://localhost:8000
    uv run python scripts/dream_inbox_daemon.py --poll 60  # polling mode, no watchdog
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dream.inbox import InboxProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def run_watchdog(processor: InboxProcessor) -> None:
    """Run with watchdog filesystem watcher."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        logger.error(
            "watchdog not installed. Install with: pip install 'mycelium[dream]'\n"
            "Or use --poll mode: dream_inbox_daemon.py --poll 60"
        )
        sys.exit(1)

    class InboxHandler(FileSystemEventHandler):
        def __init__(self) -> None:
            self._debounce: dict[str, float] = {}

        def on_created(self, event):
            if event.is_directory:
                return
            path = Path(event.src_path)
            if path.suffix not in {".md", ".txt", ".json"}:
                return
            # Debounce: wait 1s for file to finish writing
            self._debounce[str(path)] = time.time()

        def on_modified(self, event):
            self.on_created(event)

    handler = InboxHandler()
    observer = Observer()
    observer.schedule(handler, str(processor.inbox_dir), recursive=False)
    observer.start()

    logger.info("Watching %s (watchdog mode)", processor.inbox_dir)

    # Process any existing files on startup
    processor.process_and_dream()

    try:
        while True:
            time.sleep(1)
            # Process debounced files
            now = time.time()
            ready = [
                p for p, t in handler._debounce.items()
                if now - t >= 1.0
            ]
            for p in ready:
                del handler._debounce[p]
                path = Path(p)
                if path.exists():
                    processor.process_file(path)
                    # Trigger dream after each file
                    processor._api_post("/api/consolidate", {"pairs_per_cycle": 2000})
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def run_polling(processor: InboxProcessor, interval: int) -> None:
    """Run with simple polling (no watchdog dependency)."""
    logger.info("Watching %s (polling every %ds)", processor.inbox_dir, interval)

    try:
        while True:
            processor.process_and_dream()
            time.sleep(interval)
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Dream Inbox Daemon")
    parser.add_argument(
        "--api", type=str, default="http://localhost:8000",
        help="Mycelium API base URL",
    )
    parser.add_argument(
        "--poll", type=int, default=0,
        help="Polling interval in seconds (0 = use watchdog)",
    )
    args = parser.parse_args()

    processor = InboxProcessor(api_base=args.api)

    print("\n" + "=" * 50)
    print("DREAM INBOX DAEMON")
    print("=" * 50)
    print(f"  Inbox:     {processor.inbox_dir}")
    print(f"  Processed: {processor.processed_dir}")
    print(f"  API:       {args.api}")
    print(f"  Mode:      {'polling' if args.poll else 'watchdog'}")
    print("=" * 50 + "\n")

    if args.poll > 0:
        run_polling(processor, args.poll)
    else:
        run_watchdog(processor)


if __name__ == "__main__":
    main()
