"""Dream Agent — Inbox processor.

Monitors ~/dream/inbox/ for new files, extracts semantically significant
content, ingests via the Mycelium API, and archives processed files.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from dream.extractor import IngestPayload, extract

logger = logging.getLogger(__name__)

DEFAULT_INBOX = Path.home() / "dream" / "inbox"
DEFAULT_PROCESSED = Path.home() / "dream" / "processed"
DEFAULT_API_BASE = "http://localhost:8000"


class InboxProcessor:
    """Processes files from the dream inbox."""

    def __init__(
        self,
        inbox_dir: Path = DEFAULT_INBOX,
        processed_dir: Path = DEFAULT_PROCESSED,
        api_base: str = DEFAULT_API_BASE,
    ) -> None:
        self.inbox_dir = inbox_dir
        self.processed_dir = processed_dir
        self.api_base = api_base.rstrip("/")
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _api_post(self, endpoint: str, data: dict) -> dict | None:
        """POST JSON to the Mycelium API. Returns response or None on failure."""
        url = f"{self.api_base}{endpoint}"
        body = json.dumps(data).encode("utf-8")
        req = Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except (URLError, OSError, json.JSONDecodeError) as exc:
            logger.warning("API call failed (%s): %s", url, exc)
            return None

    def _ingest_payload(self, payload: IngestPayload) -> bool:
        """Send a single payload to the API for ingestion."""
        result = self._api_post(
            "/api/ingest",
            {
                "text": payload.text,
                "source": payload.source,
                "participant_id": payload.participant_id,
                "domain": payload.domain,
            },
        )
        if result is not None:
            logger.info(
                "Ingested: domain=%s source=%s (%d chars)",
                payload.domain,
                payload.source,
                len(payload.text),
            )
            return True
        return False

    def process_file(self, path: Path) -> bool:
        """Process a single file: extract, ingest, move to processed.

        Returns True if file was successfully processed and moved.
        """
        if not path.exists():
            return False

        logger.info("Processing: %s", path.name)

        # Extract payloads
        try:
            payloads = extract(path)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError) as exc:
            logger.error("Extraction failed for %s: %s", path.name, exc)
            self._move_to_processed(path, success=False)
            return False

        if not payloads:
            logger.info("No significant content in %s, skipping", path.name)
            self._move_to_processed(path, success=True)
            return True

        # Ingest each payload
        ingested = 0
        for payload in payloads:
            if self._ingest_payload(payload):
                ingested += 1

        if ingested == 0:
            logger.warning(
                "Failed to ingest any payloads from %s (API down?)", path.name
            )
            return False  # Leave in inbox for retry

        logger.info(
            "Ingested %d/%d payloads from %s", ingested, len(payloads), path.name
        )
        self._move_to_processed(path, success=True)
        return True

    def process_all_pending(self) -> int:
        """Process all files currently in the inbox. Returns count processed."""
        extensions = {".md", ".txt", ".json"}
        files = sorted(
            f
            for f in self.inbox_dir.iterdir()
            if f.is_file() and f.suffix in extensions
        )

        if not files:
            return 0

        logger.info("Found %d files in inbox", len(files))
        processed = 0
        for f in files:
            if self.process_file(f):
                processed += 1

        return processed

    def process_and_dream(self, pairs_per_cycle: int = 2000) -> int:
        """Process inbox and trigger a dream cycle if anything was ingested."""
        count = self.process_all_pending()
        if count > 0:
            logger.info("Triggering dream consolidation...")
            result = self._api_post(
                "/api/consolidate",
                {
                    "pairs_per_cycle": pairs_per_cycle,
                },
            )
            if result is not None:
                logger.info("Dream cycle: %d new intersections", len(result))
        return count

    def last_session_context(self) -> str:
        """Get the commit message from the most recently processed commit file.

        Returns empty string if no commits have been processed.
        """
        commit_files = sorted(
            (
                f
                for f in self.processed_dir.iterdir()
                if f.is_file() and "_ok_commit-" in f.name and f.suffix == ".json"
            ),
            reverse=True,
        )
        if not commit_files:
            return ""
        try:
            data = json.loads(commit_files[0].read_text(encoding="utf-8"))
            message = data.get("message", "")
            body = data.get("body", "")
            repo = data.get("repo", "")
            parts = [f"Commit in {repo}"] if repo else []
            if message:
                parts.append(message)
            if body:
                parts.append(body)
            return " — ".join(parts)
        except (json.JSONDecodeError, OSError):
            return ""

    def _move_to_processed(self, path: Path, success: bool) -> None:
        """Move file to processed directory with timestamp prefix."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "ok" if success else "err"
        dest = self.processed_dir / f"{ts}_{prefix}_{path.name}"
        shutil.move(str(path), str(dest))
        logger.info("Moved %s → %s", path.name, dest.name)
