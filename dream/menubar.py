"""Dream Agent — macOS Menubar App.

Shows a mushroom icon in the menu bar. Polls the Mycelium API
for wake-verified dream discoveries and shows a badge when
Claude has confirmed genuine insights.

Requires: pip install 'dream-agent[menubar]'
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import threading
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

DEFAULT_API = "http://localhost:8000"
POLL_INTERVAL = 30  # seconds
MIN_SIGNIFICANCE = 0.50
ICON_NORMAL = "🍄"
ICON_ACTIVE = "🍄✨"


def _api_get(api_base: str, endpoint: str) -> list | dict | None:
    """GET from API, return parsed JSON or None."""
    try:
        url = f"{api_base}{endpoint}"
        with urlopen(url, timeout=10) as resp:
            return json.loads(resp.read())
    except (URLError, OSError, json.JSONDecodeError):
        return None


def _api_post(api_base: str, endpoint: str, data: dict) -> dict | None:
    """POST JSON to API, return parsed JSON or None."""
    try:
        url = f"{api_base}{endpoint}"
        body = json.dumps(data).encode("utf-8")
        req = Request(url, data=body, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except (URLError, OSError, json.JSONDecodeError):
        return None


class DreamMenubar:
    """macOS menubar app for Dream agent.

    Shows only wake-verified connections — insights that Claude
    confirmed as genuine, not raw dream noise.
    """

    def __init__(self, api_base: str = DEFAULT_API) -> None:
        import rumps

        self.api_base = api_base
        self._unseen_count = 0
        self._unseen_entries: list[dict] = []
        self._notified_ids: set[int] = set()
        self._insight_menu_items: list[rumps.MenuItem] = []

        self.app = rumps.App(
            ICON_NORMAL,
            quit_button="Quit Dream",
        )

        # Menu items
        self._status_item = rumps.MenuItem("Dream: connecting...")
        self._insights_header = rumps.MenuItem("No verified insights")
        self._mark_seen_item = rumps.MenuItem(
            "Mark All Seen", callback=self._on_mark_seen,
        )
        self._refresh_item = rumps.MenuItem(
            "Refresh Now", callback=self._on_refresh,
        )

        self.app.menu = [
            self._status_item,
            None,
            self._insights_header,
            self._mark_seen_item,
            None,
            self._refresh_item,
        ]

        # Timer for polling
        self._timer = rumps.Timer(self._poll, POLL_INTERVAL)
        self._timer.start()

        # Initial poll in background
        threading.Thread(target=self._poll, args=(None,), daemon=True).start()

    def _poll(self, _sender) -> None:
        """Poll API for wake-verified unseen discoveries."""
        # Check health
        health = _api_get(self.api_base, "/api/health")
        if health is None:
            self._status_item.title = "Dream: API offline"
            self.app.title = ICON_NORMAL
            return

        cells = health.get("cells", 0)
        ix = health.get("intersections", 0)
        self._status_item.title = f"Substrate: {cells} cells, {ix} ix"

        # Get only wake-verified unseen entries
        unseen = _api_get(
            self.api_base,
            f"/api/dream/unseen?min_significance={MIN_SIGNIFICANCE}"
            f"&limit=20&verified_only=true",
        )

        if unseen is None:
            return

        self._unseen_entries = unseen
        self._unseen_count = len(unseen)

        if self._unseen_count > 0:
            self.app.title = f"{ICON_ACTIVE} {self._unseen_count}"
            self._update_insights_menu()
            self._notify_new_insights()
        else:
            self.app.title = ICON_NORMAL
            # Clear insight menu items
            for item in self._insight_menu_items:
                if item.title in self.app.menu:
                    del self.app.menu[item.title]
            self._insight_menu_items.clear()
            self._insights_header.title = "No verified insights"

    def _update_insights_menu(self) -> None:
        """Update the insights menu with individual clickable items."""
        import rumps

        # Remove old insight menu items
        for item in self._insight_menu_items:
            if item.title in self.app.menu:
                del self.app.menu[item.title]
        self._insight_menu_items.clear()

        if not self._unseen_entries:
            self._insights_header.title = "No verified insights"
            return

        count = len(self._unseen_entries)

        # Add individual insight items before "Mark All Seen" (stable key)
        for entry in reversed(self._unseen_entries):
            reasoning = entry.get("wake_reasoning", "")
            sig = entry.get("significance", 0)
            label = reasoning[:70] + "..." if len(reasoning) > 70 else reasoning
            if not label:
                label = entry.get("description", "Connection found")[:70]
            title = f"  {sig:.2f} -- {label}"

            item = rumps.MenuItem(
                title,
                callback=self._make_explore_callback(entry),
            )
            self._insight_menu_items.append(item)
            self.app.menu.insert_before(
                "Mark All Seen", item,
            )

        # Update header AFTER inserting items (avoids key mismatch)
        self._insights_header.title = f"{count} insight{'s' if count != 1 else ''}"

    def _notify_new_insights(self) -> None:
        """Send macOS notification for NEW insights not yet notified."""
        import rumps

        new_entries = [
            e for e in self._unseen_entries
            if e.get("id") not in self._notified_ids
        ]

        if not new_entries:
            return

        # Notify top 3 new insights max
        for entry in new_entries[:3]:
            self._notified_ids.add(entry.get("id"))
            reasoning = entry.get("wake_reasoning", "New connection found")
            sig = entry.get("significance", 0)

            rumps.notification(
                title="Dream",
                subtitle=f"Verified insight (sig={sig:.3f})",
                message=reasoning,
            )

    def _make_explore_callback(self, entry: dict):
        """Return a callback closure for a specific insight entry."""
        def callback(_sender):
            threading.Thread(
                target=self._explore_insight,
                args=(entry,),
                daemon=True,
            ).start()
        return callback

    def _explore_insight(self, entry: dict) -> None:
        """Open a Terminal window with Claude exploring this insight."""
        # Fetch cell details for parent_a and parent_b
        parent_a_id = entry.get("parent_a", "")
        parent_b_id = entry.get("parent_b", "")

        cell_a = _api_get(self.api_base, f"/api/cells/{parent_a_id}")
        cell_b = _api_get(self.api_base, f"/api/cells/{parent_b_id}")

        domain_a = cell_a.get("domain", "unknown") if cell_a else "unknown"
        domain_b = cell_b.get("domain", "unknown") if cell_b else "unknown"
        text_a = (cell_a.get("text", "")[:200] if cell_a else "").strip()
        text_b = (cell_b.get("text", "")[:200] if cell_b else "").strip()

        sig = entry.get("significance", 0)
        reasoning = entry.get("wake_reasoning", "")

        prompt = (
            f"Dream found a connection between {domain_a} and {domain_b} "
            f"(significance: {sig:.3f}):\n\n"
            f"{reasoning}\n\n"
            f"Cell A ({domain_a}): {text_a}\n"
            f"Cell B ({domain_b}): {text_b}\n\n"
            f"Explore this connection -- what patterns or applications do you see?"
        )

        # Write prompt to temp file, then a launcher .command script.
        # .command files open natively in Terminal.app (no AppleScript needed).
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="dream_insight_",
            suffix=".txt",
            delete=False,
        ) as f:
            f.write(prompt)
            prompt_path = f.name

        script_path = prompt_path.replace(".txt", ".command")
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f'claude "$(cat \'{prompt_path}\')"\n')

        os.chmod(script_path, 0o755)

        try:
            subprocess.run(["open", script_path], timeout=10)
        except Exception:
            logger.warning("Failed to open Terminal for insight exploration")

    def _on_mark_seen(self, _sender) -> None:
        """Mark all unseen dreams as seen."""
        result = _api_post(
            self.api_base, "/api/dream/mark-seen", {"dream_ids": None},
        )
        if result:
            marked = result.get("marked", 0)
            self._unseen_count = 0
            self._unseen_entries = []
            self.app.title = ICON_NORMAL
            # Clear insight menu items
            for item in self._insight_menu_items:
                if item.title in self.app.menu:
                    del self.app.menu[item.title]
            self._insight_menu_items.clear()
            self._insights_header.title = "No verified insights"

            import rumps
            rumps.notification("Dream", f"Marked {marked} as seen", "")

    def _on_refresh(self, _sender) -> None:
        """Manually trigger a poll."""
        threading.Thread(target=self._poll, args=(None,), daemon=True).start()

    def run(self) -> None:
        """Start the menubar app (blocking)."""
        self.app.run()
