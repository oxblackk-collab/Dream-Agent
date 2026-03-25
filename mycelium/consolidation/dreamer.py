"""Dreamer — oneiric consolidation daemon.

Runs in a background thread and performs systematic cross-domain Recognize
during idle periods (when the substrate hasn't received new input).

This is how the substrate 'dreams': exploring its own geometry to discover
latent connections that active interaction never surfaced.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mycelium.core.substrate import Substrate

logger = logging.getLogger(__name__)


class Dreamer:
    """Background consolidation daemon.

    Runs substrate.consolidate() when the substrate has been idle
    for at least min_idle_seconds without new ingestion.
    Uses wall-clock time for idle detection (not tick count).
    """

    def __init__(
        self,
        substrate: Substrate,
        min_idle_seconds: float = 30.0,
        pairs_per_cycle: int = 5000,
        poll_interval_seconds: float = 10.0,
        on_discoveries: callable | None = None,
        # Legacy parameter — ignored, kept for compatibility
        min_idle_ticks: int = 50,
    ) -> None:
        self._substrate = substrate
        self._min_idle_seconds = min_idle_seconds
        self._pairs_per_cycle = pairs_per_cycle
        self._poll_interval = poll_interval_seconds
        self._on_discoveries = on_discoveries

        self._last_activity_time: float = time.time()
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def notify_activity(self) -> None:
        """Call this whenever the substrate ingests new data.

        Resets the idle timer, preventing consolidation while active.
        """
        with self._lock:
            self._last_activity_time = time.time()

    def start(self) -> None:
        """Start the dreamer daemon thread."""
        if self._running:
            return
        self._running = True
        self._last_activity_time = time.time()
        self._thread = threading.Thread(
            target=self._loop,
            name="DreamerDaemon",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Dreamer started (idle_threshold=%.0fs, poll=%.0fs, pairs=%d)",
            self._min_idle_seconds, self._poll_interval, self._pairs_per_cycle,
        )

    def stop(self) -> None:
        """Signal the dreamer to stop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Dreamer stopped")

    def _loop(self) -> None:
        """Main daemon loop."""
        while self._running:
            time.sleep(self._poll_interval)

            now = time.time()
            with self._lock:
                idle_seconds = now - self._last_activity_time

            if idle_seconds >= self._min_idle_seconds:
                logger.debug(
                    "Dreamer: substrate idle for %.0fs, starting consolidation",
                    idle_seconds,
                )
                new_discoveries = self._substrate.consolidate(
                    pairs_per_cycle=self._pairs_per_cycle
                )
                if new_discoveries:
                    logger.info(
                        "Dreamer: %d new intersections discovered",
                        len(new_discoveries),
                    )
                    if self._on_discoveries:
                        try:
                            self._on_discoveries(new_discoveries)
                        except Exception as exc:
                            logger.error(
                                "Dreamer on_discoveries callback failed: %s", exc
                            )
                # Reset idle timer after dreaming
                with self._lock:
                    self._last_activity_time = time.time()
