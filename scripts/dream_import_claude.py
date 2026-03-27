"""Dream Import — Filter and import Claude.ai conversation exports.

Reads the JSON export from claude.ai, filters conversations by cognitive
value (length, semantic movement, cognitive patterns), and writes
significant ones to ~/dream/inbox/ for substrate ingestion.

Usage:
    uv run python scripts/dream_import_claude.py conversations.json --dry-run
    uv run python scripts/dream_import_claude.py conversations.json
    uv run python scripts/dream_import_claude.py conversations.json --min-turns 15 --min-distance 0.4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.error import URLError
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from mycelium.embedding.embedder import SentenceTransformerEmbedder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

INBOX_DIR = Path.home() / "dream" / "inbox"
DEFAULT_API = "http://localhost:8000"
CHUNK_SIZE = 500  # words per chunk
CHUNK_OVERLAP = 50


# ── Export parsing ─────────────────────────────────────


def _extract_message_text(msg: dict) -> str:
    """Extract full text from a message, preferring content blocks."""
    content = msg.get("content", [])
    if isinstance(content, list) and content:
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    parts.append(text)
        if parts:
            return "\n\n".join(parts)
    # Fallback to text field
    return msg.get("text", "")


def parse_export(path: Path) -> list[dict]:
    """Parse the claude.ai export JSON into normalized conversations.

    Returns list of:
        {
            "uuid": str,
            "name": str,
            "created_at": str,
            "messages": [{"sender": "human"|"assistant", "text": str}, ...]
        }
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.error("Expected list of conversations, got %s", type(data).__name__)
        return []

    conversations = []
    for conv in data:
        raw_msgs = conv.get("chat_messages", [])
        messages = []
        for msg in raw_msgs:
            sender = msg.get("sender", "")
            text = _extract_message_text(msg)
            if sender and text.strip():
                messages.append({"sender": sender, "text": text.strip()})

        conversations.append(
            {
                "uuid": conv.get("uuid", "unknown"),
                "name": conv.get("name", "Untitled"),
                "created_at": conv.get("created_at", ""),
                "messages": messages,
            }
        )

    return conversations


# ── Filters ────────────────────────────────────────────


def filter_by_length(conv: dict, min_turns: int) -> bool:
    """Filter 1: conversation must have > min_turns assistant messages."""
    assistant_count = sum(1 for m in conv["messages"] if m["sender"] == "assistant")
    return assistant_count > min_turns


def filter_by_semantic_distance(
    conv: dict,
    embedder: SentenceTransformerEmbedder,
    min_distance: float,
) -> tuple[bool, float]:
    """Filter 2: semantic distance between first and last assistant message.

    Returns (passes, distance).
    """
    assistant_msgs = [m["text"] for m in conv["messages"] if m["sender"] == "assistant"]
    if len(assistant_msgs) < 2:
        return False, 0.0

    first_text = assistant_msgs[0][:500]  # Cap for embedding
    last_text = assistant_msgs[-1][:500]

    vec_first = embedder.embed(first_text)
    vec_last = embedder.embed(last_text)
    distance = embedder.semantic_distance(vec_first, vec_last)

    return distance > min_distance, round(distance, 4)


def _detect_unanswered_question(assistant_texts: list[str]) -> bool:
    """Detect if assistant ends with a genuine open question."""
    for msg in assistant_texts:
        stripped = msg.strip()
        if stripped.endswith("?"):
            last_line = stripped.split("\n")[-1].strip()
            if last_line.endswith("?") and len(last_line) > 20:
                return True
    return False


def _detect_direction_change(full_text: str) -> bool:
    """Detect direction-change patterns in assistant text."""
    patterns = [
        r"\bactually\b",
        r"\bwait\b",
        r"\ben realidad\b",
        r"\bpensándolo mejor\b",
        r"\bpensandolo mejor\b",
        r"\bno,? mirá\b",
        r"\bno,? mira\b",
        r"\blet me reconsider\b",
        r"\bahora que lo pienso\b",
        r"\bi was wrong\b",
        r"\bme equivoqué\b",
        r"\bcorrection\b",
        r"\bcorrección\b",
    ]
    return any(re.search(pat, full_text) for pat in patterns)


def _detect_uncertainty(full_text: str) -> bool:
    """Detect uncertainty-recognition patterns in assistant text."""
    patterns = [
        r"\bi'?m not sure\b",
        r"\bno estoy segur[oa]\b",
        r"\bi don'?t know\b",
        r"\bno sé\b",
        r"\bno se\b",
        r"\bmight be wrong\b",
        r"\bpodría estar equivocad\b",
        r"\bhonestly[,.]? i\b",
        r"\bla verdad es que no\b",
        r"\bno tengo certeza\b",
        r"\bi'?m uncertain\b",
        r"\bthis is speculative\b",
        r"\besto es especulativ\b",
    ]
    return any(re.search(pat, full_text) for pat in patterns)


def _detect_emergence(
    human_msgs: list[str],
    assistant_texts: list[str],
) -> bool:
    """Detect unsolicited emergence — long response with signal words."""
    signal_words = [
        "insight",
        "pattern",
        "connection",
        "emerge",
        "tension",
        "patrón",
        "conexión",
        "tensión",
        "descubr",
        "surprising",
        "unexpected",
        "interesante",
        "fascinating",
    ]
    for h, a in zip(human_msgs, assistant_texts):
        h_words = len(h.split())
        a_words = len(a.split())
        if a_words > h_words * 3 and a_words > 100:
            a_lower = a.lower()
            if any(sw in a_lower for sw in signal_words):
                return True
    return False


def filter_by_patterns(conv: dict) -> tuple[bool, list[str]]:
    """Filter 3: cognitive patterns in assistant messages.

    Returns (passes, list_of_patterns_found).
    """
    assistant_texts = [
        m["text"] for m in conv["messages"] if m["sender"] == "assistant"
    ]
    full_text = "\n".join(assistant_texts).lower()
    human_msgs = [m["text"] for m in conv["messages"] if m["sender"] == "human"]

    patterns_found: list[str] = []
    if _detect_unanswered_question(assistant_texts):
        patterns_found.append("pregunta_sin_respuesta")
    if _detect_direction_change(full_text):
        patterns_found.append("cambio_de_direccion")
    if _detect_uncertainty(full_text):
        patterns_found.append("incertidumbre")
    if _detect_emergence(human_msgs, assistant_texts):
        patterns_found.append("emergencia_no_pedida")

    return len(patterns_found) > 0, patterns_found


# ── Conversion ─────────────────────────────────────────


def conversation_to_markdown(conv: dict) -> str:
    """Convert a conversation to markdown format for the extractor."""
    lines = [f"# {conv['name']}", f"# Date: {conv['created_at'][:10]}", ""]

    for msg in conv["messages"]:
        sender = msg["sender"].capitalize()
        # Use **Human**/**Assistant** format that extractor detects
        lines.append(f"**{sender}**: {msg['text']}")
        lines.append("")

    return "\n".join(lines)


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping chunks by sentences."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: list[str] = []
    current_words: list[str] = []
    current_count = 0

    for sentence in sentences:
        words = sentence.split()
        word_count = len(words)
        if current_count + word_count > chunk_size and current_words:
            chunk = " ".join(current_words)
            if len(chunk) > 50:
                chunks.append(chunk)
            overlap_words = current_words[-overlap:] if overlap > 0 else []
            current_words = overlap_words + words
            current_count = len(current_words)
        else:
            current_words.extend(words)
            current_count += word_count

    if current_words:
        chunk = " ".join(current_words)
        if len(chunk) > 50:
            chunks.append(chunk)

    return chunks


def _api_post(api_base: str, endpoint: str, data: dict) -> dict | None:
    """POST JSON to the Mycelium API."""
    url = f"{api_base}{endpoint}"
    body = json.dumps(data).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except (URLError, OSError, json.JSONDecodeError) as exc:
        logger.warning("API call failed (%s): %s", url, exc)
        return None


def ingest_conversation(
    conv: dict,
    api_base: str,
    dry_run: bool = False,
) -> int:
    """Ingest a conversation directly via API, chunked. Returns cells created."""
    md = conversation_to_markdown(conv)
    chunks = chunk_text(md)

    if dry_run:
        return len(chunks)

    cells_created = 0
    conv_name = conv["name"][:50]
    uuid_short = conv["uuid"][:8]

    for chunk in chunks:
        result = _api_post(
            api_base,
            "/api/ingest",
            {
                "text": chunk,
                "source": f"dream:claude_export:{uuid_short}",
                "participant_id": os.environ.get("DREAM_PARTICIPANT_ID", "anonymous"),
                "domain": "chat_session",
            },
        )
        if result is not None:
            cells_created += len(result)

    return cells_created


# ── Main ───────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dream import."""
    parser = argparse.ArgumentParser(
        description="Import Claude.ai export into Dream substrate"
    )
    parser.add_argument("export_path", type=str, help="Path to conversations.json")
    parser.add_argument(
        "--dry-run", action="store_true", help="Report only, don't write files"
    )
    parser.add_argument(
        "--min-turns", type=int, default=10, help="Min assistant turns (default: 10)"
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=0.3,
        help="Min semantic distance (default: 0.3)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show per-conversation details"
    )
    parser.add_argument(
        "--api", type=str, default=DEFAULT_API, help="Mycelium API base URL"
    )
    return parser.parse_args()


def _assistant_count(conv: dict) -> int:
    """Count assistant messages in a conversation."""
    return sum(1 for m in conv["messages"] if m["sender"] == "assistant")


def _discover_conversations(
    conversations: list[dict],
    embedder: SentenceTransformerEmbedder,
    min_turns: int,
    min_distance: float,
    verbose: bool,
) -> tuple[list[dict], int, int, int]:
    """Apply all filters. Returns (passed, d_length, d_distance, d_patterns)."""
    d_len = d_dist = d_pat = 0
    passed: list[dict] = []
    for conv in conversations:
        name = conv["name"][:60]

        if not filter_by_length(conv, min_turns):
            d_len += 1
            if verbose:
                logger.info("    x [%d turns] %s", _assistant_count(conv), name)
            continue

        ok_dist, distance = filter_by_semantic_distance(conv, embedder, min_distance)
        if not ok_dist:
            d_dist += 1
            if verbose:
                logger.info("    x [dist=%s] %s", distance, name)
            continue

        ok_pat, patterns = filter_by_patterns(conv)
        if not ok_pat:
            d_pat += 1
            if verbose:
                logger.info("    x [no patterns] %s", name)
            continue

        turns = _assistant_count(conv)
        entry = {
            "conv": conv,
            "distance": distance,
            "patterns": patterns,
            "assistant_turns": turns,
        }
        passed.append(entry)
        logger.info(
            "    + [%d turns, dist=%s, %s] %s",
            turns,
            distance,
            "+".join(patterns),
            name,
        )

    return passed, d_len, d_dist, d_pat


def _ingest_conversations(
    passed: list[dict],
    api_base: str,
) -> int:
    """Ingest filtered conversations via API, return total cells created."""
    logger.info("  Ingesting via API (%s)...", api_base)
    total_cells = 0

    for item in passed:
        conv = item["conv"]
        name = conv["name"][:50]
        cells = ingest_conversation(conv, api_base)
        total_cells += cells
        logger.info("    %s -> %d cells", name, cells)

    logger.info("  Dreaming (1 cycle)...")
    pairs = min(total_cells * 50, 50000) if total_cells > 0 else 5000
    result = _api_post(api_base, "/api/consolidate", {"pairs_per_cycle": pairs})
    disc = len(result) if result else 0
    logger.info("    %d discoveries (%d pairs examined)", disc, pairs)
    logger.info("    (Dreamer daemon will continue consolidating in background)")
    return total_cells


def _log_filter_results(
    d_len: int,
    d_dist: int,
    d_pat: int,
    passed: int,
    min_turns: int,
    min_distance: float,
) -> None:
    """Log per-filter discard counts and total passed."""
    logger.info("    x Discarded by length (<%d turns): %d", min_turns, d_len)
    logger.info("    x Discarded by semantic distance (<%s): %d", min_distance, d_dist)
    logger.info("    x Discarded by patterns (no cognitive signal): %d", d_pat)
    logger.info("    + Passed all filters: %d", passed)


def _log_dry_run_details(passed: list[dict]) -> None:
    """Log per-conversation details during a dry run."""
    logger.info("  Would ingest %d conversations via API (dry-run)", len(passed))
    for item in passed:
        conv = item["conv"]
        md = conversation_to_markdown(conv)
        chunks = chunk_text(md)
        logger.info(
            "    %s (%d turns, %d chunks)",
            conv["name"][:50],
            item["assistant_turns"],
            len(chunks),
        )


def _print_summary(
    total: int,
    passed: int,
    d_len: int,
    d_dist: int,
    d_pat: int,
    *,
    dry_run: bool,
    total_cells: int,
) -> None:
    """Print the final summary block."""
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info("  Total conversations: %d", total)
    logger.info("  Filtered (pass): %d", passed)
    logger.info("  Discarded: %d", d_len + d_dist + d_pat)
    logger.info("    - By length: %d", d_len)
    logger.info("    - By semantic distance: %d", d_dist)
    logger.info("    - By patterns: %d", d_pat)
    if not dry_run:
        logger.info("  Cells created: %d", total_cells)
    else:
        logger.info("  Conversations that would be ingested: %d", passed)


def _log_banner(
    filename: str,
    total: int,
    min_turns: int,
    min_distance: float,
    dry_run: bool,
) -> None:
    """Log the import header banner."""
    logger.info("=" * 60)
    logger.info("DREAM — Claude.ai Export Import")
    logger.info("=" * 60)
    logger.info("  Source: %s (%d conversations)", filename, total)
    logger.info("  Filters: turns>%d, distance>%s, patterns", min_turns, min_distance)
    if dry_run:
        logger.info("  Mode: DRY RUN (no files written)")


def main() -> None:
    args = _parse_args()

    export_path = Path(args.export_path)
    if not export_path.exists():
        logger.error("Export file not found: %s", export_path)
        sys.exit(1)

    conversations = parse_export(export_path)
    total = len(conversations)
    _log_banner(
        export_path.name, total, args.min_turns, args.min_distance, args.dry_run
    )

    logger.info("  Loading embedder...")
    from mycelium.embedding.embedder import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder()
    logger.info("  Filtering...")

    passed, d_len, d_dist, d_pat = _discover_conversations(
        conversations,
        embedder,
        args.min_turns,
        args.min_distance,
        args.verbose,
    )
    _log_filter_results(
        d_len, d_dist, d_pat, len(passed), args.min_turns, args.min_distance
    )

    total_cells = 0
    if passed and not args.dry_run:
        total_cells = _ingest_conversations(passed, args.api)
    elif args.dry_run and passed:
        _log_dry_run_details(passed)

    _print_summary(
        total,
        len(passed),
        d_len,
        d_dist,
        d_pat,
        dry_run=args.dry_run,
        total_cells=total_cells,
    )


if __name__ == "__main__":
    main()
