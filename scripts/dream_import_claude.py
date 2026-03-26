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
import re
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

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

        conversations.append({
            "uuid": conv.get("uuid", "unknown"),
            "name": conv.get("name", "Untitled"),
            "created_at": conv.get("created_at", ""),
            "messages": messages,
        })

    return conversations


# ── Filters ────────────────────────────────────────────


def filter_by_length(conv: dict, min_turns: int) -> bool:
    """Filter 1: conversation must have > min_turns assistant messages."""
    assistant_count = sum(1 for m in conv["messages"] if m["sender"] == "assistant")
    return assistant_count > min_turns


def filter_by_semantic_distance(
    conv: dict,
    embedder,
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


def filter_by_patterns(conv: dict) -> tuple[bool, list[str]]:
    """Filter 3: cognitive patterns in assistant messages.

    Returns (passes, list_of_patterns_found).
    """
    assistant_texts = [m["text"] for m in conv["messages"] if m["sender"] == "assistant"]
    full_text = "\n".join(assistant_texts).lower()

    patterns_found = []

    # Pattern 1: Question without direct answer (assistant ends with ?)
    for msg in assistant_texts:
        stripped = msg.strip()
        if stripped.endswith("?"):
            # Check it's not just a rhetorical question followed by an answer
            lines = stripped.split("\n")
            last_line = lines[-1].strip()
            if last_line.endswith("?") and len(last_line) > 20:
                patterns_found.append("pregunta_sin_respuesta")
                break

    # Pattern 2: Direction change
    direction_patterns = [
        r"\bactually\b", r"\bwait\b", r"\ben realidad\b",
        r"\bpensándolo mejor\b", r"\bpensandolo mejor\b",
        r"\bno,? mirá\b", r"\bno,? mira\b",
        r"\blet me reconsider\b", r"\bahora que lo pienso\b",
        r"\bi was wrong\b", r"\bme equivoqué\b",
        r"\bcorrection\b", r"\bcorrección\b",
    ]
    for pat in direction_patterns:
        if re.search(pat, full_text):
            patterns_found.append("cambio_de_direccion")
            break

    # Pattern 3: Uncertainty recognition
    uncertainty_patterns = [
        r"\bi'?m not sure\b", r"\bno estoy segur[oa]\b",
        r"\bi don'?t know\b", r"\bno sé\b", r"\bno se\b",
        r"\bmight be wrong\b", r"\bpodría estar equivocad\b",
        r"\bhonestly[,.]? i\b", r"\bla verdad es que no\b",
        r"\bno tengo certeza\b", r"\bi'?m uncertain\b",
        r"\bthis is speculative\b", r"\besto es especulativ\b",
    ]
    for pat in uncertainty_patterns:
        if re.search(pat, full_text):
            patterns_found.append("incertidumbre")
            break

    # Pattern 4: Emergence — assistant response much longer than human question
    # + contains signal words
    signal_words = [
        "insight", "pattern", "connection", "emerge", "tension",
        "patrón", "conexión", "tensión", "emerge", "descubr",
        "surprising", "unexpected", "interesante", "fascinating",
    ]
    human_msgs = [m["text"] for m in conv["messages"] if m["sender"] == "human"]
    for i, (h, a) in enumerate(zip(human_msgs, assistant_texts)):
        h_words = len(h.split())
        a_words = len(a.split())
        if a_words > h_words * 3 and a_words > 100:
            a_lower = a.lower()
            if any(sw in a_lower for sw in signal_words):
                patterns_found.append("emergencia_no_pedida")
                break

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


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
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
        result = _api_post(api_base, "/api/ingest", {
            "text": chunk,
            "source": f"dream:claude_export:{uuid_short}",
            "participant_id": "claude_y_0xblackk",
            "domain": "chat_session",
        })
        if result is not None:
            cells_created += len(result)

    return cells_created


# ── Main ───────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import Claude.ai export into Dream substrate"
    )
    parser.add_argument("export_path", type=str, help="Path to conversations.json")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't write files")
    parser.add_argument("--min-turns", type=int, default=10, help="Min assistant turns (default: 10)")
    parser.add_argument("--min-distance", type=float, default=0.3, help="Min semantic distance (default: 0.3)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-conversation details")
    parser.add_argument("--api", type=str, default=DEFAULT_API, help="Mycelium API base URL")
    args = parser.parse_args()

    export_path = Path(args.export_path)
    if not export_path.exists():
        print(f"Error: {export_path} not found")
        sys.exit(1)

    # Parse export
    conversations = parse_export(export_path)
    total = len(conversations)

    print("\n" + "=" * 60)
    print("DREAM — Claude.ai Export Import")
    print("=" * 60)
    print(f"  Source: {export_path.name} ({total} conversations)")
    print(f"  Filters: turns>{args.min_turns}, distance>{args.min_distance}, patterns")
    if args.dry_run:
        print("  Mode: DRY RUN (no files written)")
    print()

    # Initialize embedder for semantic distance
    print("  Loading embedder...")
    from mycelium.embedding.embedder import SentenceTransformerEmbedder
    embedder = SentenceTransformerEmbedder()
    print()

    # ── Filter ──

    print("  Filtering...")

    discarded_length = 0
    discarded_distance = 0
    discarded_patterns = 0
    passed = []

    for conv in conversations:
        name = conv["name"][:60]
        msg_count = len(conv["messages"])

        # Filter 1: Length
        if not filter_by_length(conv, args.min_turns):
            discarded_length += 1
            if args.verbose:
                assistant_count = sum(1 for m in conv["messages"] if m["sender"] == "assistant")
                print(f"    ✗ [{assistant_count} turns] {name}")
            continue

        # Filter 2: Semantic distance
        passes_distance, distance = filter_by_semantic_distance(
            conv, embedder, args.min_distance,
        )
        if not passes_distance:
            discarded_distance += 1
            if args.verbose:
                print(f"    ✗ [dist={distance}] {name}")
            continue

        # Filter 3: Cognitive patterns
        passes_patterns, patterns = filter_by_patterns(conv)
        if not passes_patterns:
            discarded_patterns += 1
            if args.verbose:
                print(f"    ✗ [no patterns] {name}")
            continue

        # Passed all filters
        assistant_count = sum(1 for m in conv["messages"] if m["sender"] == "assistant")
        passed.append({
            "conv": conv,
            "distance": distance,
            "patterns": patterns,
            "assistant_turns": assistant_count,
        })
        print(f"    ✓ [{assistant_count} turns, dist={distance}, {'+'.join(patterns)}] {name}")

    print()
    discarded_total = discarded_length + discarded_distance + discarded_patterns
    print(f"    ✗ Discarded by length (<{args.min_turns} turns): {discarded_length}")
    print(f"    ✗ Discarded by semantic distance (<{args.min_distance}): {discarded_distance}")
    print(f"    ✗ Discarded by patterns (no cognitive signal): {discarded_patterns}")
    print(f"    ✓ Passed all filters: {len(passed)}")

    # ── Ingest ──

    if passed and not args.dry_run:
        print(f"\n  Ingesting via API ({args.api})...")
        total_cells = 0

        for item in passed:
            conv = item["conv"]
            name = conv["name"][:50]
            cells = ingest_conversation(conv, args.api)
            total_cells += cells
            print(f"    {name} → {cells} cells")

        # Trigger one dream cycle — the Dreamer daemon handles the rest
        print("\n  Dreaming (1 cycle)...")
        # Scale pairs to substrate size: ~10% of possible pairs, capped at 50k
        health = _api_post(args.api, "/api/health", {}) if False else None
        pairs = min(total_cells * 50, 50000) if total_cells > 0 else 5000
        result = _api_post(args.api, "/api/consolidate", {"pairs_per_cycle": pairs})
        disc = len(result) if result else 0
        print(f"    {disc} discoveries ({pairs} pairs examined)")
        print("    (Dreamer daemon will continue consolidating in background)")

    elif args.dry_run and passed:
        print(f"\n  Would ingest {len(passed)} conversations via API (dry-run)")
        for item in passed:
            conv = item["conv"]
            md = conversation_to_markdown(conv)
            chunks = chunk_text(md)
            print(f"    {conv['name'][:50]} ({item['assistant_turns']} turns, {len(chunks)} chunks)")

    # ── Summary ──

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Total conversations: {total}")
    print(f"  Filtered (pass): {len(passed)}")
    print(f"  Discarded: {discarded_total}")
    print(f"    - By length: {discarded_length}")
    print(f"    - By semantic distance: {discarded_distance}")
    print(f"    - By patterns: {discarded_patterns}")
    if not args.dry_run:
        print(f"  Cells created: {total_cells if passed else 0}")
    else:
        print(f"  Conversations that would be ingested: {len(passed)}")
    print()


if __name__ == "__main__":
    main()
