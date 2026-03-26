"""Semantic extraction for Dream agent inbox processing.

Extracts the meaningful signal from raw inputs — decisions made, problems found,
tensions unresolved, connections unexpected. Filters out mechanical noise.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class IngestPayload:
    """Ready-to-ingest content for the substrate."""
    text: str
    source: str
    participant_id: str
    domain: str


ContentType = Literal["commit", "chat", "freetext"]


def detect_content_type(path: Path) -> ContentType:
    """Detect content type from file extension and content heuristics."""
    text = path.read_text(encoding="utf-8", errors="replace")

    # JSON with type field
    if path.suffix == ".json":
        try:
            data = json.loads(text)
            if isinstance(data, dict) and data.get("type") == "commit":
                return "commit"
        except json.JSONDecodeError:
            pass

    # Chat transcript patterns
    chat_patterns = [
        r"^(Human|Assistant|User|Claude)\s*:",
        r"^\*\*(Human|Assistant|User|Claude|0x\w+)\*\*:",
        r"^<(human|assistant)>",
    ]
    lines = text.split("\n")[:50]  # Check first 50 lines
    chat_matches = sum(
        1 for line in lines
        if any(re.match(p, line.strip(), re.IGNORECASE) for p in chat_patterns)
    )
    if chat_matches >= 3:
        return "chat"

    return "freetext"


def extract_from_commit(data: dict) -> list[IngestPayload]:
    """Extract semantically significant content from a commit.

    Filters out mechanical changes. Combines commit message with
    significant diff hunks into ingestible chunks.
    """
    message = data.get("message", "").strip()
    diff = data.get("diff", "").strip()
    repo = data.get("repo", "unknown")
    branch = data.get("branch", "")

    # Skip trivial commits
    skip_patterns = [
        r"^merge\s",
        r"^bump\s+version",
        r"^update\s+lock",
        r"^chore\(deps\)",
        r"^auto-format",
        r"^wip$",
    ]
    if any(re.match(p, message, re.IGNORECASE) for p in skip_patterns):
        return []

    # Skip if diff is only lockfiles, generated files, etc.
    noise_files = {
        "package-lock.json", "yarn.lock", "Cargo.lock", "go.sum",
        "uv.lock", "poetry.lock", ".gitignore", "Pipfile.lock",
    }
    diff_files = re.findall(r"^diff --git a/(.+?) b/", diff, re.MULTILINE)
    significant_files = [f for f in diff_files if Path(f).name not in noise_files]
    if not significant_files and diff_files:
        return []

    # Build the ingestible text
    parts = [f"Commit in {repo}" + (f" ({branch})" if branch else "")]
    parts.append(f"Message: {message}")

    # Extract significant diff hunks (skip binary, skip huge diffs)
    if diff and len(diff) < 10000:
        # Keep only added/removed lines (context is noise for semantics)
        meaningful_lines = []
        for line in diff.split("\n"):
            if line.startswith("diff --git"):
                file_path = line.split(" b/")[-1] if " b/" in line else ""
                if Path(file_path).name not in noise_files:
                    meaningful_lines.append(f"\nFile: {file_path}")
            elif line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
                meaningful_lines.append(line)

        if meaningful_lines:
            diff_summary = "\n".join(meaningful_lines[:100])  # Cap at 100 lines
            parts.append(f"Changes:\n{diff_summary}")

    text = "\n".join(parts)

    return [IngestPayload(
        text=text,
        source=f"dream:commit:{repo}",
        participant_id=data.get("author", ""),
        domain="session_work",
    )]


def extract_from_chat(text: str, source_file: str = "") -> list[IngestPayload]:
    """Extract semantically significant fragments from a chat transcript.

    Looks for moments of tension, decision, discovery, or self-recognition.
    Filters out routine code exchange and pleasantries.
    """
    # Split into exchanges (Human + Assistant pairs)
    exchanges = _split_chat_exchanges(text)

    # Filter to significant exchanges
    significant = []
    for exchange in exchanges:
        content = exchange.lower()
        # Skip short exchanges
        if len(content.split()) < 30:
            continue
        # Skip pure code dumps
        code_ratio = content.count("```") / max(len(content.split()), 1)
        if code_ratio > 0.01:  # More backtick-heavy than text
            continue
        # Look for signal words that indicate meaningful content
        signal_words = [
            "decision", "decided", "chose", "because", "tradeoff", "tension",
            "problem", "issue", "discovered", "realized", "insight",
            "architecture", "pattern", "design", "why", "important",
            "mistake", "wrong", "correct", "learned", "surprising",
            "decisión", "elegí", "porque", "tensión", "problema",
            "descubrí", "aprendí", "importante", "error", "sorprendente",
        ]
        signal_count = sum(1 for w in signal_words if w in content)
        if signal_count >= 2:
            significant.append(exchange)

    if not significant:
        # If nothing matched signal words, take the whole text as-is
        # (it may be a curated transcript)
        return [IngestPayload(
            text=text[:3000],  # Cap at ~500 words
            source=f"dream:chat:{source_file}" if source_file else "dream:chat",
            participant_id="",
            domain="chat_session",
        )]

    payloads = []
    for exchange in significant:
        payloads.append(IngestPayload(
            text=exchange[:3000],
            source=f"dream:chat:{source_file}" if source_file else "dream:chat",
            participant_id="",
            domain="chat_session",
        ))

    return payloads


def extract_from_freetext(text: str, source_file: str = "") -> list[IngestPayload]:
    """Extract from notes, ideas, journal entries. Minimal filtering."""
    text = text.strip()
    if len(text) < 50:
        return []

    return [IngestPayload(
        text=text[:5000],  # Cap at ~800 words
        source=f"dream:note:{source_file}" if source_file else "dream:note",
        participant_id="",
        domain="free_form",
    )]


def extract(path: Path) -> list[IngestPayload]:
    """Main entry point: detect content type and extract."""
    content_type = detect_content_type(path)

    if content_type == "commit":
        data = json.loads(path.read_text(encoding="utf-8"))
        return extract_from_commit(data)
    elif content_type == "chat":
        text = path.read_text(encoding="utf-8", errors="replace")
        return extract_from_chat(text, source_file=path.name)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        return extract_from_freetext(text, source_file=path.name)


def _split_chat_exchanges(text: str) -> list[str]:
    """Split chat text into individual exchanges."""
    # Try common patterns
    patterns = [
        r"(?=^(?:Human|User|Assistant|Claude)\s*:)",  # "Human:" style
        r"(?=^\*\*(?:Human|User|Assistant|Claude|0x\w+)\*\*:)",  # "**Human**:" style
        r"(?=^#{1,3}\s)",  # Markdown headers as separators
        r"(?=^---\s*$)",  # Horizontal rules
    ]

    for pattern in patterns:
        parts = re.split(pattern, text, flags=re.MULTILINE)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 1:
            # Group into pairs (human + assistant)
            exchanges = []
            for i in range(0, len(parts) - 1, 2):
                exchange = parts[i]
                if i + 1 < len(parts):
                    exchange += "\n\n" + parts[i + 1]
                exchanges.append(exchange)
            return exchanges

    # Fallback: return whole text as single exchange
    return [text]
