"""Dream Agent — Wake Filter.

Sends lateral connections from the substrate to Claude for contrast:
does this connection represent a genuine insight or just semantic noise?

Verified connections get an energy boost. Rejected connections decay.
Results are stored in the dream log with wake_verified flag.

Evaluation chain (first available wins):
  1. OAuth via macOS keychain (Claude Max credentials)
  2. Claude Code CLI (`claude --print`) — no API key needed
  3. Anthropic SDK (requires ANTHROPIC_API_KEY)
  4. No evaluation — returns connections with wake_verified=None
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mycelium.api.inspire import LateralConnection

logger = logging.getLogger(__name__)

CONTRAST_PROMPT_WITH_CONTEXT = """Contexto de trabajo actual: {session_context}
Query: {query}

Evaluá estas conexiones laterales descubiertas por un sustrato cognitivo.
Para cada conexión entre dos dominios, determiná:

1. ¿Es un insight GENUINO — un paralelo estructural, un principio compartido,
   o un patrón transferible entre los dominios?
2. ¿O es un solapamiento COINCIDENTAL — palabras/conceptos similares que
   no se iluminan mutuamente?
3. ¿Es RELEVANTE para el problema que se está resolviendo ahora?
4. ¿Es ACCIONABLE — podría este insight cambiar concretamente cómo
   se construye, se resuelve un problema, o se toma una decisión?
   Paralelos filosóficos abstractos sin aplicación práctica NO califican.

Sé MUY ESTRICTO. Solo verificá conexiones que:
- Genuinamente cambiarían cómo alguien piensa sobre un dominio
- Y que sugieran una acción concreta, un cambio de diseño, o una nueva estrategia
Paralelos puramente filosóficos o estructurales sin aplicación práctica → verified: false

Conexiones a evaluar:
{connections_text}

Respondé en JSON estricto (sin markdown, sin backticks):
[
  {{"index": 0, "verified": true/false, "confidence": 0.0-1.0, "reasoning": "una oración"}},
  ...
]"""

CONTRAST_PROMPT_NO_CONTEXT = """Query: {query}

Evaluá estas conexiones laterales descubiertas por un sustrato cognitivo.
Para cada conexión entre dos dominios, determiná:

1. ¿Es un insight GENUINO — un paralelo estructural, un principio compartido,
   o un patrón transferible entre los dominios?
2. ¿O es un solapamiento COINCIDENTAL — palabras/conceptos similares que
   no se iluminan mutuamente?
3. ¿Es ACCIONABLE — podría este insight cambiar concretamente cómo
   se construye, se resuelve un problema, o se toma una decisión?
   Paralelos filosóficos abstractos sin aplicación práctica NO califican.

Sé MUY ESTRICTO. Solo verificá conexiones que:
- Genuinamente cambiarían cómo alguien piensa sobre un dominio
- Y que sugieran una acción concreta, un cambio de diseño, o una nueva estrategia
Paralelos puramente filosóficos o estructurales sin aplicación práctica → verified: false

Conexiones a evaluar:
{connections_text}

Respondé en JSON estricto (sin markdown, sin backticks):
[
  {{"index": 0, "verified": true/false, "confidence": 0.0-1.0, "reasoning": "una oración"}},
  ...
]"""


@dataclass
class WakeResult:
    """Result of evaluating a single connection."""
    index: int
    verified: bool
    confidence: float
    reasoning: str


def _format_connections_for_prompt(connections: list[LateralConnection]) -> str:
    """Format connections as numbered text for the contrast prompt."""
    lines = []
    for i, conn in enumerate(connections):
        lines.append(
            f"{i}. [{conn.cell_domain}] ↔ [{conn.connected_to_domain}] "
            f"(sig={conn.intersection_significance}, overlap={conn.intersection_overlap})\n"
            f"   A: {conn.cell_text[:200]}\n"
            f"   B: {conn.connected_to_text[:200]}"
        )
    return "\n\n".join(lines)


def _build_prompt(
    query: str,
    connections: list[LateralConnection],
    session_context: str = "",
) -> str:
    """Build the contrast prompt from query, connections, and optional context."""
    connections_text = _format_connections_for_prompt(connections)
    if session_context:
        return CONTRAST_PROMPT_WITH_CONTEXT.format(
            session_context=session_context,
            query=query,
            connections_text=connections_text,
        )
    return CONTRAST_PROMPT_NO_CONTEXT.format(
        query=query,
        connections_text=connections_text,
    )


def _parse_claude_response(text: str, expected_count: int) -> list[WakeResult]:
    """Parse Claude's JSON response into WakeResults."""
    text = text.strip()

    # Extract JSON array from response — may be wrapped in markdown or prose
    json_match = re.search(r"\[[\s\S]*\]", text)
    if json_match:
        text = json_match.group(0)
    else:
        # Strip markdown fences as fallback
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to recover truncated JSON — find all complete objects
        recovered = []
        for m in re.finditer(r'\{[^{}]*\}', text):
            try:
                recovered.append(json.loads(m.group(0)))
            except json.JSONDecodeError:
                continue
        if recovered:
            logger.info("Recovered %d items from truncated JSON response", len(recovered))
            data = recovered
        else:
            logger.warning("Failed to parse Claude response as JSON: %s", text[:300])
            return []

    if not isinstance(data, list):
        logger.warning("Claude response is not a list: %s", type(data))
        return []

    results = []
    for item in data:
        if not isinstance(item, dict):
            continue
        results.append(WakeResult(
            index=item.get("index", len(results)),
            verified=bool(item.get("verified", False)),
            confidence=float(item.get("confidence", 0.0)),
            reasoning=str(item.get("reasoning", "")),
        ))

    return results


# ── OAuth helpers ──────────────────────────────────────

_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
_OAUTH_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
_OAUTH_API_URL = "https://api.anthropic.com/v1/messages?beta=true"
_OAUTH_BETA = "claude-code-20250219,interleaved-thinking-2025-05-14,oauth-2025-04-20"
_OAUTH_USER_AGENT = "claude-cli/2.1.83 (external, cli)"
_REFRESH_BUFFER_MS = 5 * 60 * 1000


def _read_keychain_credentials() -> dict | None:
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        creds = json.loads(result.stdout.strip())
        oauth = creds.get("claudeAiOauth", {})
        access_token = oauth.get("accessToken")
        refresh_token = oauth.get("refreshToken")
        expires_at = oauth.get("expiresAt", 0)
        if not access_token or not refresh_token:
            return None
        return {"accessToken": access_token, "refreshToken": refresh_token, "expiresAt": expires_at}
    except (json.JSONDecodeError, KeyError):
        return None


def _refresh_oauth_token(refresh_token: str) -> dict | None:
    body = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": _OAUTH_CLIENT_ID,
    }).encode()
    req = urllib.request.Request(
        _OAUTH_TOKEN_URL, data=body, method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": _OAUTH_USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return {
            "accessToken": data["access_token"],
            "refreshToken": data.get("refresh_token", refresh_token),
            "expiresAt": int(time.time()) + data.get("expires_in", 3600),
        }
    except Exception as exc:
        logger.warning("OAuth token refresh failed: %s", exc)
        return None


def _get_oauth_access_token() -> str | None:
    creds = _read_keychain_credentials()
    if creds is None:
        return None
    now_ms = int(time.time() * 1000)
    expires_ms = creds["expiresAt"]
    if expires_ms < now_ms + _REFRESH_BUFFER_MS:
        logger.info("OAuth token expired or expiring soon, refreshing")
        refreshed = _refresh_oauth_token(creds["refreshToken"])
        if refreshed is None:
            return None
        return refreshed["accessToken"]
    return creds["accessToken"]


def _call_anthropic_oauth(access_token: str, prompt: str, model: str = "claude-haiku-4-5-20251001") -> str | None:
    payload = json.dumps({
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        _OAUTH_API_URL, data=payload, method="POST",
        headers={
            "Authorization": f"Bearer {access_token}",
            "anthropic-beta": _OAUTH_BETA,
            "user-agent": _OAUTH_USER_AGENT,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        content = data.get("content", [])
        if content and content[0].get("type") == "text":
            return content[0]["text"]
        return None
    except Exception as exc:
        logger.error("OAuth API call failed: %s", exc)
        return None


# ── Evaluation backends ────────────────────────────────


def _evaluate_via_oauth(
    prompt: str,
    connections_count: int,
    model: str = "claude-haiku-4-5-20251001",
) -> list[WakeResult] | None:
    if platform.system() != "Darwin":
        return None
    access_token = _get_oauth_access_token()
    if access_token is None:
        logger.info("OAuth credentials not available, skipping OAuth backend")
        return None
    logger.info("Wake filter: using OAuth (keychain)")
    text = _call_anthropic_oauth(access_token, prompt, model)
    if text is None:
        return None
    results = _parse_claude_response(text, connections_count)
    if not results:
        logger.warning("Could not parse OAuth response")
        return None
    return results


def _evaluate_via_cli(
    prompt: str,
    connections_count: int,
) -> list[WakeResult] | None:
    """Evaluate via Claude Code CLI. Returns None if CLI not available."""
    claude_path = shutil.which("claude")
    if claude_path is None:
        logger.info("Claude CLI not found in PATH, skipping CLI backend")
        return None

    logger.info("Wake filter: using Claude Code CLI (%s)", claude_path)

    try:
        result = subprocess.run(
            [claude_path, "--print", prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        logger.info("Claude CLI not executable")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("Claude CLI timed out after 120s")
        return None

    if result.returncode != 0:
        logger.warning(
            "Claude CLI returned %d: %s",
            result.returncode, result.stderr[:200],
        )
        return None

    raw_output = result.stdout
    logger.info("Claude CLI response (%d chars)", len(raw_output))

    results = _parse_claude_response(raw_output, connections_count)
    if not results:
        logger.warning("Could not parse CLI response, raw output:\n%s", raw_output[:500])
        return None

    return results


def _evaluate_via_api(
    prompt: str,
    connections_count: int,
    model: str = "claude-sonnet-4-20250514",
) -> list[WakeResult] | None:
    """Evaluate via Anthropic SDK. Returns None if not available."""
    try:
        import anthropic
    except ImportError:
        logger.info("Anthropic SDK not installed, skipping API backend")
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.info("ANTHROPIC_API_KEY not set, skipping API backend")
        return None

    logger.info("Wake filter: using Anthropic API (model=%s)", model)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
    except Exception as exc:
        logger.error("Anthropic API call failed: %s", exc)
        return None

    return _parse_claude_response(text, connections_count)


# ── Async evaluation backends ─────────────────────────


async def _read_keychain_credentials_async() -> dict | None:
    if platform.system() != "Darwin":
        return None
    try:
        proc = await asyncio.create_subprocess_exec(
            "security", "find-generic-password", "-s", "Claude Code-credentials", "-w",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
    except (FileNotFoundError, asyncio.TimeoutError):
        return None
    if proc.returncode != 0 or not stdout:
        return None
    try:
        creds = json.loads(stdout.decode().strip())
        oauth = creds.get("claudeAiOauth", {})
        access_token = oauth.get("accessToken")
        refresh_token = oauth.get("refreshToken")
        expires_at = oauth.get("expiresAt", 0)
        if not access_token or not refresh_token:
            return None
        return {"accessToken": access_token, "refreshToken": refresh_token, "expiresAt": expires_at}
    except (json.JSONDecodeError, KeyError):
        return None


async def _refresh_oauth_token_async(refresh_token: str) -> dict | None:
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _refresh_oauth_token, refresh_token)
    except Exception as exc:
        logger.warning("Async OAuth token refresh failed: %s", exc)
        return None


async def _call_anthropic_oauth_async(access_token: str, prompt: str, model: str = "claude-haiku-4-5-20251001") -> str | None:
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _call_anthropic_oauth, access_token, prompt, model)
    except Exception as exc:
        logger.error("Async OAuth API call failed: %s", exc)
        return None


async def _evaluate_via_oauth_async(
    prompt: str,
    connections_count: int,
    model: str = "claude-haiku-4-5-20251001",
) -> list[WakeResult] | None:
    if platform.system() != "Darwin":
        return None
    creds = await _read_keychain_credentials_async()
    if creds is None:
        logger.info("OAuth credentials not available (async), skipping OAuth backend")
        return None
    now_ms = int(time.time() * 1000)
    expires_ms = creds["expiresAt"]
    access_token = creds["accessToken"]
    if expires_ms < now_ms + _REFRESH_BUFFER_MS:
        logger.info("OAuth token expired or expiring soon, refreshing (async)")
        refreshed = await _refresh_oauth_token_async(creds["refreshToken"])
        if refreshed is None:
            return None
        access_token = refreshed["accessToken"]
    logger.info("Wake filter (async): using OAuth (keychain)")
    text = await _call_anthropic_oauth_async(access_token, prompt, model)
    if text is None:
        return None
    results = _parse_claude_response(text, connections_count)
    if not results:
        logger.warning("Could not parse OAuth response (async)")
        return None
    return results


async def _evaluate_via_cli_async(
    prompt: str,
    connections_count: int,
) -> list[WakeResult] | None:
    """Evaluate via Claude Code CLI without blocking the event loop."""
    claude_path = shutil.which("claude")
    if claude_path is None:
        logger.info("Claude CLI not found in PATH, skipping CLI backend")
        return None

    logger.info("Wake filter (async): using Claude Code CLI (%s)", claude_path)

    try:
        proc = await asyncio.create_subprocess_exec(
            claude_path, "--print", prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
    except FileNotFoundError:
        logger.info("Claude CLI not executable")
        return None
    except asyncio.TimeoutError:
        logger.warning("Claude CLI timed out after 120s")
        proc.kill()
        await proc.wait()
        return None

    if proc.returncode != 0:
        logger.warning(
            "Claude CLI returned %d: %s",
            proc.returncode, stderr.decode()[:200],
        )
        return None

    raw_output = stdout.decode()
    logger.info("Claude CLI response (%d chars)", len(raw_output))

    results = _parse_claude_response(raw_output, connections_count)
    if not results:
        logger.warning("Could not parse CLI response, raw output:\n%s", raw_output[:500])
        return None

    return results


async def _evaluate_via_api_async(
    prompt: str,
    connections_count: int,
    model: str = "claude-sonnet-4-20250514",
) -> list[WakeResult] | None:
    """Evaluate via Anthropic SDK without blocking the event loop."""
    try:
        import anthropic
    except ImportError:
        logger.info("Anthropic SDK not installed, skipping API backend")
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.info("ANTHROPIC_API_KEY not set, skipping API backend")
        return None

    logger.info("Wake filter (async): using Anthropic API (model=%s)", model)

    try:
        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
    except Exception as exc:
        logger.error("Anthropic API call failed: %s", exc)
        return None

    return _parse_claude_response(text, connections_count)


# ── Main interface ─────────────────────────────────────


class WakeFilter:
    """Evaluates substrate connections via Claude contrast.

    Tries backends in order:
      1. OAuth via macOS keychain (Claude Max credentials)
      2. Claude Code CLI (`claude --print`)
      3. Anthropic SDK (requires ANTHROPIC_API_KEY)
      4. Returns empty (no evaluation available)

    Usage:
        wake = WakeFilter()
        results, backend = wake.evaluate(query, connections)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        oauth_model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self.model = model
        self.oauth_model = oauth_model

    def evaluate(
        self,
        query: str,
        connections: list[LateralConnection],
        session_context: str = "",
    ) -> tuple[list[WakeResult], str]:
        """Evaluate connections. Returns (results, backend_used).

        backend_used is one of: "cli", "api", "none"
        """
        if not connections:
            return [], "none"

        prompt = _build_prompt(query, connections, session_context)

        # Backend 1: OAuth (macOS keychain)
        results = _evaluate_via_oauth(prompt, len(connections), self.oauth_model)
        if results is not None:
            verified = sum(1 for r in results if r.verified)
            logger.info("Wake filter (OAuth): %d/%d verified", verified, len(results))
            return results, "oauth"

        # Backend 2: Claude Code CLI
        results = _evaluate_via_cli(prompt, len(connections))
        if results is not None:
            verified = sum(1 for r in results if r.verified)
            logger.info("Wake filter (CLI): %d/%d verified", verified, len(results))
            return results, "cli"

        # Backend 3: Anthropic SDK
        results = _evaluate_via_api(prompt, len(connections), self.model)
        if results is not None:
            verified = sum(1 for r in results if r.verified)
            logger.info("Wake filter (API): %d/%d verified", verified, len(results))
            return results, "api"

        # Backend 4: No evaluation available
        logger.warning("Wake filter: no backend available, returning unverified")
        return [], "none"

    async def evaluate_async(
        self,
        query: str,
        connections: list[LateralConnection],
        session_context: str = "",
    ) -> tuple[list[WakeResult], str]:
        """Async evaluate — does not block the event loop.

        Same fallback chain as evaluate(), but uses async subprocess
        and async Anthropic client.
        """
        if not connections:
            return [], "none"

        prompt = _build_prompt(query, connections, session_context)

        # Backend 1: OAuth (macOS keychain, async)
        results = await _evaluate_via_oauth_async(prompt, len(connections), self.oauth_model)
        if results is not None:
            verified = sum(1 for r in results if r.verified)
            logger.info("Wake filter (OAuth async): %d/%d verified", verified, len(results))
            return results, "oauth"

        # Backend 2: Claude Code CLI (async)
        results = await _evaluate_via_cli_async(prompt, len(connections))
        if results is not None:
            verified = sum(1 for r in results if r.verified)
            logger.info("Wake filter (CLI async): %d/%d verified", verified, len(results))
            return results, "cli"

        # Backend 3: Anthropic SDK (async)
        results = await _evaluate_via_api_async(prompt, len(connections), self.model)
        if results is not None:
            verified = sum(1 for r in results if r.verified)
            logger.info("Wake filter (API async): %d/%d verified", verified, len(results))
            return results, "api"

        # Backend 4: No evaluation available
        logger.warning("Wake filter: no backend available, returning unverified")
        return [], "none"


# ── Auto-wake (called by Dreamer callback) ─────────────


def auto_wake(
    discoveries,
    substrate,
    store,
    session_context: str = "",
    min_significance: float = 0.55,
) -> int:
    """Auto-evaluate high-significance discoveries via Claude contrast.

    Called by the Dreamer's on_discoveries callback after consolidation.
    Filters discoveries by significance, evaluates via Claude, applies
    boost/decay, persists results. Returns count of verified connections.
    """
    from mycelium.api.inspire import LateralConnection

    # Filter high-significance discoveries
    significant = [ix for ix in discoveries if ix.significance >= min_significance]
    if not significant:
        return 0

    # Get already-verified domain pairs to skip duplicates
    verified_pairs: set[frozenset[str]] = set()
    try:
        with store._conn() as conn:
            rows = conn.execute(
                """SELECT DISTINCT c1.domain, c2.domain
                   FROM dream_log dl
                   JOIN intersections ix ON dl.intersection_id = ix.id
                   JOIN cells c1 ON ix.parent_a = c1.id
                   JOIN cells c2 ON ix.parent_b = c2.id
                   WHERE dl.wake_verified = 1
                   AND c1.domain != '' AND c2.domain != ''""",
            ).fetchall()
            for r in rows:
                verified_pairs.add(frozenset({r[0], r[1]}))
    except Exception as exc:
        logger.warning("Failed to load verified domain pairs: %s", exc)

    # Build LateralConnection objects from parent cells
    connections = []
    ix_map = {}  # index → intersection for later dream_log update
    for ix in significant[:10]:  # Cap at 10 per batch
        cell_a = substrate.get_cell(ix.parent_a_id)
        cell_b = substrate.get_cell(ix.parent_b_id)
        if cell_a is None or cell_b is None:
            continue
        # Skip intra-domain (same domain, less interesting)
        if cell_a.domain and cell_b.domain and cell_a.domain == cell_b.domain:
            continue
        # Skip domain pairs that already have a verified insight
        if cell_a.domain and cell_b.domain:
            pair = frozenset({cell_a.domain, cell_b.domain})
            if pair in verified_pairs:
                continue
        idx = len(connections)
        ix_map[idx] = (ix, cell_a, cell_b)
        connections.append(LateralConnection(
            cell_text=cell_a.text[:300] if cell_a.text else "",
            cell_domain=cell_a.domain,
            cell_source=cell_a.origin.source,
            cell_confidence=round(cell_a.confidence, 3),
            cell_energy=round(cell_a.energy, 3),
            connected_to_text=cell_b.text[:300] if cell_b.text else "",
            connected_to_domain=cell_b.domain,
            connected_to_source=cell_b.origin.source,
            intersection_significance=round(ix.significance, 4),
            intersection_overlap=round(ix.overlap, 4),
            intersection_novelty=round(ix.novelty, 4),
            cell_id=str(cell_a.id),
            connected_to_id=str(cell_b.id),
        ))

    if not connections:
        return 0

    # Build query from the domains involved
    domains = set()
    for c in connections:
        if c.cell_domain:
            domains.add(c.cell_domain)
        if c.connected_to_domain:
            domains.add(c.connected_to_domain)
    query = f"cross-domain connections: {', '.join(sorted(domains))}"

    logger.info(
        "Auto-wake: evaluating %d connections (sig >= %.2f)",
        len(connections), min_significance,
    )

    # Evaluate
    wake = WakeFilter()
    results, backend = wake.evaluate(query, connections, session_context)

    if backend == "none" or not results:
        logger.info("Auto-wake: no backend available, skipping")
        return 0

    # Apply boost/decay
    BOOST_VERIFIED = 0.09
    DECAY_REJECTED = 0.03
    verified_count = 0

    for wr in results:
        if wr.index not in ix_map:
            continue
        ix, cell_a, cell_b = ix_map[wr.index]

        if wr.verified:
            cell_a.boost_energy(BOOST_VERIFIED)
            cell_b.boost_energy(BOOST_VERIFIED)
            verified_count += 1
        else:
            cell_a.energy = max(0.0, cell_a.energy - DECAY_REJECTED)
            cell_b.energy = max(0.0, cell_b.energy - DECAY_REJECTED)

        # Persist updated cells
        store.save_cell(cell_a)
        store.save_cell(cell_b)

        # Find dream_log entry for this intersection and record verification
        try:
            # Search recent entries for matching intersection_id
            with store._conn() as conn:
                row = conn.execute(
                    "SELECT id FROM dream_log WHERE intersection_id = ? "
                    "ORDER BY id DESC LIMIT 1",
                    (str(ix.id),),
                ).fetchone()
                if row:
                    store.set_wake_verification(
                        row["id"], wr.verified, wr.reasoning,
                    )
        except Exception as exc:
            logger.warning("Failed to record wake verification: %s", exc)

    logger.info(
        "Auto-wake complete: %d/%d verified (backend=%s)",
        verified_count, len(results), backend,
    )

    return verified_count
