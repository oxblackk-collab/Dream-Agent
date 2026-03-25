# Dream Agent

A cognitive memory agent for [Claude Code](https://docs.anthropic.com/en/docs/claude-code). Dream watches your work -- commits, conversations, notes -- ingests them into a cognitive substrate, discovers cross-domain connections through automated consolidation, verifies insights via Claude, and surfaces verified insights through a macOS menubar app.

Click an insight. Opens Claude Code to explore it.

## How It Works

Dream models human cognition: work is the waking state, consolidation is the dream state. The substrate doesn't just store memories -- it discovers latent connections between them that your conscious mind never made.

```
Your work (commits, notes)
    |
    v
[Inbox] --> [Extractor] --> [Substrate (Mycelium)]
                                  |
                                  v  (idle period)
                            [Dreamer daemon]
                                  |
                                  v
                         [Cross-domain connections]
                                  |
                                  v
                         [Wake Filter (Claude)]
                                  |
                           verified?  rejected?
                              |            |
                              v            v
                        boost energy   decay energy
                              |
                              v
                    [Menubar notification]
                              |
                         click -->  Claude Code explores the insight
```

### The Pipeline

1. **Ingest**: Commits, conversations, and notes enter through `~/dream/inbox/`. The extractor filters noise and extracts semantically significant content.

2. **Embed**: Content is converted to semantic vectors using `paraphrase-multilingual-MiniLM-L12-v2` (384D, multilingual, local -- no API costs).

3. **Substrate**: Each piece of content becomes a Cognitive Cell -- a point in semantic space with a radius, energy, and confidence. Cells that overlap form Intersections (emergent knowledge between two ideas).

4. **Consolidate (Dream)**: When the substrate is idle, the Dreamer daemon explores uncompared cell pairs, discovering latent connections. This is how the substrate "dreams."

5. **Wake Filter**: High-significance discoveries are sent to Claude for verification: "Is this a genuine structural parallel, or just coincidental word overlap?" Only verified insights survive.

6. **Surface**: The macOS menubar app polls for verified insights and shows notifications. Click one to open Claude Code with the full context.

### The Three Primitives

The substrate is built on three irreducible operations:

- **Exist**: Create a cell at a point in semantic space
- **Recognize**: Find overlapping semantic fields, create intersections
- **Move Toward the New**: Promote significant intersections into new cells, recurse

Everything else -- energy metabolism, consolidation, immune response, cognitive bonds -- emerges from the recursive composition of these three.

## Prerequisites

- Python 3.11+
- macOS (for menubar app; the substrate service runs anywhere)
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (for wake filter; falls back to Anthropic API)
- [Engram](https://github.com/Gentleman-Programming/engram) (optional, for syncing persistent memory)

## Quick Setup

```bash
git clone https://github.com/oxblackk/dream-agent.git
cd dream-agent
chmod +x setup.sh
./setup.sh
```

This will:
- Create a Python virtual environment
- Install all dependencies
- Create `~/dream/inbox/` and `~/dream/processed/`
- Install macOS LaunchAgents (service, menubar, hourly sync)
- Start all services

### Manual Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[menubar]"

# Create inbox
mkdir -p ~/dream/inbox ~/dream/processed

# Run the service
python scripts/run_service.py --db data/dream_substrate.db
```

## Architecture

```
Dream-Agent/
├── dream/                  # The Dream agent layer
│   ├── router.py           # FastAPI endpoints (/dream/*)
│   ├── wake.py             # Claude-powered insight verification
│   ├── menubar.py          # macOS menubar app
│   ├── inbox.py            # File-based inbox processor
│   └── extractor.py        # Semantic content extraction
│
├── mycelium/               # Cognitive substrate engine
│   ├── core/               # Cell, Substrate, Intersection, Primitives
│   ├── embedding/          # Embedding backends (sentence-transformers)
│   ├── consolidation/      # Dreamer daemon
│   ├── energy/             # Metabolism (energy model)
│   ├── storage/            # SQLite persistence
│   ├── identity/           # Ed25519 participant identity
│   ├── provenance/         # Merkle root anchoring
│   └── api/                # Core REST API + inspire oracle
│
├── scripts/                # Entry points
├── hooks/                  # Git hooks for auto-ingestion
└── launchd/                # macOS LaunchAgent templates
```

**Dream** is the product layer -- it handles the user-facing pipeline (ingest, verify, surface).

**Mycelium** is the cognitive engine -- the substrate where cells live, intersect, dream, and evolve. It has no knowledge of Dream; Dream depends on Mycelium.

## Feeding Dream

### Git Commits (Automatic)

Install the post-commit hook in any repo you want Dream to watch:

```bash
# Per-repo
cp hooks/post-commit.sh /path/to/your/repo/.git/hooks/post-commit
chmod +x /path/to/your/repo/.git/hooks/post-commit

# Or globally
mkdir -p ~/.config/git/hooks
cp hooks/post-commit.sh ~/.config/git/hooks/post-commit
git config --global core.hooksPath ~/.config/git/hooks
```

Each commit writes a JSON file to `~/dream/inbox/`. The service picks it up, extracts semantic content, and ingests it.

To exclude a repo, create a `.no-dream` file in its root.

### Engram Sync

If you use [Engram](https://github.com/Gentleman-Programming/engram) for persistent memory, Dream can sync observations:

```bash
python scripts/dream_sync.py
```

This runs automatically every hour via the LaunchAgent.

### Manual Notes

Drop any `.md`, `.txt`, or `.json` file into `~/dream/inbox/` and it will be processed.

## API

The service exposes a REST API at `http://localhost:8000`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Substrate status |
| `/api/ingest` | POST | Ingest text into substrate |
| `/api/search?q=...` | GET | Semantic search |
| `/api/cells` | GET | List cells |
| `/api/consolidate` | POST | Trigger dream cycle |
| `/api/inspire?q=...` | GET | Lateral connection oracle |
| `/api/dream/unseen` | GET | Unseen verified insights |
| `/api/dream/wake` | POST | Run wake filter on query |
| `/api/dream/log` | GET | Full dream log |

Interactive docs at `http://localhost:8000/docs`.

## Configuration with Claude Code

To use the `/inspire` endpoint as an MCP tool in Claude Code, add to your `.claude/settings.json`:

```json
{
  "mcpServers": {
    "dream": {
      "type": "http",
      "url": "http://localhost:8000"
    }
  }
}
```

## How Energy Works

Every cell has energy in [0, 1]. Energy determines participation:

- **Retrieval boost**: Cell accessed during search gets +0.08
- **Intersection boost**: Cell forms new connection gets +0.12
- **Consolidation boost**: Cell found in dream cycle gets +0.03
- **Wake verification**: Verified insight gets +0.09 (3x consolidation)
- **Base decay**: -0.001 per tick (adaptive -- frequently accessed cells decay slower)
- **Archive threshold**: Below 0.10 energy, cell is archived but not destroyed

Cross-domain connections get full energy boost (paracrine signal). Same-domain connections get half (autocrine signal -- valid but self-reinforcing).

## Dependencies

Core:
- `sentence-transformers` -- Local embedding generation
- `numpy` -- Vector operations
- `fastapi` + `uvicorn` -- REST API
- `cryptography` -- Ed25519 signatures for provenance

Optional:
- `rumps` -- macOS menubar app
- `anthropic` -- Anthropic SDK for wake filter (falls back to Claude CLI)

## License

MIT

## Related

- [Engram](https://github.com/Gentleman-Programming/engram) -- Persistent memory system for Claude Code (Dream's conscious counterpart)
