# AGENTS.md

Codex (and any other coding agent): read [`CLAUDE.md`](CLAUDE.md) in full at
session start. It is the agent operating contract for this project — orientation,
the non-negotiable invariants, the dev loop, test/lint/run commands, git workflow,
review loop, and never-do list — and the same rules apply to you.

`CLAUDE.md` points into [`docs/`](docs/) for the authoritative design docs; read
them in the order it gives. At the bottom it instructs loading
`internal/docs/CLAUDE-private.md` if it exists (private development repo only — it
holds project-specific state and private planning context); do that too. Loading
both files gives you the same picture Claude Code gets at session start.

## Tool-specific notes

None at this time. Behaviour/sandbox/interaction tweaks that apply to *only* one
tool (not all agents) belong here.
