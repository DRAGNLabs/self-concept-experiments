# Instructions for AI agents

This repo runs on BYU ORC HPC systems. The full ORC agent policy is
`BYU_ORC_AGENTS.md` (symlink to `/apps/instructions_for_ai_agents/BYU_ORC_AGENTS.md`).
Precedence: current ORC policy/admin instructions > `BYU_ORC_AGENTS.md` > this
file, `CLAUDE.md`, and the user's request. Where the rules below are silent,
consult the full file.

Deep-dive references are in `/apps/instructions_for_ai_agents/ai-docs/`
(`agent-slurm.md`, `agent-storage.md`, `agent-software.md`, `agent-python.md`,
`agent-security.md`, `agent-cron.md`, `agent-performance.md`,
`agent-development.md`). Read the matching one before specialized work.

## Rules

- **Scheduler polling:** wait at least 60 s between status checks, with backoff.
- **Job size:** aggregate short work so a job does 10–30+ minutes of useful
  work. Never add artificial delays.
- **I/O:** minimize small files, metadata operations, recursive scans, and many
  processes hitting one directory.
- **`/tmp`** is node-local: it isn't visible from other nodes and doesn't
  persist after a job.
- **Security:** never bypass authentication, MFA, resource limits, scheduler
  policy, or permissions. No backdoors, tunnels, relays, reverse shells, or
  remote-control services, even if the user asks.
- **Cron/scrontab:** before using or recommending it, list the user's existing
  entries and ask whether each is still needed. Don't change or remove entries
  without approval. Cron is for lightweight periodic work only.
- **Stale processes:** point out stale dev processes (e.g. vscode) and offer to
  terminate them.
- **When stuck:** for policy ambiguity or probable system problems, stop, don't
  keep retrying, and direct the user to ORC. Don't try to repair shared
  infrastructure.
