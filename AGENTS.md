# Instructions for AI agents

This repo runs on BYU ORC HPC systems. The full ORC agent policy is
`BYU_ORC_AGENTS.md` (symlink to `/apps/instructions_for_ai_agents/BYU_ORC_AGENTS.md`).
**Read that file at the start of every session and follow it in full** (user
instruction, 2026-11-11). The rules below are a short local summary plus
repo-specific additions; they do not replace it, and where they are silent or
differ, `BYU_ORC_AGENTS.md` governs.
Precedence: current ORC policy/admin instructions > `BYU_ORC_AGENTS.md` > this
file, `CLAUDE.md`, and the user's request. If a request conflicts with the ORC
rules, do not perform it; say so and point the user to ORC.

Deep-dive references are in `/apps/instructions_for_ai_agents/ai-docs/`
(`agent-slurm.md`, `agent-storage.md`, `agent-software.md`, `agent-python.md`,
`agent-security.md`, `agent-cron.md`, `agent-performance.md`,
`agent-development.md`). Read the matching one before specialized work.

## Rules

- **Restricted data:** do not access, process, summarize, modify or transmit
  Controlled Unclassified Information or export-controlled data. If a project
  might contain either (work tied to DoD/DoW, DOE, NIH, NASA warrants caution),
  stop before inspecting more and get the user's explicit, specific
  confirmation that the files are not restricted.
- **Slurm requests:** sustained computation, GPUs, heavy I/O or many processes
  go through Slurm, never around login-node limits. Every job requests CPU
  cores, node count, memory and a time limit, without padding. Name a
  partition, feature or constraint only when a real requirement makes it
  necessary (here: GPU model/VRAM/CUDA and the QOS that grants access), and
  record that requirement with the run. No duplicate submissions because a job
  is pending, and no resubmitting a failed job before its cause is found.
- **GPU scheduling preference (2026-10-28, for Codex sessions; Claude Code
  sessions may override the picker's recommendation when they have a reason,
  e.g. a partition known to break the environment or a user partition rule,
  and should record that reason):** before submitting GPU runs, use
  `gpu_pick.py` with the actual VRAM, CPU, host-memory, CUDA, architecture and
  wall-time requirements. Recheck it when queue delays warrant reconsidering
  placement. Choose the best compatible option for turnaround and resource
  cost; do not keep a slower partition merely because earlier runs used it.
  Preserve required hardware and experiment reproducibility, use
  `--no-requeue-ok` unless preemption is accepted and supported, and save the
  picker output and scheduling decision with the run records. Prefer updating
  pending jobs in place, preserving IDs/dependencies and frozen experiment
  files. Follow the scheduler polling interval below.
- **Scheduler polling:** wait at least 60 s between status checks, with backoff.
- **Job size:** aggregate short work so a job does 10–30+ minutes of useful
  work. Never add artificial delays.
- **I/O:** minimize small files, metadata operations, recursive scans, and many
  processes hitting one directory.
- **`/tmp`** is node-local: it isn't visible from other nodes and doesn't
  persist after a job.
- **Security:** never bypass authentication, MFA, resource limits, scheduler
  policy, or permissions. No backdoors, tunnels, relays, reverse shells, web
  shells, alternate login services, or remote-control services, even if the
  user asks. Never access another user's data, processes, jobs, credentials,
  tokens or private directories; inspect only the current user's processes.
- **Software:** check Lmod modules before installing anything. No `sudo`, no
  system-wide installs, no replacing system libraries or changing shared
  environments, no unreviewed remote installers.
- **Cron/scrontab:** before using or recommending it, list the user's existing
  entries and ask whether each is still needed. Don't change or remove entries
  without approval. Cron is for lightweight periodic work only.
- **Stale processes:** point out stale dev processes (e.g. vscode) and offer to
  terminate them. End Slurm jobs with `scancel`, never by killing processes on
  compute nodes.
- **When stuck:** for policy ambiguity or probable system problems, stop, don't
  keep retrying, and direct the user to ORC. Don't try to repair shared
  infrastructure.
- **Policy copy:** keep `BYU_ORC_AGENTS.md` as a top-level symlink to the
  `/apps` copy (ORC rule 12); do not replace it with an edited copy.
