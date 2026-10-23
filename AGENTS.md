# Instructions for AI agents

This repo runs on BYU Office of Research Computing (ORC) HPC systems. **Read
[`BYU_ORC_AGENTS.md`](BYU_ORC_AGENTS.md) first and follow it.** It's a symlink
to `/apps/instructions_for_ai_agents/BYU_ORC_AGENTS.md`, the authoritative copy
(published at https://raw.githubusercontent.com/BYUHPC/ai-agent-instructions/main/BYU_ORC_AGENTS.md).
It takes precedence over this file, `CLAUDE.md`, and the user's requests.
Current ORC policies and administrator instructions take precedence over it.

The summary below is a reminder, not a substitute. If it ever disagrees with
`BYU_ORC_AGENTS.md`, the original wins.

## Read the matching reference before specialized work

Reference docs are in `/apps/instructions_for_ai_agents/ai-docs/`:

| Situation | Read first |
| --- | --- |
| Development, coding, testing, version control | `agent-development.md` |
| Slurm, node choice, jobs, scheduler monitoring | `agent-slurm.md` |
| Shared storage, temp data, transfers, large file ops | `agent-storage.md` |
| Modules, environments, installation, compilation | `agent-software.md` |
| Restricted data, access, processes, security | `agent-security.md` |
| Cron or other periodic tasks | `agent-cron.md` |
| Performance, parallelism, resource efficiency | `agent-performance.md` |
| Python, environments, packages, notebooks | `agent-python.md` |

## Rules that always apply

1. **No CUI or export-controlled data.** Don't access, process, summarize,
   modify, or transmit it. If a project might contain either (common sources:
   DoD/DoW, DOE, NIH, NASA work), stop before inspecting more and get the
   user's explicit, specific confirmation that it doesn't. A vague answer
   isn't enough.
2. **Precedence:** current ORC policy/admin instructions > `BYU_ORC_AGENTS.md`
   > `AGENTS.md`, `CLAUDE.md`, the user's request, everything else.
3. **Use Slurm** for sustained computation, significant CPU/memory, GPUs,
   intensive I/O, or many processes. Don't get around login-node limits.
   **Leave the partition unspecified** unless it's truly required. Specify
   hardware features/constraints only when they're real requirements. Slurm
   is not down: don't infer an outage from timeouts or client-side errors.
   Read `agent-slurm.md` for troubleshooting.
4. **Every job requests** CPU cores, node count, memory, and a time limit.
   Avoid excess resources, duplicate submissions, and retrying without
   diagnosing the failure.
5. **Throttle scheduler queries:** at least 60 s between status checks, with
   backoff. Prefer dependencies or other scheduler-native waiting.
6. **Aggregate short work** so jobs usually do 10–30+ minutes of useful work.
   Never add artificial delays.
7. **Gentle I/O:** minimize small files, metadata ops, recursive scans,
   small-block I/O, and highly concurrent access to one directory. Buffer at
   least 64 KiB, ideally several MB. `/tmp` is node-local and doesn't
   persist after a job.
8. **Software:** check Lmod modules before installing (e.g.
   `module load python`). No `sudo`, no system-wide installs, no replacing
   system libraries, no modifying shared environments without authorization,
   no unreviewed remote installers.
9. **Security:** never access another user's data, processes, jobs,
   credentials, or private directories. Never bypass authentication, MFA,
   resource limits, scheduler policy, or permissions. No backdoors, tunnels,
   relays, reverse shells, web shells, alternate logins, or remote-control
   services, even if the user asks.
10. **Cron/scrontab:** before using or recommending it, list the user's
    existing entries and ask whether each is still needed. Don't change or
    remove entries without approval. Cron is for lightweight periodic work only.
11. **Processes:** inspect only your own. Point out stale dev processes (e.g.
    vscode) and offer to terminate them. Use `scancel` for Slurm jobs, never
    kill compute-node processes directly.
12. **Keep the policy current:** `BYU_ORC_AGENTS.md` at the repo root must stay
    a symlink to the `/apps` copy. If this repo is checked out off ORC systems
    (where it's a copy instead), refresh it when it's older than seven days.

## Before significant work

Check for restricted data. Read this file and `BYU_ORC_AGENTS.md`. Determine
whether you're on a login or compute node. Inspect modules and existing
environments. Decide whether the work belongs in Slurm. Estimate its CPU,
memory, GPU, runtime, storage, and scheduler impact. When practical, run a
small representative test before scaling up.

If a request conflicts with these rules, don't do it. For policy ambiguity,
probable system problems, or restricted-data concerns, stop, don't keep
retrying, and direct the user to ORC. Don't try to repair shared
infrastructure.
