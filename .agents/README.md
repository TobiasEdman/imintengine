# `.agents/` — vendor-neutral task-claim convention

Coordination substrate for multi-agent work in this repo. Any runtime — Claude Code, OpenAI Codex CLI, Mistral, a human with an editor — reads and writes the same JSON files. Coordination via git push collisions, not vendor-specific orchestration.

Spec: [`multi-agentic-public`](https://github.com/TobiasEdman/multi-agentic-public) — `docs/specs/tier2_multi_agent.md` §A.

## Layout

```
.agents/
├── README.md     # this file
├── schema.json   # JSON Schema for tasks/*.json (draft 2020-12)
└── tasks/
    ├── 0001.json   # one task = one file (zero-padded 4-digit id)
    ├── 0002.json
    └── archive/
        └── YYYY-MM/   # tasks completed > 30 days ago
```

## Lock protocol

1. **Claim.** Agent picks a `pending` task, writes `status=claimed`, `claimed_by`, `runtime` (`claude|codex|mistral`), `claimed_at`. Commits, pushes.
2. **Push wins → lock acquired.** Push fails with non-fast-forward → another agent got there first. Run `git pull --rebase` and pick a different task.
3. **Complete.** When the work lands (PR merged), set `status=completed`, `completed_at`, `updated_at`. Commit, push.

Reference CLI: install `agentic-task` from the public mirror with `uv tool install --from /path/to/multi-agentic-public .` (or `pip install -e .` from within it). Then:

```bash
AGENT_ID=te-claude AGENT_RUNTIME=claude agentic-task claim  .
agentic-task list  . [--status pending|claimed|in_progress|completed|blocked|abandoned]
agentic-task complete  . <task-id>
```

## Why this exists in this repo

Per `~/.claude/CLAUDE.md` §10 (one session per branch per cwd), concurrent Claude Code sessions in the same repo race on the git index. The `.agents/tasks/` lock protocol prevents two sessions from claiming the same work — combined with `git worktree`, makes parallel agent work mechanically safe.

## What this is *not*

- **Not a queue** with priorities or ordering beyond lowest-id-first.
- **Not authoritative for in-flight work.** Once `claimed`, work happens on a branch and the PR is source of truth.
- **Not a replacement for an issue tracker.** Tasks here are short-lived coordination tickets.

Bootstrapped 2026-06-10 per `docs/lessons/concurrent_sessions_diagnosis.md`.
