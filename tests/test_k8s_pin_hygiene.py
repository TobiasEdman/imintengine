"""Long-lived k8s workloads that clone this repo must match the tree.

A Deployment/CronJob that does ``git clone --branch <b>`` at pod start and
then invokes ``scripts/X.py --foo`` carries an invisible dependency: a
rename or flag change in ANOTHER branch surfaces only when a pod restarts
— possibly months later, with no CI signal. Commit 33d44dd made this
concrete: it replaced the 1118-line fetch monitor under its own filename
with a tool sharing zero flags, while the campaign-dashboard Deployment
cloned main and passed nine flags none of which survived; the failure
mode was HTTP-200-over-empty-www, indefinitely, invisibly.

These tests make the dependency CI's problem instead of the cluster's
(governance rule 12: every rule gets a test). Design note: no ``git
fetch`` — in a PR against main, the working tree IS the candidate main,
so a main-pinned manifest must be satisfiable by the tree in front of us.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

LONGLIVED_KINDS = {"Deployment", "StatefulSet", "DaemonSet", "CronJob"}
_KIND_RE = re.compile(r"^kind:\s*(\w+)", re.M)
_BRANCH_RE = re.compile(r"--branch\s+([A-Za-z0-9/_.-]+)")
_SCRIPT_RE = re.compile(r"(scripts/[a-zA-Z0-9_]+\.py)")
_FLAG_RE = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]+")
_GIT_FLAGS = {"--branch", "--depth", "--sparse", "--filter",
              "--quiet", "--no-cache-dir", "--upgrade", "--index-url"}

# Flags a long-lived main-pinned manifest passes that the script does not
# declare. Each entry is a KNOWN break with an owner — not a waiver to
# hide behind. Fix the manifest or the script, then delete the row.
KNOWN_PIN_BREAKS = {
    # scheduled-fetch-cronjob passes --tiles-from, which has NEVER existed
    # in fetch_unified_tiles.py (git log -S confirms, and the manifest's
    # own comment claiming support is false). The CronJob is not applied
    # in prithvi-training-default (verified 2026-08-31), so the break is
    # latent. Tracked for a fix-or-delete decision.
    ("scheduled-fetch-cronjob.yaml", "scripts/fetch_unified_tiles.py",
     "--tiles-from"),
}

# Long-lived workloads pinned OFF main. A pin without a stated reason and
# an exit condition is a pin nobody will ever remove — the deployed
# artifact tracks a branch nobody rebases and CI never builds. Key:
# manifest filename → (branch, why, what un-pins it).
DECLARED_NON_MAIN_PINS: dict[str, tuple[str, str, str]] = {
    # EMPTY since PR #27 merged (2026-08-31): every long-lived workload
    # clones main. A new entry here needs a reason and an exit condition.
}

def _longlived_manifests():
    for f in sorted((REPO / "k8s").glob("*.yaml")):
        text = f.read_text(errors="ignore")
        if set(_KIND_RE.findall(text)) & LONGLIVED_KINDS:
            yield f, re.sub(r"\\\s*\n\s*", " ", text)  # join continuations


def test_longlived_main_pinned_manifests_match_the_tree() -> None:
    violations: list[str] = []
    for f, joined in _longlived_manifests():
        branches = _BRANCH_RE.findall(joined)
        if not branches or branches[0] != "main":
            continue  # off-main pins: see the declaration test below
        for line in joined.splitlines():
            if line.lstrip().startswith("#"):
                continue
            m = _SCRIPT_RE.search(line)
            if not m:
                continue
            rel = m.group(1)
            src_path = REPO / rel
            if not src_path.is_file():
                violations.append(f"{f.name}: invokes {rel}, absent from tree")
                continue
            src = src_path.read_text(errors="ignore").replace("'", '"')
            for flag in sorted(set(_FLAG_RE.findall(line)) - _GIT_FLAGS):
                if f'"{flag}"' in src:
                    continue
                if (f.name, rel, flag) in KNOWN_PIN_BREAKS:
                    continue
                violations.append(
                    f"{f.name}: passes {flag} to {rel}, which does not "
                    f"declare it")

    assert not violations, (
        "Långlivade main-pinnade workloads är inte kompatibla med detta "
        "träd — en pod-restart skulle tyst degradera:\n  "
        + "\n  ".join(violations)
        + "\nFixa manifestet i SAMMA commit som skriptändringen, eller döp "
          "om skriptet istället för att återanvända namnet."
    )


def test_longlived_non_main_pins_are_declared() -> None:
    undeclared: list[str] = []
    for f, joined in _longlived_manifests():
        branches = _BRANCH_RE.findall(joined)
        if not branches or branches[0] == "main":
            continue
        declared = DECLARED_NON_MAIN_PINS.get(f.name)
        if declared is None or declared[0] != branches[0]:
            undeclared.append(f"{f.name} -> {branches[0]}")

    assert not undeclared, (
        "Långlivade workloads pinnade off-main utan deklaration:\n  "
        + "\n  ".join(undeclared)
        + "\nLägg till i DECLARED_NON_MAIN_PINS med skäl + "
          "avpinnings-villkor, eller repointa till main."
    )
