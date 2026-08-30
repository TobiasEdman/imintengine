# Finding — what the VPP phenology repair actually bought

**Status:** measured · **Date:** 2026-08-29 · **Owner:** Tobias
**Question:** the 2026-08-26/27 repair filled the VPP channels on the ~28% of
tiles that had been training on five zero-filled channels. Are we better or
worse than before?

**Answer in one line:** better, but by less than a single run can certify —
**+0.0039** aggregate val mIoU, with the gain concentrated in *bare and
sparsely-vegetated* classes rather than the crops the hypothesis predicted.

## The only valid comparison

`v8b_nmd2023_long`, pre-repair vs post-repair. Same checkpoint dir, same
28-class NMD2023 sidecar labels, same 30-epoch warm-start schedule, same
backbone. One variable.

| Estimator | pre | post | Δ |
|---|---|---|---|
| best epoch | 0.4846 | 0.4892 | +0.0046 |
| mean of last 5 epochs | 0.4826 | 0.4866 | **+0.0039** |

The last-5 mean is the estimator to quote — the best-epoch snapshot is noisy
enough to invert individual classes (see below). Both sit below the 0.005
threshold used to certify the ERA5 effect, and unlike ERA5 there is one run per
arm with no seed replication.

**But the direction is not noise.** 21 of 27 classes improved, 6 declined
(worst: trindsäd −0.0058). Under a no-effect null that split is ~p 0.004 on a
sign test. The effect is real and small.

## Where it landed

| Group | mean Δ | n |
|---|---|---|
| **open / bare land** | **+0.0072** | 6 |
| crops | +0.0038 | 11 |
| forest | +0.0015 | 5 |

Largest individual gains: öppen mark utan vegetation +0.0158, majs +0.0150,
torvtäkt +0.0149, sockerbetor +0.0107, korn +0.0086, öppen mark +0.0085,
råg +0.0078.

## Why bare ground, not crops

The hypothesis was that phenology timing separates crops, so crops should gain
most. They gained, but half as much as bare and sparse land. The mechanism that
explains the ordering:

**For bare ground, peat extraction and unvegetated surfaces, "no growing
season" is the discriminating signal — and a zero-filled channel is
indistinguishable from it.** With 28% of tiles zero-filled, absence-of-phenology
meant either "genuinely no vegetation" or "we failed to fetch", so the model
could not use it. The repair made absence informative again. That is a larger
correction than sharpening an already-usable signal, which is what the repair
did for crops.

Consistent with this: the crops that gained are the ones with *distinctive*
seasons — majs (C4, late), sockerbetor (late), råg (winter cereal, so the
year-1 autumn frame carries it). The ones that did not move are either
saturated (vete at 0.756) or spectrally confusable grass/legume types
(slåttervall, bete, trindsäd). Forest barely moved, as expected: evergreen
phenology is flat and forest is separated mostly by spectra plus the height
aux.

## Method traps (both cost a wrong conclusion first)

**1. Do not read per-class deltas off the best epoch.** At best-epoch the same
comparison reports potatis −0.0185 and lövskog −0.0123; averaged over the last
five epochs they are +0.0027 and −0.0002. Single-epoch per-class IoU on
low-support classes is mostly variance.

**2. Do not epoch-match runs with different total budgets.** Comparing the
30-epoch post-repair run against 10- and 20-epoch pre-repair runs at equal
epoch counts makes the repair look like a *regression* (−0.014 at ep10,
−0.006 at ep20). It is an artefact: each cosine schedule anneals to its own
horizon, so at epoch 10 the 10-epoch run is at lr 2.2e-11 — fully converged —
while the 30-epoch run is at 7.96e-05, still hot. Only same-schedule
comparisons are valid, which is why `v8b_nmd2023_long` is the only usable pair.

## What this does and does not justify

The repair was a **correctness** fix: `unified_dataset` fills a missing channel
with the nodata sentinel and reports nothing, so 28% of tiles trained on five
zero channels while holdout was 100% populated — silent train/serve skew. That
justified it regardless of the metric.

It does **not** justify an expectation of large mIoU gains, and the number here
should be quoted as +0.004, not rounded up. For scale, the soil-moisture aux
(`--enable-markfukt`) measures at **~+0.008 to +0.012** on matched pairs —
two to three times larger. See
[`label_source_ladder.md`](label_source_ladder.md).

## Reproducing

Both logs are on the PVC; the pre-repair copy is the reason the backup was
taken before the retrain overwrote it.

```
pre:  /cephfs/checkpoints/backup_prephenology_20260828/v8b_nmd2023_long/training_log.json
post: /cephfs/checkpoints/v8b_nmd2023_long/training_log.json
```

Per-class IoU lives on each epoch entry (`epochs[].per_class_iou`); take the
mean of the last five epochs on both sides and difference them. Note that
reading these over the dashboard's HTTP endpoint returns a *normalized* legacy
shape — read the files directly.
