# Finding — wetness aux (markfukt) is neutral on NFI forest-type

Ablation (2026-08-06): finetune v8b + SLU markfukt (soil moisture) as an 11th
aux channel, identical 600M/504 regime, warm-start (input-embed 10→11), 23-class
NMD2018 labels — markfukt the only variable.

| Metric | v8b | v8b+markfukt | Δ |
|---|---|---|---|
| val mIoU (23-class) | 0.535 | 0.5527 | **+0.018** |
| NFI overall (5-class) | 0.4650 | 0.4661 | +0.001 |
| NFI kappa | 0.3392 | 0.3389 | −0.0003 |
| NFI forest-type acc | 0.4310 | 0.4321 | +0.001 |

Per-class F1 (NFI): tall +0.000, gran −0.014, löv +0.007, bland +0.006,
non-forest +0.006.

**Conclusion:** markfukt lifts val mIoU (+0.018) but the gain is in the
wetland/open-land classes — which collapse into the single "non-forest" bucket
in the NFI forest-type suite, so field-truth forest accuracy is flat (within
noise on 944 plots). Keep markfukt if wetland/open-land mapping matters; it does
NOT change the v8b-vs-NMD forest-type story. Checkpoint:
/cephfs/checkpoints/v8b_markfukt/best_model.pt.
