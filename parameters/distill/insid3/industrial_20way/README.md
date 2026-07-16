# Industrial 20-way experiment (ARCHIVED)

Superseded by standard single-fold FSS (`../Industrial.yaml`, fold 0). Kept for the
record because the design worked and produced a clean trend; it was rolled back only
because 320-shot is compute-infeasible (OOM at every resolution that keeps quality).

## Design
- **fold=null**: train the student on ALL 20 defect classes in ONE run per shot.
- At **test**, discard the non-query-fold class logits via
  `DatasetIndustrial.restrict_logits_to_fold(logits, gt)` (query fold = `(c-1)%nfolds`;
  masks disallowed classes to `-inf`), called in `distillfss/test.py` before argmax.
- Industrial is MVTec-style single-class-per-image (`industrial.py` `read_mask` sets the
  whole region to the image's one label); 20 classes, 4 folds of 5.

## Key finding
20-way needs **`logit_mode: logits`, NOT `double_softmax`**. double_softmax (21-way
softmax over bg+20fg, then FocalLoss softmaxes again = capped confidence) is
winner-take-all: one class (hazelnut) soaks all mass, the other 19 collapse to 0
(flat fg 3.3). Harmless at 2-3 classes (ISIC/Nucleus), fatal at 20. `logits`
(bg=0, raw fg logits, single softmax in loss) un-collapses it. **Class count, not the
single-class-per-image shape, drives the collapse.**

## Results (fold-restricted fg Jaccard, logit_mode=logits)
| shots | @512 | @384 |
|------:|:----:|:----:|
| 40    | 6.99 | 5.33 |
| 80    | 13.90| 8.21 |
| 160   | 17.27| 10.08|
| 320   | OOM  | OOM  |

512 clearly beats 384 (resolution-sensitive). 320-shot OOMs at both (paired
substitutor clones the full support batch; ~22.6 GB, short by ~720 MiB on a 24 GB card).

## Files
- `Industrial_20way.yaml` — base config, shots [40,80,160,320] @512.
- `Industrial_highshot.yaml` — heavy shots [160,320] @384 (attempt to fit 320; still OOM'd).

## Run directories (tagged `EXPERIMENT_20WAY.txt`)
- `out/2026-07-04_01-17-22_DistillINSID3Industrial` — @512: 40/80/160 done (6.99/13.90/17.27), 320 OOM. **Best 512 results.**
- `out/2026-07-04_02-57-57_DistillINSID3Industrial` — @512 rerun (reap-affected): 40/80/160 = 6.46/12.14/13.82.
- `out/2026-07-06_16-23-53_DistillINSID3Industrial` — @384: 40/80/160 = 5.33/8.21/10.08, 320 OOM.
