# Phase4 Compare Pack

Runs: A (no conf gate/loss gate), B (conf loss gate only), C (conf gate+loss gate).

|Run|ID|Last WL1|Last PSNR|Last SSIM|Best WL1|Best PSNR|source_fg_key|cover_fg|cover_valid|
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
|A|CoreView_390_20260207_173011|0.093358|18.448|0.6784|0.093352|18.136|tgt_fg|0.0639|1.0000|
|B|CoreView_390_20260207_180500|0.080270|21.097|0.7911|0.079942|21.306|tgt_fg|0.0639|1.0000|
|C|CoreView_390_20260207_185302|0.060663|21.427|0.8537|0.059884|21.642|tgt_fg|0.0639|1.0000|

Artifacts:
- compare_pred_tgt_val_e011_step004212.png
- compare_fgmask_pred_tgt_val_e011_step004212.png
- compare_overlay_val_e011_step004212.png