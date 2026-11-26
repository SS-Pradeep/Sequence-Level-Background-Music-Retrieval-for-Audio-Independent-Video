# results/titlesplit/

Results on the **title-grouped film split**: 5 folds in which every film title appears in only one fold, so no film is
in both train and test. This is the evaluation protocol of the final system.

| File | Contents | Script |
|---|---|---|
| `final.md` | Final results table. | `bgm report --final` |
| `zeroshot.md` | Zero-shot baselines (no training). | `bgm eval` |
| `eval_suite.md` | Retrieval, alignment, transitions and diversity (D111). | `bgm.analysis.eval_suite_report` |
| `eval_lb.md` | LanguageBind score of the final system (D117). | `bgm.analysis.eval_lb` |
| `train_vs_test.md` | Retrieval on training vs test clips (overfitting check). | `bgm.sweep.train_vs_test` |
| `planned_losses.md` | Planned loss functions compared. | `bgm.sweep.titlesplit` |
| `improve_losses.md` | Exploratory extra loss configs. | `bgm.sweep.titlesplit` |
| `loss_bank.md` | Tag-free loss bank (inter / intra / adaptive). | `src/bgm/sweep/titlesplit_bank*.sh` |
| `similarity.md` | Similarity functions beyond mean-pooled cosine (D145). | `bgm.sweep.similarity` |
| `cue_switch.md`, `cue_switch_long.md` | Learned music-change predictor in Viterbi stitching (normal / long sequences). | `bgm cue report` |
| `cut_changes.md` | Music change at shot cuts (D103). | `bgm.cue.cut_model` |
| `cut_viterbi.md` | Learned change costs inside the Viterbi stitcher (D104). | `bgm.cue.cut_hook` |
| `window_switch_cost.md` | Per-window key and tempo in the switch cost (D137). | `bgm.analysis.window_switch_cost` |
