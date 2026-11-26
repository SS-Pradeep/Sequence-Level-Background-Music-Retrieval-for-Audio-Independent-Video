"""Person-focused branch (docs/person_branch_plan.md, PROGRESS.md D87+).

SigLIP encodes the whole frame, so the setting dominates. This package adds a branch that embeds only the people on
screen (face crops, or person boxes when no face is found) with the same SigLIP encoder, and learns from the
video-music pairs with InfoNCE. Opt-in only: `person.enabled: false` keeps every existing command bit-identical.

  subset.py     Step 1 "same setting, different music" test subsets
  evaluate.py   per-query retrieval and pipeline metrics for any system (reused by every run)
  diagnose.py   Step 1 report: the D75 system on the subsets vs all segments
"""
