# src/bgm/listening/

**Listening study** (SPEC M12): a kit for human raters to compare soundtracks, and the analysis of their ratings.

| File | What it does |
|---|---|
| `kit.py` | Builds the kit: 20 test videos (natural clips and concatenations), randomised pairwise sessions, instructions and a rating template. |
| `analysis.py` | Win rates and 1–5 ratings from the completed sheets, with a two-way bootstrap over raters and videos. `python -m bgm.listening.analysis [--ratings kit/ratings.csv]` → `kit/analysis.md` |
