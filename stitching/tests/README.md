# stitching/tests/

Sanity tests for the stitching methods, on synthetic lattices only (no dataset needed).

```bash
.venv/bin/python -m pytest -q stitching/tests
```

| File | What it tests |
|---|---|
| `test_stitching.py` | Every decoder returns valid paths and agrees with brute force / expected behaviour on small examples. |
