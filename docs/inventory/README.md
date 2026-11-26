# docs/inventory/

A record of the disk cleanup on 2026-11-26: which files of the full working directory (code, datasets, features,
runs) were kept and which were deleted.

| File | What it is |
|---|---|
| `CURRENT_VERSION.md` | Summary: what is kept and why, what was deleted and why, sizes. |
| `keep_files.tsv.gz` | Every kept file (`path`, `bytes`), about 139k files / 268 GB. |
| `deleted_files.tsv.gz` | Every deleted file (`path`, `bytes`), about 111k files / 294 GB. No backup exists. |

Read a list with:

```bash
zcat docs/inventory/keep_files.tsv.gz | less
```
