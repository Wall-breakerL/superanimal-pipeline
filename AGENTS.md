# SuperAnimal Pipeline boundaries

- This is a standalone, single-mouse pose-processing package, not AutoCBE or MoSeq.
- Preserve `src/superanimal_pipeline/legacy.py` byte-for-byte unless explicitly changing the scientific method. Its source hash is in `docs/provenance.md`.
- Do not change cleaning defaults silently: interpolation includes long gaps and likelihood; smoothing is centered rolling MEAN, not median.
- Treat all example/test poses and mocked model outputs as synthetic, never as biological evidence.
- Keep real videos, weights, server addresses, credentials and task-local logs under ignored paths. Never stage `.local/`, `.venv/`, outputs or weights.
- Keep raw inputs and previous attempts immutable. Resume must validate provenance and receipts; do not add an existence-only skip.
- Run `python -m unittest discover -s tests -v`, `ruff check src tests` and `ruff format --check src tests` after changes. Full inference and historical-data regression need separate evidence.
- GitHub publication and license selection require the owner's decision; creating local code does not authorize public upload.
