# Pipeline Verification

## Reader GC And OCR Publication

Fast, offline, deterministic verification for the three-bucket graph, upload
protection, current-bucket OCR publication and existing staged/binary contracts:

```bash
python3 -B -m unittest tests.test_reader_gc_graph tests.test_reader_bucket_store tests.test_reader_lifecycle tests.test_pdf_page_contracts tests.test_pdf_ocr_stages tests.test_pdf_reader_presentation -q
```

The graph fixtures cover PDF render-range-only PNG/JXL dependencies, complete
OCR text, split PDF/stream buckets, all five ebook extensions, native EPUB
fallback, chapter fonts/images, web/office/text/spreadsheet bundles, static PDFs,
audio/video/SWF, image manifests, DJVU derivatives, partial shard checkpoints,
resource-based review records and failed uploads. Missing/invalid manifests,
inventory permissions failures, root changes and new roots prevent acceptance.
An expired lease does not authorize deletion: all deletion code was removed from
the collector. Processing-root release and catalog generation retention are
tested as protection rules. Shared remote coordination of all producers and
actual deletion remain pending.

This group does not invoke OCR models, rebuild indexes, run browsers, upload to
HF or delete objects. It verifies reference protection, not OCR accuracy.

## Explicit Live Inventory

Read-only full inventory, potentially expensive because it lists all keys twice
and reads/rechecks every root plus live manifests:

```bash
python3 -B scripts/reader_gc_graph.py --output output/gc/reader-gc-report.json
```

Default Hub access requires `HF_TOKEN` and separate `HF_INPUT_TOKEN` for `melsm`.
Optional `--storage s3` uses main `HF_S3_ACCESS_KEY_ID` / `HF_S3_SECRET_ACCESS_KEY`
and separate `HF_S3_INPUT_ACCESS_KEY_ID` / `HF_S3_INPUT_SECRET_ACCESS_KEY`.
Bucket references always use the three current qualified names;
`HF_S3_INPUT_BUCKET` pointing to an old bucket does not remap those references.
`--skip-input-bucket` produces an explicitly incomplete report and exit status 1.

The report contains exact listed object counts, observed suffix formats, roots,
missing dependencies, ambiguous paths, unreferenced paths and unmanaged paths.
`candidates` are graph-unreferenced hints only, not deletion authorizations or
objects past a retention deadline. Failed graph acceptance suppresses candidates
globally. By default no remote state is written. `--record-observations` persists
first-seen orphan dates only after complete graph acceptance, using a positive
14-day default grace. Objects past grace are reported, not deleted.

The deployed `reader-gc.yml` schedules this report and retains the artifact for
30 days. Default and scheduled runs are read-only.

## Observations On 2026-10-09

Read-only S3 shallow production inventory confirmed these actual prefixes:

- `vomebook/reader-assets-v2`: `chapters/`, `documents/`, `media/`, `native/`,
  `pages/`, `reader-index/`.
- Ebook categories: `azw3`, `chm`, `epub`, `fb2`, `mobi`.
- Document categories: `office`, `pdf`, `spreadsheet`, `text`, `web`.
- Media categories: `audio`, `swf`, `video`; image category: `pages/image/`;
  native fallback category: `native/ebook/`.
- `vomebook/pdf-pages-v2`: `derived/`, `objects/`, `reader-index/`.
- Main-bucket root indexes included the sidecar, render/OCR registries and two
  lifecycle files; the PDF bucket had the derived-PDF root manifest.
- `melsm/pdf-archive-v2` returned S3 `NoSuchBucket` with the available credentials.
  This does not distinguish missing/private/inaccessible storage. No empty-bucket
  assumption, alternate legacy bucket read or deletion was performed.

Subsequent authenticated Hub inspection found the input bucket present and empty
before the account rollout. Real worker uploads using `HF_INPUT_TOKEN` succeeded;
the 368-page devondunn7 render manifest verified every input reference. GC now
uses the current bucket Hub API by default rather than treating inaccessible S3
storage as empty. S3 access remains a separate unaccepted optional path.

This was a shallow format inventory, not a complete production reference graph.
No production orphan totals or production deletion-safety conclusions follow.

The obsolete concurrency suite and PDF asset/render/OCR workflow-specific tests
were removed along with their replaced entrypoints. Shared manifest and binary
preservation behavior remains covered by the shared-page, staged and GC suites.

During development, a newly added catalog hook was not mocked in two tests and
wrote a test-only catalog to the live bucket. The identified catalog contained
only `objects/old/ocr-manifest.json` fixture references and was removed after
checking its exact generation, timestamp and resources. A subsequent read
confirmed its absence; no production Reader sidecar or content objects were
modified. The final 295-test focused verification passed with socket connections
blocked, so an accidentally unmocked network operation fails the test run.
