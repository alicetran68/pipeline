# Pipeline Agent Guide

This directory owns source mirroring, parsing, generated search data, Reader
asset conversion, PDF image/OCR inputs, proofreading automation, and the
publication steps that feed the HF Search and GitHub Pages projects.

Read this file before changing a workflow or a generator. The workflow files
are the operational contract; this guide explains their boundaries and the
safe order in which they are used.

## Scope And Data Ownership

- Work only in `pipeline/` for pipeline changes. Changes to `CCRD-Search/`,
  `bha-search-lite/`, `huggingface-Search/`, or `github-Search/` require an
  explicit cross-project reason.
- `data/`-like generated outputs, compressed indexes, Reader sidecars, PDF
  manifests, and generated archive files must be produced by the existing
  scripts. Do not hand-edit compressed JSON, manifests, sidecars, or generated
  search indexes.
- `output/` is a run workspace. It is disposable and must not be uploaded or
  committed unless a workflow explicitly publishes an artifact from it.
- `state/` contains workflow checkpoints and publication state. A state change
  must be committed with the workflow's documented bot commit path; do not
  casually reset or overwrite it.
- A local count, manifest, revision, or generated index describes the local
  snapshot only. It is not a production fact until the target endpoint or
  dataset revision has been checked.
- Never commit secrets, HF tokens, GitHub tokens, S3 credentials, source URLs
  containing credentials, temporary archives, partial bundles, or
  `__pycache__/`.

### Reader Bucket Names

- Current buckets: `vomebook/reader-assets-v2`, `vomebook/pdf-pages-v2` and
  `melsm/pdf-archive-v2` (lossless OCR working inputs). Do not restore legacy
  bucket/dataset dual publication as recovery for a failed current-bucket read.
- Older workflow descriptions below include historical paths and concurrency
  limits. Inspect the actual checked-out workflow before acting on them.
- `PDF_ACCOUNT_WORKERS.md` describes local account lanes and binary protection.
  `PDF_PROCESSING.md` describes the current stages and lifecycle limitations.
  `PDF_READING_CYCLE_V3.md` specifies the proposed PDF-primary full-coverage stream,
  OCR/correction and cross-bucket lifecycle redesign; it is not deployed behavior.
- `reader_gc_graph.py` now provides the only three-bucket inventory and qualified
  reference graph. It follows render ranges, progress, catalog generations and
  processing roots, but has no deletion stage. Observation recording is allowed
  only after a complete graph; producer coordination is still required before
  any future deletion implementation.

## Shared Publication Rules

### Credentials

The workflows use these secret or variable families:

- `HF_TOKEN`: Hugging Face datasets, spaces, buckets, and API publication.
- `HF_USERNAME`: optional Hugging Face account override; most defaults use
  `VoiceOfML`.
- `PAGES_TOKEN` and `PAGES_REPO`: GitHub Pages repository publication and
  workflow dispatch.
- `GH_PAT`: GitHub archive/fork/proofreading operations.
- `TRACKER_TOKEN`: the repository's issue/PR comment API token, normally
  `github.token`.
- `HF_S3_ACCESS_KEY_ID` and `HF_S3_SECRET_ACCESS_KEY`: Reader bucket and
  staging-object operations.
- `BHA_REINDEX_TOKEN`: authenticated BHA index rebuild requests.
- `READER_CONVERSION_PASSWORD`: password passed to protected document
  conversion paths.

Never print a token or put one in a generated URL. Keep source and bucket
validation enabled when changing a publisher.

### Concurrency Groups

- `reader-assets` serializes Reader sidecar publication, Reader dataset asset
  conversion, PDF asset publication, OCR publication, manifest migration,
  native-text backfill, pruning, and bucket-related operations that must not
  race. These workflows use `queue: max` and do not cancel an active run.
- `publish-ccrd-index` serializes CCRD index publication, generation promotion,
  and CCRD source updates.
- `update-bha-index` serializes parsed-branch updates, OCR conflict commands,
  and BHA reindex state.
- `mirror-bha-source-files` serializes source mirroring and source optimization.
- Proofreading submission, review, and revert workflows have separate locks;
  do not assume that a review command and a publication job are atomic with
  each other.

Do not remove a lock merely to make a run start faster. Most races here create
conflicting manifests or lose checkpoint state rather than producing a simple
failed job.

## Workflow Map

The normal data paths are:

```text
Daily Sync
  -> fetch_and_parse.py
  -> sync_to_space.py / sync_to_pages.py
  -> HF Search and GitHub Pages generated search data

Reader Assets
  -> fetch source metadata
  -> scan_reader_assets.py
  -> convert_reader_assets.py shards
  -> publish_reader_assets.py
  -> publish_search_reader_index.py

PDF render inputs
  -> plan-render
  -> render PNG/WebP/JXL inputs
  -> publish-render
  -> PDF OCR or PDF asset workflows consume the published inputs

PDF OCR
  -> plan-ocr
  -> OCR verified PNG pages
  -> publish-ocr

BHA source files
  -> mirror_bha_source_files
  -> optimize_bha_source_files
  -> build_fork_parsed.py / update BHA index

CCRD
  -> source hash check
  -> publish/promote generation
  -> optional CCRD source TXT refresh
```

The PDF image/OCR workflows and the general Reader Assets workflow are not
interchangeable. PDF page manifests, OCR page payloads, native text streams,
ebook chapter bundles, media, and converted documents have different source
and completeness rules.

## Workflow Reference

Each section below describes one file in `.github/workflows/`.

### `action.yaml` - Keep HuggingFace Spaces Alive

- **Trigger:** daily at `00:00 UTC`, or manual dispatch.
- **Action:** curls Hitokoto, both search Spaces, and BHA Search with a 30
  second timeout and two retries.
- **Success rule:** every URL must return HTTP 200. Any non-200 or request
  failure fails the workflow after all URLs have been checked.
- **Side effects:** none; this is a read-only wake-up/availability check.
- **Change carefully:** adding an endpoint changes the health contract and can
  make a healthy pipeline fail because of an unrelated service.

### `daily-sync.yml` - Daily Sync

- **Trigger:** daily at `00:00 UTC`, or manual dispatch.
- **Input:** `force_sync` (default false). Manual force bypasses unchanged
  source revision checks.
- **Stages:** `fetch_and_parse.py` obtains current source metadata and sets
  `data_changed`; only then do `sync_to_space.py` and `sync_to_pages.py` run.
- **Publication:** HF Search receives generated metadata/index data; the Pages
  repository receives its generated data and a Pages workflow dispatch is sent
  through the GitHub API.
- **State:** when data changes, `state/commits.json` is committed as
  `chore: update commit state [skip ci]`. The workflow retries a racing push by
  fetching and rebasing; if the state file conflicts, the remote state wins.
- **Secrets:** `HF_TOKEN`, `PAGES_TOKEN`, `PAGES_REPO`.
- **Do not:** run `sync_to_space.py` or `sync_to_pages.py` by hand with partial
  `output/` data and call it a complete publication.

### `fork-archives.yml` - Maintain Archive Forks

- **Trigger:** Mondays at `02:00 UTC`, or manual dispatch.
- **Action:** `fork_repositories.py` maintains the archive forks and data
  branches for `banned-historical-archives` into the `anftm` mirror owner.
- **Lock:** `maintain-archive-forks`; active runs are not cancelled.
- **Secret:** `GH_PAT`.
- **Side effects:** GitHub forks, branches, and archive synchronization. Check
  the script's dry-run/support flags before changing repository selection.

### `reader-gc.yml` - Reader Cross-Bucket GC Report

- **Trigger:** daily at `04:13 UTC`, or manual dispatch; local change, not deployed.
- **Action:** `reader_gc_graph.py` inventories all three current buckets, including
  every existing Reader format, and writes an artifact-only JSON report.
- **Safety:** default report has no remote writes or deletes. Explicit
  `record_observations` persists only first-seen orphan dates. Missing/invalid roots, unreadable
  buckets, skipped input inventory and changing root snapshots block candidates.
  Unknown prefixes remain unmanaged; unknown manifest schemas block acceptance.
- **Roots:** canonical sidecar, category indexes and shard checkpoints, render/OCR
  registries and progress, processing records and known resource-based review
  records. Whole live bundles retain fonts, images and other companion files.
- **Lock/secrets:** `reader-sidecar`; main HF S3 credentials and separate
  `HF_S3_INPUT_ACCESS_KEY_ID` / `HF_S3_INPUT_SECRET_ACCESS_KEY` for `melsm`.
  Other publishers still have partitioned locks, so this lock alone does not
  establish global deletion safety. The report rechecks root keys and bytes.
- **Cleanup:** the former asset-only, category-only, static-PDF and dataset-prune
  collector scripts/workflows were removed. There is no force-JXL deletion path.
- **Verification:** use the explicit offline commands in `TESTING.md`; do not
  infer production orphan counts from fixture reports.

### `lin-native-text.yml` - Backfill Lin Yizhang PDF Text

- **Trigger:** manual dispatch only.
- **Inputs:** exact `source_path` and `dry_run` (default true).
- **Action:** `publish_lin_native_text.py` validates a verified Teachers repair
  path and, when `dry_run` is false, publishes native text/Reader metadata.
- **Lock:** `reader-assets`; maximum runtime 350 minutes.
- **Secret:** `HF_TOKEN`.
- **Safety:** run dry-run first. The path must be a verified Lin Yizhang repair
  folder; do not generalize this workflow to arbitrary PDFs.

### `migrate-pdf-page-manifests.yml` - Migrate PDF Page Manifests

- **Trigger:** manual dispatch.
- **Inputs:** `limit` (default 500), zero-based `checkpoint` (default 0), and
  `apply` (default false).
- **Action:** `migrate_pdf_page_manifests.py` validates a batch of v1 manifests;
  only `apply=true` uploads v2 manifests.
- **Artifact:** `pdf-manifest-migration-<checkpoint>/report.json`, retained
  even when the job fails.
- **Lock/secrets:** `reader-assets`; `HF_TOKEN`; target is
  `vomebook/Reader-Assets`.
- **Safety:** checkpoints identify batches, not an arbitrary page offset. Keep
  reports for each applied batch and do not run overlapping checkpoints.

### `mirror-bha-source-files.yml` - Mirror BHA Source Files

- **Trigger:** manual dispatch.
- **Input:** `archive_id`, either `0`-`31` or `all` (default `all`).
- **Action:** `mirror_source_files.py` downloads/mirrors source archives into
  `vomebook/BHA-Source-Files` using `GH_PAT` and `HF_TOKEN`.
- **Lock:** `mirror-bha-source-files`.
- **Variables:** `REPO_START` and `REPO_END` are derived from the selected
  archive ID.
- **Side effects:** updates the source-file mirror. Optimization is a separate
  workflow and must not be assumed to have run.

### `ocr-conflict-review.yml` - OCR Conflict Review Commands

- **Trigger:** a new issue comment.
- **Guard:** only non-PR issues containing the proofreading OCR conflict
  marker and a `/ocr-keep` or `/ocr-drop` command are accepted.
- **Action:** `review_ocr_patch_conflicts.py` applies the reviewed OCR decision.
- **Lock:** `update-bha-index`.
- **Permissions/secrets:** contents/issues write, `GH_PAT`, `TRACKER_TOKEN`,
  and optional `PROOFREAD_REVIEW_ACTORS` allowlist.
- **Safety:** the issue marker and actor allowlist are part of the command
  authorization; do not remove them to make ad-hoc comments work.

### `optimize-bha-source-files.yml` - Optimize BHA Source Files

- **Trigger:** manual dispatch.
- **Inputs:** archive selection, WebP quality, re-encode flag, PDF preview DPI,
  and PDF preview quality.
- **Action:** installs qpdf, Poppler, and WebP tools, then runs
  `optimize_source_files.py` against `vomebook/BHA-Source-Files`.
- **Defaults:** WebP quality 85, max image dimension 2400, PDF preview DPI
  150, PDF quality 85, PDF concurrency 2, object concurrency 8, paced at 200
  objects/minute.
- **Lock:** `mirror-bha-source-files`.
- **Safety:** this rewrites mirrored source files in place. Use a selected
  archive for a trial and verify the manifest/optimization flags before an
  `all` run. `reoptimize_images=true` can touch existing WebP files.

### `promote-ccrd-generation.yml` - Promote CCRD Generation

- **Trigger:** manual dispatch.
- **Input:** an existing generation directory, defaulting to the documented
  generation example in the workflow.
- **Action:** `promote_ccrd_generation.py` promotes an already built generation
  without rebuilding the corpus.
- **Lock/secret:** `publish-ccrd-index`; `HF_TOKEN`.
- **Safety:** verify the generation directory and its completeness first. This
  is a pointer/promotion operation, not a repair operation.

### `proofread-review.yml` - Proofreading Review Commands

- **Trigger:** issue comment.
- **Guard:** only non-PR issues with the proofreading PR marker and `/approve`
  or `/reject` are accepted.
- **Action:** `review_proofread.py` applies the command and manages the review
  state.
- **Lock/permissions:** `proofreading-review`; issue and PR write permissions;
  `GH_PAT`, `TRACKER_TOKEN`, and `PROOFREAD_REVIEW_ACTORS`.
- **Safety:** approval/rejection is actor- and marker-gated. Do not invoke the
  script by editing an issue body without preserving its marker.

### `publish-ccrd-index.yml` - Publish CCRD Index

- **Trigger:** Mondays at `03:00 UTC`, or manual dispatch.
- **Check:** `check_ccrd_source_changed.py` compares the current CCRD source
  hash and exposes `changed`.
- **Publish when changed:** `publish_ccrd_index.py` builds/publishes indexes,
  then `update_ccrd_source_txt.py` refreshes source text and triggers the HF
  Space rebuild.
- **Lock/secret:** `publish-ccrd-index`; `HF_TOKEN` and optional HF username.
- **Safety:** an unchanged source is intentionally a no-op. Do not force a
  rebuild by hand unless the generated inputs or target revision changed.

### `publish-hf-search-index.yml` - Publish HF Search Index

- **Trigger:** manual dispatch only.
- **Action:** force-fetches current metadata, then `sync_to_space.py` publishes
  records, word postings, n-gram indexes, folder data, and initial/sidebar
  payloads to the HF Search Space.
- **Lock/secret:** `publish-hf-search-index`; `HF_TOKEN`.
- **Safety:** this publishes generated search data; it is not the same as
  deploying `huggingface-Search/app.py` or static Reader code. Run the HF
  Search project's local oracle before using a newly generated snapshot.

### `publish-proofread-upstream.yml` - Publish Proofreading Upstream

- **Trigger:** daily at `03:00 UTC`, or manual dispatch.
- **Input:** `bootstrap` (default false). Bootstrap establishes a baseline and
  does not publish existing merged corrections.
- **Action:** `publish_proofread_upstream.py` publishes merged corrections to
  the configured upstream target and records batches in
  `state/proofread-upstream.json`.
- **Lock/permissions:** `publish-proofread-upstream`; contents write;
  `GH_PAT`.
- **Side effect:** commits the publishing state. Do not delete the state file
  to force re-publication; use the script's intended bootstrap/retry behavior.

### `publish-source-txt.yml` - Publish Source TXT Once

- **Trigger:** manual dispatch only.
- **Setup:** Node 24, Python 3.11, Git LFS, and the normal requirements.
- **Stages:** `fetch_and_parse.py` generates current metadata, then
  `publish_source_txt_once.py` publishes source TXT files to the configured HF
  user/dataset exactly once according to its own state/skip rules.
- **Secrets:** `HF_TOKEN`; optional `HF_USERNAME`.
- **Safety:** this is a one-time publication path, not the daily search-data
  sync. Do not rerun casually against a changed source tree.

### `revert-proofread.yml` - Revert Proofreading

- **Trigger:** manual dispatch or issue comment.
- **Manual inputs:** correction ID and exact confirmation string `REVERT`.
- **Comment guard:** only non-PR issues with the proofreading auto-merge log
  marker and `/proofread-revert` are accepted.
- **Action:** `revert_proofread.py` creates human-reviewed revert PRs rather
  than silently rewriting source content.
- **Lock/permissions:** `proofreading-revert`; PR/issues write;
  `GH_PAT`, `TRACKER_TOKEN`, and `PROOFREAD_REVERT_ACTORS`.

### `reader-assets.yml` - Build Reader Assets

- **Trigger:** Sundays at `03:23 UTC`, or manual dispatch.
- **Scope inputs:** exact source repository, extension, path, per-shard limit,
  checkpoint count, retry/force flags, bucket migration flag, dry-run flag,
  and per-command timeout.
- **Plan:** fetches source metadata with retries, scans the incremental asset
  queue, prepares a snapshot and shard queue, and packages the plan artifact.
  `dry_run=true` stops after queue reporting.
- **Convert:** up to 20 shards install tools based on the selected format and
  run `convert_reader_assets.py` against the prepared queue. Conversion bundles
  are uploaded as short-lived artifacts.
- **Publish:** serially publishes each shard with `publish_reader_assets.py`.
  A failed conversion can still leave failure state for the publisher; complete
  successful entries are not silently treated as failed.
- **Finalize:** restores the plan, optionally removes stale mappings only when
  the scan is authoritative and all conversion/publication conditions pass,
  then publishes the Reader sidecar to Pages.
- **Lock/secrets:** `reader-assets`; HF token, HF S3 credentials, Pages
  credentials, and optional conversion password.
- **Format boundary:** PDF/OCR and ebook chapter pipelines are dedicated
  workflows. `bucket_migrate` handles final static artifacts for formats owned
  by this workflow; it must not be used as a substitute for PDF page/OCR
  publication.
- **Full rebuild:** use `READER_ASSETS_REBUILD.md`. The `clean_rebuild` input is
  authoritative and cannot be combined with a repository, extension, or path
  filter. Run its dry-run and preflight checks before enabling it.

### `submit-parse.yml` - Submit Parsing

- **Trigger:** `repository_dispatch` type `submit-parse`.
- **Action:** runs the shared `submit_proofread.py` path to create parsing pull
  requests for BHA source material.
- **Setup:** Node 24 and Python 3.11; review image dependencies are installed.
- **Lock/permissions/secrets:** `mirror-bha-source-files`; contents/issues/PR
  write; `GH_PAT`, `TRACKER_TOKEN`, `HF_TOKEN`, and the BHA source repository.
- **Safety:** this creates reviewable PRs; it does not directly publish parsed
  data or trigger the BHA reindex by itself.

### `submit-proofread.yml` - Submit Proofreading

- **Trigger:** manual dispatch or `repository_dispatch` type
  `submit-proofread`.
- **Inputs:** archive ID, correction kind, optional OCR patch fields, config
  path/content/metadata, PR title, and description.
- **Action:** `submit_proofread.py` creates the corresponding review PR.
- **Lock/permissions/secrets:** `submit-proofread`; contents/PR/issues write;
  `GH_PAT` and `TRACKER_TOKEN`.
- **Safety:** `ocr_patch`, `config`, `parse`, and ordinary `proofread` have
  different required fields. Validate the kind-specific inputs before running;
  this workflow does not apply a correction directly.

### `update-bha-index.yml` - Update BHA Index

- **Trigger:** every 15 minutes, `ocr-rebase-reviewed` repository dispatch, or
  manual dispatch.
- **Inputs:** selected archive IDs, `force_rebuild`, and explicit reviewed OCR
  baseline-rebase permission.
- **Stage 1:** `build_fork_parsed.py` updates changed anftm parsed branches and
  writes `/tmp/bha-parsed-inputs.json`.
- **Stage 2:** always maintains OCR conflict issues with
  `report_ocr_patch_conflicts.py`.
- **Stage 3:** records parsed input state, checks revisions, refreshes the
  proofreading issue queue, and requests BHA `/api/reindex` when revisions
  changed.
- **Stage 4:** polls `/api/reindex/status` until the expected archive revisions
  are ready, then records accepted revisions.
- **Lock/permissions:** `update-bha-index`; contents/issues write;
  `GH_PAT`, `TRACKER_TOKEN`, `BHA_REINDEX_TOKEN`, and BHA URL variables.
- **Safety:** do not enable OCR patch rebase without a reviewed decision. The
  workflow waits for revision identity, not just an HTTP 202 acceptance.

### `update-ccrd-source.yml` - Update CCRD Source Text

- **Trigger:** manual dispatch only.
- **Action:** runs `update_ccrd_source_txt.py` to refresh the CCRD source TXT
  publication and trigger the target HF Space rebuild.
- **Lock/secret:** `publish-ccrd-index`; `HF_TOKEN` and optional HF username.
- **Boundary:** this does not build CCRD indexes. Use `publish-ccrd-index` for
  the complete index publication path.

### `update-directories.yml` - Update HF Repository Directories

- **Trigger:** daily at `20:00 UTC`, or manual dispatch.
- **Action:** `update_dirs.py` regenerates the HF repository directory data used
  by browsing/search metadata.
- **Secret:** `HF_TOKEN`.
- **Relationship:** Daily Sync starts at `00:00 UTC` and explicitly documents
  that directory generation normally finishes before it. Do not assume a
  directory update and a search-data sync are one atomic workflow.

## Safe Development And Verification

Before changing a workflow or script:

1. Read the workflow and every script it invokes. Follow the artifact names,
   environment variables, checkpoints, and concurrency group rather than
   inventing a parallel path.
2. Run focused tests for the touched pipeline. The test suite is split by
   domain; useful groups include:

   ```bash
   python3 -m unittest tests.test_reader_lifecycle tests.test_reader_assets
   python3 -m unittest tests.test_pdf_ocr_stages tests.test_reader_gc_graph
   python3 -m unittest tests.test_pdf_render_schedule tests.test_pdf_ocr_render_integration
   python3 -m unittest tests.test_fetch_and_parse tests.test_fork_repositories
   python3 -m unittest tests.test_proofread tests.test_ocr_patch_conflicts
   python3 -m unittest tests.test_migrate_pdf_page_manifests
   python3 -m compileall -q scripts tests
   ```

3. For generated data, validate signatures, counts, revisions, manifests, and
   complete IDs/totals. A successful upload alone is not validation.
4. For a publication change, run the target project's local oracle and then
   its read-only production smoke test after deployment.
5. Inspect `git status`, `git diff`, and recent history before committing. Do
   not reset or discard changes in `output/`, `state/`, or scripts that may be
   an active user run.

## Common Failure Modes

- A source revision is unchanged: this is normally a no-op, not a failed sync.
- A shard fails conversion: inspect its result bundle and checkpoint before
  retrying; do not regenerate the entire corpus unnecessarily.
- HF S3 returns `SlowDown`: reduce concurrency and retry with the same manifest;
  never disable TLS or delete objects optimistically.
- A publication has a valid HTTP response but stale generation/manifest data:
  compare revision identity and complete object references before declaring it
  successful.
- A workflow is blocked by its concurrency group: wait for the owner run;
  dispatching a second destructive cleanup or publisher can make the first
  result ambiguous.
- A generated file is missing locally: use the owning workflow/generator. Do
  not create a hand-edited replacement and commit it.
