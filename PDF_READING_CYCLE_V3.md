# PDF Reading And OCR Cycle v3

Status: proposed architecture, 2026-10-09. This document specifies the requested
redesign. It is not a deployment report. The local HF/Pages Readers now have a
hybrid adapter for existing v2 WebP page manifests and an associated PDF. The
v3 manifest, document builder, unified publication, cross-bucket collector and
daily correction service below are not yet implemented end to end.

Local progress on 2026-10-09: a read-only three-bucket reference report, durable
staged-upload protection roots and canonical current-bucket OCR index publication
are implemented. Deletion, lease retirement/heartbeat and immutable generation
promotion remain pending. See `TESTING.md` for commands and live inventory limits.

## 1. Product Contract

The primary reading representation is a PDF document or a validated vector HTML
document. The startup/fallback stream covers every page, so any saved position
can open directly. Per-session stream fetching is bounded and demand-driven;
page images do not remain the preferred renderer once the target PDF page is
ready. OCR processing, reading and correction have independent progress.

- Reading never waits for complete OCR, correction or PDF optimization.
- PDF loading means demand-range loading, not downloading the entire document.
- Preview failure must not fail a readable PDF. PDF failure retains an already
  decoded preview and allows explicit retry or the original-document fallback.
- Saved deep positions request that page directly. Never download all earlier
  preview pages to reach a position.
- Complete stream coverage is a publication property, not a command to fetch
  or decode the entire stream in the browser. No fixed preview page cutoff.
- Native text is extracted first. OCR applies only where usable text is absent
  or demonstrably broken; the existence of any text is not proof of quality.
- Output codec follows actual page content, not source file size or account.
- Every published representation preserves page identity, order, crop, rotation,
  bookmarks and links, or explicitly declares a different representation.
- Search uses a complete validated text generation with exact totals. Missing
  text on a page is distinct from a successfully processed blank page.
- Correction creates a new version. Raw OCR, original documents and previous
  accepted versions remain recoverable.

These rules apply to HF and Pages Reader. Metadata-search matching remains
unchanged. New reading formats require both frontend acceptance layers.

## 2. Intake And Classification

Intake pins source revision and SHA-256, records the original extension and
source byte size, and deduplicates by source key. A generated PDF from DJVU, CAJ
or KDH retains that original identity. Repairing a native PDF does not move it
to the converted-input account.

Use a cheap sampling pass to prioritize jobs, then inspect every page before
publishing a whole-book classification. Store per-page evidence: raster bit depth,
filters, raster coverage, image masks, font/text usability, drawings, dimensions,
rotation and reading-order/layout signals. Keep an explicit unknown class when
inspection fails. Classification is metadata, not an excuse to skip pages.

| Page Class | Main Reading Output | Per-Page Stream Representation | Text Work |
| --- | --- | --- | --- |
| Usable native text/vector content | Original or validated optimized PDF; vector HTML candidate | Validated HTML/SVG page preserving fonts/coordinates | Extract text and geometry; OCR only broken regions |
| Bitonal scan, CCITT/JBIG2/1-bit | PDF preserving embedded image streams | Lossless PNG or validated lossless alternative, preserving text detail | Separate OCR input; retain binary source |
| Grayscale/color scan | Original or validated optimized PDF | Readable WebP/JPEG preview, explicitly not archival data | Lossless OCR input, layout detection and recognition |
| Mixed/complex layout | PDF preserving the page composition | Per-page policy, no book-wide forced codec | Region layout, native/OCR merge and review |
| Unknown/inspection failure | Original PDF | Pending until checked; complete stream cannot be advertised | Retry inspection; do not silently re-encode |

The current local rule preserves a whole book when any 1-bit image occurs. v3
refines this at page/region level: a tiny monochrome logo need not classify a
color book as a bitonal scan. Conversely, a binary-looking image stored as 8-bit
must not be destructively reduced based on size alone. Codec changes require
evidence and validation, with preservation as the fallback.

"Pure SVG PDF" here means vector PDF page content, not an SVG file hidden inside
the PDF. Export using an existing PDF engine, retaining paths, fonts and page
coordinates. Use sanitized SVG-backed HTML for fixed-layout fidelity. Reflowable
HTML is a separate candidate only when text order, tables, notes and anchors are
validated. Glyph outlines with no usable Unicode need recognition; exporting
paths to HTML alone does not create selectable text. Complex vertical pages must
not be forced into guessed horizontal reading order.

## 3. Representation Manifest

Publish one immutable `reading-manifest.json` for a book generation. Resolve
returns its current generation reference. The compact sidecar remains a discovery
index, not a multi-megabyte page table. Use a new explicit reading contract rather
than making a partial preview pass the existing full `pdf-pages` validator.

Required manifest groups:

| Group | Required Identity And Meaning |
| --- | --- |
| schema | `version: 3`, `kind: pdf-reading`, stable source key, source SHA/revision, generation |
| pages | Exact total, page-map reference, original dimensions/crop/rotation and per-page class |
| primary | `pdf` or `vector-html`, validated complete resource, PDF fallback when HTML is primary |
| preview | Whole-book stream coverage, partition index and explicit per-page number/codec/dimensions/geometry hash |
| text_layer | Optional complete raw/accepted text manifest and index, language/model/layout versions |
| review | Queue reference and raw/accepted generation identities; absence means no queued cases |
| provenance | Tool versions, recipe hash, parent generation, verification report and creation time |

Every stored resource reference includes `bucket`, `path`, `sha256`, `bytes` and
`role`. Object existence is verified before publication. References contain no
signed URLs; request-time proxy resolution obtains temporary transport URLs.
Validate allowlisted bucket names, exact path forms, digest, limits and source
identity at the server and client. Do not trust a client-selected external PDF.

`preview.complete=true` requires exactly one validated stream entry for every
page in `1..pages.total`, including blank pages, and all objects/dependencies must
exist. Store entries in immutable indexed partitions (initially 128 pages each)
so opening page 800 downloads its partition directly, not every earlier one.
The small root manifest declares exact ranges, checksums and a page-map identity.

During generation, an explicitly incomplete preview generation may expose ready
partitions while the complete original PDF remains readable. Missing partitions
are pending, not blank pages or end-of-book. Promote to complete only after
all page identities and dependencies pass verification. Do not call a source
PDF fallback a generated preview or hide failed stream pages to claim completeness.
OCR indices still require complete page identities and processed-empty entries.

Representations share one immutable page-map identity. Text records use original
page coordinates and Unicode-codepoint offsets. When a tool crops or rotates
an input, it records an invertible transform back to the original page. Preview,
PDF, SVG and text must resolve the same logical page before an in-place handoff.

## 4. Reader Startup And Handoff

One book adapter owns page shells, document loading, progress and cancellation.
Do not start two complete adapters and rebuild the DOM when switching modes.

1. Start source resolution and known PDF engine preparation concurrently.
2. Read the small reading manifest and cached reading position. Request the
   partition for the saved page or page 1, load that page's stream representation,
   and start the primary PDF's demand-range load concurrently.
3. Display a decoded preview in a correctly sized page shell immediately when
   it is available. Optional OCR text loads independently.
4. When PDF.js has parsed the document, prioritize the currently visible page
   and its immediate neighbor. Parsing alone does not trigger image removal.
5. Render into an offscreen canvas using the same shell geometry and page map.
   Commit only after rendering succeeds and book/page/zoom generations still
   match. Swap within the existing shell, then release its preview resources.
6. Prefer PDF rendering for subsequent demands, but keep the stream available
   for any page whose PDF render is still pending or has failed. A successful
   page-1 render does not globally disable the page-800 stream. Choose per page
   and current zoom: if PDF is already ready, skip its preview fetch. If it is
   not ready, show its demanded preview and replace it only after PDF success.

Once document initialization completes, start the demanded PDF page first and
delay a new preview request by 250 ms (an initial tunable value). Skip that
request if the canvas is already ready. Existing preview images stay visible
until replacement; zoom redraws retain the previous canvas and never request
another preview for an already presented PDF page. Before initialization, the
current opening/restored page starts its stream request immediately.

State is per page: `empty -> preview-loading -> preview-visible -> pdf-rendering
-> pdf-visible`. PDF may go directly from empty to visible. Failures have retry
states; cancellation never publishes a late canvas or progress update.

Preserve page number plus fractional page offset across geometry updates. Defer
the swap during an active selection or a gesture that would be disrupted; keep
selection text identities independent of image/canvas DOM. Zoom requests carry
epochs so obsolete canvases cannot replace a newer preview. Dispose cancels
preview fetches, render tasks, timers, worker and text requests exactly once.

Initial defaults, to be tuned by actual cold-network/device measurements:

- Generate representations for the entire book. There is no fixed page-count
  or total-book preview-byte budget that omits later pages. Control encoding
  quality, storage cost and partition size without reducing logical coverage.
- At runtime load the visible pages first. Adapt neighbor prefetch to viewport
  height, scroll direction/speed, network/save-data state, decoded-memory budget
  and PDF page readiness. Start with at most two stream requests in flight;
  this is concurrency, not a two-page accessibility limit.
- Give current-page PDF ranges priority over speculation. On a deep jump,
  cancel obsolete prefetch and demand the new page partition, stream and PDF
  page directly. No sequential warmup through preceding pages.
- Keep the existing bounded canvas/shell window; never retain every decoded
  page image. Image byte size is not a decoded-memory budget.
- Mobile save-data disables speculative neighbors. It still permits an actual
  demanded preview and the primary document load.

A preview's expected benefit is measured against PDF-only startup. An already
compact bitonal PDF may win the race; skip its unnecessary session preview fetch,
while keeping the generated full stream available for other positions/failures.
Binary stream previews remain lossless; no default lossy re-encoding of text.
Neither image rendering nor PDF rendering is universally faster; acceptance
compares first paint, deep-page demand latency, bytes, memory and handoff drift.

## 5. PDF Optimization And Searchable PDF

Create two independent candidates rather than tying readable PDF publication
to OCR completion:

- `display.pdf`: visual/document optimization without changing recognized text.
- `searchable.pdf`: the same visual document with a complete accepted invisible
  text layer where needed. It may become the next primary document generation.

For vector and bitonal pages, preserve existing drawing/image streams. Do not
reconstruct the document from preview WebP or OCR PNG. A full RGB raster rebuild
loses binary compression, vector detail, links and native text. OCR inputs are
working evidence, not the default source for the display PDF.

Use established PDF tools (qpdf, MuPDF, pypdf or a tested OCRmyPDF pipeline) with
format-specific recipes. Start with lossless structural cleanup and deduplication.
Treat grayscale/color image transcoding as an explicit candidate with measured
visual acceptance. No default lossy JBIG2 symbol substitution. OCRmyPDF, if used,
must not repeat already completed recognition or flatten usable native text.

OCR text insertion must map all characters through validated font and geometry
handling, including Chinese, combining characters and vertical writing. Preserve
usable existing native text instead of overlaying duplicate invisible text. The
complete separate text index remains authoritative for Reader search.

Candidate acceptance checks exact page count/order, media/crop boxes, rotation,
outline destinations, links, native text coverage, image preservation policy,
representative visual differences plus every changed page, and selectable OCR
text-to-page mapping. Record all failed checks. A larger or slower candidate is
not promoted merely because a tool returned success.

Linearization is a candidate, not a blanket requirement. Existing local Reader
acceptance observed a linearized example causing near-full PDF.js reads. Benchmark
the actual demand-range configuration before promotion. Keep the original or
previous accepted PDF if candidate startup/deep-page behavior regresses.

Optimization runs once per source/recipe. A text correction regenerates the text
index and searchable-PDF layer only; it does not rerender all input pages or
recompress the visual PDF. No automatic mid-session replacement with a newer
server generation; an open Reader pins its generation until reopen.

## 6. OCR And Review Cycle

Stream generation schedules requested/restored pages and then untouched ranges,
eventually covering the whole book; OCR and optimization can advance independently.
Extract usable native text and generate lossless working images
only for required recognition/layout regions. Preserve original pixel resolution
and transforms when extracting scans; if rendering is necessary, obey existing
pixel bounds and record the rendering recipe.

Keep the current checksummed page inputs, language-aware engines, resumable
page batches and full-book completion checks. Default Chinese/English execution
uses the configured RapidOCR ONNX backend; other languages use the supported
Paddle backend. Pin actual model/version identities, not just the package name.

Review signals include per-region low confidence, weak native-text mappings,
engine disagreement, vertical writing, columns, tables, notes, unexpected reading
order, missing text on a nonblank page and unexplained character loss. Confidence
alone is not correctness. An initial queue threshold of 0.90 can prioritize work,
but must be calibrated by model/language and is not an auto-accept threshold.

Each review task references raw OCR SHA, page-map SHA, source page, region IDs,
neighboring context, layout evidence and a lossless crop/source reference. Queue
whole-page context for order/table issues; crops alone cannot prove reading order.
Deduplicate by source, raw generation, issue, region and correction model.

Raw recognition is immutable. Store correction proposals separately with before/
after text, region ordering, evidence, provider/model, prompt recipe and task ID.
Raw reading availability is not held hostage to review completion. Unresolved
cases remain queued and their evidence remains protected from collection.

Daily OpenCode worker proposal:

- Run in one dedicated correction repository/remote worker, once per day with
  a shared job lock. Manual dispatch uses the same queue and deduplication.
- Default disabled with `OCR_CORRECTION_ENABLED=false`. Enable after the user's
  provider/model credentials and region-verification acceptance are configured.
- Pin the OpenCode CLI and model; use a provider supporting image input when
  visual checking is required. Missing visual capability leaves the task queued.
- Start with at most 20 page-equivalent tasks, one model request at a time and
  a 30-minute job deadline. Use configured request/token budgets and provider
  accounting where available; do not invent a monetary cost estimate.
- Fetch trusted queue objects with a transport process, verify their digests,
  then provide only the selected immutable task workspace to the agent. The
  correction process has no HF/GitHub publication credentials or shell tools.
- Treat book text as data, not instructions. Require bounded structured output
  referring to existing region IDs; validate it independently. Secrets never
  enter a prompt, stdout artifact or committed OpenCode configuration.
- Publish proposals/review artifacts first. Initially require acceptance before
  modifying canonical text. A later opt-in auto-accept policy may handle narrowly
  defined independently verified corrections, with an audit trail and rollback.
- A trusted publisher commits accepted text under the normal generation lock.
  Reject a stale raw-generation patch rather than applying it to new text.

This is generated Reader OCR correction, not BHA source proofreading. It must
not silently rewrite upstream source TXT or bypass existing proofreading reviews.
Assistant tokens/account credentials do not provide model quota; the remote
worker uses only explicitly configured user provider access.

## 7. Storage Contract

Do not add more buckets. Qualify every dependency by bucket and path:

| Bucket | Roles |
| --- | --- |
| `vomebook/reader-assets-v2` | Canonical immutable catalog generations, current pointer, lifecycle/leases, review state; vector HTML bundles |
| `vomebook/pdf-pages-v2` | Display/searchable/generated PDFs, reading manifests, full per-page streams, native/raw/accepted OCR text, page maps and verification reports |
| `melsm/pdf-archive-v2` | Temporary lossless OCR inputs, active processing/review evidence and retry checkpoints |

Use content-addressed `objects/<source-prefix>/<source-sha>/<recipe-id>/...`
resources and separate immutable generation manifests. Generated CAJ/DJVU/KDH
documents may retain validated `derived/.../document.pdf` paths. New preview
PNG/SVG/HTML/review paths require explicit backend/frontend allowlist and MIME
updates; this document does not authorize bypassing path validation.

Representative resources under a book recipe:

```text
reading-manifest.json
page-map.json.gz
pdf/display.pdf
pdf/searchable.pdf
preview/page-000001.png | .webp
html/page-000001.xhtml       # with validated SVG/font dependencies
text/raw/page-000001.json.gz
text/accepted/page-000001.json.gz
text/book-text.json.gz
review/queue.json
reports/document-verification.json
ocr-input/page-000001.png    # in the input bucket, not the reading bucket
```

Catalog roots live in `reader-assets-v2/reader-index/`. There is one current
generation pointer, not separate mutable "current" registries in multiple
buckets. Temporary per-stage journals are operational state and carry exact
generation/lease identities. Never recover a failed v2 read by publishing from
a stale legacy dataset fallback.

## 8. Lifecycle And Publication

Track per-book components independently: document, previews, raw text, accepted
text, searchable PDF and review. Each has `pending/running/ready/failed/skipped`
with source/recipe identity and reason. Do not use one status to imply all are
finished. A book may be readable while OCR is pending or review remains open.

Processing order:

```text
intake -> inspect/page-map -> full stream partitions + primary document candidate
                         -> native text / lossless OCR inputs -> raw OCR
                         -> lossless document optimization
raw OCR -> review proposals -> accepted text -> searchable PDF
verified ready components -> immutable generation -> current pointer promotion
superseded generation -> retained rollback -> orphan grace -> collection
```

Every worker acquires a book/stage lease before uploading, with protected bucket
prefixes, recipe, run ID and heartbeat. Initial expiry is 24 hours with 15-minute
heartbeats; expiry permits investigation/retry, not immediate deletion. Check the
owning run and durable checkpoint before releasing an abandoned lease.

Account workers use the existing exclusive size/source lanes for compute. One
central publisher owns catalogs and pointers. Retry by input/recipe generation,
not by erasing checkpoints. Partial uploads cannot become ready components.

Promotion protocol:

1. Upload immutable objects and a complete dependency manifest while lease is active.
2. Verify hashes/byte counts, all object existence, page mappings and component checks.
3. Build an immutable catalog generation referencing verified components.
4. Under the shared publication/GC lock, compare the current generation with the
   expected parent and replace the one current pointer. A racing publication
   re-merges against the new parent; it must not overwrite another book's update.
5. Derive compact sidecar and Pages/HF projections from that pinned generation.
   On projection failure retain the coherent old projection, retry and keep
   both referenced generations protected. An empty/incomplete sidecar is not recovery.
6. Record consumer acknowledgments and release only completed component leases.

A single small pointer update is the publication boundary; multiple bucket
uploads are not an atomic transaction. Read-only production acceptance checks
that both deployed consumers resolve the intended generation before retirement.

## 9. Garbage Collection

Use one cross-bucket mark-and-sweep collector. Object identity is `(bucket,path)`;
the same path in two buckets is not the same object. Manifest traversal honors
roles, input dependencies and exact bucket references.

Roots are current and retained rollback generations, active worker leases,
incomplete resumable checkpoints, open correction tasks, unacknowledged consumer
projections and pending accepted-text/searchable-PDF builds. Completed input
evidence is not a permanent runtime dependency just because an old provenance
record mentions it. Distinguish `runtime`, `processing`, `review` and `provenance`
references explicitly.

Initial retention defaults:

| Object State | Retention / Release Rule |
| --- | --- |
| Current PDF/HTML/preview/text/page map | Protect while current generation references it |
| Raw OCR, accepted correction and audit | Keep while book is active and policy retains correction history |
| Active OCR, partial retry or open review evidence | Protect; no age-only deletion |
| Full PNG inputs after completed OCR and searchable PDF, with no review | 7-day recovery retention, then 14-day orphan grace |
| Low-confidence/complex review crops/context | Retain until review is accepted or explicitly closed, then the same grace |
| Superseded generation | At least 30 days and both consumer acknowledgments |
| Abandoned upload | Confirm worker is terminal, retain recoverable checkpoint, then 14-day orphan grace |
| Original upstream sources | Never in this collector's deletion scope |

Per-page PDF takeover releases only that browser session's decoded stream resources.
It does not release the server-side stream: full page coverage must remain live
for future arbitrary history positions, slow PDF pages and failed PDF sessions.

If a searchable PDF is explicitly skipped or rejected, record that decision so
an input does not wait forever for a nonexistent build. Do not retain every
full-page PNG for all time; maintain only unresolved evidence after recovery.

Collector procedure: snapshot root generation; traverse all three buckets;
stop on missing/unreadable required indexes/manifests or unresolved dependencies;
persist first-unreferenced timestamps even in report mode; construct a bounded
deletion report; under the same promotion lock re-read the root/leases and
recheck eligibility; delete exact objects in bounded batches and verify each
deletion response. Concurrent uploads have leases; stale GC plans cannot delete
newly promoted objects. A failed traversal means no delete, not an empty graph.

Start report-only daily on the central repository. Enable apply only after
cross-bucket fixtures and repeated production reports agree. Keep a positive
grace period; no default zero-day apply. Ordinary GC has no force-JXL-delete mode.
Emergency deletion is outside this normal lifecycle.

## 10. Schedule And Admission

Suggested initial operational schedule, separate from deployment status:

- Central inventory/dispatch every six hours, using pinned sources and current
  component states. Claim before dispatch; a second tick never duplicates work.
- Each account: at most two compute jobs. Prioritize absent primary documents
  and currently demanded stream ranges; reserve progress for the remaining
  full-book stream, resumable OCR and optional optimization. Give untouched
  ranges/books a quota so repeated requests/failures cannot starve completion.
- Stage completion notifies central publication. Ready render inputs enqueue
  OCR, accepted text enqueues searchable-PDF generation; do not wait for the
  next daily batch to advance a healthy book.
- Dedicated OpenCode correction once daily under the bounded policy above.
- GC report daily after publication/correction admission, apply separately under
  the same global lock. Reports do not release active input dependencies.

The eight-account ownership expansion remains configuration-driven. Unverified
accounts are not scheduled. Model quota and GPU availability are independent
of the number of GitHub tokens; CPU render/OCR account lanes do not grant them.

## 11. Existing Implementation Gaps

These are local code observations, not production claims:

- `static/reader.js` now supports same-shell preview-to-PDF handoff inside the
  existing `pdf-pages` adapter. It initializes the associated PDF in parallel,
  verifies page count, keeps demand-range options and provides delayed image
  backfill. PDF-only and image-only adapters remain distinct. New SVG/PNG stream
  representations and the v3 partitioned manifest are still pending.
- Existing compact sidecars now carry optional `pd` (document path) and `pdb`
  (current bucket), projected as `pdf_document` / `ReaderPdfDocument` through
  API, search/session metadata and direct Reader links. Original PDF streams
  may use their original PDF download URL when no generated document is supplied.
  Converted inputs require an explicit associated PDF, not the CAJ/DJVU download.
  Only validated current-bucket PDF paths survive backend decoding. The v3
  page-map/digest identity checks are still stronger than this v2 page-count gate.
- `pdf_ocr_stages.py` currently persists lossless PNG inputs and independent
  text but builds full WebP page manifests for the non-preserved branch. It has
  no display/searchable-PDF assembly stage or mixed-codec indexed reading manifest.
- `publish_pdf_ocr_assets.py` now publishes OCR registry and sidecar together to
  `reader-assets-v2` for real HF API clients, without dataset fallback or a PDF
  bucket index copy. A matching OCR registry alone no longer skips sidecar repair.
  Publication now writes an immutable content-addressed sidecar snapshot before
  advancing the supplementary catalog. Idempotent retries preserve its generation.
  This is not an atomic v3 resolver promotion: applications still consume the
  existing sidecar. Other publication paths still need separate migration.
- `reader_gc_graph.py` is the only collector entrypoint. It follows current
  catalog generations, qualified references, render-range/OCR progress,
  processing roots and all known format indexes. Ambiguous unqualified references
  retain both copies. Unknown prefixes, missing roots/manifests and unreadable
  inventories suppress all observations.
 - Staged PDF object upload writes a unique `reader-index/processing/*.json` root
   before transferring either bucket's objects. Failed transfers retain that root.
   Successful upload changes its status to `uploaded`; central OCR publication
   advances a catalog generation and marks the record `released`. Released roots
   no longer retain objects, while processing/uploaded roots do. Heartbeats,
   rollback acknowledgments and protection records from the remaining producers
   are pending.
 - The GC workflow emits a report artifact and can record first-seen observations
   only after a complete graph. Deletion remains disabled until all producers use
   shared mutation coordination.
- Account workers and the new binary policy are local changes. There is no
  deployed daily OpenCode correction service or configured provider/model here.

The new report follows existing recorded OCR input dependencies, but cannot prove
that all historical unregistered work or future v3 review dependencies are recorded.
This distinction is why deletion remains disabled rather than equating an apparently
unreferenced object with a safely retired one.

## 12. Implementation And Acceptance Order

1. Contract and ownership: validate bucket-qualified resources, immutable page
   maps and catalogs, one canonical pointer, stage leases and partial progress.
   Eliminate legacy dataset publication/fallback in this pipeline path.
2. Reader handoff: deterministic preview/PDF fixtures for fresh/saved positions,
   zoom, selection, fast scroll, cancelled book replacement and failures. Keep
   existing search/history/download behavior. Implement both HF and Pages.
3. Classification and full stream partitions: real vector, CCITT, JBIG2, grayscale,
   color, mixed and malformed samples. No re-encoding of protected image streams;
   verify every page and resume point remains reachable. Explicitly test direct
   late-page restoration, incomplete partitions and a late PDF range after
   an earlier page has already transitioned to PDF.
4. Document candidates: pin tools; preserve full page identity and source pixels;
   prove searchable text mapping for Chinese/vertical/table examples. Compare
   original and candidates under real PDF.js ranges before automatic promotion.
5. OCR review: immutable task queue and crops, independently validated agent
   proposals, stale-patch rejection, idempotent daily budgets and acceptance audit.
   Run without provider access in offline tests; enable a configured pilot only.
6. GC: cross-bucket fake stores with a live PNG referenced only through a PDF
   render/range manifest, partial workers, expired leases, pending review,
   broken catalogs and promotion races. No deletion in these protected cases.
7. Small publication pilot: one book per representation family, verify remote
   generation and both production Readers, inspect GC dry-run, then scale lanes.

Measure first preview, first primary canvas, handoff scroll drift, page-100/deep
demand latency, viewport memory, bytes, PDF range behavior and exact text IDs/
totals separately. A first-page timing improvement alone does not accept the
cycle. Retain the known-good reading generation while later components fail.
