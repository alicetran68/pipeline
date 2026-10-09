# Account PDF Workers

The requested end-to-end redesign is specified in `PDF_READING_CYCLE_V3.md`.
The account-worker path was deployed and started on 2026-10-09. This file describes
that render/OCR and binary-preservation implementation. The proposed v3 document
builder, correction service and deleting GC are not deployed features.

## Ownership

`pdf-worker-lanes.json` defines exclusive render/OCR ownership. GitHub-hosted
jobs run in each account's own `pipeline` repository; authenticating a dispatch
with another account's token does not move the job out of the target repository.

| Account | Worker Repository | Lane | Original Input |
| --- | --- | --- | --- |
| vomebook | vomebook/pdf-pipeline | native-small | Native PDF below 5 MiB |
| rioholland79 | rioholland79/pipeline | native-small-upper | Native PDF from 5 MiB inclusive to 10 MiB exclusive |
| devondunn7 | devondunn7/pipeline | native-small-mid | Native PDF from 10 MiB inclusive to 50 MiB exclusive |
| dellamcastillo | dellamcastillo/pipeline | native-medium-lower | Native PDF from 50 MiB inclusive to 100 MiB exclusive |
| anftm | anftm/pipeline | native-medium | Native PDF from 100 MiB inclusive to 250 MiB exclusive |
| brodievsalas | brodievsalas/pipeline | native-medium-upper | Native PDF from 250 MiB inclusive to 500 MiB exclusive |
| alicetran68 | alicetran68/pipeline | native-large | Native PDF at least 500 MiB, or unknown native size |
| ambrossee768 | ambrossee768/pdf-pipeline | converted | Generated PDFs whose original source is not PDF |

DJVU, CAJ, KDH and other converted inputs retain their original extension even
when the readable artifact is PDF. Repaired native PDFs remain in the native
lanes; their original size takes precedence over the repaired artifact's size.
Planning freezes that routing size before probing so render and OCR agree.

### Binary PDF Presentation

Size lanes assign compute ownership, not output codecs. The planner inspects
image metadata on every page using PyMuPDF, including inline images. A 1-bit
raster (including CCITT/JBIG2 images) selects `preserve-pdf` for the book, even
when it also contains color pages or embedded text. Original PDF bytes are not
modified. OCR receives separate lossless PNG inputs and publishes independent
page text and a complete book-text index; no WebP/JXL reading stream is produced.

Failed image inspection also selects `preserve-pdf` with an explicit incomplete
inspection reason. Small RGB PDFs are not labeled binary merely because of their
size. Existing text-only native PDF handling remains distinct. This is a
conservative binary-preservation rule, not a complete classifier for vector,
grayscale and complex-layout publications.

The render profile includes the preservation-policy version so old image ranges
cannot be reused under the new policy. Publishing a preserved PDF retires an old
page-stream route while retaining completed OCR text; generated sources return
to their generated PDF document. The older monolithic OCR builder rejects
preserved scanned PDFs and directs them through the independent render/OCR stages.

These workers consume ready PDF derivatives. They do not replace the initial
DJVU/CAJ/KDH converters or enable missing conversion backends. Moving those
converter workflows to the dedicated account is a separate deployment step.

Additional verified accounts can split `native_lanes` into more contiguous
ranges. The first range must start at zero, the last must have `max_bytes: null`,
and account owners and lane names must be unique. Bounds are integer bytes,
lower-inclusive and upper-exclusive. Unknown sizes go to the last native lane.
All worker repositories and the publisher must use the same configuration
generation; queues with an old configuration hash are rejected.
Optional repository overrides are checked independently by the central run
validator. They do not change routing fingerprints or invalidate already queued
work whose source ownership is unchanged.

## Execution And Publication

`pdf-account-worker.yml` is manually dispatched with `stage=render` or `stage=ocr`
and a book limit between 1 and 100. It resolves ownership from the repository
owner, filters before source downloads or OCR manifest scans, and admits at most
two build jobs. Only render preparation fetches source metadata and installs
Poppler; OCR consumes checksummed PNG inputs from the published render registry.
It also admits render work every two hours; central render publication dispatches
the same account's OCR queue afterward. Scheduled runs use the same ownership
validation as manual runs.

Workers upload immutable objects and result artifacts, then notify the configured
publisher, currently `anftm/pipeline`. Workers do not update shared registries or
the search sidecar. They now write their own uniquely named processing protection
record to `reader-assets-v2/reader-index/processing/` before object upload. The
worker token therefore also needs write access to that current bucket. A failed
upload retains protection; the central publisher releases records only after
the registry, sidecar and catalog generation are published. Heartbeats and
rollback acknowledgments remain pending, so GC deletion is still disabled.
The central `pdf-account-publish.yml` workflow:

1. Checks the configured account, repository, main branch, workflow identity,
   dispatch event and completed run. Failed runs may publish verified partial
   progress; cancelled runs are rejected.
2. Downloads the run's queue and checks stage, owner, lane, configuration hash
   and all planned books before downloading results.
3. Uses the existing render/OCR publisher under the central `reader-sidecar`
   concurrency lock, then publishes the search Reader sidecar.

Current buckets remain `vomebook/reader-assets-v2`, `vomebook/pdf-pages-v2` and
`melsm/pdf-archive-v2`. There is no legacy bucket fallback in this ownership path.

## Configuration

Deploy identical scripts, workflows and lane configuration to each account's
`pipeline` repository and to the central publisher before dispatching workers.
Required secrets are:

| Repository | Secret | Purpose |
| --- | --- | --- |
| Each worker | HF_TOKEN | Source/index reads and immutable object uploads |
| Each worker and publisher | HF_INPUT_TOKEN | Separate melsm input-bucket writes |
| Each worker | PDF_PUBLISH_TOKEN | Dispatch the central publication workflow |
| Central publisher | PDF_WORKER_READ_TOKEN | Read worker run identities and artifacts |
| Central publisher | HF_TOKEN | Publish verified progress, registries and objects |
| Central publisher | PAGES_TOKEN, PAGES_REPO | Existing search sidecar publication |

Keep credentials in Actions secrets. No account passwords or token values belong
in the lane configuration. Once the account-worker path is deployed, use it for
account-owned batches rather than concurrently starting the older unscoped
manual render/OCR workflows against the same inputs.

On 2026-10-09, supplied PATs authenticated `devondunn7`, `dellamcastillo` and
`rioholland79`; all three are now included in the local lane configuration. The earlier screenshot
username `devonduhn7` was a transcription error; the authenticated login is
`devondunn7`. Browser/email verification is not required for PAT API access.
The additional token authenticated `brodievsalas`, which is now assigned the
250-500 MiB lane. All eight account repositories, Actions secrets and workflows
were deployed on 2026-10-09 and admitted actual source books.

The old vomebook/ambrossee768 fork repositories rejected dispatch despite enabled
API settings. Dedicated `pdf-pipeline` repositories were created for those two
accounts and their obsolete worker schedules were disabled.

The first verified complete render/central publication was devondunn7 run
`37954433237` and central run `37955525757`. Its 368-page mixed CCITT/JBIG2 book
published verified PNG inputs without WebP reading recompression, then automatically
started OCR run `37955635656`. See `PDF_WORKER_DEPLOYMENT.md` for run evidence.

## Verification

```bash
python3 -B -m unittest tests.test_pdf_worker_lanes tests.test_pdf_reader_presentation tests.test_pdf_ocr_stages tests.test_pdf_render_schedule tests.test_pdf_ocr -q
python3 scripts/pdf_worker_lanes.py --owner vomebook
python3 scripts/pdf_worker_lanes.py --owner anftm
```

The offline tests check exclusive and complete boundary assignment, unknown
sizes, repaired/generated sources, eight-account configuration expansion,
filtering before expensive work in both planners, queue generation rejection,
run identity, shared publication ownership and workflow shell syntax. They do
not claim a remote workflow run or production page-stream publication passed.
The binary tests use real tiny CCITT PDFs and Poppler PNG rendering, with the OCR
engine mocked. They check complete text assembly, unchanged source bytes, absent
reader image objects, range reassembly and retirement of older page-stream routes.
