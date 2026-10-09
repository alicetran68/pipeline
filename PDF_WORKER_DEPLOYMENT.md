# Account Worker Deployment

Production rollout: 2026-10-09. This records authenticated remote observations,
not results from the local corpus or mocked OCR tests. Runs can advance after
these observations; their Actions links are the current status authority.

## Deployed Topology

- Central publisher: `anftm/pipeline`, initially deployed at `0671de53`, then
  advanced at `c7934c24` with explicit alternate worker repository validation.
- Workers share the same routing configuration and source modules. Changing a
  repository override does not invalidate a queue's unchanged source ownership.
- Each account runs at most two build jobs. Render admission is scheduled every
  two hours at minute 17 UTC. Central render publication dispatches that account's
  OCR stage; OCR publication retains incomplete progress and updates Pages.
- `HF_TOKEN` writes Reader/PDF objects. Separate `HF_INPUT_TOKEN` writes inputs
  to `melsm/pdf-archive-v2`. All credentials are Actions secrets, not source files.
- GC deletion remains absent. The daily `reader-gc.yml` reports through the current
  bucket Hub API; optional observation recording requires a complete graph.

## Initial Actual Work

Every initial queue had one real book, at least one build shard, and no failed
planning entries. Counts here are those books' verified planned page totals.

| Account | Scope | Pages | First Render Run |
| --- | --- | ---: | --- |
| vomebook | <5 MiB native | 68 | [37955524212](https://github.com/vomebook/pdf-pipeline/actions/runs/37955524212) |
| rioholland79 | 5-10 MiB native | 460 | [37954431059](https://github.com/rioholland79/pipeline/actions/runs/37954431059) |
| devondunn7 | 10-50 MiB native | 368 | [37954433237](https://github.com/devondunn7/pipeline/actions/runs/37954433237) |
| dellamcastillo | 50-100 MiB native | 242 | [37954432942](https://github.com/dellamcastillo/pipeline/actions/runs/37954432942) |
| anftm | 100-250 MiB native | 603 | [37954435405](https://github.com/anftm/pipeline/actions/runs/37954435405) |
| brodievsalas | 250-500 MiB native | 604 | [37954430279](https://github.com/brodievsalas/pipeline/actions/runs/37954430279) |
| alicetran68 | >=500 MiB native | 514 | [37954436884](https://github.com/alicetran68/pipeline/actions/runs/37954436884) |
| ambrossee768 | converted DJVU | 97 | [37955527954](https://github.com/ambrossee768/pdf-pipeline/actions/runs/37955527954) |

The vomebook and ambrossee768 old fork repositories rejected dispatch with
"Actions has been disabled for this repository" despite active permission API
settings. Dedicated `pdf-pipeline` repositories were created for those two
accounts; their obsolete worker schedules in `pipeline` were disabled.

## Verified Completion Boundaries

- Devondunn7's first render and notification succeeded. Central publication
  [37955525757](https://github.com/anftm/pipeline/actions/runs/37955525757) validated
  the source/range descriptors and published the complete 368-page book.
- Its read-only render-manifest verification found all 368 PNG references and
  no WebP references. Presentation evidence contained CCITT and JBIG2 filters;
  it remained `preserve-pdf`. 359 pages entered actual image OCR, while usable
  native pages retained extraction. OCR run
  [37955635656](https://github.com/devondunn7/pipeline/actions/runs/37955635656)
  was observed executing, not reported as a completed text generation.
- Vomebook's 68-page native book reached OCR registry `status=ready` with a
  complete text manifest, without image recognition for its valid native text.
- Rioholland79's 460-page mixed book reached `status=ready`; its progress registry
  contained all 13 recognized pages. Native pages were assembled independently.
- Ambrossee768's 97-page DJVU derivative reached `status=ready`; its progress
  registry contained all 97 recognized pages with no pending error entries.
- Anftm and brodievsalas completed first render/notification; their following OCR
  stages were observed executing. Dellamcastillo still had active render shards.
  Alicetran68 had completed one range and was executing/queuing remaining ranges
  under the two-build-job cap. Admission is not whole-book completion.

## Acceptance

- Deployment preparation passed 313 focused offline tests (one optional skip).
  Subsequent repository/schema/Hub changes passed their focused contract suites.
- HF production: all four `tests.test_live_smoke` cases passed with the documented
  explicit production URL. GitHub Pages `tests/test_live_smoke.js` passed.
- Those smoke suites verify availability and baseline contracts. They do not
  establish that every new book has completed OCR or that every browser/device
  has loaded the latest sidecar.
- Real worker writes and checksum-verified manifests establish current input
  bucket access via Hub credentials. Earlier S3 `NoSuchBucket` is a separate
  storage-access limitation and is not treated as an empty input bucket.
- No original upstream files or Reader content objects were garbage-collected.

The v3 mixed-codec builder, optimized/searchable PDF generation, daily correction
service and coordinated deleting GC remain separate pending implementation work.
