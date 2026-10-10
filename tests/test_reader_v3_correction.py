import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import httpx

from scripts import reader_v3_correction as correction, pdf_text_layer as text, publish_reader_v3 as publication
from tests.test_pdf_text_layer import raw_page
from tests import test_publish_reader_v3 as fixtures


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.layer = text.from_page(raw_page("wrong", confidence=.8), "a" * 64)
        self.answer = {"replacements": [{"region_id": "b0", "before": "wrong", "after": "right", "reason": "image"}],
                       "unresolved": []}

    def test_unknown_duplicate_and_unchanged_model_regions_rejected(self):
        for changes in ([{**self.answer["replacements"][0], "region_id": "b99"}],
                        self.answer["replacements"] * 2,
                        [{**self.answer["replacements"][0], "after": "wrong"}]):
            with self.assertRaises(ValueError):
                correction.validate_answer(self.layer, {"replacements": changes, "unresolved": []})
        proposal = correction.validate_answer(self.layer, self.answer)
        accepted = text.accept_proposal(self.layer, proposal, actor="reviewer")
        self.assertEqual(accepted["text"], "right")
        self.assertEqual(self.layer["text"], "wrong")

    def test_unique_literal_patch_expands_without_changing_other_text(self):
        layer = text.from_page(raw_page("prefix wrong suffix"), "a" * 64)
        proposal = correction.validate_answer(layer, self.answer)
        self.assertEqual(proposal["replacements"][0]["before"], "prefix wrong suffix")
        self.assertEqual(proposal["replacements"][0]["after"], "prefix right suffix")
        for content in ("wrong wrong", "unrelated"):
            layer = text.from_page(raw_page(content), "a" * 64)
            unresolved = correction.validate_answer(layer, self.answer)
            self.assertEqual(unresolved["replacements"], [])
            self.assertTrue(unresolved["unresolved"])

    def test_provider_has_image_exact_model_bounds_and_no_redirects(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "task.json").write_text('{"text":"untrusted book text"}')
            (workspace / "page.png").write_bytes(b"image")
            def handler(request):
                body = json.loads(request.content)
                self.assertEqual(body["model"], "gpt-6-luna")
                self.assertEqual(body["max_output_tokens"], 4096)
                self.assertEqual(body["input"][0]["content"][1]["type"], "input_image")
                self.assertNotIn("tools", body)
                return httpx.Response(200, json={"status": "completed", "model": correction.MODEL,
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(self.answer)}]}],
                    "usage": {"input_tokens": 12, "output_tokens": 5, "secret": "never persisted"}})
            with patch.dict(os.environ, {"OCR_CORRECTION_API_BASE": "https://provider.example/v1", "OCR_CORRECTION_API_KEY": "test"}):
                result = correction.model_request(workspace, transport=httpx.MockTransport(handler))
            self.assertEqual(result["answer"], self.answer)
            self.assertNotIn("secret", result["usage"])

    def test_isolated_worker_environment_excludes_publication_credentials(self):
        with patch.dict(os.environ, {"HF_TOKEN": "secret", "GH_TOKEN": "secret", "OCR_CORRECTION_API_KEY": "model"}):
            with patch.object(correction.subprocess, "run") as run:
                run.return_value.returncode = 0
                run.return_value.stdout = b'{"answer":{}}'
                correction.isolated_model(Path("/tmp/workspace"))
                env = run.call_args.kwargs["env"]
                self.assertNotIn("HF_TOKEN", env)
                self.assertNotIn("GH_TOKEN", env)
                self.assertEqual(env["OCR_CORRECTION_API_KEY"], "model")

    def test_worker_failure_exposes_only_a_bounded_provider_code(self):
        with patch.object(correction.subprocess, "run") as run:
            run.return_value.returncode = 1
            run.return_value.stderr = b'{"code":"provider-http-403"}'
            with self.assertRaisesRegex(correction.ModelRequestError, "provider-http-403"):
                correction.isolated_model(Path("/tmp/workspace"))
            run.return_value.stderr = b"unexpected provider body with private details"
            with self.assertRaisesRegex(correction.ModelRequestError, "isolated-request-failed"):
                correction.isolated_model(Path("/tmp/workspace"))

    def test_proposal_does_not_publish_until_explicit_accept_and_visual_resources_reused(self):
        fixture = fixtures.V3PublicationTests()
        fixture.setUp()
        try:
            ref = fixture.candidate(label="wrong")
            staged = publication.stage(fixture.store, ref, fixture.bundle, apply=True)
            publication.promote(fixture.store, staged["candidate"], None, apply=True)
            # Fixture has confidence .99; add a bounded task directly to exercise the service.
            reading = publication.v3.decode(fixture.store.read_bytes(ref["bucket"], ref["path"]))
            manifest = publication.v3.decode(fixture.store.read_bytes(reading["text_layer"]["bucket"], reading["text_layer"]["path"]))
            partition = publication.v3.decode(fixture.store.read_bytes(manifest["partitions"][0]["resource"]["bucket"], manifest["partitions"][0]["resource"]["path"]))
            page_ref = partition["pages"][0]["resource"]
            layer = publication.v3.decode(fixture.store.read_bytes(page_ref["bucket"], page_ref["path"]))
            identity = "b" * 64
            state = correction.load_state(fixture.store)
            state["tasks"][identity] = {"source_key": "repo\0book.pdf", "source_sha256": layer["source_sha256"],
                "page": 1, "base_generation": layer["generation"], "reading": ref, "text_layer": page_ref,
                "issues": ["low-recognition-confidence"], "status": "pending", "attempts": 0}
            cached = {"version": 1, "kind": "reader-v3-rejected-correction", "task_id": identity,
                      "resources": [ref, page_ref], "result": {"answer": self.answer, "usage": {"input_tokens": 100}}}
            raw_cached = publication.encode(cached)
            cached_path = "reader-index/v3/corrections/rejected/" + identity + "/result.json"
            fixture.store.put_bytes(publication.ASSETS, cached_path, raw_cached)
            state["tasks"][identity]["rejected_result"] = publication.metadata(publication.ASSETS, cached_path, raw_cached)
            correction.save(fixture.store, state)
            before = publication.current(fixture.store)[0]
            provider = Mock(side_effect=AssertionError("must reuse the saved provider output"))
            result = correction.correct(fixture.store, {"limit": 1}, apply=True,
                invoke=provider)
            self.assertEqual(result["processed"][0]["status"], "proposed")
            self.assertEqual(result["budget"]["requests"], 0)
            provider.assert_not_called()
            self.assertEqual(publication.current(fixture.store)[0], before)
            writes = len(fixture.store.writes)
            with patch.dict(os.environ, {"GITHUB_ACTOR": "reviewer"}):
                options = {"task_ids": [identity], "regions": {identity: ["b0"]}}
                accepted = correction.decide(fixture.store, "accept", options, apply=True)
                retried = correction.decide(fixture.store, "accept", options, apply=True)
            self.assertTrue(accepted["visual_resources_reused"])
            self.assertTrue(retried["unchanged"])
            self.assertFalse(any(path.endswith((".png", ".webp", ".pdf")) for bucket, path in fixture.store.writes[writes:]))
            active = publication.current(fixture.store)[1]["files"]["repo\0book.pdf"]
            verified = publication.v3.verify_reading(active["resource"], lambda r: fixture.store.read_bytes(r["bucket"], r["path"]))
            self.assertNotEqual(verified["text_layer"], reading["text_layer"])
            self.assertEqual(verified["primary"], reading["primary"])
        finally:
            fixture.temp.cleanup()
