import io
import json
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts import download_pdf_render_results as results


def archive(name, payload):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as zipped:
        zipped.writestr(name, json.dumps(payload))
    return output.getvalue()


class RenderArtifactDownloadTests(unittest.TestCase):
    def test_paginated_results_keep_shard_order_and_validate_json(self):
        pages = [json.dumps({"total_count": 101, "artifacts": [
            {"name": f"pdf-render-results-{i}", "id": i + 1, "expired": False}
            for i in range(100)]}).encode(),
            json.dumps({"total_count": 101, "artifacts": [
                {"name": "pdf-render-results-100", "id": 101, "expired": False}]}).encode()]
        with patch.object(results, "api_get", side_effect=pages):
            found = results.result_artifacts("test/repo", 1, "token")
        self.assertEqual([a["name"] for a in found], [f"pdf-render-results-{i}" for i in range(101)])
        with tempfile.TemporaryDirectory() as temp:
            results.unpack_result(archive("results-100.json", {"version": 1, "results": []}), 100, Path(temp))
            self.assertEqual(json.loads((Path(temp) / "pdf-render-results-100/results-100.json").read_text()),
                             {"version": 1, "results": []})

    def test_invalid_and_unsafe_zip_entries_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            for name, payload in (("../results-1.json", {"version": 1, "results": []}),
                                  ("results-1.json", {"version": 0, "results": []})):
                with self.subTest(name=name, payload=payload), self.assertRaises(ValueError):
                    results.unpack_result(archive(name, payload), 1, Path(temp))

    def test_signed_redirect_does_not_receive_authorization(self):
        redirect = urllib.error.HTTPError("https://api.github.com/zip", 302, "redirect",
                                          {"Location": "https://blob.example/signed"}, None)
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
            def read(self):
                return b"zip"
        with patch.object(results.OPENER, "open", side_effect=redirect) as original, \
                patch.object(results.urllib.request, "urlopen", return_value=Response()) as signed:
            self.assertEqual(results.api_get("/zip", "secret", zip_archive=True), b"zip")
        self.assertIn("Authorization", original.call_args.args[0].headers)
        self.assertNotIn("Authorization", signed.call_args.args[0].headers)

    def test_secondary_limit_retries_without_hiding_other_403s(self):
        def error(message):
            return urllib.error.HTTPError("https://api.github.com/zip", 403, "forbidden", {},
                                          io.BytesIO(message.encode()))
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
            def read(self):
                return b"ok"
        with patch.object(results.OPENER, "open", side_effect=[error("secondary rate limit"), Response()]), \
                patch.object(results.time, "sleep") as sleep:
            self.assertEqual(results.api_get("/zip", "secret"), b"ok")
            sleep.assert_called_once_with(10)
        with patch.object(results.OPENER, "open", side_effect=error("permission denied")), \
                self.assertRaises(RuntimeError):
            results.api_get("/zip", "secret")


if __name__ == "__main__":
    unittest.main()
