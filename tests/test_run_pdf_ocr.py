import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import requests
from huggingface_hub.errors import HfHubHTTPError

from scripts import run_pdf_ocr


class RunPdfOcrTests(unittest.TestCase):
    def test_upload_lists_only_each_book_prefix_and_preserves_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            relative = Path("objects") / "aa" / ("a" * 64) / ("b" * 16)
            root = bundle / relative
            root.mkdir(parents=True)
            (root / "ocr-manifest.json").write_text("{}")
            with patch.object(run_pdf_ocr, "sync_bucket") as sync:
                run_pdf_ocr.upload_ocr_objects(bundle)
            self.assertEqual(sync.call_args.args, (str(root), f"hf://buckets/vomebook/pdf-pages-v2/{relative.as_posix()}"))
            self.assertIsNone(sync.call_args.kwargs["include"])

    def test_bucket_sync_honors_retry_after(self):
        response = requests.Response()
        response.status_code = 429
        response.headers["Retry-After"] = "17"
        response.request = requests.Request(
            "POST", "https://huggingface.co/api/buckets/vomebook/pdf-pages/tree"
        ).prepare()
        error = HfHubHTTPError("rate limited", response=response)

        with patch.object(run_pdf_ocr, "sync_bucket", side_effect=[error, None]) as sync, \
                patch.object(run_pdf_ocr.time, "sleep") as sleep:
            run_pdf_ocr._sync_bucket_with_retry("/tmp/book", "hf://buckets/vomebook/pdf-pages", "token")

        self.assertEqual(sync.call_count, 2)
        sleep.assert_called_once_with(17)

    def test_bucket_sync_retries_transient_server_error(self):
        response = requests.Response()
        response.status_code = 503
        response.request = requests.Request(
            "POST", "https://huggingface.co/api/buckets/vomebook/pdf-pages/tree"
        ).prepare()
        error = HfHubHTTPError("busy", response=response)

        with patch.object(run_pdf_ocr, "sync_bucket", side_effect=[error, None]), \
                patch.object(run_pdf_ocr.time, "sleep") as sleep:
            run_pdf_ocr._sync_bucket_with_retry("/tmp/book", "hf://buckets/vomebook/pdf-pages", "token")

        sleep.assert_called_once_with(5)


if __name__ == "__main__":
    unittest.main()
