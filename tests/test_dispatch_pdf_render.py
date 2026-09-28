import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from scripts import dispatch_pdf_render as controller


class Response(io.BytesIO):
    def __init__(self, data, status=200):
        super().__init__(data)
        self.status = status


class DispatchPdfRenderTests(unittest.TestCase):
    def test_pending_queued_and_running_batches_prevent_duplicate_dispatch(self):
        for status in controller.ACTIVE:
            with self.subTest(status=status), patch.object(controller, "urlopen", return_value=Response(
                    json.dumps({"workflow_runs": [{"status": status}]}).encode())) as open_url:
                self.assertFalse(controller.dispatch("anftm/pipeline", "token"))
                self.assertEqual(open_url.call_count, 1)

    def test_idle_dispatches_one_book_from_main(self):
        def respond(request, timeout):
            self.assertEqual(timeout, 30)
            self.assertEqual(request.get_header("Authorization"), "Bearer token")
            if request.get_method() == "GET":
                self.assertIn("/runs?per_page=100", request.full_url)
                return Response(b'{"workflow_runs":[{"status":"completed"}]}')
            self.assertEqual(request.full_url, "https://api.github.com/repos/anftm/pipeline/actions/workflows/pdf-render-inputs.yml/dispatches")
            self.assertEqual(json.loads(request.data), {"ref": "main", "inputs": {
                "limit": "1", "checkpoint": "0", "retry_failed": "true",
                "continue_queue": "true"}})
            return Response(b"", status=204)
        with patch.object(controller, "urlopen", side_effect=respond) as open_url:
            self.assertTrue(controller.dispatch("anftm/pipeline", "token"))
            self.assertEqual(open_url.call_count, 2)

    def test_errors_or_missing_credentials_cannot_start_another_batch(self):
        with patch.object(controller, "urlopen") as open_url:
            with self.assertRaisesRegex(ValueError, "required"):
                controller.dispatch("anftm/pipeline", "")
            open_url.assert_not_called()
        for body in (b"{}", b'{"workflow_runs":{}}', b'{"workflow_runs":[{}]}'):
            with self.subTest(body=body), patch.object(controller, "urlopen", return_value=Response(body)) as open_url:
                with self.assertRaisesRegex(ValueError, "invalid render workflow"):
                    controller.dispatch("anftm/pipeline", "token")
                self.assertEqual(open_url.call_count, 1)
        with patch.object(controller, "urlopen", side_effect=OSError("GitHub unavailable")) as open_url:
            with self.assertRaises(OSError):
                controller.dispatch("anftm/pipeline", "token")
            self.assertEqual(open_url.call_count, 1)

    def test_only_controller_is_scheduled_and_uses_serial_dispatch(self):
        root = Path(__file__).resolve().parents[1]
        controller_workflow = yaml.safe_load((root / ".github/workflows/scheduled-pdf-render-inputs.yml").read_text())
        renderer = yaml.safe_load((root / ".github/workflows/pdf-render-inputs.yml").read_text())
        self.assertIn("schedule", controller_workflow[True])
        self.assertNotIn("schedule", renderer[True])
        self.assertEqual(controller_workflow["concurrency"]["queue"], "max")
        self.assertEqual(renderer["concurrency"]["group"],
                         "${{ inputs.render_lane && format('pdf-render-inputs-{0}', inputs.render_lane) || 'pdf-render-inputs' }}")
        self.assertEqual(renderer[True]["workflow_dispatch"]["inputs"]["render_lane"]["default"], "")
        self.assertIn("scripts/dispatch_pdf_render.py", controller_workflow["jobs"]["dispatch"]["steps"][-1]["run"])
        self.assertFalse((root / ".github/workflows/scheduled-pdf-render.yml").exists())
        self.assertIn("continue_queue", renderer[True]["workflow_dispatch"]["inputs"])
        self.assertIn("notify-controller", renderer["jobs"])
        self.assertIn("book_count", renderer["jobs"]["plan"]["outputs"])


if __name__ == "__main__":
    unittest.main()
