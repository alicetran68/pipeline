import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import pilot_office_stream


class OfficeStreamTests(unittest.TestCase):
    def test_doc_text_fallback_writes_sanitized_html(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(pilot_office_stream.subprocess, "run") as run:
            run.return_value.stdout = "legacy text".encode()
            source = Path(temporary) / "source.doc"
            output = Path(temporary) / "document.html"
            source.write_bytes(b"legacy")
            pilot_office_stream.doc_to_html_fallback(source, output)
            self.assertIn("legacy text", output.read_text(encoding="utf-8"))
            self.assertEqual(run.call_args.args[0][:3], ["antiword", "-m", "UTF-8.txt"])

    def test_detects_html_saved_with_doc_extension(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.doc"
            source.write_bytes(b"\xef\xbb\xbf<html><body>text</body></html>")
            self.assertTrue(pilot_office_stream.looks_like_html(source))

    def test_converted_output_accepts_normalized_basename(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "normalized.docx"
            output.write_bytes(b"docx")
            self.assertEqual(pilot_office_stream.converted_output(Path(temporary), "docx"), output)

    def test_libreoffice_uses_private_profile(self):
        process = Mock(returncode=0)
        process.communicate.return_value = ("", "")
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(pilot_office_stream.subprocess, "Popen", return_value=process) as popen:
            pilot_office_stream.run_libreoffice(
                ["libreoffice", "--headless", "--convert-to", "docx"],
                Path(temporary) / "profile",
            )

        command = popen.call_args.args[0]
        self.assertEqual(command[0], "libreoffice")
        self.assertTrue(command[1].startswith("-env:UserInstallation=file://"))
        self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_libreoffice_kills_process_group_after_timeout(self):
        process = Mock(pid=123, returncode=0)
        process.communicate.side_effect = [
            pilot_office_stream.subprocess.TimeoutExpired(["libreoffice"], 600),
            ("", ""),
        ]
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(pilot_office_stream.subprocess, "Popen", return_value=process), \
                patch.object(pilot_office_stream.os, "killpg") as killpg:
            with self.assertRaises(pilot_office_stream.subprocess.TimeoutExpired):
                pilot_office_stream.run_libreoffice(["libreoffice"], Path(temporary) / "profile")

        killpg.assert_called_once_with(123, pilot_office_stream.signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
