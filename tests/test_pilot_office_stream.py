import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts import pilot_office_stream


class OfficeStreamTests(unittest.TestCase):
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
