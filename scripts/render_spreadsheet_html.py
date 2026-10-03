#!/usr/bin/env python3
"""Export every Calc sheet separately as HTML, without creating a PDF."""

import argparse
import csv
from pathlib import Path
import socket
import subprocess
import tempfile
import time


def property_value(uno, name, value):
    result = uno.createUnoStruct("com.sun.star.beans.PropertyValue")
    result.Name = name
    result.Value = value
    return result


def csv_filter_options(source: Path) -> str:
    raw = source.read_bytes()[:8192]
    sample = None
    for encoding in ("utf-8-sig", "gb18030", "cp1252"):
        try:
            sample = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if sample is None:
        raise RuntimeError("spreadsheet CSV text encoding is unreadable")
    try:
        separator = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        separator = ","
    return f"{ord(separator)},34,76,1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    import uno

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="lo-spreadsheet-render-") as profile:
        process = subprocess.Popen([
            "libreoffice", "--headless", "--nologo", "--nodefault", "--nofirststartwizard",
            "-env:UserInstallation=" + Path(profile).resolve().as_uri(),
            f"--accept=socket,host=127.0.0.1,port={port};urp;StarOffice.ServiceManager",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        document = None
        try:
            local = uno.getComponentContext()
            resolver = local.ServiceManager.createInstanceWithContext(
                "com.sun.star.bridge.UnoUrlResolver", local
            )
            remote = None
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                try:
                    remote = resolver.resolve(
                        f"uno:socket,host=127.0.0.1,port={port};urp;StarOffice.ComponentContext"
                    )
                    break
                except Exception:
                    if process.poll() is not None:
                        raise RuntimeError("LibreOffice stopped before spreadsheet export")
                    time.sleep(0.25)
            if remote is None:
                raise RuntimeError("LibreOffice UNO service did not start")
            desktop = remote.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", remote)
            source_url = uno.systemPathToFileUrl(str(args.source.resolve()))
            load_properties = (
                property_value(uno, "Hidden", True),
                property_value(uno, "ReadOnly", True),
                property_value(uno, "UpdateDocMode", 3),
            )
            if args.source.suffix.lower() == ".csv":
                filter_options = csv_filter_options(args.source)
                load_properties += (
                    property_value(uno, "FilterName", "Text - txt - csv (StarCalc)"),
                    property_value(uno, "FilterOptions", filter_options),
                )
            try:
                document = desktop.loadComponentFromURL(source_url, "_blank", 0, load_properties)
            except Exception as automatic_error:
                if args.source.suffix.lower() != ".xls":
                    raise
                try:
                    document = desktop.loadComponentFromURL(
                        source_url, "_blank", 0,
                        (*load_properties, property_value(uno, "FilterName", "MS Excel 97")),
                    )
                except Exception:
                    raise automatic_error
            if document is None:
                raise RuntimeError("LibreOffice could not open the spreadsheet")
            sheets = document.getSheets()
            names = list(sheets.getElementNames())
            if not names:
                raise RuntimeError("spreadsheet has no worksheets")
            for index, name in enumerate(names, 1):
                sheets.getByName(name).IsVisible = True
                for other in names:
                    if other != name:
                        sheets.getByName(other).IsVisible = False
                document.getCurrentController().setActiveSheet(sheets.getByName(name))
                directory = args.output / f"sheet-{index:04d}"
                directory.mkdir(parents=True, exist_ok=True)
                target = directory / "sheet.html"
                document.storeToURL(
                    uno.systemPathToFileUrl(str(target.resolve())),
                    (property_value(uno, "FilterName", "HTML (StarCalc)"),
                     property_value(uno, "Overwrite", True)),
                )
                if not target.is_file() or target.stat().st_size == 0:
                    raise RuntimeError(f"LibreOffice produced no HTML for worksheet {index}")
            print(len(names))
            return 0
        finally:
            if document is not None:
                try:
                    document.close(True)
                except Exception:
                    pass
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
