"""Fetch the exact benchmark files used by DCSN. No model dependencies."""

import argparse
from contextlib import ExitStack
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile


MANIFEST = Path(__file__).resolve().with_name("dataset_manifest.json")


def verify(path, entry):
    if path.stat().st_size == entry["bytes"]:
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() == entry["sha256"]:
                return
    raise ValueError(f"{entry['dataset']}: size or SHA-256 mismatch for {path}")


def fetch(root, entry):
    target = root / entry["path"]
    if target.exists():
        verify(target, entry)
        return "already verified"
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temp:
        payload = Path(temp) / "payload"
        with ExitStack() as stack:
            stream = stack.enter_context(urlopen(entry["url"], timeout=120))
            if "member" in entry:
                archive = stack.enter_context(tempfile.TemporaryFile(dir=temp))
                shutil.copyfileobj(stream, archive)
                zip_file = stack.enter_context(ZipFile(archive))
                stream = stack.enter_context(zip_file.open(entry["member"]))
            elif entry["url"].endswith(".gz"):
                stream = stack.enter_context(gzip.GzipFile(fileobj=stream))
            with payload.open("wb") as output:
                shutil.copyfileobj(stream, output)
        verify(payload, entry)
        # Publish a complete, verified file without replacing a concurrent download.
        try:
            target.hardlink_to(payload)
        except FileExistsError:
            verify(target, entry)
    return "downloaded and verified"


def main():
    entries = {row["dataset"]: row for row in json.loads(MANIFEST.read_text())}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", metavar="DATASET", nargs="*", help=", ".join(entries))
    parser.add_argument("--data-root", type=Path, default=MANIFEST.parents[1] / "datasets")
    args = parser.parse_args()
    for name in args.datasets:
        if name not in entries:
            parser.error(f"unknown dataset {name!r}; choose from {', '.join(entries)}")
    for name in args.datasets or entries:
        try:
            result = fetch(args.data_root, entries[name])
        except (OSError, ValueError) as exc:
            parser.exit(1, f"{name}: {exc}\n")
        print(f"{name}: {result}", flush=True)


if __name__ == "__main__":
    main()
