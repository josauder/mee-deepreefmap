from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path


URL = "https://zenodo.org/records/10624794/files/example_data.zip?download=1"
ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "example_data"
ARCHIVE = DESTINATION / "example_data.zip"
PARTIAL_ARCHIVE = ARCHIVE.with_suffix(".zip.part")
EXPECTED_OUTPUTS = [
    DESTINATION / "input_videos" / "GX_SINGLE_VIDEO.MP4",
    DESTINATION / "input_videos" / "GX_VIDEO_1_OF_2.MP4",
    DESTINATION / "input_videos" / "GX_VIDEO_2_OF_2.MP4",
]


def format_gib(size: int) -> str:
    return f"{size / 1024**3:.1f} GiB"


def download() -> None:
    with urllib.request.urlopen(URL) as response, PARTIAL_ARCHIVE.open("wb") as archive:
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        next_report = 0

        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            archive.write(chunk)
            downloaded += len(chunk)

            if downloaded >= next_report:
                if total:
                    print(f"Downloaded {format_gib(downloaded)} / {format_gib(total)}", flush=True)
                else:
                    print(f"Downloaded {format_gib(downloaded)}", flush=True)
                next_report = downloaded + 100 * 1024 * 1024

    PARTIAL_ARCHIVE.replace(ARCHIVE)


def normalize_extracted_layout() -> None:
    nested = DESTINATION / "example_data"
    if not nested.exists():
        return

    for path in nested.iterdir():
        destination = DESTINATION / path.name
        if destination.exists():
            continue
        path.rename(destination)


def main() -> int:
    DESTINATION.mkdir(exist_ok=True)

    if all(path.exists() for path in EXPECTED_OUTPUTS):
        print("Example data already exists; nothing to download.")
        return 0

    print(f"Downloading example data to {ARCHIVE}")
    download()

    print(f"Extracting {ARCHIVE} to {DESTINATION}")
    with zipfile.ZipFile(ARCHIVE) as archive:
        archive.extractall(DESTINATION)
    normalize_extracted_layout()

    ARCHIVE.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
