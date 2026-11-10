"""Download and verify the NASA C-MAPSS turbofan degradation dataset.

Raw files are written to ``data/raw/cmapss/`` and treated as immutable.
Downloads are idempotent: an archive that exists and matches its SHA-256
checksum is never fetched again.

Run with: ``uv run python -m predmaint.data.download``
"""

import argparse
import hashlib
import io
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Official NASA Prognostics Center of Excellence (PCoE) download, linked from
# https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
# ("6. Turbofan Engine Degradation Simulation Data Set").
DATASET_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"

# SHA-256 of the archive above, computed from a verified download on 2026-11-10
# (12,429,152 bytes, Last-Modified 2022-09-18).
EXPECTED_SHA256 = "c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2"

# Anchor to the project root (src/predmaint/data/download.py -> 3 levels up), so the
# data lands in the same place no matter which directory the code is run from.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cmapss"
ARCHIVE_NAME = "cmapss.zip"

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
EXPECTED_FILES = tuple(
    f"{kind}_{subset}.txt" for subset in SUBSETS for kind in ("train", "test", "RUL")
)


class ChecksumError(Exception):
    """Raised when a file's SHA-256 digest does not match the expected value."""


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 hex digest of a file, reading it in chunks.

    Args:
        path: File to hash.
        chunk_size: Bytes read per iteration; keeps memory use constant.

    Returns:
        Lowercase hexadecimal digest (64 characters).
    """
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected: str) -> None:
    """Check that a file's SHA-256 digest matches the expected value.

    Args:
        path: File to verify.
        expected: Expected hex digest (case-insensitive).

    Raises:
        ChecksumError: If the digests differ.
    """
    actual = sha256_of_file(path)
    if actual != expected.lower():
        raise ChecksumError(f"Checksum mismatch for {path}: expected {expected}, got {actual}")


def is_valid_file(path: Path, expected: str) -> bool:
    """Return True if ``path`` exists and its SHA-256 digest matches ``expected``."""
    if not path.is_file():
        return False
    try:
        verify_checksum(path, expected)
    except ChecksumError:
        return False
    return True


def download_file(url: str, dest: Path, *, timeout: float = 60.0) -> Path:
    """Stream a URL to disk atomically.

    Writes to ``dest`` with a ``.part`` suffix first and renames it to ``dest``
    only after the transfer completes, so an interrupted download never leaves
    a file that looks complete. Creates parent directories as needed.

    Args:
        url: HTTP(S) URL to fetch.
        dest: Final path of the downloaded file.
        timeout: Socket timeout in seconds.

    Returns:
        The path ``dest``.

    Raises:
        urllib.error.URLError: On network failure (the partial file is removed).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response, part.open("wb") as f:
            shutil.copyfileobj(response, f)
        part.replace(dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return dest


def _extract_zip(zf: zipfile.ZipFile, dest_root: Path) -> list[Path]:
    """Extract every member of an open zip into ``dest_root``, recursing into nested zips."""
    extracted: list[Path] = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        if info.filename.lower().endswith(".zip"):
            # Nested archive: read it into memory and extract its contents
            # directly into dest_root (the nested zip itself is not kept).
            with zipfile.ZipFile(io.BytesIO(zf.read(info))) as inner:
                extracted.extend(_extract_zip(inner, dest_root))
            continue
        target = (dest_root / info.filename).resolve()
        if not target.is_relative_to(dest_root):
            raise ValueError(f"Unsafe path in archive: {info.filename!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        extracted.append(target)
    return extracted


def extract_archive(archive: Path, dest_dir: Path) -> list[Path]:
    """Extract a zip archive safely, including any zip files nested inside it.

    Contents of nested zips are extracted into ``dest_dir`` itself rather than
    next to the nested zip. Every member's resolved path must stay inside
    ``dest_dir`` (protection against "zip slip" path traversal).

    Args:
        archive: Zip file to extract.
        dest_dir: Directory to extract into.

    Returns:
        Paths of all extracted regular files (not directories or nested zips).

    Raises:
        ValueError: If a member would be written outside ``dest_dir``.
    """
    dest_root = dest_dir.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        return _extract_zip(zf, dest_root)


def download_cmapss(
    raw_dir: Path = DEFAULT_RAW_DIR,
    *,
    url: str = DATASET_URL,
    expected_sha256: str = EXPECTED_SHA256,
    force: bool = False,
) -> Path:
    """Ensure the C-MAPSS dataset is downloaded, verified and extracted.

    Idempotent: if the archive already exists and its checksum matches, it is
    not downloaded again, and extraction is skipped when all data files exist.

    Args:
        raw_dir: Directory holding the raw archive and extracted files.
        url: Where to download the archive from.
        expected_sha256: SHA-256 the archive must match.
        force: Re-download and re-extract even if valid files are present.

    Returns:
        The directory containing the extracted ``train_FD00x.txt``,
        ``test_FD00x.txt`` and ``RUL_FD00x.txt`` files.

    Raises:
        ChecksumError: If the downloaded archive fails verification
            (the bad file is deleted).
    """
    archive = raw_dir / ARCHIVE_NAME

    if not force and is_valid_file(archive, expected_sha256):
        logger.info("Archive already present and verified: %s", archive)
    else:
        logger.info("Downloading %s -> %s", url, archive)
        download_file(url, archive)
        try:
            verify_checksum(archive, expected_sha256)
        except ChecksumError:
            archive.unlink()
            raise
        logger.info("Checksum verified")

    missing = [name for name in EXPECTED_FILES if not (raw_dir / name).is_file()]
    if force or missing:
        extract_archive(archive, raw_dir)
        logger.info("Extracted archive into %s", raw_dir)
    else:
        logger.info("All data files already extracted")

    return raw_dir


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Download the NASA C-MAPSS dataset.")
    parser.add_argument("--force", action="store_true", help="re-download and re-extract")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    path = download_cmapss(force=args.force)
    print(f"C-MAPSS data ready in {path}")


if __name__ == "__main__":
    main()
