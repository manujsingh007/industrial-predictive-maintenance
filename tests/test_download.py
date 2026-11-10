"""Offline tests for predmaint.data.download (no network access)."""

import hashlib
import io
import shutil
import zipfile
from pathlib import Path

import pytest

from predmaint.data import download
from predmaint.data.download import (
    EXPECTED_FILES,
    ChecksumError,
    download_cmapss,
    download_file,
    extract_archive,
    is_valid_file,
    sha256_of_file,
    verify_checksum,
)


def make_nested_archive(path: Path) -> Path:
    """Build a small zip shaped like NASA's: an outer folder holding CMAPSSData.zip."""
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        for name in EXPECTED_FILES:
            zf.writestr(name, f"contents of {name}\n")
        zf.writestr("readme.txt", "readme\n")
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("6. Turbofan Engine Degradation Simulation Data Set/", "")
        zf.writestr(
            "6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip", inner.getvalue()
        )
    return path


@pytest.fixture
def source_archive(tmp_path: Path) -> Path:
    return make_nested_archive(tmp_path / "source.zip")


@pytest.fixture
def fake_download(monkeypatch: pytest.MonkeyPatch, source_archive: Path) -> list[str]:
    """Replace the network download with a local copy; returns the list of calls."""
    calls: list[str] = []

    def fake(url: str, dest: Path, *, timeout: float = 60.0) -> Path:
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_archive, dest)
        return dest

    monkeypatch.setattr(download, "download_file", fake)
    return calls


# --- checksums ---------------------------------------------------------------


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    data = b"turbofan" * 1000
    path = tmp_path / "file.bin"
    path.write_bytes(data)
    assert sha256_of_file(path) == hashlib.sha256(data).hexdigest()


def test_sha256_is_independent_of_chunk_size(tmp_path: Path) -> None:
    path = tmp_path / "file.bin"
    path.write_bytes(bytes(range(256)) * 50)
    assert sha256_of_file(path, chunk_size=7) == sha256_of_file(path)


def test_verify_checksum_accepts_match_case_insensitive(tmp_path: Path) -> None:
    path = tmp_path / "file.bin"
    path.write_bytes(b"abc")
    verify_checksum(path, hashlib.sha256(b"abc").hexdigest().upper())


def test_verify_checksum_rejects_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "file.bin"
    path.write_bytes(b"abc")
    with pytest.raises(ChecksumError):
        verify_checksum(path, "0" * 64)


def test_is_valid_file(tmp_path: Path) -> None:
    path = tmp_path / "file.bin"
    good = hashlib.sha256(b"abc").hexdigest()
    assert not is_valid_file(path, good)  # missing
    path.write_bytes(b"abc")
    assert is_valid_file(path, good)
    assert not is_valid_file(path, "0" * 64)


# --- download_file (file:// URL, so no network) -------------------------------


def test_download_file_writes_atomically(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    dest = tmp_path / "out" / "dest.bin"
    download_file(src.as_uri(), dest)
    assert dest.read_bytes() == b"payload"
    assert not dest.with_name("dest.bin.part").exists()


# --- extraction ----------------------------------------------------------------


def test_extract_archive_flattens_nested_zip(source_archive: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    extracted = extract_archive(source_archive, out)
    for name in EXPECTED_FILES:
        assert (out / name).read_text() == f"contents of {name}\n"
    assert not any(p.suffix == ".zip" for p in extracted)


def test_extract_archive_blocks_path_traversal(tmp_path: Path) -> None:
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escaped.txt", "nope")
    with pytest.raises(ValueError, match="Unsafe path"):
        extract_archive(evil, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


# --- download_cmapss idempotency -------------------------------------------------


def test_first_run_downloads_and_extracts(
    tmp_path: Path, source_archive: Path, fake_download: list[str]
) -> None:
    raw = tmp_path / "raw"
    result = download_cmapss(raw, expected_sha256=sha256_of_file(source_archive))
    assert result == raw
    assert len(fake_download) == 1
    assert all((raw / name).is_file() for name in EXPECTED_FILES)


def test_second_run_skips_download(
    tmp_path: Path, source_archive: Path, fake_download: list[str]
) -> None:
    raw = tmp_path / "raw"
    expected = sha256_of_file(source_archive)
    download_cmapss(raw, expected_sha256=expected)
    download_cmapss(raw, expected_sha256=expected)
    assert len(fake_download) == 1


def test_second_run_skips_extraction_when_files_present(
    tmp_path: Path, source_archive: Path, fake_download: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    expected = sha256_of_file(source_archive)
    download_cmapss(raw, expected_sha256=expected)

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("extraction should be skipped")

    monkeypatch.setattr(download, "extract_archive", fail)
    download_cmapss(raw, expected_sha256=expected)


def test_corrupt_existing_archive_is_redownloaded(
    tmp_path: Path, source_archive: Path, fake_download: list[str]
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / download.ARCHIVE_NAME).write_bytes(b"truncated garbage")
    download_cmapss(raw, expected_sha256=sha256_of_file(source_archive))
    assert len(fake_download) == 1


def test_force_redownloads(tmp_path: Path, source_archive: Path, fake_download: list[str]) -> None:
    raw = tmp_path / "raw"
    expected = sha256_of_file(source_archive)
    download_cmapss(raw, expected_sha256=expected)
    download_cmapss(raw, expected_sha256=expected, force=True)
    assert len(fake_download) == 2


def test_bad_download_raises_and_is_removed(tmp_path: Path, fake_download: list[str]) -> None:
    raw = tmp_path / "raw"
    with pytest.raises(ChecksumError):
        download_cmapss(raw, expected_sha256="0" * 64)
    assert not (raw / download.ARCHIVE_NAME).exists()
