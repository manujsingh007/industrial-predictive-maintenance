"""Tests for the deployment bootstrap (no network, temporary MLflow store)."""

from pathlib import Path

import pytest

from predmaint.training import bootstrap


def test_champion_exists_is_none_for_an_empty_registry(tmp_path: Path) -> None:
    assert bootstrap.champion_exists(f"sqlite:///{tmp_path / 'mlflow.db'}") is None


def test_skips_everything_when_champion_exists(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(bootstrap, "champion_exists", lambda: "3")
    monkeypatch.setattr(bootstrap, "download_cmapss", lambda *a: pytest.fail("must not download"))
    monkeypatch.setattr(bootstrap.build_model, "main", lambda *a: pytest.fail("must not build"))
    assert bootstrap.main(["--subset", "FD001"]) == 0
    assert "already exists (version 3); nothing to do" in capsys.readouterr().out


def test_downloads_then_builds_when_no_champion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(bootstrap, "champion_exists", lambda: None)
    monkeypatch.setattr(bootstrap, "download_cmapss", lambda raw: calls.append("download") or raw)
    monkeypatch.setattr(
        bootstrap.build_model, "main", lambda argv: calls.append(f"build {argv}") or 0
    )
    monkeypatch.setattr(bootstrap, "TRACKING_DB", tmp_path / "mlflow" / "mlflow.db")
    monkeypatch.setattr(bootstrap, "ARTIFACT_ROOT", (tmp_path / "artifacts").as_uri())
    assert bootstrap.main(["--subset", "FD001"]) == 0
    assert calls == ["download", "build ['--subset', 'FD001']"]
    assert (tmp_path / "mlflow").is_dir() and (tmp_path / "artifacts").is_dir()
