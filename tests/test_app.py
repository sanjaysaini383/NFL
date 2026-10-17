from pathlib import Path

import pytest

from combine2field import pipeline, synthetic

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_app_renders_without_exceptions(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    raw = synthetic.write(tmp_path / "raw", n_per_pos=10, plays_per_player=40, seed=5)
    pipeline.run(raw, tmp_path / "work", n_repeats=2, make_figures=False)
    monkeypatch.setenv("BDB_WORK_DIR", str(tmp_path / "work"))
    at = testing.AppTest.from_file(str(APP), default_timeout=180).run()
    assert not at.exception
    assert len(at.tabs) == 4
    assert any(m.label == "Combine S0" for m in at.metric)
