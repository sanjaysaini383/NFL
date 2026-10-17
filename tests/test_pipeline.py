import json

from combine2field import pipeline, synthetic


def test_pipeline_end_to_end_on_synthetic(tmp_path):
    raw = synthetic.write(tmp_path / "raw", n_per_pos=12, plays_per_player=40, seed=11)
    work = tmp_path / "work"
    summary = pipeline.run(raw, work, n_repeats=2, make_figures=True)

    assert summary["n_players"] == 60
    assert summary["n_with_engine"] >= 55
    assert summary["n_with_expression"] >= 40
    for name in ("reliability", "translation", "associations", "incremental_value"):
        assert (work / "tables" / f"{name}.csv").exists()
    for fig in ("fig1_method", "fig2_reliability", "fig3_translation",
                "fig4_value_beyond_draft", "fig5_incremental_value"):
        assert (work / "figures" / f"{fig}.png").exists()
    saved = json.loads((work / "results.json").read_text())
    # synthetic expression fraction is drawn from Beta(6, 2.5) x U(0.55, 1): well below 1
    assert all(0.4 < v < 0.95 for v in saved["median_speed_expression"].values())
