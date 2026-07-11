import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from wsme_gpcr.plotting import save_figure


def test_save_figure_writes_png_and_svg(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    written = save_figure(fig, tmp_path / "myplot.png")
    plt.close(fig)

    assert len(written) == 2
    png_path, svg_path = written
    assert png_path == tmp_path / "myplot.png"
    assert svg_path == tmp_path / "myplot.svg"
    assert png_path.exists() and png_path.stat().st_size > 0
    assert svg_path.exists() and svg_path.stat().st_size > 0


def test_save_figure_replaces_any_given_extension():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0])

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        written = save_figure(fig, Path(d) / "myplot.svg")  # extension given doesn't have to be .png
        plt.close(fig)
        assert {p.suffix for p in written} == {".png", ".svg"}
        assert all(p.stem == "myplot" for p in written)
