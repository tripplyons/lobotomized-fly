"""Current renderer defaults and recorded anatomical counts."""
import shutil
import subprocess

import pytest

from flybird.render import render_video, _retained
from test_arcade import fixture


def test_recorded_counts_override_requested_fraction():
    episode = {"condition": {"retained_fraction": 1},
               "brain": {"retained_units": 61854, "total_units": 165122}}
    assert _retained(episode) == 61854 / 165122


@pytest.mark.parametrize("kwargs", [{"width": 641}, {"fps": 0}, {"style": "comparison"}])
def test_invalid_render_options(tmp_path, kwargs):
    with pytest.raises(ValueError):
        render_video(fixture(), tmp_path, **kwargs)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_default_artwork_encodes_and_decodes(tmp_path):
    result = render_video(fixture(), tmp_path, width=640, height=360, fps=2, hold_seconds=.5)
    assert result["chapters"] == 1
    assert result["frames"] == 3
    assert result["displayed_conditions"] == 4
    assert (tmp_path / "poster.png").is_file()
    decoded = subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-i", result["video"],
                              "-f", "null", "-"], capture_output=True)
    assert decoded.returncode == 0, decoded.stderr.decode()
