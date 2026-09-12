import copy
import shutil
import subprocess

import pytest

from flybird.arcade import ArcadeRenderer, fly_sprite
from flybird.render import render_video


def fixture():
    frames = [dict(t=t, y=.5, alive=True, score=i, pipes=[], flap=False)
              for i, t in enumerate((0, .5, 1))]
    return {"config": {"bird_x": .24, "pipe_width": .13, "gap_size": .32},
            "episodes": [{"condition": {"id": str(i), "kind": "amount", "retained_fraction": fraction},
                          "seed": 7, "course_sha256": "a" * 64, "frames": copy.deepcopy(frames)}
                         for i, fraction in enumerate((1, .65, .60, .55))]}


def test_recorded_playback_and_final_hold():
    data = fixture()
    data["episodes"][1]["frames"][1]["alive"] = False
    data["episodes"][1]["frames"][2].update(y=.9, score=99)
    renderer = ArcadeRenderer(data, 640, 360)
    assert renderer.state(0, .49) is data["episodes"][0]["frames"][0]
    assert renderer.state(1, 10) is data["episodes"][1]["frames"][1]
    assert renderer.frame(.5).crop((160, 0, 320, 360)).tobytes() == renderer.frame(10).crop((160, 0, 320, 360)).tobytes()
    assert renderer.frame(1).tobytes() == renderer.frame(20, results=True).tobytes()
    assert renderer.pages == [[0, 1, 2, 3]]


def test_only_percentage_and_score_text(monkeypatch):
    labels = []
    monkeypatch.setattr("flybird.arcade.pixel_text", lambda d, x, y, text, scale: labels.append((text, scale, y)))
    ArcadeRenderer(fixture()).frame(.5)
    assert labels == [("100% NEURONS", 2, 3), ("1", 4, 22), ("65% NEURONS", 2, 3), ("1", 4, 22),
                      ("60% NEURONS", 2, 3), ("1", 4, 22), ("55% NEURONS", 2, 3), ("1", 4, 22)]


@pytest.mark.parametrize("change", [
    lambda d: d["episodes"][1].update(seed=8),
    lambda d: d["episodes"][1].update(course_sha256="b" * 64),
    lambda d: d["episodes"].append(d["episodes"][0]),
    lambda d: d["episodes"][1]["condition"].update(kind="region"),
    lambda d: d["episodes"][1]["condition"].update(retained_fraction=1),
    lambda d: d["episodes"][1].update(frames=[]),
    lambda d: d["episodes"][1]["frames"][1].update(t=float("nan")),
])
def test_reject_ambiguous_selection(change):
    data = fixture()
    change(data)
    with pytest.raises(ValueError):
        ArcadeRenderer(data)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_arcade_mp4(tmp_path):
    result = render_video(fixture(), tmp_path, style="arcade", width=640, height=360, fps=2, hold_seconds=1)
    assert result["frames"] == 4
    assert result["chapters"] == 1
    decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", result["video"], "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True)
    assert len(decoded.stdout) == 4 * 640 * 360 * 3


def test_dead_pose_rotates_art_without_moving_recorded_center():
    from PIL import Image
    assert fly_sprite(False, False).tobytes() == fly_sprite(True, False).transpose(Image.Transpose.ROTATE_270).tobytes()
    assert fly_sprite(False, True).tobytes() == fly_sprite(False, False).tobytes()
    assert fly_sprite(True, True).tobytes() != fly_sprite(True, False).tobytes()


@pytest.mark.parametrize("speed", [0, -1, float("nan"), float("inf")])
def test_invalid_playback_speed(speed, tmp_path):
    with pytest.raises(ValueError, match="playback_speed"):
        render_video(fixture(), tmp_path, style="arcade", playback_speed=speed)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_playback_speed_preserves_source_times_and_video_hold(tmp_path, monkeypatch):
    from unittest.mock import Mock
    renderer = ArcadeRenderer(fixture(), 640, 360)
    frame = Mock(wraps=renderer.frame)
    renderer.frame = frame
    monkeypatch.setattr("flybird.arcade.ArcadeRenderer", lambda *a: renderer)
    result = render_video(fixture(), tmp_path, style="arcade", width=640, height=360,
                          fps=4, hold_seconds=.5, playback_speed=2)
    assert result["frames"] == 4
    assert result["hold_seconds"] == .5
    assert [call.args[0] for call in frame.call_args_list[:-1]] == [0, .5, 1, 1.5]
    assert [call.kwargs["results"] for call in frame.call_args_list[:-1]] == [False, False, True, True]


def test_fly_stays_centered_on_recorded_position(monkeypatch):
    from PIL import Image
    assert fly_sprite(True, False).size == (23, 23)
    marker = Image.new("RGBA", (23, 23))
    marker.putpixel((11, 11), (255, 0, 255, 255))
    monkeypatch.setattr("flybird.arcade.fly_sprite", lambda *a: marker)
    frame = ArcadeRenderer(fixture(), 640, 360).frame(0)
    for index in range(4):
        assert frame.getpixel((index * 160 + round(.24 * 336), round(.5 * 336))) == (255, 0, 255)


def test_neuron_label_glyphs_fit_four_lane_layout():
    from flybird.arcade import GLYPHS
    for glyph in GLYPHS.values():
        assert len(glyph) == 7
        assert len({len(row) for row in glyph}) == 1
    for label in ("100% NEURONS", "69.5% NEURONS"):
        assert (sum(len(GLYPHS[char][0]) + 1 for char in label) - 1) * 2 + 4 < 160


def test_pixel_font_has_two_pixel_dark_outline():
    from PIL import Image, ImageDraw
    from flybird.arcade import pixel_text
    image = Image.new("RGB", (100, 100), "#ff00ff")
    pixel_text(ImageDraw.Draw(image), 50, 20, "1", 4)
    assert image.getpixel((46, 18)) == (41, 40, 36)
    assert image.getpixel((45, 18)) == (255, 0, 255)
    assert image.getpixel((48, 20)) == (255, 249, 228)


def test_fly_sprite_has_wings_red_eye_striped_body_and_legs():
    from PIL import ImageColor
    from flybird.arcade import PALETTE
    sprite = fly_sprite(True, False)
    assert sprite.size == (23, 23)
    for point, color in (((18, 11), "R"), ((6, 7), "W"), ((6, 14), "A"),
                         ((12, 13), "T"), ((7, 14), "K"), ((6, 20), "K")):
        assert sprite.getpixel(point) == ImageColor.getrgb(PALETTE[color]) + (255,)


def test_fly_uses_flat_six_color_palette():
    from PIL import ImageColor
    from flybird.arcade import PALETTE
    assert set(PALETTE) == {"K", "A", "T", "R", "W", "G"}
    allowed = {ImageColor.getrgb(color) + (255,) for color in PALETTE.values()}
    for alive, flap in ((True, False), (True, True), (False, False)):
        sprite = fly_sprite(alive, flap)
        colors = {sprite.getpixel((x, y)) for x in range(sprite.width) for y in range(sprite.height)}
        assert {color for color in colors if color[3]} <= allowed
        assert {color[3] for color in colors} == {0, 255}


@pytest.mark.parametrize("lanes", [2, 3, 4])
def test_death_overlay_starts_at_collision_and_preserves_live_lanes(lanes, monkeypatch):
    data = fixture()
    data["episodes"] = data["episodes"][:lanes]
    before = copy.deepcopy(data)
    data["episodes"][1]["frames"][1].update(alive=False, score=6)
    data["episodes"][1]["frames"][2].update(alive=True, score=99)
    renderer = ArcadeRenderer(data, 640, 360)
    original = ArcadeRenderer(before, 640, 360)
    assert renderer.frame(.49).tobytes() == original.frame(.49).tobytes()
    boundary = round(640 / lanes)
    assert renderer.frame(.5).crop((0, 0, boundary, 360)).tobytes() == original.frame(.5).crop((0, 0, boundary, 360)).tobytes()
    # An unobstructed sky pixel is dimmed, not replaced by opaque black.
    live = original.frame(.5).getpixel((boundary + 5, 80))
    dead = renderer.frame(.5).getpixel((boundary + 5, 80))
    assert all(0 < d < l for d, l in zip(dead, live))
    labels = []
    monkeypatch.setattr("flybird.arcade.pixel_text", lambda d, x, y, text, scale: labels.append(text))
    renderer.frame(10)
    assert labels.count("FLY DIED") == 1
    assert labels.count("SCORE") == 1
    assert "6" in labels
    assert "99" not in labels
    assert renderer.state(1, 10)["score"] == 6
