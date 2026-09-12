"""Encode recorded trials with the current fruit-fly pixel artwork."""
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

def _condition(episode):
    condition = episode.get("condition", {})
    return condition if isinstance(condition, dict) else {"id": condition, "label": condition}


def _retained(episode):
    condition, brain = _condition(episode), episode.get("brain", {})
    for source in (brain, condition):
        total = source.get("total_units", source.get("total_nodes", source.get("total_neurons")))
        retained = source.get("retained_units", source.get("retained_nodes", source.get("retained_neurons")))
        if total and retained is not None:
            return max(0., min(1., retained / total))
        for key in ("retained_fraction", "active_fraction"):
            if key in source:
                return max(0., min(1., float(source[key])))
        for key in ("removed_fraction", "removal_fraction", "fraction_removed"):
            if key in source:
                return 1 - max(0., min(1., float(source[key])))
    return None


def render_video(experiment, output, *, width=1920, height=1080, fps=30,
                 hold_seconds=3., ffmpeg=None, style="arcade", playback_speed=1.):
    """Write H.264 MP4 and a final-state PNG; return their paths and frame count.

    ``output`` is an MP4 filename or directory (demo.mp4 + poster.png).
    ``style="arcade"`` uses one explicitly selected percentage scene; callers
    must keep attribution and audited provenance with the video (see arcade_run).
    ``playback_speed`` changes only presentation cadence, not recorded states.
    Hold duration is measured in video seconds, independent of playback speed.
    Requires Pillow and ffmpeg on PATH. No audio is synthesized.
    Existing output is replaced only after successful encoding.
    """
    if width % 2 or height % 2:
        raise ValueError("H.264 dimensions must be even")
    if fps <= 0 or not math.isfinite(fps) or hold_seconds < 0 or not math.isfinite(hold_seconds):
        raise ValueError("fps must be positive and hold_seconds nonnegative")
    if not math.isfinite(playback_speed) or playback_speed <= 0:
        raise ValueError("playback_speed must be positive and finite")
    if style == "arcade":
        from .arcade import ArcadeRenderer
        renderer = ArcadeRenderer(experiment, width, height)
    else:
        raise ValueError("Unknown render style: " + style)
    executable = ffmpeg or shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("ffmpeg is required: install it and add it to PATH")
    output = Path(output)
    directory_output = output.suffix.lower() != ".mp4"
    if directory_output:
        output = output / "demo.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    poster = output.parent / "poster.png" if directory_output else output.with_suffix(".png")
    chapter_frames = max(1, math.ceil((renderer.duration / playback_speed + hold_seconds) * fps))
    count = chapter_frames * len(renderer.pages)
    with tempfile.TemporaryDirectory(prefix=".flybird-render-", dir=output.parent) as temporary:
        video_tmp = Path(temporary) / "video.mp4"
        error_path = Path(temporary) / "ffmpeg.log"
        command = [str(executable), "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
                   "-c:v", "libx264", "-preset", "fast", "-crf", "19",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_tmp)]
        with error_path.open("wb") as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
            try:
                for i in range(count):
                    chapter, local_frame = divmod(i, chapter_frames)
                    t = local_frame / fps * playback_speed
                    process.stdin.write(renderer.frame(t, results=t >= renderer.duration, chapter=chapter).tobytes())
                process.stdin.close()
                returncode = process.wait()
            except BaseException as exc:
                process.kill()
                process.wait()
                if isinstance(exc, BrokenPipeError):
                    raise RuntimeError("ffmpeg failed: " + error_path.read_text()) from exc
                raise
        if returncode:
            raise RuntimeError("ffmpeg failed: " + error_path.read_text())
        poster_tmp = Path(temporary) / "poster.png"
        renderer.frame(renderer.duration, results=True).save(poster_tmp)
        video_tmp.replace(output)
        poster_tmp.replace(poster)
    return {"video": str(output), "poster": str(poster), "frames": count, "fps": fps,
            "chapters": len(renderer.pages), "displayed_conditions": len(renderer.episodes),
            "playback_speed": playback_speed, "hold_seconds": hold_seconds}
