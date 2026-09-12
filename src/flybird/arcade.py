"""Single-scene pixel-art playback of explicitly selected recorded trials."""
from bisect import bisect_right
from functools import lru_cache
import math

from PIL import Image, ImageDraw

from .render import _condition, _retained

# Original pixel artwork; no assets from Flappy Bird or other games are used.
PALETTE = {"K": "#302d27", "A": "#d4a454", "T": "#ad7c43",
           "R": "#e24a3c", "W": "#edf5df", "G": "#738b88"}
GLYPHS = {
    '0': ('01110', '10001', '10001', '10001', '10001', '10001', '01110'),
    '1': ('00100', '01100', '00100', '00100', '00100', '00100', '01110'),
    '2': ('01110', '10001', '00001', '00010', '00100', '01000', '11111'),
    '3': ('11110', '00001', '00001', '01110', '00001', '00001', '11110'),
    '4': ('00010', '00110', '01010', '10010', '11111', '00010', '00010'),
    '5': ('11111', '10000', '10000', '11110', '00001', '00001', '11110'),
    '6': ('01110', '10000', '10000', '11110', '10001', '10001', '01110'),
    '7': ('11111', '00001', '00010', '00100', '01000', '01000', '01000'),
    '8': ('01110', '10001', '10001', '01110', '10001', '10001', '01110'),
    '9': ('01110', '10001', '10001', '01111', '00001', '00001', '01110'),
    '%': ('11001', '11010', '00010', '00100', '01000', '01011', '10011'),
    '.': ('0', '0', '0', '0', '0', '0', '1'),
    ' ': ('000', '000', '000', '000', '000', '000', '000'),
    'N': ('10001', '11001', '11001', '10101', '10011', '10011', '10001'),
    'E': ('11111', '10000', '10000', '11110', '10000', '10000', '11111'),
    'U': ('10001', '10001', '10001', '10001', '10001', '10001', '01110'),
    'R': ('11110', '10001', '10001', '11110', '10100', '10010', '10001'),
    'O': ('01110', '10001', '10001', '10001', '10001', '10001', '01110'),
    'S': ('01111', '10000', '10000', '01110', '00001', '00001', '11110'),
    'F': ('11111', '10000', '10000', '11110', '10000', '10000', '10000'),
    'L': ('10000', '10000', '10000', '10000', '10000', '10000', '11111'),
    'Y': ('10001', '10001', '01010', '00100', '00100', '00100', '00100'),
    'D': ('11110', '10001', '10001', '10001', '10001', '10001', '11110'),
    'I': ('111', '010', '010', '010', '010', '010', '111'),
    'C': ('01111', '10000', '10000', '10000', '10000', '10000', '01111'),
}
TEXT_OUTLINE = 2


def pixel_text(draw, center, y, text, scale=3):
    glyphs = [GLYPHS[char] for char in text]
    width = sum(len(glyph[0]) + 1 for glyph in glyphs) - 1
    cursor = int(center - width * scale / 2)
    pixels = []
    for glyph in glyphs:
        pixels.extend((cursor + x * scale, y + row * scale)
                      for row, line in enumerate(glyph)
                      for x, value in enumerate(line) if value == "1")
        cursor += (len(glyph[0]) + 1) * scale
    for x, yy in pixels:
        draw.rectangle((x - TEXT_OUTLINE, yy - TEXT_OUTLINE,
                        x + scale - 1 + TEXT_OUTLINE, yy + scale - 1 + TEXT_OUTLINE), fill="#292824")
    for x, yy in pixels:
        draw.rectangle((x, yy, x + scale - 1, yy + scale - 1), fill="#fff9e4")


@lru_cache(maxsize=4)
def fly_sprite(alive, flap):
    """A stylized fruit fly; wing pose does not change recorded movement."""
    sprite = Image.new("RGBA", (23, 23))
    draw = ImageDraw.Draw(sprite)
    # Pale paired wings sit behind the body. Their motion is decorative.
    far_wing = ((1, 10), (3, 9), (7, 10), (12, 12), (10, 14), (5, 13), (2, 12))
    near_wing = (((5, 4), (8, 4), (10, 6), (13, 11), (12, 13), (9, 10), (6, 7))
                 if flap and alive else
                 ((3, 5), (6, 5), (10, 7), (13, 11), (12, 13), (9, 12), (5, 9), (3, 7)))
    for wing in (far_wing, near_wing):
        draw.polygon(wing, fill=PALETTE["W"], outline=PALETTE["G"])
    # Six short legs; segment positions remain fixed relative to the thorax.
    for leg in (((7, 15), (5, 18), (3, 18)), ((8, 15), (7, 19), (6, 20)),
                ((11, 15), (11, 18), (12, 20)), ((12, 15), (13, 17), (15, 18)),
                ((14, 14), (17, 17), (19, 17)), ((15, 13), (19, 15), (21, 15))):
        draw.line(leg, fill=PALETTE["K"])
    draw.polygon(((2, 13), (4, 11), (8, 11), (11, 13), (10, 16), (6, 17), (3, 15)),
                 fill=PALETTE["A"], outline=PALETTE["K"])
    for stripe in (((5, 12), (4, 15)), ((7, 12), (7, 16)), ((9, 13), (9, 16))):
        draw.line(stripe, fill=PALETTE["K"])
    draw.ellipse((9, 10, 15, 16), fill=PALETTE["T"], outline=PALETTE["K"])
    draw.ellipse((14, 9, 20, 14), fill=PALETTE["T"], outline=PALETTE["K"])
    draw.ellipse((16, 9, 20, 13), fill=PALETTE["R"], outline=PALETTE["K"])
    for antenna in (((19, 9), (21, 7), (22, 7)), ((20, 10), (22, 9))):
        draw.line(antenna, fill=PALETTE["K"])
    draw.line(((20, 13), (22, 14)), fill=PALETTE["K"])
    if not alive:
        return sprite.transpose(Image.Transpose.ROTATE_270)
    return sprite


class ArcadeRenderer:
    """One lane per percentage; no implicit trial selection or chapters.

    The horizontal viewport crops the right side of the recorded world. Both
    axes use the same scale; pipe geometry and bird centers use recorded states.
    The fly sprite is decorative, not a collision mask. The underlying game
    keeps its bird_x/radius fields and unchanged physics.
    """

    def __init__(self, experiment, width=1920, height=1080):
        self.episodes = experiment.get("episodes", [])
        if not 2 <= len(self.episodes) <= 4:
            raise ValueError("Arcade playback requires two to four selected episodes")
        if width < 320 or height < 180:
            raise ValueError("Render dimensions must be at least 320×180")
        self.config = experiment["config"]
        self.size = width, height
        self.pages = [list(range(len(self.episodes)))]
        self.times, self.death_indices, self.labels = [], [], []
        identities = {(e["seed"], e["course_sha256"]) for e in self.episodes}
        if len(identities) != 1:
            raise ValueError("Arcade episodes must share one seed and course")
        for episode in self.episodes:
            if _condition(episode).get("kind") not in {"intact", "amount"}:
                raise ValueError("Arcade labels require intact or amount conditions")
            fraction = _retained(episode)
            if fraction is None:
                raise ValueError("Arcade episodes require recorded retention")
            label = f"{100 * fraction:.1f}".removesuffix(".0") + "%"
            if label in self.labels:
                raise ValueError("Arcade percentages must be distinct")
            self.labels.append(label)
            times = [float(f["t"]) for f in episode["frames"]]
            if not times or any(not math.isfinite(t) for t in times) or times != sorted(set(times)) or times[0] != 0:
                raise ValueError("Recorded times must start at zero and increase strictly")
            self.times.append(times)
            self.death_indices.append(next((i for i, f in enumerate(episode["frames"]) if not f["alive"]), None))
        self.duration = max(times[-1] for times in self.times)

    def state(self, index, t):
        position = max(0, bisect_right(self.times[index], t) - 1)
        death = self.death_indices[index]
        if death is not None:
            position = min(position, death)
        return self.episodes[index]["frames"][position]

    def frame(self, t, results=False, chapter=0):
        if chapter != 0:
            raise ValueError("Arcade playback has only one scene")
        image = Image.new("RGB", (640, 360))
        for index in range(len(self.episodes)):
            left = round(index * 640 / len(self.episodes))
            right = round((index + 1) * 640 / len(self.episodes))
            lane = self._lane(right - left, self.state(index, t), self.labels[index])
            image.paste(lane, (left, 0))
        return image.resize(self.size, Image.Resampling.NEAREST)

    def _lane(self, width, state, label):
        image = Image.new("RGB", (width, 360), "#72c5ce")
        draw = ImageDraw.Draw(image)
        # Distant clouds, skyline and shrubs are scenery, not data overlays.
        for x, y in ((12, 210), (94, 188), (174, 216)):
            draw.rectangle((x, y, x + 44, y + 14), fill="#e7f5dc")
            draw.rectangle((x + 9, y - 8, x + 29, y), fill="#e7f5dc")
        for x in range(0, width, 19):
            top = 272 + (x * 7 % 23)
            draw.rectangle((x, top, x + 16, 326), fill="#b7dfc1")
            for yy in range(top + 5, 320, 9):
                draw.rectangle((x + 4, yy, x + 6, yy + 3), fill="#d8eccb")
        for x in range(-8, width, 17):
            draw.rectangle((x, 318 + x % 7, x + 20, 335), fill="#79bb68")
        scale = 336
        # Keep normalized geometry isotropic, cropping only the right-hand view.
        for pipe in state["pipes"]:
            x = round(pipe["x"] * scale)
            w = round(self.config["pipe_width"] * scale)
            gap_top = round((pipe["gap_y"] - self.config["gap_size"] / 2) * scale)
            gap_bottom = round((pipe["gap_y"] + self.config["gap_size"] / 2) * scale)
            for top, bottom, cap_y in ((0, gap_top, gap_top - 10), (gap_bottom, 336, gap_bottom)):
                draw.rectangle((x, top, x + w, bottom), fill="#568c2c", outline="#3c5428", width=2)
                draw.rectangle((x + 3, top, x + 10, bottom), fill="#b8df58")
                draw.rectangle((x + 11, top, x + w - 7, bottom), fill="#8ec640")
                draw.rectangle((x - 2, cap_y, x + w + 2, cap_y + 10), fill="#8ec640", outline="#3c5428", width=2)
                draw.line((x + 1, cap_y + 2, x + w - 1, cap_y + 2), fill="#d4ee75", width=2)
        bx = round(self.config["bird_x"] * scale)
        by = round(state["y"] * scale)
        sprite = fly_sprite(state["alive"], bool(state.get("flap")))
        image.paste(sprite, (bx - sprite.width // 2, by - sprite.height // 2), sprite)
        draw.rectangle((0, 336, width, 359), fill="#dfd598")
        draw.rectangle((0, 336, width, 340), fill="#4e6e30")
        draw.line((0, 337, width, 337), fill="#d6ed83")
        for x in range(-6, width, 10):
            draw.polygon(((x, 341), (x + 5, 341), (x + 1, 345), (x - 4, 345)), fill="#91ba4b")
        draw.line((0, 347, width, 347), fill="#b3a766")
        if not state["alive"]:
            shade = Image.new("RGBA", image.size, (10, 14, 20, 180))
            image = Image.alpha_composite(image.convert("RGBA"), shade).convert("RGB")
            draw = ImageDraw.Draw(image)
            self._death_overlay(draw, width, state["score"])
        pixel_text(draw, width / 2, 3, label + " NEURONS", 2)
        if state["alive"]:
            pixel_text(draw, width / 2, 22, str(state["score"]), 4)
        draw.line((width - 1, 0, width - 1, 359), fill="#4e6e30")
        return image

    @staticmethod
    def _death_overlay(draw, width, score):
        # Original pixel skull. The overlay reads only the frozen collision state.
        skull = (
            "000111111111000",
            "001111111111100",
            "011111111111110",
            "111111111111111",
            "111111111111111",
            "110001111100011",
            "110001111100011",
            "110001111100011",
            "111111101111111",
            "011111000111110",
            "001111111111100",
            "000110110110000",
            "000110110110000",
        )
        scale = 4
        left = round(width / 2 - len(skull[0]) * scale / 2)
        for row, line in enumerate(skull):
            for column, pixel in enumerate(line):
                if pixel == "1":
                    x, y = left + column * scale, 120 + row * scale
                    draw.rectangle((x, y, x + scale - 1, y + scale - 1), fill="#fff9e4")
        pixel_text(draw, width / 2, 190, "FLY DIED", 2)
        pixel_text(draw, width / 2, 219, "SCORE", 2)
        pixel_text(draw, width / 2, 241, str(score), 4)
