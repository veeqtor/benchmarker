"""Test image generator for vision model accuracy tests.

Generates PNG images with random 4-digit numbers in colored frames,
used to test vision model recognition accuracy.
"""

import base64
import random
import struct
import zlib

# 5×7 pixel digit bitmaps
_FONT = {
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
}

# Frame color RGB values
_COLORS = {"red": (220, 30, 30), "green": (20, 160, 40), "blue": (30, 60, 220)}


def generate_test_image(px, seed=None):
    """Generate a PNG image with a random 4-digit number in a coloured frame.

    Returns (png_bytes, number, colour_name).
    """
    rnd = random.Random(seed)
    number = "".join(rnd.choice("123456789") for _ in range(4))
    colour = rnd.choice(sorted(_COLORS))
    col = _COLORS[colour]
    w = h = px
    img = [bytearray(b"\xff" * (w * 3)) for _ in range(h)]
    border = max(6, px // 25)
    for y in range(h):
        for x in list(range(border)) + list(range(w - border, w)):
            img[y][x * 3 : x * 3 + 3] = bytes(col)
    for y in list(range(border)) + list(range(h - border, h)):
        img[y][:] = bytes(col) * w
    scale = max(4, int(w * 0.6) // (4 * 6))
    tw, th = 4 * 6 * scale - scale, 7 * scale
    ox, oy = (w - tw) // 2, (h - th) // 2
    for i, d in enumerate(number):
        for ry, row in enumerate(_FONT[d]):
            for rx, bit in enumerate(row):
                if bit == "1":
                    for y in range(oy + ry * scale, oy + (ry + 1) * scale):
                        x0 = ox + (i * 6 + rx) * scale
                        img[y][x0 * 3 : (x0 + scale) * 3] = b"\x00" * (3 * scale)
    raw = b"".join(b"\x00" + bytes(r) for r in img)
    chunk = lambda t, d: (
        struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    )
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    return png, number, colour


def image_message(png, text):
    """Build an OpenAI-style chat message containing a base64-encoded PNG image."""
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
    ]


VISION_Q = (
    "Read the number in the image and name the colour of the frame. "
    "Answer only in this form: NUMBER=<digits>, COLOUR=<red|green|blue>."
)


__all__ = [
    "VISION_Q",
    "image_message",
    "test_image",
]
