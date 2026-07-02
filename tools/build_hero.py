#!/usr/bin/env python
"""Compose the docs landing-page hero image (``docs/images/hero.webp``).

The hero is built from real package output, not stock art: the India LISA cluster map
(``explore_lisa_cluster_map`` on the 520-district nighttime-lights panel, vector
``tiles=None`` rendering) is exported with kaleido and composed with the wordmark and
tagline over a cosmo-blue gradient with a faint lattice grid — the same motif as the
logo (see ``tools/build_logo.py``).

Run locally and commit the output (CI never builds it; kaleido + the pooch-cached India
download are only needed here)::

    uv run python tools/build_hero.py

Requires the ``png`` extra (kaleido) and network on the first run (India data download).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "images" / "hero.webp"

# Canvas matches expdpy's hero (2848x1504) so the landing pages read alike.
W, H = 2848, 1504

# The logo's cosmo-blue ramp.
NAVY = (11, 46, 87)  # #0b2e57
BLUE = (39, 128, 227)  # #2780e3
GRID_ALPHA = 14

TAGLINE = (
    "Regional growth, convergence,",
    "and inequality — spatially, in Python.",
)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load the first available system font (macOS/Linux candidates)."""
    candidates = (
        [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        if bold
        else [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    raise SystemExit("No usable system font found — edit _font() candidates.")


def _background() -> Image.Image:
    """Diagonal navy-to-blue gradient with a faint lattice grid."""
    x = np.linspace(0.0, 1.0, W)[None, :]
    y = np.linspace(0.0, 1.0, H)[:, None]
    t = np.clip(0.62 * x + 0.38 * y, 0.0, 1.0) ** 1.15
    rgb = np.empty((H, W, 3), dtype=np.uint8)
    for i, (lo, hi) in enumerate(zip(NAVY, BLUE, strict=True)):
        rgb[..., i] = (lo + (hi - lo) * t).astype(np.uint8)
    img = Image.fromarray(rgb, "RGB").convert("RGBA")

    grid = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(grid)
    step = 112
    for gx in range(step, W, step):
        draw.line([(gx, 0), (gx, H)], fill=(255, 255, 255, GRID_ALPHA), width=2)
    for gy in range(step, H, step):
        draw.line([(0, gy), (W, gy)], fill=(255, 255, 255, GRID_ALPHA), width=2)
    return Image.alpha_composite(img, grid)


def _lisa_map_png() -> Image.Image:
    """Render the India LISA cluster map to a transparent PNG via kaleido."""
    import geometrics as gm

    gdf, df, df_dict = gm.data.load_india()
    df = gm.set_labels(df, df_dict, set_panel=True)
    w = gm.make_weights(gdf, method="knn", k=6)
    res = gm.explore_lisa_cluster_map(df, "log_ntl_pc_1996", gdf=gdf, w=w, tiles=None)

    fig = res.fig
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=None,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False,
        annotations=[],
    )
    fig.update_geos(bgcolor="rgba(0,0,0,0)")
    png = fig.to_image(format="png", width=1050, height=1150, scale=2)
    return Image.open(io.BytesIO(png)).convert("RGBA")


def _fit(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    """Trim transparent borders, then scale to fit inside (box_w, box_h)."""
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    ratio = min(box_w / img.width, box_h / img.height)
    return img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)


def build() -> None:
    """Compose and save the hero."""
    hero = _background()
    draw = ImageDraw.Draw(hero)

    # --- right: the LISA map on a translucent card -------------------------------
    card = (1560, 110, 2740, 1394)
    panel = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(panel).rounded_rectangle(card, radius=44, fill=(255, 255, 255, 234))
    hero = Image.alpha_composite(hero, panel)
    draw = ImageDraw.Draw(hero)

    lisa = _fit(_lisa_map_png(), card[2] - card[0] - 90, card[3] - card[1] - 150)
    cx = card[0] + (card[2] - card[0] - lisa.width) // 2
    cy = card[1] + 40 + (card[3] - card[1] - 110 - lisa.height) // 2
    hero.alpha_composite(lisa, (cx, cy))
    caption = "LISA clusters — nighttime lights across 520 Indian districts"
    cap_font = _font(34)
    cw = draw.textlength(caption, font=cap_font)
    draw.text(
        (card[0] + (card[2] - card[0] - cw) / 2, card[3] - 78),
        caption,
        font=cap_font,
        fill=(70, 90, 120, 255),
    )

    # --- left: lattice mark + wordmark + tagline ----------------------------------
    mark = Image.open(REPO / "src" / "geometrics" / "_assets" / "logo.png").convert(
        "RGBA"
    )
    mark = mark.resize((170, 170), Image.LANCZOS)
    hero.alpha_composite(mark, (170, 430))

    word_font = _font(210, bold=True)
    draw.text((380, 400), "geometrics", font=word_font, fill=(255, 255, 255, 255))

    tag_font = _font(74)
    for i, line in enumerate(TAGLINE):
        draw.text((178, 740 + i * 104), line, font=tag_font, fill=(224, 236, 250, 255))

    sub_font = _font(46)
    for i, line in enumerate(
        ("Built on the PySAL stack —", "maps, models, and plain-language readings.")
    ):
        draw.text((178, 1010 + i * 66), line, font=sub_font, fill=(190, 212, 240, 255))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    hero.convert("RGB").save(OUT, "WEBP", quality=82, method=6)
    print(f"wrote {OUT.relative_to(REPO)}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build()
