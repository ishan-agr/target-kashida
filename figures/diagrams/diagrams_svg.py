"""Hand-laid-out SVG diagrams (fixed geometry, print-legible at A4 text width).

    python figures/diagrams/diagrams_svg.py   -> figures/d_system.svg

Mermaid is kept for the small serving flowchart (d_serving); the system diagram needs control
over placement that Mermaid's auto-layout doesn't give.
"""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
FONT = "Segoe UI, Arial, sans-serif"
INK, MUTED, LINE = "#1f2933", "#616e7c", "#52606d"
BLUE = ("#eef2ff", "#4f46e5")
GREEN = ("#ecfdf5", "#059669")
AMBER = ("#fff7ed", "#d97706")
RED = ("#fef2f2", "#dc2626")
GREY = ("#f8fafc", "#cbd2d9")


class Canvas:
    def __init__(self, w: int, h: int):
        self.w, self.h, self.parts = w, h, []

    def text(self, x, y, s, size=13, weight="400", fill=INK, anchor="middle", italic=False):
        st = ' font-style="italic"' if italic else ""
        self.parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{FONT}" '
                          f'font-size="{size}" font-weight="{weight}" fill="{fill}"{st}>{s}</text>')

    def box(self, x, y, w, h, lines, style=BLUE, size=14, rx=8):
        fill, stroke = style
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="1.5"/>')
        n = len(lines)
        for i, t in enumerate(lines):
            sz = size if i == 0 else size - 2
            ty = y + h / 2 + (i - (n - 1) / 2) * (size + 4) + sz * 0.35
            self.text(x + w / 2, ty, t, sz, "600" if i == 0 else "400", INK if i == 0 else MUTED)

    def diamond(self, cx, cy, rw, rh, lines, style=BLUE, size=13):
        fill, stroke = style
        self.parts.append(f'<polygon points="{cx},{cy - rh} {cx + rw},{cy} {cx},{cy + rh} {cx - rw},{cy}" '
                          f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        n = len(lines)
        for i, t in enumerate(lines):
            self.text(cx, cy + (i - (n - 1) / 2) * (size + 3) + size * 0.35, t, size, "600" if i == 0 else "400")

    def group(self, x, y, w, h, title, style=GREY):
        fill, stroke = style
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="1.2" stroke-dasharray="5 4"/>')
        self.text(x + 12, y + 20, title, 12.5, "700", MUTED, anchor="start")

    def path(self, pts, dashed=False, color=LINE):
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        p = " ".join(f"{x},{y}" for x, y in pts)
        self.parts.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="1.6"{dash} '
                          f'marker-end="url(#ah)"/>')

    def label(self, x, y, s, anchor="middle"):
        self.text(x, y, s, 12, "400", MUTED, anchor=anchor, italic=True)

    def save(self, name: str) -> None:
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
               f'width="{self.w}" height="{self.h}">\n'
               f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
               f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/>'
               f'</marker></defs>\n<rect width="100%" height="100%" fill="white"/>\n'
               + "\n".join(self.parts) + "\n</svg>\n")
        (OUT / f"{name}.svg").write_text(svg, encoding="utf-8")
        print("wrote", name)


def system() -> None:
    c = Canvas(1000, 600)
    c.box(40, 20, 250, 64, ["Cheque scan", "PNG · TIFF · PDF"])
    c.box(375, 20, 250, 64, ["Ingestion", "deskew · sha256 · audit log"])
    c.box(710, 20, 250, 64, ["Layout", "printed key + frozen offset"])
    c.path([(290, 52), (373, 52)])
    c.path([(625, 52), (708, 52)])

    c.group(20, 125, 560, 150, "vLLM · Qwen3-VL-8B AWQ-4bit · 5 concurrent passes", BLUE)
    for i, t in enumerate(["Full image", "Payee crop", "Amount crop", "Date crop", "MICR strip"]):
        c.box(35 + i * 108, 158, 98, 66, [t, "VLM pass"], BLUE, size=13)
    c.group(600, 125, 150, 150, "Sidecar SLM", AMBER)
    c.box(615, 158, 120, 66, ["≤ 3 GB", "payee only"], AMBER, size=13)
    c.group(770, 125, 210, 150, "Deterministic · CPU", GREEN)
    c.box(782, 158, 88, 66, ["Tesseract", "E-13B MICR"], GREEN, size=13)
    c.box(880, 158, 88, 66, ["Paddle+", "Easy IBAN"], GREEN, size=13)
    c.path([(835, 84), (835, 104), (300, 104), (300, 123)])
    c.path([(835, 104), (675, 104), (675, 123)])
    c.path([(875, 84), (875, 123)])
    # payee crop -> sidecar, routed under the pass boxes
    c.path([(143, 224), (143, 248), (660, 248), (660, 226)], dashed=True)
    c.label(400, 264, "payee crop")

    c.box(40, 330, 200, 74, ["Fusion", "payee override · guards"])
    c.box(300, 330, 220, 74, ["Validation", "mod-97 · MICR grammar", "words ↔ figures"], GREEN, size=13)
    c.box(580, 330, 190, 74, ["Confidence", "agree · valid · logprob"])
    c.diamond(880, 367, 65, 40, ["Route"])
    c.path([(300, 275), (300, 295), (140, 295), (140, 328)])
    c.path([(675, 275), (675, 300), (200, 300), (200, 328)])
    c.path([(875, 275), (875, 312), (410, 312), (410, 328)])
    c.path([(240, 367), (298, 367)])
    c.path([(520, 367), (578, 367)])
    c.path([(770, 367), (813, 367)])

    c.box(560, 470, 200, 64, ["Straight-through", "auto-approved"], GREEN)
    c.box(790, 470, 190, 64, ["Maker-checker", "human review queue"], AMBER)
    c.box(300, 470, 220, 64, ["Active-learning pool", "content-addressed crops"], AMBER)
    c.path([(880, 407), (880, 440), (660, 440), (660, 468)])
    c.path([(880, 440), (880, 468)])
    c.label(760, 434, "≥ threshold")
    c.label(905, 458, "below", anchor="start")
    c.path([(885, 534), (885, 562), (410, 562), (410, 536)], dashed=True)
    c.label(650, 580, "verified crops")
    c.save("d_system")


if __name__ == "__main__":
    for f in (system,):
        f()
