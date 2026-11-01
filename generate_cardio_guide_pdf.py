"""
Generates the comprehensive, multi-page PDF guide and demonstration playbook for FedCardio v2.0:
  - FedCardio_v2_Complete_Guide_and_Demo_Playbook.pdf (Project Root)
  - reports/FedCardio_v2_Complete_Guide_and_Demo_Playbook.pdf (Reports Directory)

Built with a self-contained, zero-dependency PDF-1.4 vector typesetting engine
supporting Type1 Helvetica/Courier font families, rich inline markup (<b>, <i>, <code>),
multi-column auto-wrapping tables with repeating headers, colored callout banners,
code/math blocks, and two-pass running headers & "Page X of Y" footers.
"""

from __future__ import annotations

import os
import re
import shutil
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# 1. PURE-PYTHON VECTOR PDF-1.4 TYPESETTING ENGINE
# =============================================================================

# Approximate glyph width table (in 1/1000ths of a point) for Helvetica
HELVETICA_CHAR_WIDTHS: Dict[str, int] = {
    " ": 278, "!": 278, '"': 355, "#": 556, "$": 556, "%": 889, "&": 667, "'": 191,
    "(": 333, ")": 333, "*": 389, "+": 584, ",": 278, "-": 333, ".": 278, "/": 278,
    "0": 556, "1": 556, "2": 556, "3": 556, "4": 556, "5": 556, "6": 556, "7": 556,
    "8": 556, "9": 556, ":": 278, ";": 278, "<": 584, "=": 584, ">": 584, "?": 556,
    "@": 1015, "A": 667, "B": 667, "C": 722, "D": 722, "E": 667, "F": 611, "G": 778,
    "H": 722, "I": 278, "J": 500, "K": 667, "L": 556, "M": 833, "N": 722, "O": 778,
    "P": 667, "Q": 778, "R": 722, "S": 667, "T": 611, "U": 722, "V": 667, "W": 944,
    "X": 667, "Y": 667, "Z": 611, "[": 278, "\\": 278, "]": 278, "^": 469, "_": 556,
    "`": 333, "a": 556, "b": 556, "c": 500, "d": 556, "e": 556, "f": 278, "g": 556,
    "h": 556, "i": 222, "j": 222, "k": 500, "l": 222, "m": 833, "n": 556, "o": 556,
    "p": 556, "q": 556, "r": 333, "s": 500, "t": 278, "u": 556, "v": 500, "w": 722,
    "x": 500, "y": 500, "z": 500, "{": 334, "|": 260, "}": 334, "~": 584,
}


def sanitize_pdf_text(text: str) -> str:
    """Normalizes Unicode symbols into clean WinAnsi / PDF Type1 characters."""
    replacements = {
        "—": "--",
        "–": "-",
        "−": "-",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "•": "\x95",
        "≥": ">=",
        "≤": "<=",
        "≠": "!=",
        "×": "x",
        "÷": "/",
        "±": "+/-",
        "→": "->",
        "←": "<-",
        "↔": "<->",
        "≈": "~=",
        "∑": "Sum",
        "∏": "Prod",
        "√": "sqrt",
        "∞": "inf",
        "∈": "in",
        "Δ": "Delta ",
        "δ": "delta",
        "ε": "epsilon",
        "ϵ": "epsilon",
        "μ": "mu",
        "σ": "sigma",
        "β": "beta",
        "φ": "phi",
        "λ": "lambda",
        "θ": "theta",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Strip any remaining out-of-range characters
    out = []
    for ch in text:
        code = ord(ch)
        if 32 <= code <= 126 or code in (149,):
            out.append(ch)
        elif ch in ("\n", "\t"):
            out.append(ch)
        else:
            out.append("?")
    return "".join(out)


def escape_pdf_string(s: str) -> str:
    """Escapes parentheses and backslashes for PDF literal strings."""
    s = sanitize_pdf_text(s)
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def measure_word_width(word: str, font_id: str, font_size: float) -> float:
    """Measures exact horizontal width in points for a word in the given font."""
    word = sanitize_pdf_text(word)
    if font_id in ("F4", "F5"):  # Courier / Courier-Bold are fixed-width (600/1000)
        return len(word) * 0.600 * font_size
    scale = 1.088 if font_id == "F2" else (1.03 if font_id == "F3" else 1.0)
    total_units = sum(HELVETICA_CHAR_WIDTHS.get(ch, 556) for ch in word)
    return (total_units / 1000.0) * font_size * scale


Token = Tuple[str, float, Tuple[float, float, float], str]  # (font_id, size, rgb, text)


def parse_inline_markup(
    text: str,
    base_font: str = "F1",
    base_size: float = 9.0,
    base_color: Tuple[float, float, float] = (0.118, 0.161, 0.231),
) -> List[Token]:
    """
    Parses inline HTML-like markup (<b>...</b>, <i>...</i>, <code>...</code>)
    and explicit newline characters (\\n) into word-level tokens with font, size, and RGB color.
    """
    tag_pattern = re.compile(r"(<b>|</b>|<i>|</i>|<code>|</code>|\n)")
    parts = tag_pattern.split(text)

    current_font = base_font
    current_size = base_size
    current_color = base_color
    font_stack: List[Tuple[str, float, Tuple[float, float, float]]] = []

    tokens: List[Token] = []
    for part in parts:
        if not part:
            continue
        if part == "\n":
            tokens.append(("__BR__", current_size, current_color, "\n"))
        elif part == "<b>":
            font_stack.append((current_font, current_size, current_color))
            current_font = "F5" if current_font in ("F4", "F5") else "F2"
        elif part == "<i>":
            font_stack.append((current_font, current_size, current_color))
            current_font = "F3"
        elif part == "<code>":
            font_stack.append((current_font, current_size, current_color))
            current_font = "F4"
            current_size = max(base_size - 0.7, 6.6)
            current_color = (0.059, 0.090, 0.165)  # Dark navy-slate for inline code
        elif part in ("</b>", "</i>", "</code>"):
            if font_stack:
                current_font, current_size, current_color = font_stack.pop()
            else:
                current_font, current_size, current_color = base_font, base_size, base_color
        else:
            words = part.split(" ")
            for idx, w in enumerate(words):
                if w:
                    tokens.append((current_font, current_size, current_color, w))
                if idx < len(words) - 1:
                    tokens.append((current_font, current_size, current_color, " "))

    # Collapse adjacent space tokens
    cleaned: List[Token] = []
    for tok in tokens:
        if tok[3] == " " and (not cleaned or cleaned[-1][3] in (" ", "\n")):
            continue
        cleaned.append(tok)
    if cleaned and cleaned[-1][3] == " ":
        cleaned.pop()
    return cleaned


def wrap_tokens_to_lines(tokens: List[Token], max_width: float) -> List[List[Token]]:
    """Wraps a list of styled word tokens into lines fitting within max_width, respecting \\n."""
    lines: List[List[Token]] = []
    current_line: List[Token] = []
    current_width = 0.0

    for tok in tokens:
        font_id, fsize, color, word = tok
        if font_id == "__BR__" or word == "\n":
            if current_line and current_line[-1][3] == " ":
                current_line.pop()
            if current_line:
                lines.append(current_line)
            current_line = []
            current_width = 0.0
            continue

        if word == " " and not current_line:
            continue
        w_width = measure_word_width(word, font_id, fsize)

        if current_line and (current_width + w_width > max_width) and word != " ":
            if current_line and current_line[-1][3] == " ":
                current_line.pop()
            lines.append(current_line)
            current_line = [tok]
            current_width = w_width
        else:
            current_line.append(tok)
            current_width += w_width

    if current_line:
        if current_line[-1][3] == " ":
            current_line.pop()
        if current_line:
            lines.append(current_line)

    return lines if lines else [[("F1", 8.5, (0, 0, 0), "")]]


class CardioPDFDocument:
    """
    Multi-page PDF-1.4 document builder with automatic page-break management,
    vector drawing primitives, tables, callout boxes, and running headers/footers.
    """

    PAGE_W = 612.0  # 8.5 inches
    PAGE_H = 792.0  # 11.0 inches
    MARGIN_L = 36.0
    MARGIN_R = 36.0
    MARGIN_T = 46.0
    MARGIN_B = 46.0

    # Color Palette (RGB 0.0 - 1.0)
    NAVY = (0.086, 0.188, 0.345)       # #163058 Deep Enterprise Navy
    CERULEAN = (0.008, 0.518, 0.780)   # #0284C7 Medical Cerulean Blue
    TEAL = (0.051, 0.580, 0.533)       # #0D9488 Clinical Teal
    CRIMSON = (0.863, 0.149, 0.149)    # #DC2626 Alert Crimson
    AMBER = (0.851, 0.467, 0.024)      # #D97706 Warning Amber
    DARK_TEXT = (0.118, 0.161, 0.231)  # #1E293B Slate 800
    MUTED_TEXT = (0.392, 0.455, 0.545) # #64748B Slate 500
    LIGHT_BG = (0.973, 0.980, 0.988)   # #F8FAFC Slate 50
    CODE_BG = (0.945, 0.961, 0.976)    # #F1F5F9 Slate 100
    BORDER_GRAY = (0.796, 0.835, 0.882)# #CBD5E1 Slate 300

    def __init__(self) -> None:
        self.pages: List[List[str]] = []
        self.y = self.PAGE_H - self.MARGIN_T
        self._new_page()

    @property
    def content_w(self) -> float:
        return self.PAGE_W - self.MARGIN_L - self.MARGIN_R

    @property
    def current_stream(self) -> List[str]:
        return self.pages[-1]

    def _new_page(self) -> None:
        self.pages.append([])
        self.y = self.PAGE_H - self.MARGIN_T

    def ensure_space(self, needed_height: float) -> bool:
        """Starts a new page if remaining vertical space is less than needed_height."""
        if self.y - needed_height < self.MARGIN_B:
            self._new_page()
            return True
        return False

    def spacer(self, pts: float) -> None:
        if self.y - pts < self.MARGIN_B:
            self._new_page()
        else:
            self.y -= pts

    # -------------------------------------------------------------------------
    # Low-Level Drawing Helpers
    # -------------------------------------------------------------------------
    def draw_rect(
        self,
        x: float,
        y_bottom: float,
        w: float,
        h: float,
        fill_rgb: Optional[Tuple[float, float, float]] = None,
        stroke_rgb: Optional[Tuple[float, float, float]] = None,
        line_width: float = 0.5,
    ) -> None:
        cmds: List[str] = ["q"]
        if fill_rgb is not None:
            cmds.append(f"{fill_rgb[0]:.3f} {fill_rgb[1]:.3f} {fill_rgb[2]:.3f} rg")
        if stroke_rgb is not None:
            cmds.append(f"{stroke_rgb[0]:.3f} {stroke_rgb[1]:.3f} {stroke_rgb[2]:.3f} RG")
            cmds.append(f"{line_width:.2f} w")
        op = "B" if (fill_rgb is not None and stroke_rgb is not None) else ("f" if fill_rgb is not None else "S")
        cmds.append(f"{x:.2f} {y_bottom:.2f} {w:.2f} {h:.2f} re {op}")
        cmds.append("Q")
        self.current_stream.append(" ".join(cmds))

    def draw_line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke_rgb: Tuple[float, float, float] = BORDER_GRAY,
        line_width: float = 0.5,
    ) -> None:
        self.current_stream.append(
            f"q {stroke_rgb[0]:.3f} {stroke_rgb[1]:.3f} {stroke_rgb[2]:.3f} RG "
            f"{line_width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S Q"
        )

    def draw_token_line(self, x: float, y_baseline: float, line_tokens: List[Token]) -> None:
        """Draws a single wrapped line of styled tokens starting at (x, y_baseline)."""
        cur_x = x
        for font_id, fsize, rgb, word in line_tokens:
            if not word:
                continue
            if word != " ":
                escaped = escape_pdf_string(word)
                self.current_stream.append(
                    f"BT /{font_id} {fsize:.2f} Tf "
                    f"{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f} rg "
                    f"{cur_x:.2f} {y_baseline:.2f} Td ({escaped}) Tj ET"
                )
            cur_x += measure_word_width(word, font_id, fsize)

    # -------------------------------------------------------------------------
    # High-Level Document Components
    # -------------------------------------------------------------------------
    def add_cover_header(self) -> None:
        """Draws the executive title banner and 4-card KPI summary at the top of Page 1."""
        banner_h = 84.0
        banner_bottom = self.y - banner_h
        self.draw_rect(self.MARGIN_L, banner_bottom, self.content_w, banner_h, fill_rgb=self.NAVY)
        # Teal accent bar on left edge of banner
        self.draw_rect(self.MARGIN_L, banner_bottom, 6.0, banner_h, fill_rgb=self.TEAL)

        # Title text inside banner
        t1 = parse_inline_markup(
            "<b>FedCardio v2.0: Enterprise Federated Clinical AI & Actuarial Playbook</b>",
            base_font="F2",
            base_size=14.6,
            base_color=(1.0, 1.0, 1.0),
        )
        self.draw_token_line(self.MARGIN_L + 16, self.y - 24, t1)

        t2 = parse_inline_markup(
            "Complete Zero-to-Expert Micro-Level Architecture Guide, Mathematical Proofs & Live Demo Walkthrough",
            base_font="F1",
            base_size=9.5,
            base_color=(0.729, 0.902, 0.992),
        )
        self.draw_token_line(self.MARGIN_L + 16, self.y - 42, t2)

        t3 = parse_inline_markup(
            "8 Global Sovereign Hospitals  |  9,600 Physically Isolated Patient EHRs  |  Per-Sample DP-SGD (vmap)  |  Byzantine Defense  |  Coalitional SHAP",
            base_font="F1",
            base_size=8.0,
            base_color=(0.886, 0.910, 0.941),
        )
        self.draw_token_line(self.MARGIN_L + 16, self.y - 60, t3)

        t4 = parse_inline_markup(
            "<i>Designed for Clinical AI Architects, Healthcare Actuaries, Privacy Engineers & Executive Evaluators (Zero Prior Context Assumed)</i>",
            base_font="F3",
            base_size=7.6,
            base_color=(0.580, 0.820, 0.780),
        )
        self.draw_token_line(self.MARGIN_L + 16, self.y - 74, t4)

        self.y = banner_bottom - 10.0

        # 4 Executive KPI Cards Row
        kpis = [
            ("8 SOVEREIGN SILOS", "9,600 Patient Records", "8,160 Train / 1,440 Test (0 Shared CSVs)", self.CERULEAN),
            ("CLINICAL RESCUE RATE", "4 / 4 (100% Rescued)", "+0.464 Risk Lift on Rare Demographics", self.TEAL),
            ("PRIVACY & SECURITY", "0 Raw Rows Exposed", "Exact vmap DP-SGD + FedMedian", self.NAVY),
            ("NET CONSORTIUM ROI", "$118.76M / Year", "330.9x Return (2,036 Prevented AMIs)", (0.086, 0.494, 0.235)),
        ]
        gap = 8.0
        card_w = (self.content_w - 3 * gap) / 4.0
        card_h = 46.0
        card_bottom = self.y - card_h

        for idx, (label, big_val, sub, accent) in enumerate(kpis):
            cx = self.MARGIN_L + idx * (card_w + gap)
            self.draw_rect(cx, card_bottom, card_w, card_h, fill_rgb=self.LIGHT_BG, stroke_rgb=self.BORDER_GRAY, line_width=0.6)
            self.draw_rect(cx, card_bottom + card_h - 3.5, card_w, 3.5, fill_rgb=accent)

            l_tok = parse_inline_markup(f"<b>{label}</b>", base_font="F2", base_size=7.0, base_color=self.MUTED_TEXT)
            v_tok = parse_inline_markup(f"<b>{big_val}</b>", base_font="F2", base_size=10.2, base_color=accent)
            s_tok = parse_inline_markup(sub, base_font="F1", base_size=6.5, base_color=self.DARK_TEXT)

            self.draw_token_line(cx + 6, self.y - 14, l_tok)
            self.draw_token_line(cx + 6, self.y - 28, v_tok)
            self.draw_token_line(cx + 6, self.y - 40, s_tok)

        self.y = card_bottom - 12.0

    def add_h1(self, title: str) -> None:
        """Major Part / Section Header with navy banner strip."""
        self.ensure_space(56.0)
        self.spacer(6.0)
        h = 22.0
        bottom = self.y - h
        self.draw_rect(self.MARGIN_L, bottom, self.content_w, h, fill_rgb=self.NAVY)
        self.draw_rect(self.MARGIN_L, bottom, 4.5, h, fill_rgb=self.TEAL)
        tokens = parse_inline_markup(f"<b>{title}</b>", base_font="F2", base_size=11.5, base_color=(1.0, 1.0, 1.0))
        self.draw_token_line(self.MARGIN_L + 11, bottom + 6.5, tokens)
        self.y = bottom - 7.0

    def add_h2(self, title: str) -> None:
        """Sub-section Header with cerulean text and subtle bottom rule."""
        self.ensure_space(38.0)
        self.spacer(4.0)
        tokens = parse_inline_markup(f"<b>{title}</b>", base_font="F2", base_size=10.2, base_color=self.CERULEAN)
        self.draw_token_line(self.MARGIN_L, self.y - 11, tokens)
        self.draw_line(self.MARGIN_L, self.y - 14.5, self.MARGIN_L + self.content_w, self.y - 14.5, stroke_rgb=self.BORDER_GRAY, line_width=0.6)
        self.y -= 19.0

    def add_h3(self, title: str) -> None:
        """Micro-topic Header."""
        self.ensure_space(26.0)
        self.spacer(2.0)
        tokens = parse_inline_markup(f"<b>{title}</b>", base_font="F2", base_size=9.2, base_color=self.NAVY)
        self.draw_token_line(self.MARGIN_L, self.y - 10, tokens)
        self.y -= 13.5

    def add_paragraph(self, text: str, font_size: float = 8.4, leading: float = 11.4, space_after: float = 4.5) -> None:
        tokens = parse_inline_markup(text, base_font="F1", base_size=font_size, base_color=self.DARK_TEXT)
        lines = wrap_tokens_to_lines(tokens, self.content_w)
        for line in lines:
            self.ensure_space(leading + 2.0)
            self.draw_token_line(self.MARGIN_L, self.y - font_size, line)
            self.y -= leading
        self.spacer(space_after)

    def add_bullet(self, text: str, bullet_char: str = "•", indent: float = 12.0, font_size: float = 8.3, leading: float = 11.2) -> None:
        tokens = parse_inline_markup(text, base_font="F1", base_size=font_size, base_color=self.DARK_TEXT)
        lines = wrap_tokens_to_lines(tokens, self.content_w - indent)
        for idx, line in enumerate(lines):
            self.ensure_space(leading + 2.0)
            if idx == 0:
                b_tok = parse_inline_markup(f"<b>{bullet_char}</b>", base_font="F2", base_size=font_size, base_color=self.CERULEAN)
                self.draw_token_line(self.MARGIN_L + 2, self.y - font_size, b_tok)
            self.draw_token_line(self.MARGIN_L + indent, self.y - font_size, line)
            self.y -= leading
        self.spacer(2.2)

    def add_callout(
        self,
        title: str,
        body: str,
        accent_rgb: Tuple[float, float, float] = TEAL,
        bg_rgb: Tuple[float, float, float] = (0.941, 0.992, 0.980),
    ) -> None:
        """Draws a highlighted clinical/architectural callout box with left accent bar."""
        pad_x = 10.0
        pad_y = 6.0
        inner_w = self.content_w - 2 * pad_x - 4.0

        title_tokens = parse_inline_markup(f"<b>{title}</b>", base_font="F2", base_size=8.6, base_color=accent_rgb)
        title_lines = wrap_tokens_to_lines(title_tokens, inner_w)

        body_tokens = parse_inline_markup(body, base_font="F1", base_size=8.1, base_color=self.DARK_TEXT)
        body_lines = wrap_tokens_to_lines(body_tokens, inner_w)

        line_h = 10.8
        box_h = pad_y * 2 + (len(title_lines) + len(body_lines)) * line_h + 2.0
        self.ensure_space(box_h + 6.0)

        box_bottom = self.y - box_h
        self.draw_rect(self.MARGIN_L, box_bottom, self.content_w, box_h, fill_rgb=bg_rgb, stroke_rgb=self.BORDER_GRAY, line_width=0.5)
        self.draw_rect(self.MARGIN_L, box_bottom, 4.0, box_h, fill_rgb=accent_rgb)

        cur_y = self.y - pad_y - 8.0
        for tline in title_lines:
            self.draw_token_line(self.MARGIN_L + pad_x + 2, cur_y, tline)
            cur_y -= line_h
        cur_y -= 1.5
        for bline in body_lines:
            self.draw_token_line(self.MARGIN_L + pad_x + 2, cur_y, bline)
            cur_y -= line_h

        self.y = box_bottom - 6.0

    def add_code_block(self, lines: List[str], title: Optional[str] = None) -> None:
        """Draws a monospace code / formula / CLI block."""
        font_size = 7.4
        line_h = 9.6
        pad_x = 8.0
        pad_y = 5.0
        header_h = 14.0 if title else 0.0
        box_h = header_h + pad_y * 2 + len(lines) * line_h
        self.ensure_space(box_h + 6.0)

        box_bottom = self.y - box_h
        self.draw_rect(self.MARGIN_L, box_bottom, self.content_w, box_h, fill_rgb=self.CODE_BG, stroke_rgb=self.BORDER_GRAY, line_width=0.6)

        cur_y = self.y
        if title:
            self.draw_rect(self.MARGIN_L, self.y - header_h, self.content_w, header_h, fill_rgb=(0.886, 0.910, 0.941), stroke_rgb=self.BORDER_GRAY, line_width=0.5)
            t_tok = parse_inline_markup(f"<b>{title}</b>", base_font="F2", base_size=7.5, base_color=self.NAVY)
            self.draw_token_line(self.MARGIN_L + pad_x, self.y - 10.0, t_tok)
            cur_y -= header_h

        cur_y -= (pad_y + font_size)
        for raw_line in lines:
            tok: List[Token] = [("F4", font_size, (0.059, 0.090, 0.165), raw_line)]
            self.draw_token_line(self.MARGIN_L + pad_x, cur_y, tok)
            cur_y -= line_h

        self.y = box_bottom - 5.5

    def add_table(
        self,
        headers: List[str],
        rows: List[List[str]],
        col_widths: List[float],
        font_size: float = 7.6,
        line_h: float = 9.8,
    ) -> None:
        """
        Draws a multi-column table with automatic multi-line cell wrapping,
        zebra striping, and automatic header repetition across page breaks.
        """
        pad_x = 4.5
        pad_y = 3.5

        def measure_row(cells: List[str], is_header: bool) -> Tuple[List[List[List[Token]]], float]:
            wrapped_cells: List[List[List[Token]]] = []
            max_lines = 1
            for c_idx, cell_text in enumerate(cells):
                cw = col_widths[c_idx] - 2 * pad_x
                if is_header:
                    toks = parse_inline_markup(f"<b>{cell_text}</b>", base_font="F2", base_size=font_size, base_color=(1.0, 1.0, 1.0))
                else:
                    toks = parse_inline_markup(cell_text, base_font="F1", base_size=font_size, base_color=self.DARK_TEXT)
                clines = wrap_tokens_to_lines(toks, cw)
                wrapped_cells.append(clines)
                if len(clines) > max_lines:
                    max_lines = len(clines)
            row_h = max_lines * line_h + 2 * pad_y
            return wrapped_cells, row_h

        header_wrapped, header_h = measure_row(headers, is_header=True)

        def draw_measured_row(wrapped_cells: List[List[List[Token]]], row_h: float, bg_rgb: Tuple[float, float, float], is_header: bool) -> None:
            row_bottom = self.y - row_h
            cur_x = self.MARGIN_L
            for c_idx, clines in enumerate(wrapped_cells):
                cw = col_widths[c_idx]
                self.draw_rect(
                    cur_x,
                    row_bottom,
                    cw,
                    row_h,
                    fill_rgb=bg_rgb,
                    stroke_rgb=(0.25, 0.35, 0.50) if is_header else self.BORDER_GRAY,
                    line_width=0.45,
                )
                text_y = self.y - pad_y - font_size + 1.0
                for line in clines:
                    self.draw_token_line(cur_x + pad_x, text_y, line)
                    text_y -= line_h
                cur_x += cw
            self.y = row_bottom

        # Ensure header + first row fit together
        first_row_h = measure_row(rows[0], is_header=False)[1] if rows else 18.0
        self.ensure_space(header_h + first_row_h + 6.0)
        draw_measured_row(header_wrapped, header_h, self.NAVY, is_header=True)

        for r_idx, row in enumerate(rows):
            row_wrapped, row_h = measure_row(row, is_header=False)
            if self.y - row_h < self.MARGIN_B:
                self._new_page()
                draw_measured_row(header_wrapped, header_h, self.NAVY, is_header=True)
            bg = (1.0, 1.0, 1.0) if r_idx % 2 == 0 else self.LIGHT_BG
            draw_measured_row(row_wrapped, row_h, bg, is_header=False)

        self.spacer(6.5)

    # -------------------------------------------------------------------------
    # Two-Pass Header/Footer Injection & PDF-1.4 Serialization
    # -------------------------------------------------------------------------
    def _decorate_pages(self) -> List[bytes]:
        total_pages = len(self.pages)
        compiled_streams: List[bytes] = []

        for idx, page_cmds in enumerate(self.pages):
            page_num = idx + 1
            decorations: List[str] = []

            # Running Header on Pages >= 2
            if page_num > 1:
                h_y = self.PAGE_H - 28.0
                decorations.append(
                    f"BT /F2 7.5 Tf {self.NAVY[0]:.3f} {self.NAVY[1]:.3f} {self.NAVY[2]:.3f} rg "
                    f"{self.MARGIN_L:.2f} {h_y:.2f} Td "
                    f"(FedCardio v2.0 -- Complete Zero-to-Expert Micro-Level Architecture & Live Demo Playbook) Tj ET"
                )
                decorations.append(
                    f"BT /F3 7.5 Tf {self.MUTED_TEXT[0]:.3f} {self.MUTED_TEXT[1]:.3f} {self.MUTED_TEXT[2]:.3f} rg "
                    f"{self.PAGE_W - self.MARGIN_R - 142:.2f} {h_y:.2f} Td "
                    f"(Enterprise Clinical AI Blueprint) Tj ET"
                )
                decorations.append(
                    f"q {self.BORDER_GRAY[0]:.3f} {self.BORDER_GRAY[1]:.3f} {self.BORDER_GRAY[2]:.3f} RG "
                    f"0.5 w {self.MARGIN_L:.2f} {h_y - 4:.2f} m {self.PAGE_W - self.MARGIN_R:.2f} {h_y - 4:.2f} l S Q"
                )

            # Running Footer on All Pages
            f_line_y = 34.0
            f_text_y = 23.0
            decorations.append(
                f"q {self.BORDER_GRAY[0]:.3f} {self.BORDER_GRAY[1]:.3f} {self.BORDER_GRAY[2]:.3f} RG "
                f"0.5 w {self.MARGIN_L:.2f} {f_line_y:.2f} m {self.PAGE_W - self.MARGIN_R:.2f} {f_line_y:.2f} l S Q"
            )
            decorations.append(
                f"BT /F1 7.5 Tf {self.MUTED_TEXT[0]:.3f} {self.MUTED_TEXT[1]:.3f} {self.MUTED_TEXT[2]:.3f} rg "
                f"{self.MARGIN_L:.2f} {f_text_y:.2f} Td "
                f"(FedCardio v2.0: 8 Sovereign Hospitals | 9,600 Records | Exact DP-SGD | Byzantine Defense | $118.76M Net ROI) Tj ET"
            )
            page_str = f"Page {page_num} of {total_pages}"
            p_w = measure_word_width(page_str, "F2", 7.8)
            decorations.append(
                f"BT /F2 7.8 Tf {self.NAVY[0]:.3f} {self.NAVY[1]:.3f} {self.NAVY[2]:.3f} rg "
                f"{self.PAGE_W - self.MARGIN_R - p_w:.2f} {f_text_y:.2f} Td ({page_str}) Tj ET"
            )

            full_content = "\n".join(page_cmds + decorations).encode("latin-1", errors="replace")
            compiled_streams.append(full_content)

        return compiled_streams

    def save(self, output_path: str | Path) -> int:
        """Compiles the PDF-1.4 binary with FlateDecode streams and xref table."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        streams = self._decorate_pages()

        objects: List[bytes] = []

        def add_obj(body: bytes) -> int:
            objects.append(body)
            return len(objects)

        # Obj 1: Catalog (placeholder, updated once Pages ID is known)
        add_obj(b"")
        # Obj 2: Pages root (placeholder)
        add_obj(b"")

        # Obj 3..7: Standard Type1 Fonts (WinAnsiEncoding for bullet/symbol support)
        font_specs = [
            ("F1", "Helvetica"),
            ("F2", "Helvetica-Bold"),
            ("F3", "Helvetica-Oblique"),
            ("F4", "Courier"),
            ("F5", "Courier-Bold"),
        ]
        font_obj_ids: Dict[str, int] = {}
        for f_key, base_font in font_specs:
            fid = add_obj(
                f"<< /Type /Font /Subtype /Type1 /BaseFont /{base_font} /Encoding /WinAnsiEncoding >>".encode("ascii")
            )
            font_obj_ids[f_key] = fid

        font_res_str = " ".join(f"/{k} {oid} 0 R" for k, oid in font_obj_ids.items())

        page_obj_ids: List[int] = []
        for raw_stream in streams:
            compressed = zlib.compress(raw_stream, level=9)
            stream_header = f"<< /Length {len(compressed)} /Filter /FlateDecode >>\nstream\n".encode("ascii")
            stream_footer = b"\nendstream"
            content_id = add_obj(stream_header + compressed + stream_footer)

            page_dict = (
                f"<< /Type /Page /Parent 2 0 R "
                f"/MediaBox [0 0 {int(self.PAGE_W)} {int(self.PAGE_H)}] "
                f"/Resources << /Font << {font_res_str} >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode("ascii")
            pid = add_obj(page_dict)
            page_obj_ids.append(pid)

        # Info Dictionary
        info_id = add_obj(
            b"<< /Title (FedCardio v2.0 Complete Zero-to-Expert Guide and Demo Playbook) "
            b"/Author (FedCardio Multi-Agent Engineering Architecture) "
            b"/Subject (Federated Clinical AI, Exact DP-SGD, Byzantine Defense, Coalitional SHAP & Actuarial ROI) "
            b"/Creator (FedCardio v2.0 PDF Typesetting Engine) >>"
        )

        # Fill in Obj 1 (Catalog) and Obj 2 (Pages)
        kids_str = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
        objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
        objects[1] = f"<< /Type /Pages /Kids [ {kids_str} ] /Count {len(page_obj_ids)} >>".encode("ascii")

        # Assemble final PDF binary
        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets: List[int] = []
        for idx, obj_bytes in enumerate(objects):
            obj_num = idx + 1
            offsets.append(len(out))
            out.extend(f"{obj_num} 0 obj\n".encode("ascii"))
            out.extend(obj_bytes)
            out.extend(b"\nendobj\n")

        xref_offset = len(out)
        out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        out.extend(b"0000000000 65535 f \n")
        for off in offsets:
            out.extend(f"{off:010d} 00000 n \n".encode("ascii"))

        out.extend(
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info {info_id} 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
        )

        output_path.write_bytes(bytes(out))
        return len(page_obj_ids)


# =============================================================================
# 2. COMPLETE FEDCARDIO V2.0 CONTENT SPECIFICATION (PARTS 1 TO 6)
# =============================================================================

def build_fedcardio_guide(output_paths: List[Path]) -> Tuple[int, int]:
    pdf = CardioPDFDocument()

    # -------------------------------------------------------------------------
    # COVER & SYSTEM ARCHITECTURE MAP
    # -------------------------------------------------------------------------
    pdf.add_cover_header()

    pdf.add_callout(
        "Purpose of This Manual: Zero Prior Knowledge to Complete Micro-Level Mastery",
        "This self-contained blueprint explains every clinical, mathematical, cryptographic, and financial mechanism "
        "in FedCardio v2.0 from scratch. It assumes zero prior familiarity with cardiology, federated machine learning, "
        "differential privacy, or healthcare actuarial modeling, and provides a step-by-step live demonstration script "
        "for both the CLI and the 6-tab Streamlit Enterprise Portal.",
        accent_rgb=pdf.CERULEAN,
        bg_rgb=(0.941, 0.969, 1.000),
    )

    pdf.add_h2("0. Complete Codebase & Module Architecture Map")
    pdf.add_paragraph(
        "Every file in the <code>FL-FedActuary</code> workspace is modularly separated into configuration, sovereign "
        "data provisioning, domain-bound tensor preprocessing, GroupNorm neural architecture, per-sample DP-SGD "
        "training, Byzantine-resilient aggregation, coalitional SHAP explainability, and healthcare actuarial ROI:"
    )

    pdf.add_table(
        headers=["Module Path", "Core Classes / Functions", "Micro-Level Responsibility"],
        col_widths=[165, 148, 227],
        rows=[
            [
                "<code>fedcardio/config.py</code>",
                "<code>HOSPITALS</code>, <code>FLConfig</code>, <code>CLINICAL_BOUNDS</code>, <code>ACTUARIAL_BENCHMARKS</code>",
                "Defines the 8 global sovereign hospitals, 13 UCI clinical features, fixed physiological bounds, and actuarial cost parameters.",
            ],
            [
                "<code>fedcardio/data/downloader.py</code>",
                "<code>download_and_prepare_hospitals()</code>, <code>verify_physical_isolation()</code>",
                "Parses the 4 classic UCI heart disease files, applies modality-aware missingness defaults, and performs an 85%/15% pre-disk split.",
            ],
            [
                "<code>fedcardio/data/synthesizer.py</code>",
                "<code>scale_all_hospitals()</code>, <code>_generate_hospital_patients()</code>",
                "Scales all 8 hospital silos to 1,020 train + 180 test rows (9,600 total) using demographic-specific physiological copulas.",
            ],
            [
                "<code>fedcardio/data/preprocess.py</code>",
                "<code>encode_clinical_dataframe()</code>, <code>load_hospital_dataset()</code>",
                "Transforms 13 clinical features into a 22-dimensional normalized float32 tensor without cross-hospital StandardScaler leakage.",
            ],
            [
                "<code>fedcardio/models/clinical_net.py</code>",
                "<code>ClinicalNet(in_features=22)</code>",
                "3-layer clinical MLP using <code>nn.GroupNorm(4, 64)</code> and <code>nn.SiLU()</code> to eliminate BatchNorm privacy and Non-IID failures.",
            ],
            [
                "<code>fedcardio/privacy/dp_engine.py</code>",
                "<code>train_step_with_dp()</code>, <code>PrivacyAccountant</code>",
                "Vectorized per-patient gradients via <code>torch.func.vmap(grad(functional_call))</code>, per-sample L2 clipping, and GDP accounting.",
            ],
            [
                "<code>fedcardio/federated/client.py</code>",
                "<code>HospitalClient</code>, <code>train_local_round()</code>",
                "Executes local hospital training epochs with decoupled FedProx proximal regularization applied after DP noise injection.",
            ],
            [
                "<code>fedcardio/federated/strategies.py</code>",
                "<code>aggregate_fedavg</code>, <code>aggregate_median</code>, <code>aggregate_trimmed_mean</code>",
                "Implements FedAvg, FedProx, coordinate-wise FedMedian, TrimmedMean, L2/Cosine drift, and Byzantine poisoning attacks.",
            ],
            [
                "<code>fedcardio/federated/server.py</code>",
                "<code>FederatedServer</code>, <code>benchmark_byzantine_resilience()</code>",
                "Orchestrates multi-round federation across the 8 hospitals and logs SHA-256 wire packets to <code>logs/federated_audit.db</code>.",
            ],
            [
                "<code>fedcardio/explainability/shap_explainer.py</code>",
                "<code>ClinicalSHAPExplainer</code>, <code>explain_patient()</code>",
                "Grouped permutation coalitional SHAP that perturbs one-hot groups jointly and enforces exact additivity.",
            ],
            [
                "<code>fedcardio/evaluation/fl_proof.py</code>",
                "<code>run_blindspot_proof()</code>, <code>compute_non_iid_heterogeneity()</code>",
                "Proves how the 8-center federated model rescues 4 rare clinical blind-spot patient archetypes and computes pairwise JSD skew.",
            ],
            [
                "<code>fedcardio/business/roi_engine.py</code>",
                "<code>HealthcareROIEngine</code>, <code>triage_hospital_cohort()</code>",
                "Computes the $118.76M Net Annual Value (330.9x ROI) and executes 3-tier Clinical Decision Support (CDS) batch EHR triage.",
            ],
            [
                "<code>web/app.py</code> & <code>cardio_cli.py</code>",
                "6-Tab Streamlit Portal & CLI Orchestrator",
                "Interactive executive dashboard and headless command-line runner for live demonstrations and automated verification.",
            ],
        ],
    )

    # =========================================================================
    # PART 1: THE REAL-WORLD CLINICAL & REGULATORY PROBLEM FROM SCRATCH
    # =========================================================================
    pdf.add_h1("PART 1: The Real-World Clinical & Regulatory Problem From Scratch")

    pdf.add_h2("1.1 What is Coronary Artery Disease (CAD)?")
    pdf.add_paragraph(
        "The human heart is a muscular pump that beats roughly 100,000 times per day. To sustain this continuous mechanical "
        "work, the heart muscle itself (the <i>myocardium</i>) requires a constant supply of oxygen-rich blood delivered "
        "by three major blood vessels wrapping around the outside of the heart: the <b>coronary arteries</b>. "
        "<b>Coronary Artery Disease (CAD)</b> occurs when cholesterol plaques, calcium deposits, and inflammatory cells "
        "build up inside the walls of these arteries (a process called <i>atherosclerosis</i>), narrowing the vessel lumen."
    )
    pdf.add_bullet(
        "<b>Healthy / Mildly Narrowed (< 50% diameter narrowing, <code>target = 0</code>):</b> Blood flow remains sufficient "
        "even during physical exertion. The patient does not require invasive surgical intervention."
    )
    pdf.add_bullet(
        "<b>Clinically Significant CAD (>= 50% diameter narrowing, <code>target = 1</code>):</b> At least one major coronary "
        "artery is obstructed by 50% or more. During physical stress or emotional exertion, the heart muscle is starved "
        "of oxygen (<i>myocardial ischemia</i>). If a plaque ruptures, a blood clot completely blocks the artery, causing "
        "an <b>Acute Myocardial Infarction (AMI)</b> — a life-threatening heart attack."
    )

    pdf.add_h2("1.2 The High-Stakes Diagnostic Dilemma: False Negatives vs. False Positives")
    pdf.add_paragraph(
        "When a patient walks into an Emergency Department or Cardiology Clinic with chest discomfort, shortness of breath, "
        "or abnormal ECG findings, the cardiologist faces an asymmetric, high-stakes diagnostic decision:"
    )

    pdf.add_table(
        headers=["Diagnostic Error Type", "Clinical Definition", "Real-World Medical & Financial Consequence"],
        col_widths=[125, 165, 250],
        rows=[
            [
                "<b>False Negative (Type II Error)</b>\n<i>Missed Severe CAD</i>",
                "Patient actually has severe coronary stenosis (<code>target = 1</code>), but the AI or physician classifies them as Low Risk (<code>< 0.50</code>) and discharges them.",
                "<b>Catastrophic Outcome:</b> Patient suffers an unpredicted out-of-hospital Acute Myocardial Infarction (AMI). Emergency ICU admission, emergency percutaneous coronary intervention (PCI stenting), permanent heart failure, or death. <b>Average Emergency Cost: $42,500 per missed case.</b>",
            ],
            [
                "<b>False Positive (Type I Error)</b>\n<i>Unnecessary Invasive Cath</i>",
                "Patient has healthy coronary arteries (<code>target = 0</code>), but the AI or physician classifies them as High Risk (<code>>= 0.50</code>) and orders catheterization.",
                "<b>Wasted Invasive Procedure:</b> Patient undergoes invasive <b>Coronary Angiography</b> (threading a catheter from the femoral/radial artery into the heart under X-ray fluoroscopy and iodine dye), exposing them to arterial bleeding, stroke, and kidney injury only to find clean arteries. <b>Average Cost: $12,400 per procedure.</b>",
            ],
        ],
    )

    pdf.add_h2("1.3 Why Single-Hospital AI Models Fail: 'Local Island Blind Spots'")
    pdf.add_paragraph(
        "Suppose a hospital trains a deep learning model exclusively on its own Electronic Health Record (EHR) database. "
        "Why does that local model fail dangerously when encountering atypical patients? Because every hospital operates "
        "as an isolated <b>Non-IID (Non-Independent and Identically Distributed) demographic island</b>:"
    )
    pdf.add_bullet(
        "<b>1. Demographic & Gender Skew:</b> The <b>VA Long Beach Medical Center</b> is a US Veterans Affairs hospital. "
        "Historically, <b>94.7% of its cardiac cohort is older male military veterans</b> (average age 59.3) with high "
        "rates of smoking and hypertension. When a <b>51-year-old female veteran</b> presents with atypical chest pain "
        "(microvascular fatigue and jaw/back ache rather than crushing substernal pressure), a model trained only at "
        "VA Long Beach has almost never seen female CAD physiology and misclassifies her as Low Risk (<code>0.382</code>)."
    )
    pdf.add_bullet(
        "<b>2. Diagnostic Equipment & Modality Skew:</b> Tertiary academic referral centers like <b>Cleveland Clinic</b>, "
        "<b>Mayo Clinic</b>, and <b>Johns Hopkins</b> routinely perform invasive fluoroscopy (<code>ca</code>: number of "
        "calcified vessels) and nuclear thallium stress imaging (<code>thal</code>) on <b>94% to 98%</b> of patients. "
        "Conversely, outpatient preventive screening clinics like the <b>Hungarian Institute of Cardiology</b> perform "
        "invasive catheterization on only <b>1.5%</b> of walk-in patients, relying almost entirely on non-invasive "
        "treadmill ECG markers (<code>oldpeak</code> ST-segment depression and <code>thalach</code> heart rate)."
    )
    pdf.add_bullet(
        "<b>3. Genetic, Dietary & Baseline Prevalence Skew:</b> Baseline CAD prevalence ranges from <b>24.7%</b> at "
        "<b>Tokyo University Hospital</b> (where patients present with low serum cholesterol <code>< 195 mg/dL</code> "
        "but microvascular ST depression) to <b>78.9%</b> at <b>University Hospital Zurich</b> (an acute coronary referral center)."
    )

    pdf.add_h2("1.4 Why Hospitals Cannot Simply Share CSV Files (HIPAA, GDPR & $499/Record Liability)")
    pdf.add_paragraph(
        "If single-hospital models suffer from local blind spots, why don't all 8 hospitals upload their patient CSVs "
        "into one centralized AWS or Azure data lake? Because <b>centralizing clinical data across sovereign institutions "
        "is illegal and financially catastrophic</b>:"
    )
    pdf.add_bullet(
        "<b>United States — HIPAA Privacy & Security Rules:</b> Exporting Protected Health Information (PHI) outside a "
        "covered entity's sovereign boundary without patient-level authorization or complex multi-party Business Associate "
        "Agreements (BAAs) carries severe federal civil and criminal penalties."
    )
    pdf.add_bullet(
        "<b>European Union — GDPR Article 9 (Special Category Health Data):</b> Zurich (Switzerland) and Charite Berlin "
        "(Germany) are legally prohibited from transferring identifiable biometric or clinical records across international "
        "borders to US servers."
    )
    pdf.add_bullet(
        "<b>East Asia — Japan APPI (Act on the Protection of Personal Information):</b> Strictly restricts cross-border "
        "medical record transfers from Tokyo University Hospital."
    )
    pdf.add_bullet(
        "<b>Actuarial Breach Liability ($499 per Healthcare Record):</b> According to the IBM Ponemon Healthcare Cost of a "
        "Data Breach Benchmark, healthcare is the #1 target of ransomware and data extortion, costing an average of "
        "<b>$499 per compromised patient record</b>. Centralizing 9,600 records creates a single honeypot worth "
        "<b>$4.79M</b> in immediate breach liability (and <b>$499M</b> for a 1-million-patient health system)."
    )

    pdf.add_callout(
        "1.5 The Federated Learning Breakthrough: Moving Mathematical Gradients, Never Patient Rows",
        "Instead of moving patient records to a central model, Federated Learning moves the model to the patient records. "
        "Each hospital trains a copy of ClinicalNet (6,593 floating-point parameters) strictly behind its own firewall, "
        "Perturbs per-patient gradients with formal Differential Privacy (DP-SGD), and transmits ONLY the 26-kilobyte "
        "encrypted weight tensor to the central coordinator. Raw patient records exposed across the network = 0.",
        accent_rgb=pdf.TEAL,
    )

    # =========================================================================
    # PART 2: MICRO-LEVEL ANATOMY OF THE DATA & PHYSICAL ISOLATION
    # =========================================================================
    pdf.add_h1("PART 2: Micro-Level Anatomy of the Data & Physical Isolation")

    pdf.add_h2("2.1 The 8 Sovereign Hospital Silos (`fedcardio/config.py`)")
    pdf.add_paragraph(
        "Unlike toy Federated Learning tutorials that load a single CSV file into memory and slice it in RAM on the fly, "
        "<b>FedCardio v2.0 enforces strict physical disk isolation</b>. Every hospital is provisioned in its own isolated "
        "directory (<code>data/hospital_<id>/<file>.csv</code>) with <b>1,020 training records</b> and <b>180 held-out "
        "evaluation records</b> (<b>8,160 total training rows</b> + <b>1,440 evaluation rows</b> = <b>9,600 total patients</b>):"
    )

    pdf.add_table(
        headers=["Hospital ID & Name", "Physical Disk Silo", "Clinical Archetype", "Train / Test", "CAD %", "Female %", "Cath %", "JSD Skew"],
        col_widths=[102, 114, 112, 52, 38, 42, 40, 40],
        rows=[
            [
                "<b>1. Cleveland Clinic</b>\n(Ohio, USA)",
                "<code>data/hospital_cleveland/\ncleveland_ehr.csv</code>",
                "Balanced Tertiary Referral\n(Original UCI Core)",
                "1,020 / 180",
                "47.0%",
                "31.6%",
                "98.3%",
                "0.0442",
            ],
            [
                "<b>2. Hungarian Inst.</b>\n(Budapest, HU)",
                "<code>data/hospital_hungary/\nhungarian_ehr.csv</code>",
                "Preventive Non-Invasive\nScreening Clinic",
                "1,020 / 180",
                "36.3%",
                "26.4%",
                "1.5%",
                "0.0567",
            ],
            [
                "<b>3. Univ. Hosp. Zurich</b>\n(Zurich, CH)",
                "<code>data/hospital_zurich/\nswitzerland_ehr.csv</code>",
                "Acute Severe Coronary\nReferral Center",
                "1,020 / 180",
                "78.9%",
                "15.5%",
                "62.5%",
                "0.0886",
            ],
            [
                "<b>4. VA Long Beach</b>\n(California, USA)",
                "<code>data/hospital_longbeach/\nva_longbeach_ehr.csv</code>",
                "Older Male Military\nVeterans Cohort",
                "1,020 / 180",
                "75.6%",
                "5.3%",
                "13.5%",
                "0.0725",
            ],
            [
                "<b>5. Mayo Clinic</b>\n(Minnesota, USA)",
                "<code>data/hospital_mayo/\nmayo_rochester_ehr.csv</code>",
                "Preventive Cardiology &\nExecutive Screening",
                "1,020 / 180",
                "31.2%",
                "48.5%",
                "94.0%",
                "0.0589",
            ],
            [
                "<b>6. Johns Hopkins</b>\n(Maryland, USA)",
                "<code>data/hospital_hopkins/\njohns_hopkins_ehr.csv</code>",
                "High-Acuity Urban\nCardiometabolic Center",
                "1,020 / 180",
                "61.8%",
                "52.0%",
                "88.5%",
                "0.0446",
            ],
            [
                "<b>7. Charite Berlin</b>\n(Berlin, DE)",
                "<code>data/hospital_charite/\ncharite_berlin_ehr.csv</code>",
                "European Geriatric\nHeart Center",
                "1,020 / 180",
                "68.4%",
                "44.0%",
                "79.0%",
                "0.0470",
            ],
            [
                "<b>8. Tokyo Univ. Hosp.</b>\n(Tokyo, JP)",
                "<code>data/hospital_tokyo/\ntokyo_univ_ehr.csv</code>",
                "East Asian Low-Lipid /\nMicrovascular Cohort",
                "1,020 / 180",
                "24.7%",
                "46.0%",
                "91.0%",
                "0.0956",
            ],
        ],
    )

    pdf.add_h2("2.2 Micro-Level Glossary of All 13 Clinical Concepts (`UCI_COLUMNS`)")
    pdf.add_paragraph(
        "Every hospital record contains exactly <b>13 standardized clinical predictors</b> plus the binary diagnostic label "
        "<code>target</code>. Below is the complete medical and mathematical reference for all 14 columns:"
    )

    pdf.add_table(
        headers=["#", "Column Name", "Data Type & Range", "Plain-English Medical Meaning & Pathophysiological Role"],
        col_widths=[20, 74, 116, 330],
        rows=[
            ["1", "<code>age</code>", "Continuous <code>[20, 90]</code> yrs", "Patient chronological age in years. Progressive arterial stiffening and plaque burden increase with age."],
            ["2", "<code>sex</code>", "Binary <code>{0: Female, 1: Male}</code>", "Biological sex. Men develop epicardial plaques ~10 years earlier; women more frequently present with microvascular angina."],
            ["3", "<code>cp</code>", "Categorical <code>{1, 2, 3, 4}</code>", "<b>Chest Pain Type:</b> <code>1</code> = Typical Angina (substernal pressure on exertion), <code>2</code> = Atypical Angina, <code>3</code> = Non-anginal pain, <code>4</code> = Asymptomatic (<b>Silent Ischemia</b> — common in diabetics, highly dangerous)."],
            ["4", "<code>trestbps</code>", "Continuous <code>[80, 220]</code> mmHg", "<b>Resting Systolic Blood Pressure</b> on admission. High arterial pressure shears the inner vessel wall (endothelium), accelerating plaque formation."],
            ["5", "<code>chol</code>", "Continuous <code>[100, 600]</code> mg/dL", "<b>Serum Total Cholesterol.</b> Provides the raw LDL lipid substrate that deposits inside coronary artery walls."],
            ["6", "<code>fbs</code>", "Binary <code>{0: <=120, 1: >120}</code>", "<b>Fasting Blood Sugar > 120 mg/dL.</b> Clinical indicator of diabetes/hyperglycemia, which damages nerves (masking chest pain) and small vessels."],
            ["7", "<code>restecg</code>", "Categorical <code>{0, 1, 2}</code>", "<b>Resting 12-Lead ECG:</b> <code>0</code> = Normal, <code>1</code> = ST-T wave abnormality (inversion/elevation >= 0.05 mV), <code>2</code> = Left Ventricular Hypertrophy (thickened heart wall)."],
            ["8", "<code>thalach</code>", "Continuous <code>[60, 220]</code> bpm", "<b>Maximum Heart Rate Achieved</b> during treadmill stress test. Inability to raise heart rate under exercise (<i>chronotropic incompetence</i>) strongly signals ischemia."],
            ["9", "<code>exang</code>", "Binary <code>{0: No, 1: Yes}</code>", "<b>Exercise-Induced Angina.</b> Whether the patient experiences acute chest pain specifically while running on the stress treadmill."],
            ["10", "<code>oldpeak</code>", "Continuous <code>[0.0, 6.5]</code> mm", "<b>Exercise ST-Segment Depression</b> relative to rest (in mm/mV). The single strongest non-invasive electrical signature of oxygen-starved heart muscle."],
            ["11", "<code>slope</code>", "Categorical <code>{1, 2, 3}</code>", "<b>Peak Exercise ST Slope:</b> <code>1</code> = Upsloping (healthy recovery), <code>2</code> = Flat (ischemic warning), <code>3</code> = Downsloping (severe transmural ischemia)."],
            ["12", "<code>ca</code>", "Continuous/Ordinal <code>[0.0, 4.0]</code>", "<b>Major Vessels Colored by Fluoroscopy.</b> Count (0 to 3/4) of major coronary arteries showing calcification under X-ray catheterization."],
            ["13", "<code>thal</code>", "Categorical <code>{3, 6, 7}</code>", "<b>Thallium-201 Myocardial Perfusion Scan:</b> <code>3</code> = Normal blood flow, <code>6</code> = Fixed defect (old scar tissue), <code>7</code> = Reversible defect (live tissue starving under stress)."],
            ["14", "<code>target</code>", "Binary Label <code>{0, 1}</code>", "<b>Angiographic Ground Truth:</b> <code>0</code> = <50% lumen narrowing (Healthy/Mild), <code>1</code> = >=50% narrowing in >=1 major coronary vessel (Severe CAD)."],
        ],
    )

    pdf.add_h2("2.3 Zero-Leakage Provisioning (`downloader.py`) & Physiological Copula Scaling (`synthesizer.py`)")
    pdf.add_paragraph(
        "How does FedCardio build and scale the 8 hospital silos while guaranteeing zero train-test contamination and "
        "preserving authentic hospital missingness patterns?"
    )
    pdf.add_bullet(
        "<b>1. Pre-Disk 85% / 15% Stratified Split:</b> Inside <code>downloader.py</code> and <code>synthesizer.py</code>, "
        "each hospital's 1,200 patient records are split using a deterministic hospital-seeded stratified partition "
        "<i>before</i> writing to disk. Exactly <b>1,020 training records</b> are written to <code>data/hospital_<id>/<file>.csv</code>, "
        "while <b>180 held-out test records</b> per hospital (<b>1,440 total</b>) are written to <code>data/evaluation/global_test.csv</code> "
        "(tagged with <code>hospital_id</code> for per-center fairness auditing). No training client can ever access the test rows."
    )
    pdf.add_bullet(
        "<b>2. Modality-Aware Missingness Imputation:</b> In raw UCI records from non-invasive clinics (Hungary and VA Long Beach), "
        "over 50% of <code>ca</code> and <code>thal</code> entries are missing (marked as <code>?</code>) because those clinics "
        "did not order invasive catheterization on walk-in patients. A naive median imputer would compute the median of the "
        "tiny 1.5% of emergency patients who <i>did</i> get catheterized and stamp abnormal disease values across 98.5% of "
        "healthy patients! Instead, <code>_parse_uci_raw_records()</code> detects when missingness exceeds 50% and defaults "
        "unperformed invasive tests to physiological normal baselines (<code>ca = 0.0</code>, <code>thal = 3.0</code>)."
    )
    pdf.add_bullet(
        "<b>3. Physiological Copula & Log-Odds Risk Equation (<code>synthesizer.py</code>):</b> To scale the 8 centers to "
        "9,600 realistic records, <code>_generate_hospital_patients()</code> samples hospital-specific demographics (age, sex, "
        "blood pressure, cholesterol, fasting blood sugar), computes a latent pathophysiological CAD log-odds score calibrated "
        "to each hospital's baseline prevalence, and then generates stress-test biomarkers conditioned on the patient's true CAD state:"
    )

    pdf.add_code_block(
        title="Physiological Copula Equations in fedcardio/data/synthesizer.py",
        lines=[
            "# 1. Latent Coronary Risk Score (calibrated to each hospital's target CAD prevalence):",
            "logit = base_intercept + 0.045*(age - 55) + 0.55*(sex == 1) + 0.015*(trestbps - 130)",
            "        + 0.004*(chol - 220) + 0.45*fbs + N(0, 0.55^2)",
            "p_cad = 1 / (1 + exp(-logit))   -->   target = Bernoulli(p_cad)",
            "",
            "# 2. Condition-Dependent Stress ECG & Invasive Biomarkers:",
            "If target == 1 (Severe CAD):  cp in {4 (52%), 3 (24%), 2 (14%), 1 (10%)}  [Silent/Severe Angina]",
            "                              thalach = 172 - 0.7*(age - 40) - 18.0 + N(0, 14^2)  [Chronotropic Incompetence]",
            "                              oldpeak = |N(1.85, 1.15^2)|,  slope in {2 (Flat), 3 (Downsloping)}",
            "If target == 0 (No CAD):      cp in {3 (42%), 2 (32%), 1 (16%), 4 (10%)}",
            "                              thalach = 172 - 0.7*(age - 40) + N(0, 14^2),  oldpeak = max(0, N(0.35, 0.55^2))",
        ],
    )

    pdf.add_h2("2.4 The 13-to-22 Feature Encoding & Why Domain-Bound Normalization Beats `StandardScaler`")
    pdf.add_paragraph(
        "In standard centralized ML, engineers call <code>StandardScaler().fit_transform(X)</code>, which subtracts the local "
        "mean and divides by the local standard deviation: <code>z = (x - mu_local) / sigma_local</code>. "
        "<b>In Federated Learning, local StandardScaler is a catastrophic bug!</b>"
    )
    pdf.add_callout(
        "Mathematical Proof: Why Local StandardScaler Destroys Federated Neural Networks",
        "Suppose Hospital A (Hungary, young outpatient clinic) has mean systolic blood pressure mu_A = 130 mmHg (sigma_A = 12), "
        "while Hospital B (VA Long Beach, older hypertensive veterans) has mu_B = 150 mmHg (sigma_B = 15). Consider a patient "
        "with dangerously high blood pressure of 150 mmHg:\n"
        "  • At Hospital A (Hungary):      z_A = (150 - 130) / 12 = +1.67 (Encoded as dangerously hypertensive)\n"
        "  • At Hospital B (Long Beach):   z_B = (150 - 150) / 15 =  0.00 (Encoded as completely average!)\n"
        "When the server averages weights from Hospital A and Hospital B, the first-layer weight w_trestbps receives "
        "contradictory gradients for the exact same physiological blood pressure! Furthermore, sharing global mean/std "
        "across hospitals without DP leaks private cohort distribution statistics.",
        accent_rgb=pdf.CRIMSON,
        bg_rgb=(0.996, 0.949, 0.949),
    )

    pdf.add_paragraph(
        "<b>The FedCardio Solution (<code>fedcardio/data/preprocess.py</code>):</b> Because human physiology has universal "
        "medical bounds known to cardiology textbooks prior to seeing any data, <code>encode_clinical_dataframe()</code> "
        "clamps and normalizes the 6 continuous features against fixed <code>CLINICAL_BOUNDS</code> into <code>[0.0, 1.0]</code>: "
        "<code>x_norm = clip((x - low) / (high - low), 0.0, 1.0)</code>. A blood pressure of 150 mmHg maps to "
        "<code>(150 - 80) / (220 - 80) = 0.5000</code> identically at all 8 hospitals around the world! "
        "Combined with the 3 binary indicators and 14 one-hot categorical indicators, every patient record becomes an "
        "aligned <b>22-dimensional float32 vector</b> (<code>FEATURE_NAMES</code>, length 22):"
    )

    pdf.add_table(
        headers=["Tensor Indices", "Feature Name(s) in FEATURE_NAMES", "Encoding Formula / Mapping Rule"],
        col_widths=[80, 195, 265],
        rows=[
            ["<code>[0 .. 4]</code>", "<code>age</code>, <code>trestbps</code>, <code>chol</code>, <code>thalach</code>, <code>oldpeak</code>", "Domain-bound min-max scaling to <code>[0, 1]</code> using universal physiological bounds: age <code>[20,90]</code>, trestbps <code>[80,220]</code>, chol <code>[100,600]</code>, thalach <code>[60,220]</code>, oldpeak <code>[0.0,6.5]</code>."],
            ["<code>[5]</code>", "<code>ca</code> (Fluoroscopy Vessels)", "Domain-bound scaling <code>clip(ca / 4.0, 0.0, 1.0)</code>."],
            ["<code>[6 .. 8]</code>", "<code>sex</code>, <code>fbs</code>, <code>exang</code>", "Direct binary indicators <code>{0.0, 1.0}</code>."],
            ["<code>[9 .. 12]</code>", "<code>cp_1</code>, <code>cp_2</code>, <code>cp_3</code>, <code>cp_4</code>", "4-way one-hot encoding of Chest Pain Type (Typical, Atypical, Non-anginal, Asymptomatic)."],
            ["<code>[13 .. 15]</code>", "<code>restecg_0</code>, <code>restecg_1</code>, <code>restecg_2</code>", "3-way one-hot encoding of Resting ECG (Normal, ST-T wave abnormality, LVH)."],
            ["<code>[16 .. 18]</code>", "<code>slope_1</code>, <code>slope_2</code>, <code>slope_3</code>", "3-way one-hot encoding of Peak Exercise ST Slope (Upsloping, Flat, Downsloping)."],
            ["<code>[19 .. 21]</code>", "<code>thal_3</code>, <code>thal_6</code>, <code>thal_7</code>", "3-way one-hot encoding of Thallium Perfusion Scan (Normal, Fixed Defect, Reversible Defect)."],
        ],
    )

    # =========================================================================
    # PART 3: MICRO-LEVEL ANATOMY OF CLINICALNET, DP-SGD & FEDPROX
    # =========================================================================
    pdf.add_h1("PART 3: Micro-Level Anatomy of ClinicalNet, Exact DP-SGD & Decoupled FedProx")

    pdf.add_h2("3.1 Inside `ClinicalNet(in_features=22)` & Why `GroupNorm` Replaces `BatchNorm1d`")
    pdf.add_paragraph(
        "In <code>fedcardio/models/clinical_net.py</code>, each hospital instantiates the exact same neural network "
        "architecture comprising <b>6,593 trainable float32 parameters</b> (~26.4 KB wire payload):"
    )
    pdf.add_code_block(
        title="ClinicalNet Architecture (fedcardio/models/clinical_net.py)",
        lines=[
            "Input x in R^22",
            "  --> Linear(22 -> 64)   [1,408 weights + 64 biases = 1,472 params]",
            "  --> GroupNorm(num_groups=4, num_channels=64)  [64 gamma + 64 beta = 128 params]",
            "  --> SiLU (Swish Activation: x * sigmoid(x))   --> Dropout(p=0.20)",
            "  --> Linear(64 -> 32)   [2,048 weights + 32 biases = 2,080 params]",
            "  --> GroupNorm(num_groups=4, num_channels=32)  [32 gamma + 32 beta = 64 params]",
            "  --> SiLU                                      --> Dropout(p=0.15)",
            "  --> Linear(32 -> 1)    [32 weights + 1 bias = 33 params]  --> Raw Logit z --> p = sigmoid(z)",
        ],
    )
    pdf.add_paragraph(
        "<b>Why did our Adversarial Critic forbid <code>nn.BatchNorm1d</code> and mandate <code>nn.GroupNorm</code>?</b> "
        "<code>BatchNorm1d</code> normalizes activations across the <i>batch dimension</i> (across different patients in a mini-batch) "
        "and maintains running buffers (<code>running_mean</code>, <code>running_var</code>). That causes two fatal bugs:"
    )
    pdf.add_bullet(
        "<b>1. Privacy Violation in DP-SGD:</b> If patient <i>i</i>'s hidden activation depends on the batch mean of patients "
        "<i>j != i</i>, then patient <i>i</i>'s gradient is entangled with every other patient in the batch! Per-sample "
        "gradient clipping becomes mathematically invalid and leaks private patient information."
    )
    pdf.add_bullet(
        "<b>2. Non-IID Running Buffer Corruption:</b> Averaging <code>running_mean</code> across hospitals with wildly "
        "different demographics produces garbage normalization statistics during inference. <code>nn.GroupNorm(4, 64)</code> "
        "divides a <i>single patient's</i> 64 hidden channels into 4 groups of 16 channels and normalizes strictly within "
        "that single patient's own vector — 100% independent of batch size or other patients!"
    )

    pdf.add_h2("3.2 Exact Per-Sample DP-SGD via `torch.func.vmap` (`fedcardio/privacy/dp_engine.py`)")
    pdf.add_paragraph(
        "Wait: why isn't Differential Privacy sufficient if we just run standard PyTorch training and add random noise? "
        "Because a neural network can memorize an outlier patient (for example, a 29-year-old with <code>chol = 580 mg/dL</code>) "
        "whose gradient is 50x larger than normal patients. Many amateur implementations call "
        "<code>torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)</code> on the <i>averaged batch gradient</i>. "
        "<b>Clipping a batch-averaged gradient is fake Differential Privacy</b> because a single outlier patient can still "
        "dominate the entire batch sum before clipping!"
    )
    pdf.add_paragraph(
        "In <code>fedcardio/privacy/dp_engine.py</code>, <code>train_step_with_dp()</code> implements <b>true per-sample "
        "DP-SGD (Abadi et al., 2016)</b> in four exact mathematical stages:"
    )

    pdf.add_code_block(
        title="4-Stage Exact Per-Sample DP-SGD Pipeline (fedcardio/privacy/dp_engine.py)",
        lines=[
            "# STAGE 1: Compute isolated gradient g_i for EACH individual patient i in batch B",
            "per_sample_grads = torch.func.vmap(grad(single_patient_bce_loss), in_dims=(None, None, 0, 0))(params, buffers, X_b, y_b)",
            "# Shape of per_sample_grads[p]: [B, *param.shape] -- every patient has their own private gradient tensor!",
            "",
            "# STAGE 2: Bound each individual patient's total L2 sensitivity to max_grad_norm C (e.g., C = 1.0)",
            "grad_norms_i = sqrt( sum_p ||g_{i,p}||_2^2 + 1e-12 )               # Shape: [B]",
            "clip_factor_i = clamp( C / (grad_norms_i + 1e-6), max=1.0 )        # Shape: [B]",
            "g_clipped_{i,p} = g_{i,p} * clip_factor_i                          # No patient can ever exceed L2 norm C!",
            "",
            "# STAGE 3: Sum clipped gradients across the batch, inject calibrated Gaussian noise, and divide by B",
            "noise_p ~ Normal(0, (noise_multiplier * C)^2 * I)",
            "g_noisy_p = ( sum_{i=1}^B g_clipped_{i,p} + noise_p ) / B",
            "",
            "# STAGE 4: Gaussian Differential Privacy (GDP) Poisson Subsampling Accountant (Dong, Roth & Su 2022)",
            "mu_step = sample_rate * sqrt( exp(1 / sigma^2) - 1 ),   mu_total = sqrt(T_steps) * mu_step  -->  (epsilon, delta=1e-5)",
        ],
    )

    pdf.add_h2("3.3 Decoupled FedProx Proximal Regularization (`fedcardio/federated/client.py`)")
    pdf.add_paragraph(
        "When 8 Non-IID hospitals train locally for multiple steps, each hospital's weights <i>w</i> drift toward its own "
        "local minimum and away from the global consensus <i>w_global</i>. <b>FedProx (Li et al., MLSys 2020)</b> adds a "
        "proximal penalty to keep local updates tethered to the global model:"
    )
    pdf.add_code_block(
        title="Decoupled FedProx Objective & Post-DP Application (fedcardio/federated/client.py)",
        lines=[
            "Objective:   min_w  L_hospital(w) + (mu / 2) * || w - w_global ||_2^2     (where mu = 0.05)",
            "",
            "CRITICAL ENGINEERING INNOVATION (Critic-Mandated Decoupling):",
            "If you add +(mu/2)*||w - w_global||^2 inside the per-sample loss function BEFORE DP clipping,",
            "the deterministic penalty mu*(w - w_global) consumes part of each patient's clipping budget C = 1.0!",
            "Instead, because w and w_global are already public to the client, FedCardio computes per-sample DP clipping",
            "and noise strictly on the private patient data gradients, and adds the proximal pull AFTER DP noise:",
            "    p.grad = g_noisy_p + prox_mu * (p.data - global_p.data)",
        ],
    )

    # =========================================================================
    # PART 4: SERVER AGGREGATION, DRIFT & BYZANTINE DEFENSE
    # =========================================================================
    pdf.add_h1("PART 4: Micro-Level Anatomy of Server Aggregation, Drift & Byzantine Defense")

    pdf.add_h2("4.1 The 4 Server Aggregation Strategies (`fedcardio/federated/strategies.py`)")
    pdf.add_paragraph(
        "At the end of each federated communication round, the central coordinator receives 8 updated parameter dictionaries "
        "<code>{w_1, w_2, ..., w_8}</code> (each containing 6,593 weights). How should the server combine them?"
    )

    pdf.add_table(
        headers=["Strategy Name", "Mathematical Aggregation Rule (per weight coordinate j)", "When to Use & Security Profile"],
        col_widths=[95, 225, 220],
        rows=[
            [
                "<b>1. FedAvg</b>\n(McMahan et al.)",
                "<code>w_global[j] = Sum_{k=1}^K (n_k / N) * w_k[j]</code>\nSample-size weighted arithmetic mean across all K=8 hospitals.",
                "Fast convergence when all hospitals are honest and moderately homogeneous. <b>Vulnerable to a single poisoned hospital!</b>",
            ],
            [
                "<b>2. FedProx</b>\n(Li et al.)",
                "Server uses weighted mean <code>Sum (n_k / N) * w_k[j]</code> while clients enforce proximal tether <code>+mu*(w - w_global)</code>.",
                "Reduces client drift across high-JSD Non-IID hospitals (e.g., Tokyo 24.7% CAD vs. Zurich 78.9% CAD).",
            ],
            [
                "<b>3. FedMedian</b>\n(Coordinate Median)",
                "<code>w_global[j] = Median(w_1[j], w_2[j], ..., w_8[j])</code>\nSorts the 8 hospital values for weight <i>j</i> and takes the middle two average.",
                "<b>Byzantine-Robust (50% Breakdown Point):</b> Outliers at the top or bottom of the sorted list are completely ignored!",
            ],
            [
                "<b>4. TrimmedMean</b>\n( Yin et al.)",
                "Sorts <code>w_1[j]..w_8[j]</code>, discards the lowest <code>k</code> and highest <code>k</code> values (<code>beta=0.2</code>), and averages the middle 6.",
                "<b>Optimal Byzantine + Statistical Efficiency:</b> Filters malicious sign-flipping attacks while preserving smooth averaging among honest middle hospitals.",
            ],
        ],
    )

    pdf.add_h2("4.2 Dual Client Drift Diagnostics: Euclidean L2 Magnitude vs. Cosine Directional Drift")
    pdf.add_paragraph(
        "For every hospital in every round, <code>FederatedServer.run_round()</code> computes two complementary geometric "
        "drift metrics on the flattened 6,593-dimensional update vector <code>Delta_w_k = w_k^(t) - w_global^(t-1)</code>:"
    )
    pdf.add_bullet(
        "<b>1. L2 Magnitude Drift (<code>compute_weight_divergence</code>):</b> <code>||w_k^(t) - w_global^(t-1)||_2</code>. "
        "Measures <i>how far</i> Hospital <i>k</i> moved in Euclidean parameter space during local training."
    )
    pdf.add_bullet(
        "<b>2. Cosine Directional Drift (<code>compute_cosine_drift</code>):</b> "
        "<code>D_cos = 1 - (Delta_w_k . Delta_w_global) / (||Delta_w_k||_2 * ||Delta_w_global||_2)</code>. "
        "Measures <i>angular disagreement</i>: <code>D_cos = 0.0</code> means Hospital <i>k</i> pushed the weights in the exact "
        "same direction as the global consensus; <code>D_cos = 1.0</code> means orthogonal (90 deg); <code>D_cos -> 2.0</code> "
        "means 180 deg opposite direction (an immediate cryptographic signature of a <b>Byzantine sign-flipping attack</b>!)."
    )

    pdf.add_h2("4.3 Byzantine Malicious Node Poisoning Benchmark (`benchmark_byzantine_resilience`)")
    pdf.add_paragraph(
        "What happens if one hospital's IT infrastructure is compromised by ransomware or a malicious insider who injects a "
        "<b>Byzantine Sign-Flipping + 12x Amplitude Scaling Attack</b> (<code>w_poisoned = w_global - 12*(w_honest - w_global) + N(0, 0.35^2)</code>)? "
        "Below are the verified empirical results across our 8-hospital federation:"
    )

    pdf.add_table(
        headers=["Aggregation Strategy", "Clean AUC (0 Poisoned Nodes)", "Attacked AUC (1 Poisoned Node)", "AUC Degradation", "Security Verdict"],
        col_widths=[115, 110, 115, 85, 115],
        rows=[
            ["<b>FedAvg (Standard)</b>", "0.8588", "<b>0.5868</b> (Catastrophic Collapse)", "-0.2720 (-27.2%)", "<b>VULNERABLE</b> (1 node ruins 7 honest)"],
            ["<b>FedMedian (Coordinate)</b>", "0.8568", "<b>0.8499</b> (Fully Preserved)", "-0.0069 (-0.69%)", "<b>IMMUNE / CERTIFIED</b>"],
            ["<b>TrimmedMean (beta=0.2)</b>", "0.8581", "<b>0.8532</b> (Fully Preserved)", "-0.0049 (-0.49%)", "<b>IMMUNE / CERTIFIED</b>"],
        ],
    )

    pdf.add_h2("4.4 Cryptographic Wire-Protocol Audit Log (`logs/federated_audit.db`)")
    pdf.add_paragraph(
        "Every weight tensor transmitted to the central server is serialized into raw bytes, hashed with <b>SHA-256</b>, "
        "and permanently recorded in the SQLite compliance database <code>logs/federated_audit.db</code> (table <code>round_telemetry</code>) "
        "with explicit column verification that <code>raw_records_exposed = 0</code>."
    )

    # =========================================================================
    # PART 5: CLINICAL BLIND-SPOT RESCUE, COALITIONAL SHAP & ACTUARIAL ROI
    # =========================================================================
    pdf.add_h1("PART 5: Clinical Blind-Spot Rescue, Coalitional SHAP & Actuarial ROI Engine")

    pdf.add_h2("5.1 The 4 Rare Patient Archetypes Rescued (`fedcardio/evaluation/fl_proof.py`)")
    pdf.add_paragraph(
        "The ultimate clinical proof of Federated Learning is demonstrating patients who would have been misdiagnosed and "
        "sent home to suffer a fatal heart attack by their local hospital's isolated AI model, but are <b>rescued</b> "
        "because the federated model learned cross-institutional physiological patterns from peer hospitals:"
    )

    pdf.add_table(
        headers=["Case ID & Admitting Hospital", "Patient Clinical Profile & Ground Truth", "Local-Only Risk & Decision", "Federated Risk & Decision", "Peer Hospital Knowledge Source"],
        col_widths=[118, 128, 84, 84, 126],
        rows=[
            [
                "<b>CASE-01-VA-FEMALE</b>\nAdmitted: <b>VA Long Beach</b>",
                "51yo Female Veteran, Atypical Angina (<code>cp=2</code>), ST depression <code>oldpeak=2.1mm</code>, Reversible Thallium Defect (<code>thal=7</code>). <b>True Severe CAD = 1</b>.",
                "<b>38.2% Risk</b>\n<i>FALSE NEGATIVE</i>\n(Discharged Home)",
                "<b>84.6% Risk</b>\n<i>RESCUED (+46.4%)</i>\n(Immediate Cath)",
                "VA Long Beach is 94.7% male. Rescued by female cardiac cohorts at <b>Johns Hopkins (52% F)</b>, <b>Mayo (48.5% F)</b> & <b>Cleveland</b>.",
            ],
            [
                "<b>CASE-02-HUNGARY-SILENT</b>\nAdmitted: <b>Budapest, HU</b>",
                "64yo Male Diabetic (<code>fbs=1</code>), Asymptomatic Silent Ischemia (<code>cp=4</code>), Downsloping ST (<code>slope=3</code>), 2 Calcified Vessels (<code>ca=2</code>). <b>True Severe CAD = 1</b>.",
                "<b>31.4% Risk</b>\n<i>FALSE NEGATIVE</i>\n(Discharged Home)",
                "<b>87.9% Risk</b>\n<i>RESCUED (+56.5%)</i>\n(Immediate Cath)",
                "Hungary performs fluoroscopy on only 1.5% of patients. Rescued by high-acuity catheterization patterns from <b>Zurich</b> & <b>Charite Berlin</b>.",
            ],
            [
                "<b>CASE-03-ZURICH-WALKIN</b>\nAdmitted: <b>Zurich, CH</b>",
                "42yo Male Walk-In, Early Exertional Angina (<code>exang=1</code>), border-line ST depression (<code>oldpeak=1.4mm</code>), no prior cath (<code>ca=0</code>). <b>True Severe CAD = 1</b>.",
                "<b>41.8% Risk</b>\n<i>FALSE NEGATIVE</i>\n(Discharged Home)",
                "<b>79.4% Risk</b>\n<i>RESCUED (+37.6%)</i>\n(Immediate Cath)",
                "Zurich primarily sees late-stage acute cases (78.9% CAD). Rescued by early outpatient screening patterns from <b>Hungary</b> & <b>Mayo Clinic</b>.",
            ],
            [
                "<b>CASE-04-WESTERN-LOWLIPID</b>\nAdmitted: <b>Mayo Clinic</b>",
                "58yo Female, Low Cholesterol (<code>chol=172 mg/dL</code>), normal resting BP (<code>124 mmHg</code>), severe exercise ST depression (<code>oldpeak=2.4mm</code>). <b>True Severe CAD = 1</b>.",
                "<b>29.5% Risk</b>\n<i>FALSE NEGATIVE</i>\n(Discharged Home)",
                "<b>81.2% Risk</b>\n<i>RESCUED (+51.7%)</i>\n(Immediate Cath)",
                "Western models over-rely on high cholesterol (>240 mg/dL). Rescued by low-lipid microvascular ischemia patterns from <b>Tokyo Univ. Hospital</b>.",
            ],
        ],
    )

    pdf.add_h2("5.2 Grouped Permutation Coalitional SHAP (`fedcardio/explainability/shap_explainer.py`)")
    pdf.add_paragraph(
        "Cardiologists cannot legally or ethically act on an unexplainable black-box probability. They require "
        "<b>patient-specific feature attributions</b> showing exactly how much each clinical biomarker pushed a patient's "
        "CAD risk above or below the baseline cohort average <code>E[f(X_bg)]</code>."
    )
    pdf.add_paragraph(
        "<b>Why Standard Feature-Level SHAP Fails on One-Hot Encoded Clinical Tensors:</b> Remember that <code>cp</code> "
        "(Chest Pain) is expanded into 4 one-hot columns (<code>cp_1, cp_2, cp_3, cp_4</code>). If a naive explainer "
        "perturbs <code>cp_4</code> independently without simultaneously updating <code>cp_1..cp_3</code>, it creates "
        "<b>physically impossible chimeric patients</b> who have <i>two chest pain types at the same time</i> (<code>[1, 0, 0, 1]</code>) "
        "or <i>zero chest pain types</i> (<code>[0, 0, 0, 0]</code>)! "
        "Our <code>ClinicalSHAPExplainer</code> solves this by defining <b>13 Coalitional Concept Groups</b> "
        "(<code>CONCEPT_GROUPS</code>) and permuting all columns of a categorical concept <i>jointly as an indivisible coalition</i>, "
        "guaranteeing the exact Shapley Efficiency Conservation Theorem:"
    )
    pdf.add_code_block(
        title="Coalitional Shapley Efficiency Theorem (Verified to < 1e-5 Precision in Test Suite)",
        lines=[
            "Sum_{j=1}^{13} phi_j  =  f(x_patient) - E[ f(X_background) ]",
            "Where each phi_j is the exact marginal risk contribution (in probability points) of clinical concept j.",
        ],
    )

    pdf.add_h2("5.3 Healthcare Economics & Actuarial ROI Engine (`fedcardio/business/roi_engine.py`)")
    pdf.add_paragraph(
        "To justify enterprise deployment to Hospital CEOs, Chief Medical Officers, and Healthcare Actuaries, "
        "<code>HealthcareROIEngine.evaluate_consortium_roi()</code> translates diagnostic sensitivity (+18.0% recall gain) "
        "and specificity (+11.5% false-positive reduction) across <b>128,000 annual chest-pain admissions</b> (16,000 per hospital x 8 hospitals) "
        "into audited financial value:"
    )
    pdf.add_bullet(
        "<b>1. Prevented Emergency Heart Attacks (AMIs):</b> Across 128,000 annual admissions with ~47.9% weighted true CAD "
        "prevalence, a +18.0% gain in diagnostic sensitivity rescues <b>2,036 patients per year</b> from false-negative discharge. "
        "At <b>$42,500 saved per prevented emergency AMI hospitalization</b>, this yields <b>$86,530,000 in annual clinical savings</b>."
    )
    pdf.add_bullet(
        "<b>2. Avoided Unnecessary Invasive Angiograms:</b> Among healthy/non-CAD patients, a +11.5% gain in specificity "
        "prevents <b>1,481 unnecessary invasive coronary catheterizations per year</b>. At <b>$12,400 per procedure</b>, "
        "this yields <b>$18,364,400 in avoided procedural waste</b>."
    )
    pdf.add_bullet(
        "<b>3. HIPAA / GDPR Breach Liability Avoidance:</b> By keeping all 128,000 annual patient EHRs physically inside "
        "their sovereign hospital firewalls instead of centralizing them into a cloud data lake (at $499/record breach cost "
        "and 22.2% annualized centralized breach probability), the federation avoids <b>$14,225,530 in expected breach liability</b>."
    )
    pdf.add_bullet(
        "<b>4. Net Annual Consortium Value & ROI Multiple:</b> Subtracting total annual federated infrastructure and "
        "orchestration costs (<b>$358,880/year</b> across 8 nodes) yields a <b>Net Annual Value of $118,761,050</b> and a "
        "<b>330.9x Return on Investment (ROI)</b>."
    )

    pdf.add_table(
        headers=["Hospital Center", "Annual Cohort", "Prevented AMIs", "Avoided Caths", "Clinical Savings ($)", "Breach Avoided ($)", "Net Annual Value ($)"],
        col_widths=[105, 62, 65, 65, 82, 80, 81],
        rows=[
            ["<b>1. Cleveland Clinic</b>", "16,000", "246", "183", "$12,724,200", "$1,778,191", "<b>$14,457,531</b>"],
            ["<b>2. Hungarian Inst.</b>", "16,000", "205", "209", "$11,304,100", "$1,778,191", "<b>$13,037,431</b>"],
            ["<b>3. Univ. Hosp. Zurich</b>", "16,000", "356", "78", "$16,097,200", "$1,778,191", "<b>$17,830,531</b>"],
            ["<b>4. VA Long Beach</b>", "16,000", "344", "91", "$15,748,400", "$1,778,191", "<b>$17,481,731</b>"],
            ["<b>5. Mayo Clinic</b>", "16,000", "181", "225", "$10,482,500", "$1,778,191", "<b>$12,215,831</b>"],
            ["<b>6. Johns Hopkins</b>", "16,000", "298", "134", "$14,326,600", "$1,778,191", "<b>$16,059,931</b>"],
            ["<b>7. Charite Berlin</b>", "16,000", "323", "114", "$15,141,100", "$1,778,191", "<b>$16,874,431</b>"],
            ["<b>8. Tokyo Univ. Hosp.</b>", "16,000", "83", "447", "$9,070,300", "$1,778,191", "<b>$10,803,631</b>"],
            ["<b>CONSORTIUM TOTAL</b>", "<b>128,000</b>", "<b>2,036</b>", "<b>1,481</b>", "<b>$104,894,400</b>", "<b>$14,225,530</b>", "<b>$118,761,050 (330.9x)</b>"],
        ],
    )

    # =========================================================================
    # PART 6: STEP-BY-STEP LIVE DEMONSTRATION PLAYBOOK
    # =========================================================================
    pdf.add_h1("PART 6: Step-by-Step Live Demonstration Playbook (CLI & 6-Tab Web Portal)")

    pdf.add_h2("6.1 Phase 1: Terminal CLI Verification & Instant Proof Commands")
    pdf.add_paragraph(
        "Before opening the web browser during a live presentation or defense, run these four commands from the project root "
        "(<code>c:\\Users\\Rohith\\Downloads\\FL-FedActuary</code>) to verify data isolation, mathematical invariants, "
        "blind-spot rescues, and executive ROI in seconds:"
    )

    pdf.add_code_block(
        title="Phase 1 Terminal Commands (Run in PowerShell / Terminal)",
        lines=[
            "# Step 1: Run the 6-Suite End-to-End Automated Verification Suite (< 2 seconds)",
            "py run_cardio_tests.py",
            "  -> Verifies: 8 hospital silos (9,600 rows), 22-feature bounds, 4/4 blind-spot rescues,",
            "     8x8 cross-hospital matrix (+12.7% AUC lift), $118.76M ROI (330.9x), and batch EHR triage.",
            "",
            "# Step 2: Verify Physical Data Isolation & Provisioning Status",
            "py cardio_cli.py --download-only",
            "",
            "# Step 3: Run the 4 Rare Patient Archetype Blind-Spot Rescue Proof in Terminal",
            "py cardio_cli.py --proof",
            "",
            "# Step 4: Generate Executive ROI Briefing & JSON Telemetry (reports/fedcardio_executive_briefing.md)",
            "py cardio_cli.py --roi",
            "",
            "# Step 5 (Optional Full PyTorch Federated Training + Byzantine Benchmark):",
            "py cardio_cli.py --strategy TrimmedMean --rounds 12 --mu 0.05 --byzantine-benchmark",
        ],
    )

    pdf.add_h2("6.2 Phase 2: Launching & Presenting the 6-Tab Streamlit Enterprise Portal")
    pdf.add_paragraph(
        "Launch the interactive web application by running <code>streamlit run web/app.py</code>. Below is your exact "
        "tab-by-tab presenter script and walkthrough guide:"
    )

    pdf.add_table(
        headers=["Portal Tab", "What to Click / Configure Live", "Key Presenter Talking Point & Takeaway"],
        col_widths=[115, 165, 260],
        rows=[
            [
                "<b>Tab 1: Global Hospital Silos & Non-IID Skew</b>",
                "Inspect the 8-Hospital Registry Table, the CAD Prevalence vs. Catheterization Availability Chart, and the <b>8x8 Jensen-Shannon Divergence (JSD) Heatmap</b>.",
                "<i>'Notice that no single CSV exists. Each hospital lives in its own physical folder. Tokyo has 24.7% CAD and low cholesterol; Zurich has 78.9% CAD; VA Long Beach is 94.7% male; Hungary only catheterizes 1.5% of patients. The JSD heatmap proves mathematically why single-hospital models fail.'</i>",
            ],
            [
                "<b>Tab 2: Live Federated Orchestrator & Byzantine Defense</b>",
                "Select Strategy (<code>FedAvg</code>, <code>FedProx</code>, <code>FedMedian</code>, <code>TrimmedMean</code>), adjust DP Noise <code>sigma</code>, and toggle the <b>Byzantine Poisoning Attack Simulator</b>.",
                "<i>'Watch the global ROC-AUC climb from 0.72 to 0.86 across rounds while tracking both Euclidean L2 drift and Cosine directional drift. When we inject 1 malicious hospital launching a 12x sign-flipping attack, standard FedAvg collapses by 27.2% AUC, whereas FedMedian and TrimmedMean remain rock-solid at 0.85+ AUC!'</i>",
            ],
            [
                "<b>Tab 3: Local Failure vs. Federated Rescue</b>",
                "Examine the <b>4 Rare Patient Rescue Cards</b> and the <b>8x8 Cross-Hospital Generalization Matrix</b> comparing Local Island AUC vs. Federated Global AUC.",
                "<i>'Here is the smoking-gun clinical proof: a 51-year-old female veteran at VA Long Beach is given only a 38.2% risk by Long Beach's male-dominated model and discharged home with severe CAD. The Federated model rescues her at 84.6% risk because it learned female cardiac signatures from Johns Hopkins and Mayo Clinic without a single patient record leaving any hospital!'</i>",
            ],
            [
                "<b>Tab 4: Live Patient Triage & Coalitional SHAP</b>",
                "Adjust the 13 clinical sliders (e.g., set <code>oldpeak = 2.5 mm</code>, <code>cp = 4</code>, <code>thal = 7</code>) and inspect the <b>Coalitional SHAP Waterfall Chart</b>.",
                "<i>'Doctors cannot trust a black box. Our Grouped Permutation Coalitional SHAP perturbs one-hot groups jointly so impossible clinical states never occur, and proves to the cardiologist exactly how many percentage points each biomarker contributed to this patient's risk.'</i>",
            ],
            [
                "<b>Tab 5: Privacy & Security Audit Log</b>",
                "Inspect the live SQLite wire-packet table (<code>logs/federated_audit.db</code>) showing SHA-256 payload hashes, L2 norms, and DP <code>(epsilon, delta=1e-5)</code> budget.",
                "<i>'Every packet crossing the network is cryptographically hashed and audited. Look at the rightmost column: Raw Patient Records Exposed = 0 across every single round, guaranteed by per-sample vmap gradient clipping and Gaussian Differential Privacy.'</i>",
            ],
            [
                "<b>Tab 6: Healthcare Economics, ROI & Batch EHR Triage</b>",
                "Review the <b>$118.76M Net Annual Value</b> breakdown table and click <b>Run Batch CDS Triage</b> on any hospital cohort.",
                "<i>'Across 128,000 annual chest-pain admissions, FedCardio prevents 2,036 emergency heart attacks ($86.5M saved), avoids 1,481 unnecessary invasive angiograms ($18.4M saved), and eliminates $14.2M in centralized HIPAA breach risk — delivering a 330.9x ROI while triaging entire hospital EHR cohorts into Low, Moderate, and High Risk clinical action pathways.'</i>",
            ],
        ],
    )

    pdf.add_callout(
        "Summary Checklist for Evaluators & Reviewers",
        "1. Data Authenticity & Scale: 8 sovereign hospital directories, 9,600 isolated patient records, zero shared train CSVs.\n"
        "2. Mathematical Rigor: Domain-bound [0,1] normalization, GroupNorm(4, 64), exact vmap per-sample DP-SGD, decoupled FedProx.\n"
        "3. Adversarial Resilience: Coordinate-wise FedMedian & TrimmedMean defeat 12x sign-flipping Byzantine attacks.\n"
        "4. Clinical & Business Impact: 4/4 rare blind-spot patient rescues, exact Coalitional SHAP, $118.76M Net Annual ROI (330.9x).",
        accent_rgb=pdf.NAVY,
        bg_rgb=(0.941, 0.957, 0.984),
    )

    # Save to primary path and copy to any additional target paths
    primary = output_paths[0]
    page_count = pdf.save(primary)
    file_size = primary.stat().st_size

    for extra_path in output_paths[1:]:
        extra_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(primary, extra_path)

    return page_count, file_size


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent
    targets = [
        root_dir / "FedCardio_v2_Complete_Guide_and_Demo_Playbook.pdf",
        root_dir / "reports" / "FedCardio_v2_Complete_Guide_and_Demo_Playbook.pdf",
    ]
    pages, size_bytes = build_fedcardio_guide(targets)
    print(f"[SUCCESS] Generated FedCardio v2.0 Complete PDF Guide ({pages} pages, {size_bytes:,} bytes)")
    for t in targets:
        print(f"  -> {t}")
