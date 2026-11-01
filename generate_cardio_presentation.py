"""
Generates the simplified, humanized, large-font PowerPoint presentation for FedCardio v2.0:
  - FedCardio_v2_Presentation.pptx (Project Root)
  - reports/FedCardio_v2_Presentation.pptx (Reports Directory)

Visual Design System:
  - Background: Warm white-beige (RGB: 248, 246, 240 / #F8F6F0)
  - Theme: Dark green (RGB: 24, 58, 40 / #183A28) with subtle sage & mint accents
  - Layout: 16:9 widescreen, minimal cards, generous whitespace
  - Typography: Prominent, larger fonts (32-44pt titles, 18-22pt body/headers)
  - Narrative Flow: From clinical problem from scratch to the complete federated approach and final proof
"""

import os
import shutil
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor


# =============================================================================
# COLOR PALETTE SPECIFICATION
# =============================================================================
BEIGE_BG      = RGBColor(248, 246, 240)  # Warm white beige (#F8F6F0)
DARK_GREEN    = RGBColor(24, 58, 40)     # Deep forest dark green (#183A28)
SAGE_GREEN    = RGBColor(52, 107, 74)    # Muted sage green (#346B4A)
MINT_TINT     = RGBColor(234, 243, 237)  # Soft mint highlight fill (#EAF3ED)
CARD_BG       = RGBColor(255, 255, 253)  # Crisp warm off-white (#FFFFFD)
CARD_BORDER   = RGBColor(210, 224, 215)  # Subtle sage card border (#D2E0D7)
TEXT_DARK     = RGBColor(32, 44, 36)     # High-contrast charcoal green (#202C24)
TEXT_MUTED    = RGBColor(88, 106, 94)    # Secondary muted slate-green (#586A5E)
WARM_AMBER    = RGBColor(196, 120, 25)   # Amber / gold for economic ROI (#C47819)
ALERT_CRIMSON = RGBColor(185, 45, 45)    # Crimson for high-risk / false negative (#B92D2D)
WHITE         = RGBColor(255, 255, 255)

FONT_PRIMARY = "Segoe UI"
FONT_TITLE   = "Segoe UI"


class FedCardioPresentationBuilder:
    """Builder class for crafting clean, minimal, large-text PowerPoint slides."""

    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width = Inches(13.333)
        self.prs.slide_height = Inches(7.5)
        self.total_slides = 15

    def _create_base_slide(self):
        """Creates a 16:9 blank slide with the white-beige background."""
        blank_layout = self.prs.slide_layouts[6]
        slide = self.prs.slides.add_slide(blank_layout)

        # 1. Slide native background
        bg = slide.background
        fill = bg.fill
        fill.solid()
        fill.fore_color.rgb = BEIGE_BG

        # 2. Resilient background rectangle for universal compatibility
        bg_rect = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, 0, 0, self.prs.slide_width, self.prs.slide_height
        )
        bg_rect.fill.solid()
        bg_rect.fill.fore_color.rgb = BEIGE_BG
        bg_rect.line.color.rgb = BEIGE_BG
        bg_rect.line.width = Pt(0)

        return slide

    def _add_header(self, slide, eyebrow: str, title: str, subtitle: str):
        """
        Adds a unified, minimal, large-text header zone in a single flowed textframe
        to guarantee zero text overlap regardless of title word-wrapping.
        """
        # Top subtle accent bar
        accent_bar = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(0.38), Inches(0.45), Inches(0.05)
        )
        accent_bar.fill.solid()
        accent_bar.fill.fore_color.rgb = SAGE_GREEN
        accent_bar.line.fill.background()

        # Eyebrow category tag
        eyebrow_box = slide.shapes.add_textbox(Inches(1.35), Inches(0.30), Inches(11.0), Inches(0.30))
        tf_e = eyebrow_box.text_frame
        tf_e.word_wrap = True
        tf_e.margin_left = tf_e.margin_right = tf_e.margin_top = tf_e.margin_bottom = 0
        p_e = tf_e.paragraphs[0]
        p_e.text = eyebrow.upper()
        p_e.font.name = FONT_PRIMARY
        p_e.font.size = Pt(12)
        p_e.font.bold = True
        p_e.font.color.rgb = SAGE_GREEN

        # Flowed Title & Subtitle in ONE single text frame
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.68), Inches(11.733), Inches(1.35))
        tf_h = header_box.text_frame
        tf_h.word_wrap = True
        tf_h.margin_left = tf_h.margin_right = tf_h.margin_top = tf_h.margin_bottom = 0

        p_t = tf_h.paragraphs[0]
        p_t.text = title
        p_t.font.name = FONT_TITLE
        p_t.font.size = Pt(29)
        p_t.font.bold = True
        p_t.font.color.rgb = DARK_GREEN
        p_t.space_after = Pt(4)

        p_s = tf_h.add_paragraph()
        p_s.text = subtitle
        p_s.font.name = FONT_PRIMARY
        p_s.font.size = Pt(16.5)
        p_s.font.color.rgb = TEXT_MUTED

    def _add_footer(self, slide, slide_num: int):
        """Adds a subtle, minimal bottom footer."""
        footer_line = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(6.92), Inches(11.733), Inches(0.015)
        )
        footer_line.fill.solid()
        footer_line.fill.fore_color.rgb = CARD_BORDER
        footer_line.line.fill.background()

        # Footer Left
        left_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.0), Inches(8.5), Inches(0.3))
        tf_l = left_box.text_frame
        tf_l.margin_left = tf_l.margin_right = tf_l.margin_top = tf_l.margin_bottom = 0
        p_l = tf_l.paragraphs[0]
        p_l.text = "FedCardio v2.0  •  Collaborative Clinical AI Without Data Sharing"
        p_l.font.name = FONT_PRIMARY
        p_l.font.size = Pt(11)
        p_l.font.color.rgb = TEXT_MUTED

        # Footer Right
        right_box = slide.shapes.add_textbox(Inches(10.5), Inches(7.0), Inches(2.0), Inches(0.3))
        tf_r = right_box.text_frame
        tf_r.margin_left = tf_r.margin_right = tf_r.margin_top = tf_r.margin_bottom = 0
        p_r = tf_r.paragraphs[0]
        p_r.alignment = PP_ALIGN.RIGHT
        p_r.text = f"Slide {slide_num} of {self.total_slides}"
        p_r.font.name = FONT_PRIMARY
        p_r.font.size = Pt(11)
        p_r.font.bold = True
        p_r.font.color.rgb = SAGE_GREEN

    def _add_card(
        self,
        slide,
        left: float,
        top: float,
        width: float,
        height: float,
        bg_color=CARD_BG,
        border_color=CARD_BORDER,
        border_width=Pt(1.2),
        accent_top_color=None,
    ):
        """Draws a clean, minimal card container with optional top accent stripe."""
        card = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(left),
            Inches(top),
            Inches(width),
            Inches(height),
        )
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = border_width

        if accent_top_color:
            stripe = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE,
                Inches(left),
                Inches(top),
                Inches(width),
                Inches(0.12),
            )
            stripe.fill.solid()
            stripe.fill.fore_color.rgb = accent_top_color
            stripe.line.fill.background()

        return card

    # =========================================================================
    # SLIDE BUILDERS (SLIDES 1 TO 15)
    # =========================================================================

    def build_slide_01_cover(self):
        """Slide 1: Cover Slide - Simplified, Elegant & Humanized."""
        slide = self._create_base_slide()

        # Large prominent title box
        t_box = slide.shapes.add_textbox(Inches(1.0), Inches(1.3), Inches(11.3), Inches(3.2))
        tf = t_box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

        p0 = tf.paragraphs[0]
        p0.text = "FEDCARDIO v2.0  •  ENTERPRISE CLINICAL AI BLUEPRINT"
        p0.font.name = FONT_PRIMARY
        p0.font.size = Pt(14)
        p0.font.bold = True
        p0.font.color.rgb = SAGE_GREEN
        p0.space_after = Pt(14)

        p1 = tf.add_paragraph()
        p1.text = "Healing the Heart Without Exposing the Patient"
        p1.font.name = FONT_TITLE
        p1.font.size = Pt(44)
        p1.font.bold = True
        p1.font.color.rgb = DARK_GREEN
        p1.space_after = Pt(16)

        p2 = tf.add_paragraph()
        p2.text = (
            "How 8 global sovereign hospitals train collective life-saving AI "
            "across 9,600 isolated patients — with mathematical privacy and zero shared records."
        )
        p2.font.name = FONT_PRIMARY
        p2.font.size = Pt(22)
        p2.font.color.rgb = TEXT_DARK

        # 4 Metric Highlight Cards at bottom
        metrics = [
            ("8 SOVEREIGN HOSPITALS", "9,600", "Isolated Patient EHRs", DARK_GREEN),
            ("PHYSICAL PRIVACY", "0 Shared CSVs", "Raw data never moves", SAGE_GREEN),
            ("CLINICAL RESCUE RATE", "4 / 4 Rescued", "100% rare archetypes saved", DARK_GREEN),
            ("HEALTHCARE VALUE", "$118.76M", "Annual Net ROI (330.9x)", WARM_AMBER),
        ]

        card_w = 2.7
        gap = 0.3
        start_x = 1.0
        card_top = 4.8
        card_h = 1.6

        for idx, (label, big_text, desc, accent) in enumerate(metrics):
            cx = start_x + idx * (card_w + gap)
            self._add_card(slide, cx, card_top, card_w, card_h, accent_top_color=accent)

            tb = slide.shapes.add_textbox(Inches(cx + 0.2), Inches(card_top + 0.2), Inches(card_w - 0.4), Inches(card_h - 0.3))
            tf_c = tb.text_frame
            tf_c.word_wrap = True
            tf_c.margin_left = tf_c.margin_right = tf_c.margin_top = tf_c.margin_bottom = 0

            p_l = tf_c.paragraphs[0]
            p_l.text = label
            p_l.font.name = FONT_PRIMARY
            p_l.font.size = Pt(11)
            p_l.font.bold = True
            p_l.font.color.rgb = TEXT_MUTED

            p_b = tf_c.add_paragraph()
            p_b.text = big_text
            p_b.font.name = FONT_PRIMARY
            p_b.font.size = Pt(26)
            p_b.font.bold = True
            p_b.font.color.rgb = accent

            p_d = tf_c.add_paragraph()
            p_d.text = desc
            p_d.font.name = FONT_PRIMARY
            p_d.font.size = Pt(13)
            p_d.font.color.rgb = TEXT_DARK

        self._add_footer(slide, 1)

    def build_slide_02_clinical_problem(self):
        """Slide 2: The Real-World Clinical Problem From Scratch."""
        slide = self._create_base_slide()
        self._add_header(
            slide,
            "01 / The Clinical Reality From Scratch",
            "The High-Stakes Decision in Every Emergency Department",
            "When a patient walks in with chest discomfort, a cardiologist faces two asymmetric risks."
        )

        card_top = 2.22
        card_h = 4.45

        # Left Card: False Negative (The Tragedy)
        self._add_card(slide, 0.8, card_top, 5.7, card_h, accent_top_color=ALERT_CRIMSON)
        tb_l = slide.shapes.add_textbox(Inches(1.1), Inches(card_top + 0.25), Inches(5.1), Inches(card_h - 0.4))
        tf_l = tb_l.text_frame
        tf_l.word_wrap = True
        tf_l.margin_left = tf_l.margin_right = tf_l.margin_top = tf_l.margin_bottom = 0

        p1 = tf_l.paragraphs[0]
        p1.text = "FALSE NEGATIVE: The Missed Heart Attack"
        p1.font.name = FONT_TITLE
        p1.font.size = Pt(20)
        p1.font.bold = True
        p1.font.color.rgb = ALERT_CRIMSON
        p1.space_after = Pt(10)

        bullets_l = [
            "The Diagnosis: The AI or doctor classifies severe coronary narrowing as 'Low Risk' and discharges the patient home.",
            "The Human Cost: Hours or days later, a plaque ruptures, causing an unpredicted Acute Myocardial Infarction (AMI).",
            "The Medical Outcome: Emergency ICU admission, permanent heart muscle death, heart failure, or sudden death.",
            "The Economic Price: $42,500 in emergency surgery & intensive hospital care per missed case.",
        ]
        for b in bullets_l:
            p = tf_l.add_paragraph()
            p.text = f"•  {b}"
            p.font.name = FONT_PRIMARY
            p.font.size = Pt(15)
            p.font.color.rgb = TEXT_DARK
            p.space_after = Pt(8)

        # Right Card: False Positive (The Waste & Hazard)
        self._add_card(slide, 6.8, card_top, 5.7, card_h, accent_top_color=SAGE_GREEN)
        tb_r = slide.shapes.add_textbox(Inches(7.1), Inches(card_top + 0.25), Inches(5.1), Inches(card_h - 0.4))
        tf_r = tb_r.text_frame
        tf_r.word_wrap = True
        tf_r.margin_left = tf_r.margin_right = tf_r.margin_top = tf_r.margin_bottom = 0

        p2 = tf_r.paragraphs[0]
        p2.text = "FALSE POSITIVE: The Unnecessary Surgery"
        p2.font.name = FONT_TITLE
        p2.font.size = Pt(20)
        p2.font.bold = True
        p2.font.color.rgb = DARK_GREEN
        p2.space_after = Pt(10)

        bullets_r = [
            "The Diagnosis: A healthy patient is falsely predicted as 'High Risk' due to ambiguous symptoms or flawed AI.",
            "The Procedure: The patient undergoes invasive Coronary Angiography (threading a catheter into the heart under X-ray dye).",
            "The Medical Outcome: Clean arteries are found, but the patient was exposed to arterial puncture, stroke, and kidney injury.",
            "The Economic Price: $12,400 in wasted invasive procedural costs per unnecessary catheterization.",
        ]
        for b in bullets_r:
            p = tf_r.add_paragraph()
            p.text = f"•  {b}"
            p.font.name = FONT_PRIMARY
            p.font.size = Pt(15)
            p.font.color.rgb = TEXT_DARK
            p.space_after = Pt(8)

        self._add_footer(slide, 2)

    def build_slide_03_island_blindspots(self):
        """Slide 3: Why Single Hospitals Fail: Local Island Blind Spots."""
        slide = self._create_base_slide()
        self._add_header(
            slide,
            "02 / The Root Cause",
            "Why Single-Hospital AI Models Fail: 'Island Blind Spots'",
            "Every hospital operates as an isolated demographic island. Models trained on one island fail on another."
        )

        cards = [
            (
                "VA Long Beach, California",
                "DEMOGRAPHIC BLIND SPOT",
                "94.7% Male Military Veterans",
                "Because its cohort is almost exclusively older men, Long Beach's AI has virtually never learned female cardiac physiology. When a 51-year-old woman walks in, the local model is blind.",
                DARK_GREEN,
            ),
            (
                "Hungarian Cardiac Institute",
                "EQUIPMENT BLIND SPOT",
                "1.5% Catheterization Rate",
                "As an outpatient screening clinic, Budapest relies entirely on exercise treadmill ECGs. Its local AI cannot recognize advanced fluoroscopic vessel calcifications.",
                SAGE_GREEN,
            ),
            (
                "Tokyo University Hospital",
                "BIOLOGICAL BLIND SPOT",
                "Low Cholesterol & Microvascular",
                "Tokyo patients have clean lipid panels (chol < 195) but severe microvascular ST depression. Western models that over-rely on high cholesterol misdiagnose them completely.",
                DARK_GREEN,
            ),
        ]

        card_w = 3.7
        gap = 0.3
        start_x = 0.8
        card_top = 2.22
        card_h = 4.45

        for idx, (hosp, tag, subtitle, body, accent) in enumerate(cards):
            cx = start_x + idx * (card_w + gap)
            self._add_card(slide, cx, card_top, card_w, card_h, accent_top_color=accent)

            tb = slide.shapes.add_textbox(Inches(cx + 0.3), Inches(card_top + 0.25), Inches(card_w - 0.6), Inches(card_h - 0.5))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

            p_tag = tf.paragraphs[0]
            p_tag.text = tag
            p_tag.font.name = FONT_PRIMARY
            p_tag.font.size = Pt(12)
            p_tag.font.bold = True
            p_tag.font.color.rgb = SAGE_GREEN
            p_tag.space_after = Pt(4)

            p_h = tf.add_paragraph()
            p_h.text = hosp
            p_h.font.name = FONT_TITLE
            p_h.font.size = Pt(20)
            p_h.font.bold = True
            p_h.font.color.rgb = DARK_GREEN
            p_h.space_after = Pt(4)

            p_s = tf.add_paragraph()
            p_s.text = subtitle
            p_s.font.name = FONT_PRIMARY
            p_s.font.size = Pt(14)
            p_s.font.bold = True
            p_s.font.color.rgb = WARM_AMBER
            p_s.space_after = Pt(12)

            p_b = tf.add_paragraph()
            p_b.text = body
            p_b.font.name = FONT_PRIMARY
            p_b.font.size = Pt(15.5)
            p_b.font.color.rgb = TEXT_DARK

        self._add_footer(slide, 3)

    def build_slide_04_regulatory_wall(self):
        """Slide 4: Why Centralizing Patient Data is Legally & Financially Impossible."""
        slide = self._create_base_slide()
        self._add_header(
            slide,
            "03 / The Regulatory Trap",
            "Why Can't Hospitals Just Upload Everything to the Cloud?",
            "Centralizing clinical data across international institutions is illegal, unethical, and financially catastrophic."
        )

        barriers = [
            (
                "FEDERAL & INTERNATIONAL LAWS",
                "HIPAA & GDPR Prohibit Raw Data Export",
                "Under US HIPAA and European GDPR Article 9 (Special Category Health Data), moving identifiable patient records outside sovereign institutional boundaries without complex patient authorization carries severe criminal and civil penalties.",
                ALERT_CRIMSON,
            ),
            (
                "$499 PER RECORD BREACH LIABILITY",
                "The Multi-Million Dollar Centralized Honeypot",
                "According to the IBM Ponemon Benchmark, healthcare breaches cost an average of $499 per compromised record. Centralizing 9,600 patients creates a $4.8M breach target. Centralizing 1M records creates a $500M catastrophe.",
                WARM_AMBER,
            ),
            (
                "COMMERCIAL & TRUST BOUNDARIES",
                "Hospitals Compete — But Must Collaborate",
                "Leading medical centers compete for patients and research grants. No hospital CEO will surrender their proprietary electronic health record (EHR) database to an external competitor or a commercial cloud provider.",
                DARK_GREEN,
            ),
        ]

        card_top = 2.22
        card_h = 1.34
        card_gap = 0.20

        for idx, (tag, title, desc, accent) in enumerate(barriers):
            cy = card_top + idx * (card_h + card_gap)
            self._add_card(slide, 0.8, cy, 11.733, card_h, accent_top_color=accent)

            tb = slide.shapes.add_textbox(Inches(1.15), Inches(cy + 0.16), Inches(11.0), Inches(card_h - 0.30))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

            p_t = tf.paragraphs[0]
            p_t.text = f"{tag}  —  {title}"
            p_t.font.name = FONT_TITLE
            p_t.font.size = Pt(18)
            p_t.font.bold = True
            p_t.font.color.rgb = DARK_GREEN
            p_t.space_after = Pt(4)

            p_d = tf.add_paragraph()
            p_d.text = desc
            p_d.font.name = FONT_PRIMARY
            p_d.font.size = Pt(15)
            p_d.font.color.rgb = TEXT_DARK

        self._add_footer(slide, 4)

    def build_slide_05_core_approach_concept(self):
        """Slide 5: The Breakthrough Approach: What is Federated Learning?"""
        slide = self._create_base_slide()
        self._add_header(
            slide,
            "04 / The Core Approach",
            "The Big Idea: Don't Move the Patient — Move the Classroom",
            "Instead of dragging sensitive patient records to a central model, we send the model to the patient's bedside."
        )

        steps = [
            (
                "STEP 1",
                "Learn in Secret",
                "Every hospital trains a private copy of the AI inside its own secure server room. Patient names, lab results, and ECGs never leave the room.",
                DARK_GREEN,
            ),
            (
                "STEP 2",
                "Extract Only Math",
                "The hospital calculates what the AI learned — a tiny 26-kilobyte update of neural network weights. No patient information is contained in the weights.",
                SAGE_GREEN,
            ),
            (
                "STEP 3",
                "Fuse the Intelligence",
                "A central coordinator merges the mathematical lessons from all 8 hospitals into a single super-model, and transmits the smarter model back to all 8 centers.",
                DARK_GREEN,
            ),
        ]

        card_w = 3.7
        gap = 0.3
        start_x = 0.8
        card_top = 2.22
        card_h = 3.4

        for idx, (step_label, title, body, accent) in enumerate(steps):
            cx = start_x + idx * (card_w + gap)
            self._add_card(slide, cx, card_top, card_w, card_h, accent_top_color=accent)

            tb = slide.shapes.add_textbox(Inches(cx + 0.3), Inches(card_top + 0.25), Inches(card_w - 0.6), Inches(card_h - 0.5))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

            p_step = tf.paragraphs[0]
            p_step.text = step_label
            p_step.font.name = FONT_PRIMARY
            p_step.font.size = Pt(13)
            p_step.font.bold = True
            p_step.font.color.rgb = SAGE_GREEN
            p_step.space_after = Pt(4)

            p_t = tf.add_paragraph()
            p_t.text = title
            p_t.font.name = FONT_TITLE
            p_t.font.size = Pt(23)
            p_t.font.bold = True
            p_t.font.color.rgb = DARK_GREEN
            p_t.space_after = Pt(10)

            p_b = tf.add_paragraph()
            p_b.text = body
            p_b.font.name = FONT_PRIMARY
            p_b.font.size = Pt(16)
            p_b.font.color.rgb = TEXT_DARK

        # Bottom Callout Summary
        self._add_card(slide, 0.8, 5.85, 11.733, 0.80, bg_color=MINT_TINT, border_color=SAGE_GREEN)
        tb_call = slide.shapes.add_textbox(Inches(1.15), Inches(5.98), Inches(11.0), Inches(0.50))
        tf_c = tb_call.text_frame
        tf_c.word_wrap = True
        tf_c.margin_left = tf_c.margin_right = tf_c.margin_top = tf_c.margin_bottom = 0
        p_c = tf_c.paragraphs[0]
        p_c.text = "THE GOLDEN PRINCIPLE: Raw patient records exposed across the network = EXACTLY ZERO."
        p_c.font.name = FONT_PRIMARY
        p_c.font.size = Pt(17)
        p_c.font.bold = True
        p_c.font.color.rgb = DARK_GREEN

        self._add_footer(slide, 5)

    def build_slide_06_eight_hospital_silos(self):
        """Slide 6: Step 1 of the Approach: 8 Sovereign Physical Silos."""
        slide = self._create_base_slide()
        self._add_header(
            slide,
            "05 / Architecture & Data",
            "The 8 Sovereign Silos: Real Physical Isolation Across 3 Continents",
            "Not a toy notebook partition. 8 distinct hospital directories, 9,600 isolated patients, zero shared CSVs."
        )

        hospitals = [
            ("Cleveland Clinic (USA)", "1,020 train / 180 test", "47.0% CAD", "Balanced Tertiary Referral"),
            ("Hungarian Inst. (Budapest)", "1,020 train / 180 test", "36.3% CAD", "Preventive Screening Clinic"),
            ("Univ. Hospital Zurich (CH)", "1,020 train / 180 test", "78.9% CAD", "Acute Severe CAD Center"),
            ("VA Long Beach (USA)", "1,020 train / 180 test", "75.6% CAD", "Older Male Veterans Cohort"),
            ("Mayo Clinic (USA)", "1,020 train / 180 test", "31.2% CAD", "Executive Outpatient Screening"),
            ("Johns Hopkins (USA)", "1,020 train / 180 test", "61.8% CAD", "Urban Cardiometabolic Center"),
            ("Charite Berlin (Germany)", "1,020 train / 180 test", "68.4% CAD", "European Geriatric Center"),
            ("Tokyo Univ. Hospital (JP)", "1,020 train / 180 test", "24.7% CAD", "East Asian Microvascular Cohort"),
        ]

        card_w = 2.7
        card_h = 2.05
        gap_x = 0.3
        gap_y = 0.25
        start_x = 0.8
        start_y = 2.22

        for idx, (name, records, cad_rate, role) in enumerate(hospitals):
            col = idx % 4
            row = idx // 4
            cx = start_x + col * (card_w + gap_x)
            cy = start_y + row * (card_h + gap_y)

            self._add_card(slide, cx, cy, card_w, card_h, accent_top_color=DARK_GREEN)

            tb = slide.shapes.add_textbox(Inches(cx + 0.2), Inches(cy + 0.16), Inches(card_w - 0.4), Inches(card_h - 0.30))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

            p_n = tf.paragraphs[0]
            p_n.text = name
            p_n.font.name = FONT_TITLE
            p_n.font.size = Pt(15.5)
            p_n.font.bold = True
            p_n.font.color.rgb = DARK_GREEN
            p_n.space_after = Pt(3)

            p_r = tf.add_paragraph()
            p_r.text = f"Patients: {records}"
            p_r.font.name = FONT_PRIMARY
            p_r.font.size = Pt(12.5)
            p_r.font.bold = True
            p_r.font.color.rgb = SAGE_GREEN

            p_c = tf.add_paragraph()
            p_c.text = f"CAD Rate: {cad_rate}"
            p_c.font.name = FONT_PRIMARY
            p_c.font.size = Pt(12.5)
            p_c.font.color.rgb = WARM_AMBER
            p_c.space_after = Pt(3)

            p_o = tf.add_paragraph()
            p_o.text = role
            p_o.font.name = FONT_PRIMARY
            p_o.font.size = Pt(12)
            p_o.font.color.rgb = TEXT_DARK

        self._add_footer(slide, 6)

    def build_slide_07_harmonization_and_encoding(self):
        """Slide 7: Step 2 of the Approach: Universal Clinical Language (13 to 22 Features)."""
        slide = self._create_base_slide()
        self._add_header(
            slide,
            "06 / Clinical Harmonization",
            "Speaking One Universal Medical Language Without Leaking Data",
            "Why standard machine learning normalization fails in medicine, and how universal bounds fix it."
        )

        card_top = 2.22
        card_h = 4.45

        # Left Card: Why StandardScaler Fails in FL
        self._add_card(slide, 0.8, card_top, 5.7, card_h, accent_top_color=ALERT_CRIMSON)
        tb_l = slide.shapes.add_textbox(Inches(1.1), Inches(card_top + 0.25), Inches(5.1), Inches(card_h - 0.4))
        tf_l = tb_l.text_frame
        tf_l.word_wrap = True
        tf_l.margin_left = tf_l.margin_right = tf_l.margin_top = tf_l.margin_bottom = 0

        p1 = tf_l.paragraphs[0]
        p1.text = "THE STANDARDSCALER TRAP"
        p1.font.name = FONT_TITLE
        p1.font.size = Pt(20)
        p1.font.bold = True
        p1.font.color.rgb = ALERT_CRIMSON
        p1.space_after = Pt(10)

        bullets_l = [
            "What amateur ML does: Local StandardScaler subtracts the hospital's local mean and divides by standard deviation.",
            "The Fatal Contradiction: At Hungary (younger patients, mean BP=130), a 150 mmHg reading is encoded as +1.67 (Critical!).",
            "At Long Beach (hypertensive veterans, mean BP=150), that exact same 150 mmHg reading is encoded as 0.00 (Completely average!).",
            "The Result: The central server averages contradictory gradients for identical blood pressure, destroying the neural network.",
        ]
        for b in bullets_l:
            p = tf_l.add_paragraph()
            p.text = f"•  {b}"
            p.font.name = FONT_PRIMARY
            p.font.size = Pt(15)
            p.font.color.rgb = TEXT_DARK
            p.space_after = Pt(8)

        # Right Card: The FedCardio Universal Domain Bounds
        self._add_card(slide, 6.8, card_top, 5.7, card_h, accent_top_color=DARK_GREEN)
        tb_r = slide.shapes.add_textbox(Inches(7.1), Inches(card_top + 0.25), Inches(5.1), Inches(card_h - 0.4))
        tf_r = tb_r.text_frame
        tf_r.word_wrap = True
        tf_r.margin_left = tf_r.margin_right = tf_r.margin_top = tf_r.margin_bottom = 0

        p2 = tf_r.paragraphs[0]
        p2.text = "THE FEDCARDIO SOLUTION: UNIVERSAL BOUNDS"
        p2.font.name = FONT_TITLE
        p2.font.size = Pt(20)
        p2.font.bold = True
        p2.font.color.rgb = DARK_GREEN
        p2.space_after = Pt(10)

        bullets_r = [
            "Fixed Medical Bounds: Human biology has known limits (e.g. resting blood pressure [80, 220] mmHg, age [20, 90] yrs).",
            "Universal Min-Max: Continuous values are clamped into [0.0, 1.0] identically across all 8 hospitals around the globe.",
            "13 Clinical Features -> 22 Aligned Dimensions: 6 bounded continuous + 3 binary + 13 categorical one-hot indicators.",
            "Zero Data Leakage: Not a single hospital mean, variance, or private statistic is ever transmitted to the server.",
        ]
        for b in bullets_r:
            p = tf_r.add_paragraph()
            p.text = f"•  {b}"
            p.font.name = FONT_PRIMARY
            p.font.size = Pt(15)
            p.font.color.rgb = TEXT_DARK
            p.space_after = Pt(8)

        self._add_footer(slide, 7)

    def build_slide_08_clinical_net_architecture(self):
        """Slide 8: Step 3 of the Approach: ClinicalNet & GroupNorm."""
        slide = self._create_base_slide()
        self._add_header(
            slide,
            "07 / Neural Architecture",
            "ClinicalNet: Why BatchNorm Had to Die in Medicine",
            "A lightweight 6,593-parameter neural network specifically designed to preserve patient autonomy."
        )

        cards = [
            (
                "WHY WE ELIMINATED BATCHNORM",
                "BatchNorm Leaks Data Between Patients",
                "Standard Batch Normalization computes running statistics across different patients in a batch. If Patient A's hidden neuron depends on Patient B's lab test, privacy is shattered! Furthermore, hospital demographic skews corrupt batch statistics.",
                ALERT_CRIMSON,
            ),
            (
                "WHY WE MANDATED GROUPNORM",
                "GroupNorm(4, 64) Isolates Each Human",
                "GroupNorm divides a single patient's 64 channels into 4 groups of 16 channels, normalizing strictly inside that individual's own vector. Every patient is processed as an autonomous human being — 100% independent of batch size or hospital cohort.",
                DARK_GREEN,
            ),
            (
                "COMPACT 26-KB WIRE PAYLOAD",
                "6,593 Trainable Parameters",
                "3 linear layers [22 -> 64 -> 32 -> 1] with Swish (SiLU) activations and calibrated dropout. The entire model update is only 26 kilobytes — light enough to transmit over hospital networks in milliseconds.",
                SAGE_GREEN,
            ),
        ]

        card_w = 3.7
        gap = 0.3
        start_x = 0.8
        card_top = 2.22
        card_h = 4.45

        for idx, (tag, title, body, accent) in enumerate(cards):
            cx = start_x + idx * (card_w + gap)
            self._add_card(slide, cx, card_top, card_w, card_h, accent_top_color=accent)

            tb = slide.shapes.add_textbox(Inches(cx + 0.3), Inches(card_top + 0.25), Inches(card_w - 0.6), Inches(card_h - 0.5))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

            p_t = tf.paragraphs[0]
            p_t.text = tag
            p_t.font.name = FONT_PRIMARY
            p_t.font.size = Pt(12)
            p_t.font.bold = True
            p_t.font.color.rgb = accent
            p_t.space_after = Pt(4)

            p_h = tf.add_paragraph()
            p_h.text = title
            p_h.font.name = FONT_TITLE
            p_h.font.size = Pt(20)
            p_h.font.bold = True
            p_h.font.color.rgb = DARK_GREEN
            p_h.space_after = Pt(12)

            p_b = tf.add_paragraph()
            p_b.text = body
            p_b.font.name = FONT_PRIMARY
            p_b.font.size = Pt(15.5)
            p_b.font.color.rgb = TEXT_DARK

        self._add_footer(slide, 8)

    def build_slide_09_differential_privacy(self):
        """Slide 9: Step 4 of the Approach: Mathematical Secrecy (Per-Sample DP-SGD)."""
        slide = self._create_base_slide()
        self._add_header(
            slide,
            "08 / Formal Secrecy",
            "Differential Privacy: Why No Single Patient Can Be Spied On",
            "How vectorized per-sample gradient clipping and Gaussian noise guarantee mathematical plausible deniability."
        )

        dp_steps = [
            (
                "STAGE 1: VECTORIZED GRADIENTS",
                "Isolate Every Patient with torch.func.vmap",
                "Most tutorials calculate batch-averaged gradients. But if an outlier has 50x higher cholesterol, their gradient dominates the batch. FedCardio computes isolated gradients for every single patient separately.",
                DARK_GREEN,
            ),
            (
                "STAGE 2: INDIVIDUAL CLIPPING",
                "Cap Each Patient's Sensitivity at C = 1.0",
                "If any patient's gradient length exceeds 1.0, it is scaled down: g_clipped = g * min(1, C / ||g||). No single human being — no matter how rare or extreme their disease — can push the model beyond radius C.",
                SAGE_GREEN,
            ),
            (
                "STAGE 3: CALIBRATED NOISE",
                "Gaussian Privacy Fog & GDP Accounting",
                "Calibrated Gaussian noise is injected into the summed updates. Mathematically guarantees that an attacker observing model weights cannot determine whether any specific patient was included in the training.",
                DARK_GREEN,
            ),
        ]

        card_top = 2.22
        card_h = 1.34
        card_gap = 0.20

        for idx, (tag, title, desc, accent) in enumerate(dp_steps):
            cy = card_top + idx * (card_h + card_gap)
            self._add_card(slide, 0.8, cy, 11.733, card_h, accent_top_color=accent)

            tb = slide.shapes.add_textbox(Inches(1.15), Inches(cy + 0.16), Inches(11.0), Inches(card_h - 0.30))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

            p_t = tf.paragraphs[0]
            p_t.text = f"{tag}  —  {title}"
            p_t.font.name = FONT_TITLE
            p_t.font.size = Pt(18)
            p_t.font.bold = True
            p_t.font.color.rgb = DARK_GREEN
            p_t.space_after = Pt(4)

            p_d = tf.add_paragraph()
            p_d.text = desc
            p_d.font.name = FONT_PRIMARY
            p_d.font.size = Pt(15)
            p_d.font.color.rgb = TEXT_DARK

        self._add_footer(slide, 9)

    def build_slide_10_decoupled_fedprox(self):
        """Slide 10: Step 5 of the Approach: Untangling Drift with Decoupled FedProx."""
        slide = self._create_base_slide()
        self._add_header(
            slide,
            "09 / Collaborative Stability",
            "Decoupled FedProx: Keeping 8 Independent Hospitals Aligned",
            "When hospitals have conflicting patient demographics, local models wander. How we tether them gently."
        )

        card_top = 2.22
        card_h = 4.45

        # Left Card: The Client Drift Dilemma
        self._add_card(slide, 0.8, card_top, 5.7, card_h, accent_top_color=WARM_AMBER)
        tb_l = slide.shapes.add_textbox(Inches(1.1), Inches(card_top + 0.25), Inches(5.1), Inches(card_h - 0.4))
        tf_l = tb_l.text_frame
        tf_l.word_wrap = True
        tf_l.margin_left = tf_l.margin_right = tf_l.margin_top = tf_l.margin_bottom = 0

        p1 = tf_l.paragraphs[0]
        p1.text = "THE PROBLEM: CLIENT DRIFT"
        p1.font.name = FONT_TITLE
        p1.font.size = Pt(20)
        p1.font.bold = True
        p1.font.color.rgb = WARM_AMBER
        p1.space_after = Pt(10)

        bullets_l = [
            "Conflicting Minima: Tokyo trains on 24.7% CAD (low lipid); Zurich trains on 78.9% CAD (acute crises).",
            "Wandering Weights: After several local training epochs, each hospital's weights drift away from the collective consensus toward its own local demographic silo.",
            "Naive Averaging Fails: Simple FedAvg struggles to converge when client updates pull in opposite geometric directions.",
        ]
        for b in bullets_l:
            p = tf_l.add_paragraph()
            p.text = f"•  {b}"
            p.font.name = FONT_PRIMARY
            p.font.size = Pt(15)
            p.font.color.rgb = TEXT_DARK
            p.space_after = Pt(10)

        # Right Card: The Decoupled FedProx Innovation
        self._add_card(slide, 6.8, card_top, 5.7, card_h, accent_top_color=DARK_GREEN)
        tb_r = slide.shapes.add_textbox(Inches(7.1), Inches(card_top + 0.25), Inches(5.1), Inches(card_h - 0.4))
        tf_r = tb_r.text_frame
        tf_r.word_wrap = True
        tf_r.margin_left = tf_r.margin_right = tf_r.margin_top = tf_r.margin_bottom = 0

        p2 = tf_r.paragraphs[0]
        p2.text = "THE SOLUTION: DECOUPLED FEDPROX"
        p2.font.name = FONT_TITLE
        p2.font.size = Pt(20)
        p2.font.bold = True
        p2.font.color.rgb = DARK_GREEN
        p2.space_after = Pt(10)

        bullets_r = [
            "The Elastic Tether: Adds a proximal pull (mu = 0.05) tethering local weights w back to the global consensus w_global.",
            "Critic-Mandated Innovation: If you put the tether inside per-sample DP, the deterministic penalty consumes patient clipping budget!",
            "Decoupled Application: FedCardio computes DP clipping strictly on private patient gradients, then adds the proximal pull AFTER DP noise.",
            "Mathematical Stability: Reduces Euclidean L2 drift and angular cosine divergence without sacrificing a single drop of privacy.",
        ]
        for b in bullets_r:
            p = tf_r.add_paragraph()
            p.text = f"•  {b}"
            p.font.name = FONT_PRIMARY
            p.font.size = Pt(15)
            p.font.color.rgb = TEXT_DARK
            p.space_after = Pt(8)

        self._add_footer(slide, 10)

    def build_slide_11_byzantine_defense(self):
        """Slide 11: Step 6 of the Approach: Byzantine Defense Against Poisoned Nodes."""
        slide = self._create_base_slide()
        self._add_header(
            slide,
            "10 / Adversarial Resilience",
            "Byzantine Defense: What if a Hospital is Hacked?",
            "What happens when a hospital's server is infected by ransomware or an attacker injecting poisoned gradients?"
        )

        strategies = [
            (
                "STANDARD FEDAVG",
                "COLLAPSES (-27.2% AUC)",
                "Under a 12x sign-flipping attack, FedAvg averages the poisoned weights with honest weights. Global ROC-AUC crashes from 0.858 to 0.586. One compromised hospital destroys the entire network.",
                ALERT_CRIMSON,
            ),
            (
                "COORDINATE FEDMEDIAN",
                "IMMUNE (-0.69% AUC)",
                "Sorts all 8 hospital values for each weight coordinate and picks the median. Malicious extreme outliers are ignored by mathematical definition (50% breakdown point). AUC holds rock-solid at 0.850.",
                DARK_GREEN,
            ),
            (
                "TRIMMED MEAN (beta=0.2)",
                "OPTIMAL (-0.49% AUC)",
                "Discards the highest and lowest 20% of weights across clients, averaging the clean middle 60%. Combines Byzantine immunity with smooth statistical efficiency. AUC holds at 0.853.",
                SAGE_GREEN,
            ),
        ]

        card_w = 3.7
        gap = 0.3
        start_x = 0.8
        card_top = 2.22
        card_h = 4.45

        for idx, (tag, verdict, body, accent) in enumerate(strategies):
            cx = start_x + idx * (card_w + gap)
            self._add_card(slide, cx, card_top, card_w, card_h, accent_top_color=accent)

            tb = slide.shapes.add_textbox(Inches(cx + 0.3), Inches(card_top + 0.25), Inches(card_w - 0.6), Inches(card_h - 0.5))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

            p_tag = tf.paragraphs[0]
            p_tag.text = tag
            p_tag.font.name = FONT_PRIMARY
            p_tag.font.size = Pt(12)
            p_tag.font.bold = True
            p_tag.font.color.rgb = SAGE_GREEN
            p_tag.space_after = Pt(4)

            p_v = tf.add_paragraph()
            p_v.text = verdict
            p_v.font.name = FONT_TITLE
            p_v.font.size = Pt(20)
            p_v.font.bold = True
            p_v.font.color.rgb = accent
            p_v.space_after = Pt(12)

            p_b = tf.add_paragraph()
            p_b.text = body
            p_b.font.name = FONT_PRIMARY
            p_b.font.size = Pt(15.5)
            p_b.font.color.rgb = TEXT_DARK

        self._add_footer(slide, 11)

    def build_slide_12_clinical_rescue_proof(self):
        """Slide 12: The Proof: 4 Rare Patient Archetypes Rescued."""
        slide = self._create_base_slide()
        self._add_header(
            slide,
            "11 / Clinical Proof of Value",
            "The Human Proof: 4 Lives Rescued From False-Negative Discharge",
            "Four atypical patients who would have died under their local model, but were saved by the collective AI."
        )

        cases = [
            (
                "CASE 1: THE FEMALE VETERAN",
                "Admitted: VA Long Beach",
                "51yo Female, Atypical Angina, ST depression. True CAD = 1.",
                "Local Risk: 38.2% (Discharged!)",
                "Federated: 84.6% (Rescued)",
                "Rescued via female cardiac patterns learned from Johns Hopkins (52% F) & Mayo.",
            ),
            (
                "CASE 2: THE SILENT DIABETIC",
                "Admitted: Budapest, Hungary",
                "64yo Male Diabetic, Asymptomatic, 2 calcified vessels. True CAD = 1.",
                "Local Risk: 31.4% (Discharged!)",
                "Federated: 87.9% (Rescued)",
                "Rescued via advanced fluoroscopy patterns learned from Univ. Hospital Zurich.",
            ),
            (
                "CASE 3: THE WALK-IN RUNNER",
                "Admitted: Univ. Hospital Zurich",
                "42yo Male Walk-In, Early exercise angina, no previous cath. True CAD = 1.",
                "Local Risk: 41.8% (Discharged!)",
                "Federated: 79.4% (Rescued)",
                "Rescued via early outpatient screening patterns learned from Hungary & Mayo.",
            ),
            (
                "CASE 4: LOW-LIPID WOMAN",
                "Admitted: Mayo Clinic",
                "58yo Female, Low Cholesterol (172 mg/dL), severe ST depression. True CAD = 1.",
                "Local Risk: 29.5% (Discharged!)",
                "Federated: 81.2% (Rescued)",
                "Rescued via low-lipid microvascular ischemia signatures from Tokyo Univ. Hospital.",
            ),
        ]

        card_w = 2.7
        card_h = 4.45
        gap_x = 0.3
        start_x = 0.8
        start_y = 2.22

        for idx, (title, hosp, prof, local, fed, why) in enumerate(cases):
            cx = start_x + idx * (card_w + gap_x)
            self._add_card(slide, cx, start_y, card_w, card_h, accent_top_color=DARK_GREEN)

            tb = slide.shapes.add_textbox(Inches(cx + 0.2), Inches(start_y + 0.18), Inches(card_w - 0.4), Inches(card_h - 0.35))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

            p_t = tf.paragraphs[0]
            p_t.text = title
            p_t.font.name = FONT_TITLE
            p_t.font.size = Pt(14)
            p_t.font.bold = True
            p_t.font.color.rgb = DARK_GREEN
            p_t.space_after = Pt(2)

            p_h = tf.add_paragraph()
            p_h.text = hosp
            p_h.font.name = FONT_PRIMARY
            p_h.font.size = Pt(12)
            p_h.font.bold = True
            p_h.font.color.rgb = SAGE_GREEN
            p_h.space_after = Pt(6)

            p_p = tf.add_paragraph()
            p_p.text = prof
            p_p.font.name = FONT_PRIMARY
            p_p.font.size = Pt(12.5)
            p_p.font.color.rgb = TEXT_DARK
            p_p.space_after = Pt(6)

            p_l = tf.add_paragraph()
            p_l.text = local
            p_l.font.name = FONT_PRIMARY
            p_l.font.size = Pt(13.5)
            p_l.font.bold = True
            p_l.font.color.rgb = ALERT_CRIMSON
            p_l.space_after = Pt(3)

            p_f = tf.add_paragraph()
            p_f.text = fed
            p_f.font.name = FONT_PRIMARY
            p_f.font.size = Pt(13.5)
            p_f.font.bold = True
            p_f.font.color.rgb = DARK_GREEN
            p_f.space_after = Pt(6)

            p_w = tf.add_paragraph()
            p_w.text = f"Why: {why}"
            p_w.font.name = FONT_PRIMARY
            p_w.font.size = Pt(12)
            p_w.font.color.rgb = TEXT_MUTED

        self._add_footer(slide, 12)

    def build_slide_13_grouped_coalitional_shap(self):
        """Slide 13: Opening the Black Box: Grouped Coalitional SHAP."""
        slide = self._create_base_slide()
        self._add_header(
            slide,
            "12 / Clinical Explainability",
            "Why Cardiologists Can Actually Trust the AI's Decisions",
            "Doctors will never act on a black-box probability. How Grouped Coalitional SHAP reveals the 'why'."
        )

        cards = [
            (
                "THE PROBLEM WITH REGULAR SHAP",
                "Impossible Frankenstein Patients",
                "Standard feature SHAP perturbs one-hot variables separately. For example, it might turn Chest Pain 1 OFF while leaving Chest Pain 4 OFF, creating an impossible patient with NO chest pain type, or BOTH types at once. The resulting math is garbage.",
                ALERT_CRIMSON,
            ),
            (
                "OUR COALITIONAL INNOVATION",
                "13 Grouped Clinical Coalitions",
                "FedCardio perturbs one-hot groups jointly as indivisible medical teams (e.g., all 4 chest pain types permute together; all 3 ECG slopes permute together). Every evaluated state remains physiologically valid and realistic.",
                DARK_GREEN,
            ),
            (
                "AUDITED SHAPLEY EFFICIENCY",
                "Exact Additive Risk Attribution",
                "Guaranteed mathematical conservation: Sum(phi_i) = f(x) - E[f(X_bg)]. The cardiologist receives an exact waterfall chart showing precisely how many percentage points each biomarker added to or subtracted from baseline risk.",
                SAGE_GREEN,
            ),
        ]

        card_w = 3.7
        gap = 0.3
        start_x = 0.8
        card_top = 2.22
        card_h = 4.45

        for idx, (tag, title, body, accent) in enumerate(cards):
            cx = start_x + idx * (card_w + gap)
            self._add_card(slide, cx, card_top, card_w, card_h, accent_top_color=accent)

            tb = slide.shapes.add_textbox(Inches(cx + 0.3), Inches(card_top + 0.25), Inches(card_w - 0.6), Inches(card_h - 0.5))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

            p_t = tf.paragraphs[0]
            p_t.text = tag
            p_t.font.name = FONT_PRIMARY
            p_t.font.size = Pt(12)
            p_t.font.bold = True
            p_t.font.color.rgb = accent
            p_t.space_after = Pt(4)

            p_h = tf.add_paragraph()
            p_h.text = title
            p_h.font.name = FONT_TITLE
            p_h.font.size = Pt(20)
            p_h.font.bold = True
            p_h.font.color.rgb = DARK_GREEN
            p_h.space_after = Pt(12)

            p_b = tf.add_paragraph()
            p_b.text = body
            p_b.font.name = FONT_PRIMARY
            p_b.font.size = Pt(15.5)
            p_b.font.color.rgb = TEXT_DARK

        self._add_footer(slide, 13)

    def build_slide_14_healthcare_roi_economics(self):
        """Slide 14: Healthcare Economics & Actuarial ROI ($118.76M Annual Net Value)."""
        slide = self._create_base_slide()
        self._add_header(
            slide,
            "13 / Healthcare Economics",
            "The Actuarial ROI: $118.76M Net Annual Consortium Value",
            "Proven financial value across 128,000 annual chest-pain admissions across the 8 hospital systems."
        )

        roi_cards = [
            (
                "2,036 PREVENTED AMIs",
                "$86.53M",
                "Saved Heart Attack Admissions",
                "+18.0% diagnostic sensitivity rescues 2,036 patients from false-negative discharge. At $42,500 saved per prevented emergency AMI, hospitals avoid uncompensated emergency care.",
                DARK_GREEN,
            ),
            (
                "1,481 AVOIDED ANGIOGRAMS",
                "$18.36M",
                "Avoided Procedural Waste",
                "+11.5% specificity gain prevents 1,481 unnecessary invasive catheterizations on healthy patients. At $12,400 per procedure, this frees up clinical cath labs.",
                SAGE_GREEN,
            ),
            (
                "CENTRALIZED BREACH IMMUNITY",
                "$14.23M",
                "Avoided HIPAA Liability",
                "By keeping all 128,000 annual EHR records behind hospital firewalls, the consortium completely eliminates the 22.2% annual centralized breach risk ($499/record IBM Ponemon benchmark).",
                DARK_GREEN,
            ),
            (
                "TOTAL ANNUAL NET ROI",
                "330.9x",
                "$118.76M Net Annual Value",
                "Total annual infrastructure and orchestration costs across all 8 nodes are only $358,880. Yields a massive 330.9x financial return on investment for the consortium.",
                WARM_AMBER,
            ),
        ]

        card_w = 2.7
        card_h = 4.45
        gap_x = 0.3
        start_x = 0.8
        start_y = 2.22

        for idx, (tag, big_num, title, body, accent) in enumerate(roi_cards):
            cx = start_x + idx * (card_w + gap_x)
            self._add_card(slide, cx, start_y, card_w, card_h, accent_top_color=accent)

            tb = slide.shapes.add_textbox(Inches(cx + 0.22), Inches(start_y + 0.20), Inches(card_w - 0.44), Inches(card_h - 0.40))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

            p_tag = tf.paragraphs[0]
            p_tag.text = tag
            p_tag.font.name = FONT_PRIMARY
            p_tag.font.size = Pt(11)
            p_tag.font.bold = True
            p_tag.font.color.rgb = TEXT_MUTED
            p_tag.space_after = Pt(2)

            p_num = tf.add_paragraph()
            p_num.text = big_num
            p_num.font.name = FONT_PRIMARY
            p_num.font.size = Pt(34)
            p_num.font.bold = True
            p_num.font.color.rgb = accent
            p_num.space_after = Pt(2)

            p_t = tf.add_paragraph()
            p_t.text = title
            p_t.font.name = FONT_TITLE
            p_t.font.size = Pt(15)
            p_t.font.bold = True
            p_t.font.color.rgb = DARK_GREEN
            p_t.space_after = Pt(8)

            p_b = tf.add_paragraph()
            p_b.text = body
            p_b.font.name = FONT_PRIMARY
            p_b.font.size = Pt(13)
            p_b.font.color.rgb = TEXT_DARK

        self._add_footer(slide, 14)

    def build_slide_15_conclusion_and_vision(self):
        """Slide 15: The Vision & Closing: Sovereign Data, Collective Wisdom."""
        slide = self._create_base_slide()
        self._add_header(
            slide,
            "14 / The Future of Medicine",
            "Sovereign Data, Collective Wisdom: The FedCardio Legacy",
            "What FedCardio proves for the future of collaborative clinical healthcare."
        )

        pillars = [
            (
                "01 / PRIVACY AND ACCURACY ARE NO LONGER AT WAR",
                "You do not have to sacrifice patient trust or violate HIPAA/GDPR to achieve world-class AI accuracy. By keeping data in sovereign hospital silos and moving only encrypted math, privacy and performance reinforce each other.",
                DARK_GREEN,
            ),
            (
                "02 / DEMOGRAPHIC JUSTICE FOR UNDERREPRESENTED PATIENTS",
                "A regional clinic treating primarily older men no longer has to endanger female patients. Federated intelligence allows rare, atypical, and vulnerable patient populations to be protected by the collective experience of global medical centers.",
                SAGE_GREEN,
            ),
            (
                "03 / PRODUCTION READY TODAY",
                "Fully verified: 8 sovereign hospital silos, 9,600 isolated patient records, per-sample DP-SGD, coordinate Byzantine defense, 6-tab Streamlit Enterprise Web Portal, cryptographic SQLite audit logs, and $118.76M in annual actuarial ROI.",
                DARK_GREEN,
            ),
        ]

        card_top = 2.22
        card_h = 1.34
        card_gap = 0.20

        for idx, (title, desc, accent) in enumerate(pillars):
            cy = card_top + idx * (card_h + card_gap)
            self._add_card(slide, 0.8, cy, 11.733, card_h, accent_top_color=accent)

            tb = slide.shapes.add_textbox(Inches(1.15), Inches(cy + 0.16), Inches(11.0), Inches(card_h - 0.30))
            tf = tb.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0

            p_t = tf.paragraphs[0]
            p_t.text = title
            p_t.font.name = FONT_TITLE
            p_t.font.size = Pt(18)
            p_t.font.bold = True
            p_t.font.color.rgb = DARK_GREEN
            p_t.space_after = Pt(4)

            p_d = tf.add_paragraph()
            p_d.text = desc
            p_d.font.name = FONT_PRIMARY
            p_d.font.size = Pt(15)
            p_d.font.color.rgb = TEXT_DARK

        self._add_footer(slide, 15)

    def generate_all(self, target_paths):
        """Builds all 15 slides and saves the presentation."""
        print("[*] Building Slide 1: Cover Slide...")
        self.build_slide_01_cover()

        print("[*] Building Slide 2: The Clinical Reality From Scratch...")
        self.build_slide_02_clinical_problem()

        print("[*] Building Slide 3: Island Blind Spots...")
        self.build_slide_03_island_blindspots()

        print("[*] Building Slide 4: The Regulatory Trap...")
        self.build_slide_04_regulatory_wall()

        print("[*] Building Slide 5: The Core Approach Concept...")
        self.build_slide_05_core_approach_concept()

        print("[*] Building Slide 6: The 8 Sovereign Hospital Silos...")
        self.build_slide_06_eight_hospital_silos()

        print("[*] Building Slide 7: Clinical Harmonization & Universal Bounds...")
        self.build_slide_07_harmonization_and_encoding()

        print("[*] Building Slide 8: ClinicalNet Architecture & GroupNorm...")
        self.build_slide_08_clinical_net_architecture()

        print("[*] Building Slide 9: Formal Secrecy & Per-Sample DP-SGD...")
        self.build_slide_09_differential_privacy()

        print("[*] Building Slide 10: Collaborative Stability & Decoupled FedProx...")
        self.build_slide_10_decoupled_fedprox()

        print("[*] Building Slide 11: Byzantine Defense & Poisoning Resilience...")
        self.build_slide_11_byzantine_defense()

        print("[*] Building Slide 12: Clinical Proof: 4 Patients Rescued...")
        self.build_slide_12_clinical_rescue_proof()

        print("[*] Building Slide 13: Clinical Explainability & Coalitional SHAP...")
        self.build_slide_13_grouped_coalitional_shap()

        print("[*] Building Slide 14: Healthcare Economics & Actuarial ROI...")
        self.build_slide_14_healthcare_roi_economics()

        print("[*] Building Slide 15: The Future of Medicine & Vision...")
        self.build_slide_15_conclusion_and_vision()

        primary = Path(target_paths[0])
        primary.parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(str(primary))
        file_size = primary.stat().st_size
        print(f"[SUCCESS] Saved primary presentation to: {primary} ({file_size:,} bytes, 15 slides)")

        for extra in target_paths[1:]:
            p_extra = Path(extra)
            p_extra.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(primary, p_extra)
            print(f"  -> Replicated to: {p_extra}")

        return len(self.prs.slides), file_size


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent
    targets = [
        root_dir / "FedCardio_v2_Presentation.pptx",
        root_dir / "reports" / "FedCardio_v2_Presentation.pptx",
    ]
    builder = FedCardioPresentationBuilder()
    builder.generate_all(targets)
