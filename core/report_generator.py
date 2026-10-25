"""
Report Generator Module for AI Video Assistant (Phase 7).

Generates publication-grade PDF and Markdown meeting reports:
1. Multi-page executive PDF report using fpdf2 with embedded Unicode fonts (English, Hindi, Nepali).
2. Structured Markdown report with metadata, action items, decisions, and timestamped transcript.
3. Graceful font fallbacks and unicode sanitization so report generation never crashes.
"""

import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fpdf import FPDF

from core.meeting_qa import Citation, format_timestamp, parse_timestamp_range


def get_system_font_paths() -> Dict[str, str]:
    """Find available Windows/system fonts supporting Unicode and Indic scripts."""
    windir = os.environ.get("WINDIR", "C:\\Windows")
    fonts_dir = os.path.join(windir, "Fonts")

    candidates = {
        "nirmala": [os.path.join(fonts_dir, "Nirmala.ttc"), os.path.join(fonts_dir, "Nirmala.ttf")],
        "arial": [os.path.join(fonts_dir, "arial.ttf"), os.path.join(fonts_dir, "Arial.ttf")],
        "segoe": [os.path.join(fonts_dir, "segoeui.ttf"), os.path.join(fonts_dir, "SegoeUI.ttf")],
    }

    found = {}
    for key, paths in candidates.items():
        for p in paths:
            if os.path.exists(p):
                found[key] = p
                break
    return found


def sanitize_text_for_core_font(text: str) -> str:
    """Sanitize non-Latin characters if only standard Helvetica/Latin-1 core font is available."""
    if not text:
        return ""
    # Replace common typographic characters with ASCII equivalents
    replacements = {
        "—": "-",
        "–": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "•": "*",
        "…": "...",
        "\u200b": "",  # zero-width space
        "\ufeff": "",  # BOM
    }
    cleaned = text
    for k, v in replacements.items():
        cleaned = cleaned.replace(k, v)
    return cleaned


class MeetingPDF(FPDF):
    """Custom FPDF class with branded header, footer, and styling."""

    def __init__(self, meeting_title: str = "Meeting Notes", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meeting_title = meeting_title or "Meeting Notes"
        self._font_name = "Helvetica"
        self._has_unicode_font = False

    def setup_fonts(self):
        """Register best available Unicode fonts."""
        fonts = get_system_font_paths()
        # Prefer Nirmala for Indic/Devanagari scripts (Hindi/Nepali)
        if "nirmala" in fonts:
            try:
                self.add_font("MeetingUnicode", "", fonts["nirmala"])
                self.add_font("MeetingUnicode", "B", fonts["nirmala"])
                self._font_name = "MeetingUnicode"
                self._has_unicode_font = True
                return
            except Exception:
                pass

        # Fallback to Arial Unicode
        if "arial" in fonts:
            try:
                self.add_font("MeetingUnicode", "", fonts["arial"])
                self._font_name = "MeetingUnicode"
                self._has_unicode_font = True
                return
            except Exception:
                pass

        # Default fallback: core Helvetica font
        self._font_name = "Helvetica"
        self._has_unicode_font = False

    def header(self):
        if self.page_no() == 1:
            return  # First page has full cover banner
        self.set_font(self._font_name, "", 8)
        self.set_text_color(128, 128, 128)
        clean_title = self.meeting_title if self._has_unicode_font else sanitize_text_for_core_font(self.meeting_title)
        self.cell(0, 8, f"{clean_title} | AI Meeting Assistant", align="L")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font(self._font_name, "", 8)
        self.set_text_color(140, 140, 140)
        page_str = f"Page {self.page_no()}"
        self.cell(0, 10, page_str, align="C")


def generate_pdf_report(
    intelligence_data: dict,
    output_path: Optional[str] = None,
) -> bytes:
    """
    Generate an executive multi-page PDF report.
    
    Args:
        intelligence_data: dict containing:
            - title: str
            - summary: str
            - action_items: list of ActionItem or list of dict or str
            - key_decisions: list of MeetingDecision or list of dict or str
            - open_questions: list of str or str
            - transcript: str
            - segments: list of dict (optional)
            - metadata: dict (duration, language, audio_source, session_id, date)
        output_path: optional path to save the generated PDF file.
        
    Returns:
        bytes: Raw PDF file bytes.
    """
    title = intelligence_data.get("title", "Meeting Executive Report") or "Meeting Executive Report"
    summary = intelligence_data.get("summary", "")
    action_items = intelligence_data.get("action_items", [])
    key_decisions = intelligence_data.get("key_decisions", [])
    open_questions = intelligence_data.get("open_questions", [])
    transcript = intelligence_data.get("transcript", "")
    segments = intelligence_data.get("segments", [])
    meta = intelligence_data.get("metadata", {})

    pdf = MeetingPDF(meeting_title=title, orientation="P", unit="mm", format="A4")
    pdf.setup_fonts()
    font = pdf._font_name
    is_unicode = pdf._has_unicode_font

    def safe(txt: str) -> str:
        if not txt:
            return ""
        return str(txt) if is_unicode else sanitize_text_for_core_font(str(txt))

    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # 1. Title Banner
    pdf.set_font(font, "B" if is_unicode else "", 18)
    pdf.set_text_color(24, 43, 73)  # Deep navy
    pdf.multi_cell(0, 9, safe(title), align="L")
    pdf.ln(3)

    # Metadata bar
    pdf.set_font(font, "", 9)
    pdf.set_text_color(90, 100, 115)
    date_str = meta.get("date", datetime.now().strftime("%Y-%m-%d %H:%M"))
    duration_str = meta.get("duration", "N/A")
    lang_str = meta.get("language", "English")
    source_str = meta.get("audio_source", "Recording")
    meta_line = f"Date: {date_str}   |   Duration: {duration_str}   |   Language: {lang_str}   |   Source: {source_str}"
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 6, safe(meta_line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Decorative separator line
    pdf.set_draw_color(220, 225, 230)
    pdf.line(pdf.l_margin, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)

    # 2. Executive Summary
    if summary:
        pdf.set_font(font, "B" if is_unicode else "", 13)
        pdf.set_text_color(24, 43, 73)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 7, safe("1. Executive Summary"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        pdf.set_font(font, "", 10)
        pdf.set_text_color(40, 45, 55)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5.5, safe(summary.strip()), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    # 3. Action Items
    pdf.set_font(font, "B" if is_unicode else "", 13)
    pdf.set_text_color(24, 43, 73)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 7, safe("2. Action Items"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    if action_items:
        if isinstance(action_items, str):
            pdf.set_font(font, "", 10)
            pdf.set_text_color(40, 45, 55)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5.5, safe(action_items), new_x="LMARGIN", new_y="NEXT")
        elif isinstance(action_items, list):
            for i, item in enumerate(action_items, 1):
                # Check for ActionItem object or dict
                if hasattr(item, "task"):
                    task = item.task
                    owner = item.owner or "Unknown"
                    priority = item.priority or "Medium"
                    deadline = item.deadline or "Not specified"
                    ts = item.citation.timestamp_str if (item.citation and hasattr(item.citation, "timestamp_str")) else ""
                elif isinstance(item, dict):
                    task = item.get("task", "")
                    owner = item.get("owner", "Unknown")
                    priority = item.get("priority", "Medium")
                    deadline = item.get("deadline", "Not specified")
                    ts = item.get("timestamp", "")
                else:
                    task = str(item)
                    owner, priority, deadline, ts = "Unknown", "Medium", "Not specified", ""

                item_line = f"• Task #{i}: {task}"
                pdf.set_font(font, "B" if is_unicode else "", 10)
                pdf.set_text_color(30, 40, 50)
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 5, safe(item_line), new_x="LMARGIN", new_y="NEXT")

                pdf.set_font(font, "", 9)
                pdf.set_text_color(100, 110, 120)
                details = f"   Assignee: {owner}   |   Priority: {priority}   |   Deadline: {deadline}"
                if ts:
                    details += f"   |   Citation: [{ts}]"
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 4.5, safe(details), new_x="LMARGIN", new_y="NEXT")
                pdf.ln(1.5)
        pdf.ln(3)
    else:
        pdf.set_font(font, "", 10)
        pdf.set_text_color(100, 100, 100)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 6, safe("No action items identified in this meeting."), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    # 4. Key Decisions
    pdf.set_font(font, "B" if is_unicode else "", 13)
    pdf.set_text_color(24, 43, 73)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 7, safe("3. Key Decisions"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    if key_decisions:
        if isinstance(key_decisions, str):
            pdf.set_font(font, "", 10)
            pdf.set_text_color(40, 45, 55)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5.5, safe(key_decisions), new_x="LMARGIN", new_y="NEXT")
        elif isinstance(key_decisions, list):
            for i, dec in enumerate(key_decisions, 1):
                if hasattr(dec, "decision"):
                    decision = dec.decision
                    rationale = dec.context_rationale or "Not specified"
                    owner = dec.consensus_owner or "Unknown"
                    ts = dec.citation.timestamp_str if (dec.citation and hasattr(dec.citation, "timestamp_str")) else ""
                elif isinstance(dec, dict):
                    decision = dec.get("decision", "")
                    rationale = dec.get("context_rationale", "Not specified")
                    owner = dec.get("consensus_owner", "Unknown")
                    ts = dec.get("timestamp", "")
                else:
                    decision = str(dec)
                    rationale, owner, ts = "Not specified", "Unknown", ""

                dec_line = f"• Decision #{i}: {decision}"
                pdf.set_font(font, "B" if is_unicode else "", 10)
                pdf.set_text_color(30, 40, 50)
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 5, safe(dec_line), new_x="LMARGIN", new_y="NEXT")

                pdf.set_font(font, "", 9)
                pdf.set_text_color(100, 110, 120)
                details = f"   Rationale: {rationale}   |   Consensus: {owner}"
                if ts:
                    details += f"   |   Citation: [{ts}]"
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 4.5, safe(details), new_x="LMARGIN", new_y="NEXT")
                pdf.ln(1.5)
        pdf.ln(3)
    else:
        pdf.set_font(font, "", 10)
        pdf.set_text_color(100, 100, 100)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 6, safe("No explicit decisions identified in this meeting."), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    # 5. Open Questions
    if open_questions:
        pdf.set_font(font, "B" if is_unicode else "", 13)
        pdf.set_text_color(24, 43, 73)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 7, safe("4. Open Questions & Follow-ups"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        pdf.set_font(font, "", 10)
        pdf.set_text_color(40, 45, 55)
        if isinstance(open_questions, list):
            for q in open_questions:
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 5, safe(f"• {q}"), new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5.5, safe(str(open_questions)), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    # 6. Meeting Transcript
    if segments or transcript:
        pdf.add_page()
        pdf.set_font(font, "B" if is_unicode else "", 13)
        pdf.set_text_color(24, 43, 73)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 7, safe("5. Complete Meeting Transcript"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        if segments:
            for s in segments:
                t_start = format_timestamp(s.get("start", 0.0))
                t_end = format_timestamp(s.get("end", 0.0))
                txt = s.get("text", "").strip()
                if not txt:
                    continue
                pdf.set_font(font, "B" if is_unicode else "", 9)
                pdf.set_text_color(70, 90, 120)
                pdf.set_x(pdf.l_margin)
                pdf.cell(0, 5, safe(f"[{t_start} - {t_end}]"), new_x="LMARGIN", new_y="NEXT")
                pdf.set_font(font, "", 9.5)
                pdf.set_text_color(30, 35, 45)
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 5, safe(txt), new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2)
        else:
            pdf.set_font(font, "", 9.5)
            pdf.set_text_color(30, 35, 45)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5, safe(transcript), new_x="LMARGIN", new_y="NEXT")

    pdf_bytes = bytes(pdf.output())

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes


def generate_markdown_report(intelligence_data: dict) -> str:
    """
    Generate an executive Markdown report with metadata, structured tables, and citations.
    """
    title = intelligence_data.get("title", "Meeting Executive Report") or "Meeting Executive Report"
    summary = intelligence_data.get("summary", "")
    action_items = intelligence_data.get("action_items", [])
    key_decisions = intelligence_data.get("key_decisions", [])
    open_questions = intelligence_data.get("open_questions", [])
    transcript = intelligence_data.get("transcript", "")
    segments = intelligence_data.get("segments", [])
    meta = intelligence_data.get("metadata", {})

    date_str = meta.get("date", datetime.now().strftime("%Y-%m-%d %H:%M"))
    duration_str = meta.get("duration", "N/A")
    lang_str = meta.get("language", "English")
    source_str = meta.get("audio_source", "Recording")

    lines = [
        f"# {title}",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| **Date** | {date_str} |",
        f"| **Duration** | {duration_str} |",
        f"| **Language Mode** | {lang_str} |",
        f"| **Audio Source** | {source_str} |",
        "",
        "## 1. Executive Summary",
        f"{summary.strip() if summary else 'No summary provided.'}",
        "",
        "## 2. Action Items",
    ]

    if action_items:
        if isinstance(action_items, str):
            lines.append(action_items.strip())
        elif isinstance(action_items, list):
            lines.append("| # | Task | Assignee | Priority | Deadline | Citation |")
            lines.append("| :-: | :--- | :--- | :-: | :--- | :-: |")
            for i, item in enumerate(action_items, 1):
                if hasattr(item, "task"):
                    task = item.task.replace("|", "-")
                    owner = (item.owner or "Unknown").replace("|", "-")
                    priority = item.priority or "Medium"
                    deadline = (item.deadline or "Not specified").replace("|", "-")
                    ts = item.citation.timestamp_str if (item.citation and hasattr(item.citation, "timestamp_str")) else "-"
                elif isinstance(item, dict):
                    task = item.get("task", "").replace("|", "-")
                    owner = item.get("owner", "Unknown").replace("|", "-")
                    priority = item.get("priority", "Medium")
                    deadline = item.get("deadline", "Not specified").replace("|", "-")
                    ts = item.get("timestamp", "-")
                else:
                    task = str(item).replace("|", "-")
                    owner, priority, deadline, ts = "Unknown", "Medium", "Not specified", "-"
                lines.append(f"| {i} | {task} | {owner} | {priority} | {deadline} | `{ts}` |")
        lines.append("")
    else:
        lines.extend(["*No action items identified in this meeting.*", ""])

    lines.append("## 3. Key Decisions")
    if key_decisions:
        if isinstance(key_decisions, str):
            lines.append(key_decisions.strip())
        elif isinstance(key_decisions, list):
            lines.append("| # | Decision | Context / Rationale | Consensus / Owner | Citation |")
            lines.append("| :-: | :--- | :--- | :--- | :-: |")
            for i, dec in enumerate(key_decisions, 1):
                if hasattr(dec, "decision"):
                    decision = dec.decision.replace("|", "-")
                    rationale = (dec.context_rationale or "Not specified").replace("|", "-")
                    owner = (dec.consensus_owner or "Unknown").replace("|", "-")
                    ts = dec.citation.timestamp_str if (dec.citation and hasattr(dec.citation, "timestamp_str")) else "-"
                elif isinstance(dec, dict):
                    decision = dec.get("decision", "").replace("|", "-")
                    rationale = dec.get("context_rationale", "Not specified").replace("|", "-")
                    owner = dec.get("consensus_owner", "Unknown").replace("|", "-")
                    ts = dec.get("timestamp", "-")
                else:
                    decision = str(dec).replace("|", "-")
                    rationale, owner, ts = "Not specified", "Unknown", "-"
                lines.append(f"| {i} | {decision} | {rationale} | {owner} | `{ts}` |")
        lines.append("")
    else:
        lines.extend(["*No explicit decisions identified in this meeting.*", ""])

    if open_questions:
        lines.append("## 4. Open Questions & Follow-ups")
        if isinstance(open_questions, list):
            for q in open_questions:
                lines.append(f"- {q}")
        else:
            lines.append(str(open_questions).strip())
        lines.append("")

    lines.append("## 5. Complete Meeting Transcript")
    if segments:
        for s in segments:
            t_start = format_timestamp(s.get("start", 0.0))
            t_end = format_timestamp(s.get("end", 0.0))
            txt = s.get("text", "").strip()
            if txt:
                lines.append(f"**[{t_start} - {t_end}]** {txt}\n")
    elif transcript:
        lines.append(transcript.strip())
    else:
        lines.append("*No transcript text available.*")

    if intelligence_data.get("original_transcript"):
        lines.append("\n## Original Nepali Transcript (Devanagari)")
        lines.append(intelligence_data["original_transcript"].strip())

    return "\n".join(lines)
