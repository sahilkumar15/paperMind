
"""
reviewer_agent.py — conference-style review analysis for uploaded PDFs / TeX.

This module lets KatzScholarMind analyze a paper like a conference reviewer,
identify venue fit, major weaknesses, and revision priorities.
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

VENUE_PROFILES: Dict[str, str] = {
    "NeurIPS": "Broad ML venue. Values strong novelty, significance beyond a narrow use case, clear empirical or theoretical support, and clean positioning. Weaknesses often include engineering-only contributions, narrow scope without strong insight, or incomplete ablations.",
    "ICML": "Core machine learning venue. Emphasizes technical soundness, methodological clarity, strong evidence, and contributions that advance ML understanding or capability. Often skeptical of application-only novelty unless the ML contribution is clear.",
    "ICLR": "Representation learning and modern deep learning venue. Welcomes new architectures, optimization ideas, generative modeling, and strong empirical papers. Clear motivation, ablations, and narrative clarity matter a lot.",
    "CVPR": "Top computer vision venue. Expects clear visual or perceptual relevance, strong experiments, competitive baselines, and practical or conceptual novelty for vision tasks. Presentation and empirical completeness are critical.",
    "ICCV": "Top vision venue similar to CVPR but sometimes more tolerant of conceptual or analytical work if the empirical validation remains strong. Strong vision relevance is essential.",
    "ECCV": "Top vision venue with standards close to CVPR/ICCV. Good fit for well-scoped, technically solid vision papers with meaningful empirical gains and a clear narrative.",
    "ACL": "Top NLP venue. Values language-centric significance, rigorous evaluation, strong error analysis, clear comparison to prior work, and attention to data quality and reproducibility.",
    "EMNLP": "Major NLP venue. Good fit for strong empirical NLP/system papers, resource papers, and practical methods, especially if the scope is clearly articulated and evaluation is thorough.",
    "ACM MM": "Multimedia venue. Good fit for multimodal, audio-visual, cross-media, forensics, retrieval, or applied perception systems. Values real-world relevance, strong experiments, and practical significance even if the contribution is less theoretically deep than NeurIPS/ICML.",
    "AAAI": "Broad AI venue. Welcomes practical AI systems, applied ML, reasoning, multimodal, and domain-driven contributions if the novelty and experimental support are clear.",
    "TMLR": "Journal-style ML review process. Strong fit for technically solid work that may be too specialized or revision-heavy for top conference cycles but still valuable and reproducible.",
    "Interspeech": "Speech and audio venue. Expects strong audio relevance, good human/objective metrics, and careful experimental methodology for speech, paralinguistics, or audio generation/detection tasks.",
}

REVIEWER_PERSONAS: Dict[str, str] = {
    "Balanced": "Balanced and fair reviewer. Acknowledge strengths and weaknesses proportionally. Avoid overclaiming either praise or criticism.",
    "Skeptical": "Skeptical top-tier reviewer. Probe novelty inflation, unsupported claims, missing baselines, weak ablations, and narrative overreach. Still be fair and constructive.",
    "Supportive": "Constructive but standards-aware reviewer. Emphasize what is promising and how to make the paper publishable without lowering the technical bar.",
    "Empirical-Rigorous": "Empirical reviewer focused on datasets, baselines, metrics, ablations, statistical validity, reproducibility, and fairness of comparison.",
    "Theory/Method": "Method-focused reviewer who prioritizes conceptual novelty, assumptions, correctness of formulation, and whether the claimed method contribution is intellectually substantial.",
    "Area Chair": "Meta-review style analyst. Focus on overall strengths, central risks, venue fit, and what issues are likely decisive in discussion.",
}

RUBRIC_GUIDANCE = """
Evaluate the paper on the following dimensions.
- Originality: What is genuinely new? Is the novelty conceptual, methodological, analytical, or mostly engineering/integration?
- Significance: Does the problem matter? Could this influence future research or practice? Is the scope appropriate?
- Soundness: Are the claims supported by experiments/theory? Are baselines, ablations, metrics, and comparisons adequate?
- Presentation: Is the paper well written, well structured, and reproducible enough for an expert reader to follow?
""".strip()

OUTPUT_INSTRUCTIONS = """
Return a structured markdown report with exactly these top-level headings:
# Review Snapshot
# Scorecard
# Summary
# Strengths
# Weaknesses
# Selected Venue Fit
# Strongest Venue Fit Overall
# Major Questions for Authors
# Revision Priorities
# Bottom Line

Requirements:
- In #Review Snapshot include: detected title, selected conference, reviewer persona, paper type guess.
- In #Scorecard give 1-5 ratings for Originality, Significance, Soundness, Presentation.
- In #Summary give a neutral summary of the paper and claimed contributions in your own words.
- In #Strengths and #Weaknesses use concise bullets with evidence from the paper when possible.
- In #Selected Venue Fit state Strong / Moderate / Weak fit and explain why under the selected conference norms.
- In #Strongest Venue Fit Overall rank the top 3 best venues from the provided list and explain why.
- In #Major Questions for Authors provide 3-5 numbered questions whose answers could materially change the evaluation.
- In #Revision Priorities provide a prioritized numbered list of the most important fixes.
- In #Bottom Line provide a short verdict that does NOT predict acceptance probability and does NOT claim certainty.
- Do not say you cannot access the internet. Base the analysis only on the uploaded paper text.
""".strip()


def _strip_tex_commands(text: str) -> str:
    text = re.sub(r"(?m)^\s*%.*$", "", text)
    text = re.sub(r"\\begin\{abstract\}", "\nAbstract\n", text, flags=re.I)
    text = re.sub(r"\\end\{abstract\}", "\n", text, flags=re.I)
    text = re.sub(r"\\section\*?\{([^}]*)\}", lambda m: f"\n\n{m.group(1).strip()}\n", text)
    text = re.sub(r"\\subsection\*?\{([^}]*)\}", lambda m: f"\n\n{m.group(1).strip()}\n", text)
    text = re.sub(r"\\subsubsection\*?\{([^}]*)\}", lambda m: f"\n\n{m.group(1).strip()}\n", text)
    text = re.sub(r"\\caption\{([^}]*)\}", lambda m: f"\nCaption: {m.group(1).strip()}\n", text)
    text = re.sub(r"\\title\{([^}]*)\}", lambda m: f"\nTitle: {m.group(1).strip()}\n", text)
    text = re.sub(r"\\author\{([^}]*)\}", "\n", text)
    text = re.sub(r"\\cite[t|p]?\{[^}]*\}", "[cite]", text)
    text = re.sub(r"\\ref\{[^}]*\}", "[ref]", text)
    text = re.sub(r"\\label\{[^}]*\}", "", text)
    text = re.sub(r"\\url\{([^}]*)\}", lambda m: m.group(1), text)
    text = re.sub(r"\\href\{([^}]*)\}\{([^}]*)\}", lambda m: f"{m.group(2)} ({m.group(1)})", text)
    text = re.sub(r"\\[A-Za-z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", text)
    text = re.sub(r"\$[^$]*\$", " [math] ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_pdf_text(file_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except Exception as e:
            raise RuntimeError("Install pypdf to analyze uploaded PDFs.") from e

    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    text = "\n\n".join(pages)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_upload(uploaded_file) -> Tuple[str, str]:
    """Return (extracted_text, detected_title)."""
    name = getattr(uploaded_file, "name", "uploaded_paper")
    suffix = os.path.splitext(name)[1].lower()
    data = uploaded_file.read()
    if not data:
        raise ValueError("Uploaded file is empty.")

    if suffix == ".pdf":
        text = _extract_pdf_text(data)
    elif suffix in {".tex", ".txt", ".md"}:
        raw = data.decode("utf-8", errors="ignore")
        text = _strip_tex_commands(raw) if suffix == ".tex" else raw
    else:
        raise ValueError("Unsupported file type. Upload PDF, TEX, TXT, or MD.")

    if len(text.strip()) < 500:
        raise ValueError("Could not extract enough text from the uploaded file.")

    title = detect_title(text, fallback=name)
    return text, title


def detect_title(text: str, fallback: str = "Uploaded Paper") -> str:
    m = re.search(r"Title:\s*(.+?)($|\n)", text, flags=re.I)
    if m:
        return m.group(1).strip()[:200]

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[:12]:
        if 8 < len(ln) < 220 and not ln.lower().startswith(("abstract", "introduction", "authors", "1 ")):
            if len(ln.split()) >= 4:
                return ln[:200]
    return os.path.splitext(fallback)[0]


def _paper_type_guess(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ["theorem", "proof", "lemma", "corollary"]):
        return "theory/method paper"
    if any(k in low for k in ["benchmark", "dataset", "leaderboard"]):
        return "benchmark/resource paper"
    if any(k in low for k in ["system", "pipeline", "deployment", "platform"]):
        return "systems/application paper"
    return "empirical method paper"


def _truncate_text(text: str, max_chars: int = 55000) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    if len(text) <= max_chars:
        return text
    head = text[:35000]
    tail = text[-16000:]
    return head + "\n\n[... middle truncated for length ...]\n\n" + tail


def analyze_paper_text(
    paper_text: str,
    selected_conference: str,
    reviewer_persona: str,
    paper_title: str = "Uploaded Paper",
) -> str:
    from llm_config import get_openai_client, get_model_name

    venue_cards = "\n".join([f"- {k}: {v}" for k, v in VENUE_PROFILES.items()])
    persona = REVIEWER_PERSONAS.get(reviewer_persona, REVIEWER_PERSONAS["Balanced"])
    selected_profile = VENUE_PROFILES.get(selected_conference, "Use general top-tier reviewer norms.")
    paper_type = _paper_type_guess(paper_text)

    system_prompt = f"""
You are ReviewerAgent inside KatzScholarMind.
Your job is to analyze a paper like a careful conference reviewer and venue-fit advisor.

Selected conference: {selected_conference}
Conference profile: {selected_profile}
Reviewer persona: {reviewer_persona}
Persona guidance: {persona}
Detected paper title: {paper_title}
Detected paper type: {paper_type}

General rubric:
{RUBRIC_GUIDANCE}

Available venues for cross-fit comparison:
{venue_cards}

{OUTPUT_INSTRUCTIONS}
""".strip()

    user_prompt = f"""
Analyze this paper for the selected conference and also determine the strongest overall venue fit.
Paper text below:

{_truncate_text(paper_text)}
""".strip()

    client = get_openai_client()
    model = get_model_name()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=2200,
    )
    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("ReviewerAgent returned an empty response.")
    return content
