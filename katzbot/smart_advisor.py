"""
katzbot/smart_advisor.py
=========================
Smart Advisor — connects research gaps to faculty and events.
The complete academic loop: Topic → Gap → Faculty → Event → Email.
"""

from katzbot.faculty        import KATZ_FACULTY, match_faculty
from katzbot.events_fetcher import fetch_events, match_events_to_topic


def get_smart_advice(topic: str, gaps_text: str = "",
                     research_question: str = "") -> dict:
    """
    Generate smart advice connecting topic/gaps to faculty + events.

    Returns:
      faculty_matches, event_matches, advice_text, action_items
    """
    search_text   = gaps_text[:500] if gaps_text else topic
    faculty_match = match_faculty(search_text, top_k=3)

    try:
        events        = fetch_events()
        event_matches = match_events_to_topic(events, search_text, top_k=3)
    except Exception as e:
        print(f"[SmartAdvisor] Events error: {e}")
        event_matches = []

    lines = []
    if faculty_match:
        lines.append("**Recommended Faculty:**")
        for f in faculty_match:
            exp = ", ".join(f["expertise"][:3])
            lines.append(
                f"• **{f['name']}** — {f['title']}\n"
                f"  Expertise: {exp}\n"
                f"  ✉ {f['email']} · {f['profile']}\n"
                f"  → {f['note']}"
            )

    if event_matches:
        lines.append("\n**Relevant Events:**")
        for ev in event_matches:
            lines.append(
                f"• **{ev.get('title','')}** — {ev.get('date','TBD')}\n"
                f"  📍 {ev.get('location','')}\n"
                f"  {ev.get('description','')[:120]}…\n"
                f"  🔗 {ev.get('url','')}"
            )

    actions = []
    for f in faculty_match[:2]:
        actions.append(f"Email {f['name']} ({f['email']}) about {topic}")
    for ev in event_matches[:2]:
        actions.append(
            f"Attend: {ev.get('title','')} — {ev.get('date','TBD')}"
        )

    return {
        "faculty_matches": faculty_match,
        "event_matches":   event_matches,
        "advice_text":     "\n".join(lines),
        "action_items":    actions,
    }


def format_email_template(student_name: str, faculty: dict,
                           topic: str, gap: str = "") -> str:
    """Generate a professional email template to a faculty member."""
    gap_line = f"\n\nMy specific focus is on: {gap}" if gap else ""
    return (
        f"Subject: Research Inquiry — {topic}\n\n"
        f"Dear {faculty['name'].replace('Prof. ', 'Professor ')},\n\n"
        f"I am a graduate student at the Katz School of Science and Health, "
        f"currently researching {topic}.{gap_line}\n\n"
        f"I came across your work in {', '.join(faculty['expertise'][:2])} and "
        f"believe your expertise closely aligns with my research direction.\n\n"
        f"I would greatly appreciate 15 minutes of your time to discuss my "
        f"research approach and receive your guidance.\n\n"
        f"Thank you for your time and consideration.\n\n"
        f"Best regards,\n"
        f"{student_name}\n"
        f"Katz School, Yeshiva University"
    )
