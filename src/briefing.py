import os
import json


def build_facts(zones, plan, metrics):
    cols = ["zone_id", "flood_frac", "population", "road_access", "hospital_km"]
    return {"zones": zones[cols].to_dict("records"), "plan": plan.to_dict("records"), "metrics": metrics}


def template_briefing(f):
    top = sorted(f["zones"], key=lambda r: -r["flood_frac"] * r["population"])[:3]
    parts = [f"{r['zone_id']} ({r['flood_frac']:.0%} flooded, about "
             f"{int(r['population'] * r['flood_frac']):,} people exposed)" for r in top]
    return ("Highest exposure: " + "; ".join(parts) +
            f". The plan covers {f['metrics']['coverage']:.0%} of estimated need.")


def llm_briefing(f):
    if not os.getenv("ANTHROPIC_API_KEY"):
        return template_briefing(f)
    try:
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model="claude-sonnet-5-5", max_tokens=500,
            system=("You write concise flood situation briefings for emergency coordinators. "
                    "Use only the numbers in the data. Exposed people is population times flood "
                    "fraction, an estimate. State the main trade-offs and what to verify. No invented facts."),
            messages=[{"role": "user", "content": json.dumps(f)}])
        return msg.content[0].text
    except Exception:
        return template_briefing(f)