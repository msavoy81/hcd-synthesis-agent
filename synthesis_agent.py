#!/usr/bin/env python3
"""
Synthesis Agent crew for FutureLab Team 3 (Workbook B, pages 7 and 8).

Three agents, each with ONE job, and a human approval between each step:

    1. Clusterer       notes    -> rated clusters
    2. Insight Writer  clusters -> 3 "I wonder if this means..." insights
    3. HMW Framer      insights -> How Might We questions

After every step, plain Python code (no AI) checks the work:
invented note IDs, made-up quotes, ratings that break the source-counting
rule, solutions sneaking into insights or HMWs, and anything resting on D17.

Commands (run them in order, from inside the "Synthesis Agent" folder):

    python3 synthesis_agent.py cluster                 # Agent 1 runs
    python3 synthesis_agent.py approve clusters        # human says yes
    python3 synthesis_agent.py insights                # Agent 2 runs
    python3 synthesis_agent.py approve insights        # human says yes
    python3 synthesis_agent.py hmw                     # Agent 3 runs
    python3 synthesis_agent.py report                  # one combined report

Useful options:

    --run v2                   keep a second run separate (for "change one
                               instruction and run it again" on page 8)
    --feedback "..."           re-run a step with one instruction from you
    --keep C1,C3               (with approve) pass only these items forward
    --model NAME               choose a different Claude model
"""

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"
RUNS = HERE / "runs"
EVIDENCE = HERE / "evidence_packet.csv"
DEFAULT_MODEL = "claude-opus-5-5"

RATINGS = ["Emerging", "Corroborated", "Corroborated (joint interview only)", "Contradictory"]
JOINT_PEOPLE = {"Instructor A", "Instructor B"}

# Words that usually mean a solution has slipped in.
SOLUTION_WORDS = [r"\bshould\b", r"\bneeds? to\b", r"\bmust\b", r"\bcreate\b", r"\bimplement\b",
                  r"\badd\b", r"\bhire\b", r"\bintroduce\b", r"\blaunch\b", r"\brequire\b"]
# Words that usually make a How Might We too narrow (a named fix) or too broad.
HMW_NARROW_WORDS = [r"\bthrough\b", r"\bby\b", r"\bapp\b", r"\bworkshop\b", r"\bchatbot\b",
                    r"\btext(s|ing)?\b", r"\bemail\b", r"\bplatform\b", r"\bsoftware\b",
                    r"\bdashboard\b", r"\bhire\b"]
HMW_BROAD_WORDS = [r"\bimprove\b", r"\bbetter\b"]


# ---------------------------------------------------------------- setup ----

def load_api_key():
    """Get the key from the environment, or from a PRIVATE file named by ANTHROPIC_API_KEY_FILE.

    The key is never stored in this folder, because this folder is shared with the team.
    The key is never printed.
    """
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    key_file = os.environ.get("ANTHROPIC_API_KEY_FILE", "").strip()
    if not key and key_file and Path(key_file).exists():
        # Works for plain text and for files TextEdit saved as rich text (.rtf).
        found = re.search(r"sk-[A-Za-z0-9_\-]{20,}", Path(key_file).read_text(errors="ignore"))
        if found:
            key = found.group(0)
    if not key:
        sys.exit("No API key found. Keep your key in a private file OUTSIDE the shared team folder,\n"
                 "then run with:  ANTHROPIC_API_KEY_FILE=/path/to/your/key.txt python3 synthesis_agent.py ...")
    return key


def load_notes():
    """Load the evidence packet. Hidden columns (team labels, prep notes) are NOT sent to the AI."""
    notes = {}
    with open(EVIDENCE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            notes[row["note_id"]] = {
                "id": row["note_id"],
                "speaker": row["speaker_role"],
                "session": row["session"][:2],
                "type": row["note_type"],
                "text": row["note_text"],
                "uncertain": "UNCERTAIN" in row.get("prep_flags", ""),
            }
    return notes


def notes_for_prompt(notes):
    lines = ["THE 26 NOTES (format: ID | speaker | session | quote or paraphrase | text)"]
    for n in notes.values():
        flag = " [UNCERTAIN: do not build on this note alone]" if n["uncertain"] else ""
        lines.append(f'{n["id"]} | {n["speaker"]} | {n["session"]} | {n["type"]} | {n["text"]}{flag}')
    return "\n".join(lines)


def run_dir(name):
    d = RUNS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def system_prompt(agent_file):
    return (PROMPTS / "shared_charter.md").read_text() + "\n\n" + (PROMPTS / agent_file).read_text()


# ------------------------------------------------------- output formats ----
# Each agent must answer by "filling in a form" (a tool). This forces a fixed
# structure, so the code can check every field. Think of it as a scorecard
# the interviewer has to complete, instead of free-form notes.

CLUSTER_TOOL = {
    "name": "submit_clusters",
    "description": "Submit the rated clusters for human review.",
    "input_schema": {
        "type": "object",
        "properties": {
            "clusters": {"type": "array", "items": {"type": "object", "properties": {
                "id": {"type": "string", "description": "C1, C2, ..."},
                "headline": {"type": "string"},
                "note_ids": {"type": "array", "items": {"type": "string"}},
                "rating": {"type": "string", "enum": RATINGS},
                "supporting_people": {"type": "array", "items": {"type": "string"}},
                "rating_reason": {"type": "string"},
                "key_quote": {"type": "string", "description": "Copied word for word from one of the cluster's notes."},
                "sides": {"type": "array", "description": "Only for Contradictory clusters.",
                          "items": {"type": "object", "properties": {
                              "position": {"type": "string"},
                              "note_ids": {"type": "array", "items": {"type": "string"}}},
                              "required": ["position", "note_ids"]}},
            }, "required": ["id", "headline", "note_ids", "rating", "supporting_people", "rating_reason", "key_quote"]}},
            "unused_notes": {"type": "array", "items": {"type": "object", "properties": {
                "note_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["note_id", "reason"]}},
            "unclear_notes": {"type": "array", "items": {"type": "object", "properties": {
                "note_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["note_id", "reason"]}},
            "evidence_gaps": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["clusters", "unused_notes", "evidence_gaps"],
    },
}

TEST = {"type": "object", "properties": {"pass": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["pass", "reason"]}

INSIGHT_TOOL = {
    "name": "submit_insights",
    "description": "Submit exactly 3 insights for human review.",
    "input_schema": {
        "type": "object",
        "properties": {
            "insights": {"type": "array", "items": {"type": "object", "properties": {
                "id": {"type": "string", "description": "I1, I2, I3"},
                "text": {"type": "string"},
                "cluster_ids": {"type": "array", "items": {"type": "string"}},
                "note_ids": {"type": "array", "items": {"type": "string"}},
                "rating": {"type": "string", "enum": RATINGS},
                "supporting_people": {"type": "array", "items": {"type": "string"}},
                "alternative_reading": {"type": "string"},
                "insight_test": {"type": "object", "properties": {
                    "evidence": TEST, "why_or_how": TEST, "huh": TEST, "solution_free": TEST},
                    "required": ["evidence", "why_or_how", "huh", "solution_free"]},
            }, "required": ["id", "text", "cluster_ids", "note_ids", "rating", "supporting_people",
                            "alternative_reading", "insight_test"]}},
            "evidence_gaps": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["insights", "evidence_gaps"],
    },
}

HMW_TOOL = {
    "name": "submit_hmws",
    "description": "Submit one How Might We question per approved insight.",
    "input_schema": {
        "type": "object",
        "properties": {
            "hmws": {"type": "array", "items": {"type": "object", "properties": {
                "id": {"type": "string", "description": "H1, H2, ..."},
                "insight_id": {"type": "string"},
                "text": {"type": "string"},
                "goldilocks": {"type": "string", "enum": ["Too broad", "Too narrow", "Just right"]},
                "goldilocks_reason": {"type": "string"},
            }, "required": ["id", "insight_id", "text", "goldilocks", "goldilocks_reason"]}},
        },
        "required": ["hmws"],
    },
}


# ----------------------------------------------------------- the AI call ----

def call_agent(model, agent_file, tool, user_text):
    try:
        import anthropic
    except ImportError:
        sys.exit("The anthropic library is not installed. Run:  python3 -m pip install anthropic")
    client = anthropic.Anthropic(api_key=load_api_key())
    ask = (user_text + f"\n\nWhen you are finished, submit your work by calling the {tool['name']} tool "
           "exactly once. Do not answer in plain text.")
    messages = [{"role": "user", "content": ask}]
    for attempt in range(2):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=16000,
                system=system_prompt(agent_file),
                tools=[tool],
                tool_choice={"type": "auto"},
                messages=messages,
            )
        except anthropic.AuthenticationError:
            sys.exit("The API key was rejected (or the network blocked the call). Check the key file.")
        except anthropic.NotFoundError:
            sys.exit(f'Model "{model}" was not found. Try:  --model claude-sonnet-5-5')
        for block in resp.content:
            if block.type == "tool_use":
                usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
                return block.input, usage
        # The agent answered in plain text instead of filling in the form: remind it once.
        messages = messages + [{"role": "assistant", "content": resp.content},
                               {"role": "user", "content": f"Please submit your work now by calling the {tool['name']} tool."}]
    sys.exit("The agent did not return the expected form. Try running the step again.")


def stamp(model, agent_file, usage, feedback):
    return {"model": model, "ran_at": dt.datetime.now().isoformat(timespec="seconds"),
            "prompt_files": ["shared_charter.md", agent_file], "usage": usage,
            "feedback_given": feedback or None, "approved": False}


# ------------------------------------------------------------ the checks ----
# Plain code, no AI. These are the guardrails page 7 asks for, enforced
# instead of just requested.

def plain(pattern):
    """Turn a search pattern like \\bshould\\b back into the word should."""
    return pattern.replace("\\b", "")


def norm(s):
    s = s.lower().replace("“", '"').replace("”", '"').replace("’", "'").replace("…", "...")
    s = s.strip().strip('"').strip()
    return re.sub(r"\s+", " ", s)


def people_for(ids, notes):
    return sorted({notes[i]["speaker"] for i in ids if i in notes})


def expected_rating(ids, notes, claimed):
    people = set(people_for(ids, notes))
    if claimed == "Contradictory":
        return "Contradictory" if len(people) >= 2 else "Emerging"
    if len(people) <= 1:
        return "Emerging"
    if people <= JOINT_PEOPLE:
        return "Corroborated (joint interview only)"
    return "Corroborated"


def check_common(item, ids, notes, label, flags):
    unknown = [i for i in ids if i not in notes]
    if unknown:
        flags.append(f"FLAG {label}: cites note IDs that do not exist: {', '.join(unknown)} (invented evidence)")
    known = [i for i in ids if i in notes]
    if known and all(notes[i]["uncertain"] for i in known):
        flags.append(f"FLAG {label}: rests only on an UNCERTAIN note ({', '.join(known)})")
    exp = expected_rating(known, notes, item["rating"])
    if exp != item["rating"]:
        flags.append(f'FLAG {label}: rated "{item["rating"]}" but the notes come from '
                     f'{", ".join(people_for(known, notes)) or "nobody"}, so the rule says "{exp}"')
    claimed_people = {p.strip() for p in item.get("supporting_people", [])}
    actual = set(people_for(known, notes))
    if claimed_people and claimed_people != actual:
        flags.append(f"CHECK {label}: says it is supported by {', '.join(sorted(claimed_people))}, "
                     f"but its notes are from {', '.join(sorted(actual))}")


def check_clusters(data, notes):
    flags = []
    placed = {}
    for c in data["clusters"]:
        label = c["id"]
        check_common(c, c["note_ids"], notes, label, flags)
        for i in c["note_ids"]:
            placed.setdefault(i, []).append(c["id"])
        if any(re.search(w, c["headline"], re.I) for w in SOLUTION_WORDS):
            flags.append(f'FLAG {label}: headline sounds like a solution: "{c["headline"]}"')
        q = norm(c.get("key_quote", ""))
        cited = [notes[i]["text"] for i in c["note_ids"] if i in notes]
        if q and not any(q in norm(t) for t in cited):
            if any(q in norm(n["text"]) for n in notes.values()):
                flags.append(f"CHECK {label}: key quote is real but comes from a note outside this cluster")
            else:
                flags.append(f'FLAG {label}: key quote is not word for word in any note: "{c["key_quote"]}"')
        if c["rating"] == "Contradictory":
            sides = c.get("sides") or []
            if len(sides) < 2:
                flags.append(f"FLAG {label}: rated Contradictory but does not show two sides")
            else:
                side_people = [set(people_for(s["note_ids"], notes)) for s in sides]
                if len(set().union(*side_people)) < 2:
                    flags.append(f"FLAG {label}: both sides come from the same person")
    accounted = set(placed) | {u["note_id"] for u in data.get("unused_notes", [])} \
        | {u["note_id"] for u in data.get("unclear_notes", []) or []}
    missing = [i for i in notes if i not in accounted]
    if missing:
        flags.append(f"CHECK: notes neither clustered nor listed as unused: {', '.join(missing)}")
    overlap = {i: cs for i, cs in placed.items() if len(cs) > 1}
    if overlap:
        flags.append("INFO: notes in more than one cluster: "
                     + "; ".join(f"{i} in {'+'.join(cs)}" for i, cs in overlap.items()))
    return flags


def check_insights(data, clusters, notes):
    flags = []
    cluster_notes = {c["id"]: set(c["note_ids"]) for c in clusters}
    if len(data["insights"]) != 3:
        flags.append(f'FLAG: asked for exactly 3 insights, got {len(data["insights"])}')
    for ins in data["insights"]:
        label = ins["id"]
        check_common(ins, ins["note_ids"], notes, label, flags)
        if not ins["text"].lower().startswith("i wonder if"):
            flags.append(f'CHECK {label}: does not start with "I wonder if this means..."')
        hits = [plain(w) for w in SOLUTION_WORDS if re.search(w, ins["text"], re.I)]
        if hits:
            flags.append(f"FLAG {label}: solution words in the insight: {', '.join(hits)}")
        if not re.search(r"\b(because|so|since|which|why|how)\b", ins["text"], re.I):
            flags.append(f"CHECK {label}: no 'because', 'so', 'why' or 'how'. Make sure it explains why, not what.")
        bad_c = [c for c in ins["cluster_ids"] if c not in cluster_notes]
        if bad_c:
            flags.append(f"FLAG {label}: cites clusters that were not approved: {', '.join(bad_c)}")
        allowed = set().union(*[cluster_notes.get(c, set()) for c in ins["cluster_ids"]]) if ins["cluster_ids"] else set()
        outside = [i for i in ins["note_ids"] if i in notes and i not in allowed]
        if outside:
            flags.append(f"INFO {label}: uses notes outside its clusters: {', '.join(outside)}")
        failed = [k for k, v in ins["insight_test"].items() if not v.get("pass")]
        if failed:
            flags.append(f"INFO {label}: the agent says it fails its own test on: {', '.join(failed)}")
    return flags


def check_hmws(data, insights):
    flags = []
    ids = {i["id"] for i in insights}
    covered = {h["insight_id"] for h in data["hmws"]}
    if ids - covered:
        flags.append(f"FLAG: no HMW for insight(s) {', '.join(sorted(ids - covered))}")
    for h in data["hmws"]:
        label, t = h["id"], h["text"]
        if h["insight_id"] not in ids:
            flags.append(f"FLAG {label}: tied to an insight that was not approved ({h['insight_id']})")
        if not t.lower().startswith("how might we"):
            flags.append(f'CHECK {label}: does not start with "How might we"')
        if not t.strip().endswith("?"):
            flags.append(f"CHECK {label}: is not phrased as a question")
        narrow = [plain(w) for w in HMW_NARROW_WORDS if re.search(w, t, re.I)]
        if narrow:
            flags.append(f"CHECK {label}: may name a solution (too narrow): {', '.join(narrow)}")
        broad = [plain(w) for w in HMW_BROAD_WORDS if re.search(w, t, re.I)]
        if broad:
            flags.append(f"CHECK {label}: may be too broad: {', '.join(broad)}")
    return flags


# ------------------------------------------------------ readable output ----

def md_clusters(data, flags):
    out = ["# Step 1: Clusters (Agent 1, the Clusterer)", "",
           "| Cluster | Headline | Notes | Rating | Who supports it | Key quote |", "| --- | --- | --- | --- | --- | --- |"]
    for c in data["clusters"]:
        out.append(f'| {c["id"]} | {c["headline"]} | {", ".join(c["note_ids"])} | {c["rating"]} | '
                   f'{", ".join(c["supporting_people"])} | {c["key_quote"]} |')
    for c in data["clusters"]:
        if c.get("sides"):
            out += ["", f'**{c["id"]} sides:**'] + [f'- {s["position"]} ({", ".join(s["note_ids"])})' for s in c["sides"]]
    out += ["", "**Why each rating:**"] + [f'- {c["id"]}: {c["rating_reason"]}' for c in data["clusters"]]
    out += ["", "**Unused notes:**"] + [f'- {u["note_id"]}: {u["reason"]}' for u in data.get("unused_notes", [])]
    if data.get("unclear_notes"):
        out += ["", "**Unclear notes:**"] + [f'- {u["note_id"]}: {u["reason"]}' for u in data["unclear_notes"]]
    out += ["", "**Evidence gaps:**"] + [f"- {g}" for g in data.get("evidence_gaps", [])]
    return "\n".join(out + md_flags(flags))


def md_insights(data, flags):
    out = ["# Step 2: Insights (Agent 2, the Insight Writer)", ""]
    for i in data["insights"]:
        t = i["insight_test"]
        out += [f'## {i["id"]}: {i["text"]}', "",
                f'- Clusters: {", ".join(i["cluster_ids"])} | Notes: {", ".join(i["note_ids"])}',
                f'- Rating: {i["rating"]} ({", ".join(i["supporting_people"])})',
                f'- Alternative reading: {i["alternative_reading"]}',
                "- Insight Test: " + "; ".join(f'{k}: {"pass" if v["pass"] else "FAIL"} ({v["reason"]})' for k, v in t.items()),
                ""]
    out += ["**Evidence gaps:**"] + [f"- {g}" for g in data.get("evidence_gaps", [])]
    return "\n".join(out + md_flags(flags))


def md_hmws(data, flags):
    out = ["# Step 3: How Might We (Agent 3, the HMW Framer)", "",
           "| # | How might we...? | From | Goldilocks | Why |", "| --- | --- | --- | --- | --- |"]
    for h in data["hmws"]:
        out.append(f'| {h["id"]} | {h["text"]} | {h["insight_id"]} | {h["goldilocks"]} | {h["goldilocks_reason"]} |')
    return "\n".join(out + md_flags(flags))


def md_flags(flags):
    out = ["", "## Code checks (no AI)", ""]
    if not flags:
        return out + ["No problems found by the automatic checks. That does not mean the work is right; it means it broke none of the rules the code can see."]
    out += ["FLAG = breaks a rule. CHECK = a human should look. INFO = worth knowing.", ""]
    return out + [f"- {f}" for f in flags]


def show(path_md, flags):
    print(f"\nSaved: {path_md.relative_to(HERE)}")
    print(f"Code checks: {sum(f.startswith('FLAG') for f in flags)} FLAG, "
          f"{sum(f.startswith('CHECK') for f in flags)} CHECK, {sum(f.startswith('INFO') for f in flags)} INFO")
    for f in flags:
        print("  - " + f)


# ------------------------------------------------------------ the steps ----

def feedback_block(feedback, previous):
    if not feedback:
        return ""
    prev = f"\n\nYOUR PREVIOUS ATTEMPT:\n{json.dumps(previous, indent=2)}" if previous else ""
    return f"{prev}\n\nFEEDBACK FROM THE HUMAN REVIEWER (apply this one change, keep the rest of the rules):\n{feedback}"


def step_cluster(args, notes):
    d = run_dir(args.run)
    prev = read_json(d / "1_clusters.json")
    text = notes_for_prompt(notes) + "\n\nDo your one job: cluster and rate these notes." + feedback_block(args.feedback, prev)
    data, usage = call_agent(args.model, "clusterer.md", CLUSTER_TOOL, text)
    data["_meta"] = stamp(args.model, "clusterer.md", usage, args.feedback)
    flags = check_clusters(data, notes)
    data["_checks"] = flags
    write_json(d / "1_clusters.json", data)
    (d / "1_clusters.md").write_text(md_clusters(data, flags))
    show(d / "1_clusters.md", flags)
    print("\nNext: read it, then  python3 synthesis_agent.py approve clusters"
          f"{'' if args.run == 'v1' else ' --run ' + args.run}"
          "\n  or re-run with one change:  python3 synthesis_agent.py cluster --feedback \"...\"")


def approved_items(path, key, what):
    data = read_json(path)
    if not data:
        sys.exit(f"No {what} yet. Run that step first.")
    if not data["_meta"].get("approved"):
        sys.exit(f"STOP: the {what} have not been approved by a human yet.\n"
                 f"Read them, then run:  python3 synthesis_agent.py approve {what}")
    keep = data["_meta"].get("kept")
    return [x for x in data[key] if not keep or x["id"] in keep]


def step_insights(args, notes):
    d = run_dir(args.run)
    clusters = approved_items(d / "1_clusters.json", "clusters", "clusters")
    prev = read_json(d / "2_insights.json")
    used = sorted({i for c in clusters for i in c["note_ids"]}, key=lambda x: list(notes).index(x) if x in notes else 999)
    text = (notes_for_prompt(notes) + "\n\nHUMAN-APPROVED CLUSTERS:\n" + json.dumps(clusters, indent=2)
            + f"\n\nNotes inside approved clusters: {', '.join(used)}"
            + "\n\nDo your one job: write exactly 3 insights." + feedback_block(args.feedback, prev))
    data, usage = call_agent(args.model, "insight_writer.md", INSIGHT_TOOL, text)
    data["_meta"] = stamp(args.model, "insight_writer.md", usage, args.feedback)
    flags = check_insights(data, clusters, notes)
    data["_checks"] = flags
    write_json(d / "2_insights.json", data)
    (d / "2_insights.md").write_text(md_insights(data, flags))
    show(d / "2_insights.md", flags)
    print("\nNext: read it, then  python3 synthesis_agent.py approve insights")


def step_hmw(args, notes):
    d = run_dir(args.run)
    insights = approved_items(d / "2_insights.json", "insights", "insights")
    text = ("HUMAN-APPROVED INSIGHTS:\n" + json.dumps(insights, indent=2)
            + "\n\nDo your one job: one How Might We question per insight." + feedback_block(args.feedback, read_json(d / "3_hmws.json")))
    data, usage = call_agent(args.model, "hmw_framer.md", HMW_TOOL, text)
    data["_meta"] = stamp(args.model, "hmw_framer.md", usage, args.feedback)
    flags = check_hmws(data, insights)
    data["_checks"] = flags
    write_json(d / "3_hmws.json", data)
    (d / "3_hmws.md").write_text(md_hmws(data, flags))
    show(d / "3_hmws.md", flags)
    print("\nNext:  python3 synthesis_agent.py report")


def step_approve(args, notes):
    d = run_dir(args.run)
    files = {"clusters": "1_clusters.json", "insights": "2_insights.json"}
    if args.what not in files:
        sys.exit("Use:  approve clusters   or   approve insights")
    path = d / files[args.what]
    data = read_json(path)
    if not data:
        sys.exit(f"Nothing to approve: run the {args.what} step first.")
    key = args.what
    keep = [k.strip() for k in args.keep.split(",")] if args.keep else None
    if keep:
        unknown = [k for k in keep if k not in {x["id"] for x in data[key]}]
        if unknown:
            sys.exit(f"Unknown IDs in --keep: {', '.join(unknown)}")
    data["_meta"].update({"approved": True, "approved_at": dt.datetime.now().isoformat(timespec="seconds"),
                          "kept": keep})
    write_json(path, data)
    print(f"Approved {args.what}" + (f" (keeping only {', '.join(keep)})" if keep else " (all)") + ".")


def step_report(args, notes):
    d = run_dir(args.run)
    parts = [f"# Synthesis Agent report, run {args.run}", "",
             "Evidence: 26 de-identified notes (evidence_packet.csv). Human approval was required between every step.", ""]
    for name, fn in [("1_clusters", md_clusters), ("2_insights", md_insights), ("3_hmws", md_hmws)]:
        data = read_json(d / f"{name}.json")
        if data:
            m = data["_meta"]
            parts += [fn(data, data.get("_checks", [])), "",
                      f"_Model {m['model']}, run {m['ran_at']}, approved: {m.get('approved')} "
                      f"{('(kept ' + ', '.join(m['kept']) + ')') if m.get('kept') else ''}"
                      f"{', feedback: ' + m['feedback_given'] if m.get('feedback_given') else ''}_", ""]
    for f in ["shared_charter.md", "clusterer.md", "insight_writer.md", "hmw_framer.md"]:
        (d / f"prompt_snapshot_{f}").write_text((PROMPTS / f).read_text())
    (d / "report.md").write_text("\n".join(parts))
    print(f"Saved: {(d / 'report.md').relative_to(HERE)} (plus a snapshot of the prompts used)")


def step_check(args, notes):
    """Re-run the code checks on saved results (no AI call)."""
    d = run_dir(args.run)
    c, i, h = (read_json(d / f) for f in ["1_clusters.json", "2_insights.json", "3_hmws.json"])
    if c:
        show(d / "1_clusters.md", check_clusters(c, notes))
    def kept(data, key):
        keep = data["_meta"].get("kept")
        return [x for x in data[key] if not keep or x["id"] in keep]
    if c and i:
        show(d / "2_insights.md", check_insights(i, kept(c, "clusters"), notes))
    if i and h:
        show(d / "3_hmws.md", check_hmws(h, kept(i, "insights")))


def main():
    p = argparse.ArgumentParser(description="Three-agent synthesis crew with human checkpoints.")
    p.add_argument("step", choices=["cluster", "approve", "insights", "hmw", "report", "check"])
    p.add_argument("what", nargs="?", help="for approve: clusters or insights")
    p.add_argument("--run", default="v1")
    p.add_argument("--feedback")
    p.add_argument("--keep")
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()
    notes = load_notes()
    {"cluster": step_cluster, "approve": step_approve, "insights": step_insights,
     "hmw": step_hmw, "report": step_report, "check": step_check}[args.step](args, notes)


if __name__ == "__main__":
    main()
