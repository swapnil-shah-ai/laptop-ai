"""
Laptop AI - Phase 7: Multi-agent decision flow

What this does:
    You type a business decision. Two different AI models debate it
    from opposing perspectives. A judge reads the full debate and
    delivers a verdict. One file. Runs locally.

What changed from Phase 5/6:
    Phase 5: one agent calls tools.
    Phase 6: one agent evaluates itself.
    Phase 7: agents talk to EACH OTHER. Two genuinely different models
    (Mistral 7B and Phi3 3.8B) produce genuinely different reasoning.
    The debate emerges from the interaction, not from a plan.

What you learn:
    - Agent identity: LLM generates roles dynamically based on the topic
    - Agent communication: one agent's output becomes another's input
    - Convergence: tracking whether agents agree or disagree over rounds
    - Judge pattern: a third call synthesizes the debate (Microsoft Critique)

Usage:
    python debate.py                                              # interactive
    python debate.py "should we build or buy our AI stack"        # single topic
    python debate.py "should we build or buy" --rounds 5          # more rounds
"""

import sys
import os
import time
import argparse
from datetime import datetime

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434"

# Two genuinely different models for genuinely different reasoning
AGENT_A_MODEL = "mistral:latest"     # 7B parameters - deeper reasoning
AGENT_B_MODEL = "phi3:mini"          # 3.8B parameters - different perspective
JUDGE_MODEL = "phi3:mini"            # judge uses Phi3

DEFAULT_ROUNDS = 3
MAX_ROUNDS = 6


# -------------------------------------------------------------------
# HELPER - talk to Ollama
# -------------------------------------------------------------------

def call_ollama(model, prompt, max_tokens=400, temperature=0.4):
    import requests
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens}
            },
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        error_type = type(e).__name__
        if "ConnectionError" in error_type:
            return "[ERROR] Cannot connect to Ollama. Run: ollama serve"
        elif "Timeout" in error_type:
            return "[ERROR] Model timed out. Try a shorter topic."
        else:
            return f"[ERROR] {str(e)}"


def check_ollama():
    import requests
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# -------------------------------------------------------------------
# CONCEPT 1: AGENT IDENTITY
# The LLM generates roles dynamically based on the topic
# -------------------------------------------------------------------

ROLE_PROMPT = """Given this business decision: "{topic}"

Generate exactly 2 opposing perspectives for a debate.

Reply EXACTLY in this format (no other text):
ROLE A: [role name] | STANCE: [one sentence stance]
ROLE B: [role name] | STANCE: [one sentence stance]

Example:
ROLE A: enterprise CTO | STANCE: we should build our own AI stack for full control
ROLE B: CFO | STANCE: we should buy a SaaS solution for speed and cost savings
"""


def generate_roles(topic):
    """
    CONCEPT 1: Agent identity.
    The LLM decides who the debaters should be.
    Different topic = different roles. Nothing hardcoded.
    """
    print(f"  Generating debate roles...")
    response = call_ollama(JUDGE_MODEL, ROLE_PROMPT.format(topic=topic),
                           max_tokens=200, temperature=0.3)

    if response.startswith("[ERROR]"):
        print(f"  {response}")
        return None, None

    role_a = {"name": "Advocate", "stance": "argues for"}
    role_b = {"name": "Challenger", "stance": "argues against"}

    for line in response.strip().split("\n"):
        line = line.strip()
        if not line.upper().startswith("ROLE"):
            continue
        if "|" not in line:
            continue

        parts = line.split("|", 1)

        # Parse role name
        role_part = parts[0].strip()
        if ":" in role_part:
            role_name = role_part.split(":", 1)[1].strip()
        else:
            role_name = role_part

        # Parse stance
        stance_part = parts[1].strip()
        if ":" in stance_part:
            stance = stance_part.split(":", 1)[1].strip()
        else:
            stance = stance_part

        if "A" in parts[0].upper():
            role_a = {"name": role_name, "stance": stance}
        elif "B" in parts[0].upper():
            role_b = {"name": role_name, "stance": stance}

    return role_a, role_b


# -------------------------------------------------------------------
# CONCEPT 2: AGENT COMMUNICATION
# One agent's output becomes another agent's input
# -------------------------------------------------------------------

OPENING_PROMPT = """You are a {role_name}.
Your position: {stance}

The topic being debated: "{topic}"

Give your opening argument in 3-4 sentences. Be specific and direct.
Do not introduce yourself. Just argue your point."""

REBUTTAL_PROMPT = """You are a {role_name}.
Your position: {stance}

The topic being debated: "{topic}"

Your opponent just argued:
"{opponent_argument}"

Counter their argument in 3-4 sentences. Address their specific points.
Be direct. Do not repeat your previous arguments."""

CONFIDENCE_PROMPT = """You are a {role_name} debating: "{topic}"
Your position: {stance}

The debate so far:
{transcript}

On a scale of 1 to 10, how confident are you in your original position?
1 = completely changed my mind
5 = seeing merit in both sides
10 = more convinced than ever

Reply with ONLY a number between 1 and 10. Nothing else."""


def get_opening(model, role, topic):
    """Agent gives opening argument."""
    prompt = OPENING_PROMPT.format(
        role_name=role["name"],
        stance=role["stance"],
        topic=topic,
    )
    return call_ollama(model, prompt, max_tokens=300)


def get_rebuttal(model, role, topic, opponent_argument):
    """Agent responds to what the other agent just said."""
    prompt = REBUTTAL_PROMPT.format(
        role_name=role["name"],
        stance=role["stance"],
        topic=topic,
        opponent_argument=opponent_argument[:500],
    )
    return call_ollama(model, prompt, max_tokens=300)


def get_confidence(model, role, topic, transcript):
    """Agent rates how confident it still is in its position."""
    prompt = CONFIDENCE_PROMPT.format(
        role_name=role["name"],
        stance=role["stance"],
        topic=topic,
        transcript=transcript[:1000],
    )
    response = call_ollama(model, prompt, max_tokens=10, temperature=0.1)

    # Parse number from response
    for word in response.split():
        try:
            num = int(word.strip(".,!"))
            if 1 <= num <= 10:
                return num
        except ValueError:
            continue
    return 5  # default if parsing fails


# -------------------------------------------------------------------
# CONCEPT 3: CONVERGENCE
# Tracking whether agents are moving toward agreement or staying apart
# -------------------------------------------------------------------

class DebateTracker:
    """
    Tracks confidence scores across rounds.
    Convergence = both agents moving toward 5 (middle ground).
    Polarization = both staying at extremes (1-2 or 9-10).
    """

    def __init__(self):
        self.rounds = []

    def add_round(self, round_num, conf_a, conf_b):
        gap = abs(conf_a - conf_b)
        self.rounds.append({
            "round": round_num,
            "conf_a": conf_a,
            "conf_b": conf_b,
            "gap": gap,
        })

    def is_converging(self):
        """Are they getting closer over rounds?"""
        if len(self.rounds) < 2:
            return None
        first_gap = self.rounds[0]["gap"]
        last_gap = self.rounds[-1]["gap"]
        return last_gap < first_gap

    def summary(self):
        """One-line convergence summary."""
        if len(self.rounds) < 2:
            return "Not enough rounds to assess convergence"

        first_gap = self.rounds[0]["gap"]
        last_gap = self.rounds[-1]["gap"]

        if last_gap < first_gap:
            return f"Converging - gap narrowed from {first_gap} to {last_gap}"
        elif last_gap > first_gap:
            return f"Polarizing - gap widened from {first_gap} to {last_gap}"
        else:
            return f"Stable - gap stayed at {last_gap}"


# -------------------------------------------------------------------
# CONCEPT 4: JUDGE PATTERN
# A third call reads the full debate and synthesizes
# -------------------------------------------------------------------

JUDGE_PROMPT = """You are a neutral judge. Read this debate between two people on the topic: "{topic}"

{role_a_name} ({role_a_stance}):
Uses {model_a}

{role_b_name} ({role_b_stance}):
Uses {model_b}

Full debate transcript:
{transcript}

Confidence tracking:
{confidence_summary}

As the judge, provide:
1. Where they AGREED (if anywhere)
2. Where they DISAGREED and why
3. The strongest argument from each side
4. Your recommendation for a decision-maker considering this topic

Be concise and specific. 6-8 sentences total."""


def judge_debate(topic, role_a, role_b, transcript, tracker):
    """
    CONCEPT 4: Judge pattern.
    The judge reads everything and synthesizes.
    Same pattern as Microsoft Critique - one generates, another evaluates.
    """

    conf_lines = []
    for r in tracker.rounds:
        conf_lines.append(
            f"  Round {r['round']}: {role_a['name']}={r['conf_a']}/10, "
            f"{role_b['name']}={r['conf_b']}/10 (gap={r['gap']})"
        )
    conf_lines.append(f"  Trend: {tracker.summary()}")

    prompt = JUDGE_PROMPT.format(
        topic=topic,
        role_a_name=role_a["name"],
        role_a_stance=role_a["stance"],
        role_b_name=role_b["name"],
        role_b_stance=role_b["stance"],
        model_a=AGENT_A_MODEL,
        model_b=AGENT_B_MODEL,
        transcript=transcript[:2000],
        confidence_summary="\n".join(conf_lines),
    )

    return call_ollama(JUDGE_MODEL, prompt, max_tokens=400, temperature=0.2)


# -------------------------------------------------------------------
# THE DEBATE LOOP
# -------------------------------------------------------------------

def run_debate(topic, num_rounds=DEFAULT_ROUNDS):
    """
    The main debate loop. All 4 concepts come together:
    1. Generate roles dynamically (agent identity)
    2. Agents take turns responding (agent communication)
    3. Track confidence each round (convergence)
    4. Judge synthesizes at the end (judge pattern)
    """

    print()
    print("=" * 60)
    print(f"  MULTI-AGENT DEBATE")
    print(f"  Topic: {topic}")
    print("=" * 60)

    # --- CONCEPT 1: Generate roles ---
    role_a, role_b = generate_roles(topic)
    if role_a is None:
        print("  Could not generate roles. Try again.")
        return

    print(f"\n  Roles generated:")
    print(f"    Agent A ({AGENT_A_MODEL}): {role_a['name']}")
    print(f"      Stance: {role_a['stance']}")
    print(f"    Agent B ({AGENT_B_MODEL}): {role_b['name']}")
    print(f"      Stance: {role_b['stance']}")

    tracker = DebateTracker()
    transcript_parts = []
    last_a = ""
    last_b = ""

    total_start = time.time()

    for round_num in range(1, num_rounds + 1):
        print(f"\n  {'─' * 56}")
        print(f"  Round {round_num}/{num_rounds}")
        print(f"  {'─' * 56}")

        # --- CONCEPT 2: Agent communication ---

        # Agent A speaks
        print(f"\n  [{role_a['name']}] ({AGENT_A_MODEL})")
        start = time.time()
        if round_num == 1:
            last_a = get_opening(AGENT_A_MODEL, role_a, topic)
        else:
            last_a = get_rebuttal(AGENT_A_MODEL, role_a, topic, last_b)
        elapsed_a = time.time() - start

        if last_a.startswith("[ERROR]"):
            print(f"    {last_a}")
            return
        for line in last_a.split("\n"):
            print(f"    {line}")
        print(f"    ({elapsed_a:.1f}s)")

        transcript_parts.append(f"[{role_a['name']}]: {last_a}")

        # Agent B speaks (sees what Agent A just said)
        print(f"\n  [{role_b['name']}] ({AGENT_B_MODEL})")
        start = time.time()
        if round_num == 1:
            last_b = get_opening(AGENT_B_MODEL, role_b, topic)
        else:
            last_b = get_rebuttal(AGENT_B_MODEL, role_b, topic, last_a)
        elapsed_b = time.time() - start

        if last_b.startswith("[ERROR]"):
            print(f"    {last_b}")
            return
        for line in last_b.split("\n"):
            print(f"    {line}")
        print(f"    ({elapsed_b:.1f}s)")

        transcript_parts.append(f"[{role_b['name']}]: {last_b}")

        # --- CONCEPT 3: Convergence tracking ---
        full_transcript = "\n\n".join(transcript_parts)

        conf_a = get_confidence(AGENT_A_MODEL, role_a, topic, full_transcript)
        conf_b = get_confidence(AGENT_B_MODEL, role_b, topic, full_transcript)
        tracker.add_round(round_num, conf_a, conf_b)

        # Visual confidence bar
        bar_a = "█" * conf_a + "░" * (10 - conf_a)
        bar_b = "█" * conf_b + "░" * (10 - conf_b)
        print(f"\n  Confidence:")
        print(f"    {role_a['name']:>20}: {bar_a} {conf_a}/10")
        print(f"    {role_b['name']:>20}: {bar_b} {conf_b}/10")
        print(f"    Gap: {abs(conf_a - conf_b)}")

    # --- CONCEPT 4: Judge verdict ---
    print(f"\n  {'─' * 56}")
    print(f"  JUDGE VERDICT")
    print(f"  {'─' * 56}")

    print(f"\n  Analyzing full debate...")
    start = time.time()
    verdict = judge_debate(topic, role_a, role_b, full_transcript, tracker)
    elapsed_j = time.time() - start

    if verdict.startswith("[ERROR]"):
        print(f"  {verdict}")
    else:
        print()
        for line in verdict.split("\n"):
            print(f"    {line}")

    total_elapsed = time.time() - total_start

    # --- Final summary ---
    print(f"\n  {'=' * 56}")
    print(f"  SUMMARY")
    print(f"  {'=' * 56}")
    print(f"\n  Topic: {topic}")
    print(f"  Rounds: {num_rounds}")
    print(f"  Agent A: {role_a['name']} on {AGENT_A_MODEL}")
    print(f"  Agent B: {role_b['name']} on {AGENT_B_MODEL}")
    print(f"  Convergence: {tracker.summary()}")

    print(f"\n  Confidence over rounds:")
    for r in tracker.rounds:
        bar_a = "█" * r["conf_a"] + "░" * (10 - r["conf_a"])
        bar_b = "█" * r["conf_b"] + "░" * (10 - r["conf_b"])
        print(f"    R{r['round']}: {role_a['name'][:15]:>15} {bar_a} {r['conf_a']}  |  "
              f"{role_b['name'][:15]:>15} {bar_b} {r['conf_b']}  |  gap={r['gap']}")

    print(f"\n  Total time: {total_elapsed:.1f}s")
    print(f"  Judge verdict: ({elapsed_j:.1f}s)")
    print()


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Laptop AI Phase 7 - multi-agent debate"
    )
    parser.add_argument(
        "topic", nargs="?", default=None,
        help='Business decision to debate (in quotes)'
    )
    parser.add_argument(
        "--rounds", type=int, default=DEFAULT_ROUNDS,
        help=f"Number of debate rounds (default: {DEFAULT_ROUNDS}, max: {MAX_ROUNDS})"
    )
    args = parser.parse_args()

    if args.rounds > MAX_ROUNDS:
        print(f"  Max rounds is {MAX_ROUNDS}. Using {MAX_ROUNDS}.")
        args.rounds = MAX_ROUNDS

    if not check_ollama():
        print("\n  Ollama is not running. Start with: ollama serve")
        sys.exit(1)

    if args.topic is None:
        print()
        print("  MULTI-AGENT DEBATE")
        print("  Type a business decision. Two AI models will debate it.")
        print("  Type 'quit' to exit.")
        print()
        print("  Examples:")
        print('    should we build or buy our AI stack')
        print('    should India build sovereign AI or rely on US providers')
        print('    should we replace customer support with AI agents')
        print('    should startups raise VC funding or bootstrap')
        print()

        while True:
            try:
                topic = input("  Topic > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n  Bye.")
                break

            if not topic or topic.lower() in ("quit", "exit", "q"):
                print("  Bye.")
                break

            run_debate(topic, args.rounds)
    else:
        run_debate(args.topic, args.rounds)


if __name__ == "__main__":
    main()
