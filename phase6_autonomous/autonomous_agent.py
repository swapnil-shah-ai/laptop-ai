"""
Laptop AI - Phase 6: Autonomous agent with reflection

What changed from Phase 5:
    Phase 5 plans and executes. If a step produces garbage, it moves on.
    Phase 6 plans, executes, then EVALUATES its own output. If the result
    is poor, it retries with a different approach. It catches its own
    mistakes without you telling it to.

    Phase 5 = student who submits the first draft.
    Phase 6 = student who rereads, fixes weak parts, then submits.

What you learn:
    - ReAct loop (reason, act, observe, reason again)
    - Reflection (agent judges its own output quality)
    - Self-correction (agent retries with a different strategy)
    - Goal convergence (how the agent decides "I'm done")

The 4 new concepts:
    1. EVALUATE  - after each step, ask "was this good enough?"
    2. RETRY     - if not good enough, rephrase and try again
    3. REFLECT   - log WHY something failed so the next attempt is smarter
    4. CONVERGE  - max 2 retries per step, then accept and move on

Usage:
    python autonomous_agent.py                                    # interactive
    python autonomous_agent.py "find everything about pricing"    # single goal
    python autonomous_agent.py --compare "prepare for meeting"    # side by side with Phase 5
"""

import sys
import os
import time
import argparse
from datetime import datetime
from pathlib import Path

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434"
PLANNER_MODEL = "phi3:mini"
WRITER_MODEL = "swapnil-tinyllama:latest"
EMBED_MODEL = "nomic-embed-text:latest"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_db")

MAX_RETRIES = 2  # maximum retry attempts per step


# -------------------------------------------------------------------
# HELPER - talk to Ollama
# -------------------------------------------------------------------

def call_ollama(model, prompt, max_tokens=500, temperature=0.3):
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
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        error_type = type(e).__name__
        if "ConnectionError" in error_type:
            return "[ERROR] Cannot connect to Ollama. Run: ollama serve"
        elif "Timeout" in error_type:
            return "[ERROR] Model timed out."
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
# PLANNING - same as Phase 5
# -------------------------------------------------------------------

PLAN_PROMPT = """You are a task planner with 4 tools:
- SEARCH: find information in documents
- WRITE: write text in a specific style
- PREDICT: predict which files are needed based on day/time patterns
- SUMMARIZE: condense long text into key points

Break the goal into 2-4 steps. Each step uses exactly one tool.

Format EXACTLY like this (no other text before or after):
STEP 1: [what to do] | TOOL: [SEARCH or WRITE or PREDICT or SUMMARIZE]
STEP 2: [what to do] | TOOL: [SEARCH or WRITE or PREDICT or SUMMARIZE]

Example:
Goal: "prepare me for tomorrow's meeting"
STEP 1: predict which files I will need tomorrow | TOOL: PREDICT
STEP 2: search for key information in those files | TOOL: SEARCH
STEP 3: write a brief meeting prep summary | TOOL: WRITE

Goal: "{goal}"
"""


def decompose_goal(goal):
    print(f"\n  Planning...")
    prompt = PLAN_PROMPT.format(goal=goal)
    response = call_ollama(PLANNER_MODEL, prompt, max_tokens=300, temperature=0.1)

    if response.startswith("[ERROR]"):
        print(f"  {response}")
        return []

    steps = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line.upper().startswith("STEP"):
            continue
        if "|" not in line:
            continue

        parts = line.split("|", 1)
        desc_part = parts[0].strip()
        if ":" in desc_part:
            description = desc_part.split(":", 1)[1].strip()
        else:
            description = desc_part

        tool_part = parts[1].strip().upper()
        tool = "UNKNOWN"
        for t in ["SEARCH", "WRITE", "PREDICT", "SUMMARIZE"]:
            if t in tool_part:
                tool = t
                break

        steps.append({
            "step_num": len(steps) + 1,
            "description": description,
            "tool": tool,
            "status": "pending",
            "result": None,
            "attempts": 0,
            "reflections": [],
        })

    if not steps:
        print(f"  Could not parse plan.")
        print(f"  Raw: {response[:300]}")

    return steps


# -------------------------------------------------------------------
# TOOLS - same as Phase 5
# -------------------------------------------------------------------

def tool_search(query):
    try:
        import chromadb
        import requests

        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collections = client.list_collections()
        if not collections:
            return "No document collection found. Run Phase 1 scan.py first."

        collection = collections[0]
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": query},
            timeout=30,
        )
        resp.raise_for_status()
        embedding = resp.json()["embedding"]

        results = collection.query(query_embeddings=[embedding], n_results=3)
        if not results["documents"][0]:
            return "No relevant documents found."

        chunks = "\n---\n".join(results["documents"][0])
        answer = call_ollama(
            PLANNER_MODEL,
            f"Answer this question using ONLY the documents below. Be concise.\n\n"
            f"Question: {query}\n\nDocuments:\n{chunks}\n\nAnswer:",
            max_tokens=300,
        )
        return answer
    except Exception as e:
        return f"[TOOL_ERROR] Search failed: {str(e)}"


def tool_write(instruction):
    try:
        result = call_ollama(WRITER_MODEL, instruction, max_tokens=400)
        if not result or result.startswith("[ERROR]"):
            return result or "[TOOL_ERROR] Write returned empty."
        return result
    except Exception as e:
        return f"[TOOL_ERROR] Write failed: {str(e)}"


def tool_predict(context):
    try:
        predict_dir = os.path.join(PROJECT_ROOT, "phase4_predict")
        if predict_dir not in sys.path:
            sys.path.insert(0, predict_dir)

        from predict import load_models, predict_now, DAY_NAMES
        models, vectorizer, meta = load_models()
        if models is None:
            return "Prediction model not trained. Run: cd phase4_predict && python predict.py --retrain"

        now = datetime.now()
        predictions = predict_now(models, vectorizer, day=now.weekday(), hour=now.hour)
        if not predictions:
            return "No predictions for this time."

        lines = [f"Predicted files for {DAY_NAMES[now.weekday()]} {now.hour}:00:"]
        for p in predictions:
            pct = p["probability"] * 100
            lines.append(f"  - {p['filename']} ({pct:.0f}% likely)")
        return "\n".join(lines)
    except Exception as e:
        return f"[TOOL_ERROR] Predict failed: {str(e)}"


def tool_summarize(text):
    try:
        result = call_ollama(
            PLANNER_MODEL,
            f"Summarize this concisely in 3-4 bullet points:\n\n{text}",
            max_tokens=200,
        )
        if not result or result.startswith("[ERROR]"):
            return result or "[TOOL_ERROR] Summarize returned empty."
        return result
    except Exception as e:
        return f"[TOOL_ERROR] Summarize failed: {str(e)}"


TOOLS = {
    "SEARCH": tool_search,
    "WRITE": tool_write,
    "PREDICT": tool_predict,
    "SUMMARIZE": tool_summarize,
}


# -------------------------------------------------------------------
# NEW CONCEPT 1: EVALUATE - the agent judges its own output
# -------------------------------------------------------------------

EVALUATE_PROMPT = """You are a quality checker. Look at this result and answer three questions.

Task was: {task}
Result was: {result}

Question 1: Does the result contain more than 10 words of real content? (YES or NO)
Question 2: Does the result relate to the task? (YES or NO)
Question 3: Does the result contain an error message? (YES or NO)

Reply EXACTLY in this format:
WORDS: YES or NO
RELEVANT: YES or NO
ERROR: YES or NO
VERDICT: PASS or FAIL
"""


def evaluate_result(task, result):
    """
    NEW CONCEPT: The agent evaluates its own output.

    Instead of asking "is this good?" (too vague for a small model),
    we ask three binary questions any model can answer:
    1. Is there real content? (not just a few characters)
    2. Is it relevant to the task?
    3. Is there an error?

    Then we combine the answers into PASS or FAIL.
    """

    # Quick checks before even asking the model
    # These are deterministic - no LLM needed
    if not result or len(result.strip()) < 5:
        return "FAIL", "Result is empty or too short"

    if "[ERROR]" in result or "[TOOL_ERROR]" in result:
        return "FAIL", "Result contains an error"

    # Count real words (not just whitespace)
    word_count = len(result.split())
    if word_count < 10:
        return "FAIL", f"Only {word_count} words - too short to be useful"

    # Now ask the model for relevance check
    prompt = EVALUATE_PROMPT.format(
        task=task[:200],  # keep prompt short
        result=result[:300],  # keep prompt short
    )
    response = call_ollama(PLANNER_MODEL, prompt, max_tokens=50, temperature=0.1)

    # Parse the response
    response_upper = response.upper()

    # Check for explicit FAIL verdict
    if "VERDICT: FAIL" in response_upper:
        return "FAIL", "Model judged result as insufficient"

    # Check for RELEVANT: NO
    if "RELEVANT: NO" in response_upper:
        return "FAIL", "Result not relevant to the task"

    # Check for ERROR: YES
    if "ERROR: YES" in response_upper:
        return "FAIL", "Result contains errors"

    # If we got here, it passed (either explicit PASS or we couldn't parse a FAIL)
    return "PASS", "Result meets quality threshold"


# -------------------------------------------------------------------
# NEW CONCEPT 2: RETRY - try again with a different approach
# -------------------------------------------------------------------

def build_retry_instruction(step, attempt, previous_result, reason):
    """
    NEW CONCEPT: When a step fails, don't just repeat it.
    Change the approach.

    Attempt 1 (original): use the description as-is
    Attempt 2 (retry 1): simplify the instruction
    Attempt 3 (retry 2): make it very specific and short
    """
    description = step["description"]

    if attempt == 1:
        # First retry: simplify
        return f"Keep it simple and brief. {description}"

    elif attempt == 2:
        # Second retry: be very direct
        if step["tool"] == "WRITE":
            return f"Write 2-3 sentences about: {description}"
        elif step["tool"] == "SEARCH":
            # Extract key words only
            words = [w for w in description.split() if len(w) > 3]
            short_query = " ".join(words[:5])
            return short_query
        elif step["tool"] == "SUMMARIZE":
            return f"List 3 key points from this text: {previous_result[:200]}"
        else:
            return description

    return description


# -------------------------------------------------------------------
# NEW CONCEPT 3: REFLECT - log what happened and why
# -------------------------------------------------------------------

class Reflection:
    """
    A record of one evaluation cycle.
    The agent keeps a log of what it tried and what happened.
    This is the agent's "thinking about its own thinking."
    """

    def __init__(self, attempt, result, verdict, reason, elapsed):
        self.attempt = attempt
        self.result_preview = result[:100] if result else ""
        self.verdict = verdict
        self.reason = reason
        self.elapsed = elapsed

    def __str__(self):
        return f"Attempt {self.attempt}: {self.verdict} - {self.reason} ({self.elapsed:.1f}s)"


# -------------------------------------------------------------------
# CONTEXT BUILDER - pass previous results to later steps
# -------------------------------------------------------------------

def build_tool_input(step, completed_steps):
    description = step["description"]
    if step["tool"] in ("SEARCH", "PREDICT"):
        return description

    if not completed_steps:
        return description

    context_parts = []
    for prev in completed_steps:
        if prev.get("result"):
            snippet = prev["result"][:500]
            context_parts.append(f"[{prev['tool']}] {snippet}")

    context = "\n\n".join(context_parts)
    return f"{description}\n\nContext from previous steps:\n{context}"


# -------------------------------------------------------------------
# THE AUTONOMOUS LOOP - where reflection makes the difference
# -------------------------------------------------------------------

def run_autonomous_agent(goal):
    """
    The autonomous agent loop:

    1. PLAN the goal into steps (same as Phase 5)
    2. For each step:
       a. EXECUTE the tool
       b. EVALUATE the result (NEW - reflection)
       c. If FAIL: RETRY with modified approach (NEW - self-correction)
       d. If PASS or retries exhausted: MOVE ON
       e. LOG the reflection (NEW - learning record)
    3. Display results with reflection history
    """

    print()
    print("=" * 60)
    print(f"  AUTONOMOUS AGENT (with reflection)")
    print(f"  Goal: {goal}")
    print("=" * 60)

    # Plan
    steps = decompose_goal(goal)
    if not steps:
        print("\n  Could not create a plan. Try rephrasing.")
        return [], 0

    print(f"\n  Plan ({len(steps)} steps):")
    for s in steps:
        print(f"    {s['step_num']}. {s['description']}  [{s['tool']}]")
    print()

    total_start = time.time()
    completed = []

    for step in steps:
        step_num = step["step_num"]
        total = len(steps)
        tool_name = step["tool"]

        if tool_name not in TOOLS:
            step["status"] = "failed"
            step["result"] = f"Unknown tool: {tool_name}"
            print(f"  Step {step_num}/{total}: SKIPPED - unknown tool '{tool_name}'")
            print()
            continue

        print(f"  Step {step_num}/{total}: {step['description']}")
        print(f"    Tool: {tool_name}")

        # --- THE REFLECTION LOOP ---
        best_result = None
        best_verdict = "FAIL"
        attempt = 0

        while attempt <= MAX_RETRIES:
            attempt += 1
            step["attempts"] = attempt

            # Build the input
            if attempt == 1:
                tool_input = build_tool_input(step, completed)
            else:
                tool_input = build_retry_instruction(
                    step, attempt - 1, best_result,
                    step["reflections"][-1].reason if step["reflections"] else ""
                )

            # Execute
            exec_start = time.time()
            result = TOOLS[tool_name](tool_input)
            exec_elapsed = time.time() - exec_start

            # Evaluate (NEW - this is the reflection)
            eval_start = time.time()
            verdict, reason = evaluate_result(step["description"], result)
            eval_elapsed = time.time() - eval_start

            total_step_time = exec_elapsed + eval_elapsed

            # Log the reflection
            reflection = Reflection(attempt, result, verdict, reason, total_step_time)
            step["reflections"].append(reflection)

            if attempt == 1:
                attempt_label = "1st attempt"
            elif attempt == 2:
                attempt_label = "retry 1"
            else:
                attempt_label = "retry 2"

            if verdict == "PASS":
                print(f"    [{attempt_label}] PASS ({exec_elapsed:.1f}s + {eval_elapsed:.1f}s eval)")
                preview = result[:120].replace("\n", " ")
                print(f"    > {preview}...")
                best_result = result
                best_verdict = "PASS"
                break
            else:
                print(f"    [{attempt_label}] FAIL - {reason} ({exec_elapsed:.1f}s)")
                if attempt <= MAX_RETRIES:
                    print(f"    Retrying with different approach...")
                best_result = result  # keep the best we have

        # Record final result
        if best_verdict == "PASS":
            step["status"] = "done"
            step["result"] = best_result
        else:
            print(f"    Exhausted {MAX_RETRIES} retries. Accepting best available result.")
            step["status"] = "done_with_issues"
            step["result"] = best_result

        completed.append(step)
        print()

    total_elapsed = time.time() - total_start
    return steps, total_elapsed


# -------------------------------------------------------------------
# PHASE 5 AGENT (for comparison) - no reflection
# -------------------------------------------------------------------

def run_basic_agent(goal):
    """Phase 5 style - plan and execute, no reflection."""

    print()
    print("=" * 60)
    print(f"  BASIC AGENT (Phase 5 - no reflection)")
    print(f"  Goal: {goal}")
    print("=" * 60)

    steps = decompose_goal(goal)
    if not steps:
        print("\n  Could not create a plan.")
        return [], 0

    print(f"\n  Plan ({len(steps)} steps):")
    for s in steps:
        print(f"    {s['step_num']}. {s['description']}  [{s['tool']}]")
    print()

    total_start = time.time()
    completed = []

    for step in steps:
        step_num = step["step_num"]
        total = len(steps)
        tool_name = step["tool"]

        if tool_name not in TOOLS:
            step["status"] = "failed"
            step["result"] = f"Unknown tool: {tool_name}"
            continue

        print(f"  Step {step_num}/{total}: {step['description']}")
        print(f"    Tool: {tool_name}")

        tool_input = build_tool_input(step, completed)
        start = time.time()
        result = TOOLS[tool_name](tool_input)
        elapsed = time.time() - start

        if "[ERROR]" in result or "[TOOL_ERROR]" in result:
            step["status"] = "failed"
            step["result"] = result
            print(f"    x Failed ({elapsed:.1f}s): {result[:100]}")
        else:
            step["status"] = "done"
            step["result"] = result
            preview = result[:120].replace("\n", " ")
            print(f"    Done ({elapsed:.1f}s)")
            print(f"    > {preview}...")

        completed.append(step)
        print()

    total_elapsed = time.time() - total_start
    return steps, total_elapsed


# -------------------------------------------------------------------
# DISPLAY
# -------------------------------------------------------------------

def display_results(steps, elapsed, label="RESULTS"):
    print("=" * 60)
    print(f"  {label}")
    print("=" * 60)

    successes = [s for s in steps if s["status"] == "done"]
    with_issues = [s for s in steps if s["status"] == "done_with_issues"]
    failures = [s for s in steps if s["status"] == "failed"]

    for s in steps:
        if s.get("result"):
            status_tag = ""
            if s["status"] == "done_with_issues":
                status_tag = " (accepted with issues)"
            print(f"\n  [{s['tool']}] {s['description']}{status_tag}")
            print(f"  {'.' * 50}")
            for line in s["result"].split("\n"):
                print(f"    {line}")

            # Show reflection log if any
            if s.get("reflections"):
                attempts = len(s["reflections"])
                if attempts > 1:
                    print(f"\n    Reflection log ({attempts} attempts):")
                    for r in s["reflections"]:
                        print(f"      {r}")

    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"  {len(successes)} clean, {len(with_issues)} with issues, {len(failures)} failed")
    print()


def display_comparison(steps_basic, elapsed_basic, steps_auto, elapsed_auto):
    """Show side-by-side results when --compare is used."""

    print()
    print("=" * 60)
    print("  COMPARISON: Phase 5 vs Phase 6")
    print("=" * 60)

    # Count results
    basic_ok = len([s for s in steps_basic if s["status"] == "done"])
    basic_fail = len([s for s in steps_basic if s["status"] == "failed"])
    auto_ok = len([s for s in steps_auto if s["status"] == "done"])
    auto_issues = len([s for s in steps_auto if s["status"] == "done_with_issues"])
    auto_fail = len([s for s in steps_auto if s["status"] == "failed"])

    # Count retries
    total_retries = sum(
        len(s.get("reflections", [])) - 1
        for s in steps_auto
        if len(s.get("reflections", [])) > 1
    )

    # Count quality catches
    catches = sum(
        1 for s in steps_auto
        for r in s.get("reflections", [])
        if r.verdict == "FAIL"
    )

    print(f"\n  Phase 5 (basic agent):")
    print(f"    Time: {elapsed_basic:.1f}s")
    print(f"    Steps: {basic_ok} done, {basic_fail} failed")
    print(f"    Quality checks: 0 (no reflection)")

    print(f"\n  Phase 6 (autonomous agent):")
    print(f"    Time: {elapsed_auto:.1f}s")
    print(f"    Steps: {auto_ok} clean, {auto_issues} with issues, {auto_fail} failed")
    print(f"    Quality catches: {catches}")
    print(f"    Retries triggered: {total_retries}")

    # Show specific catches
    if catches > 0:
        print(f"\n  What reflection caught:")
        for s in steps_auto:
            for r in s.get("reflections", []):
                if r.verdict == "FAIL":
                    print(f"    - Step '{s['description'][:50]}': {r.reason}")

    print()


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Laptop AI Phase 6 - autonomous agent with reflection"
    )
    parser.add_argument(
        "goal", nargs="?", default=None,
        help='Your goal in quotes'
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run same goal with Phase 5 AND Phase 6, show side by side"
    )
    args = parser.parse_args()

    if not check_ollama():
        print("\n  Ollama is not running. Start with: ollama serve")
        sys.exit(1)

    # Interactive mode
    if args.goal is None:
        print()
        print("  LAPTOP AI - AUTONOMOUS AGENT")
        print("  Type a goal, or 'quit' to exit.")
        print("  Add --compare to see Phase 5 vs Phase 6 side by side.")
        print()
        print("  Examples:")
        print('    find everything about pricing and summarize it')
        print('    prepare me for tomorrow\'s meeting')
        print()

        while True:
            try:
                goal = input("  Goal > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n  Bye.")
                break

            if not goal or goal.lower() in ("quit", "exit", "q"):
                print("  Bye.")
                break

            steps, elapsed = run_autonomous_agent(goal)
            display_results(steps, elapsed)

    elif args.compare:
        # Run both agents on the same goal
        print("\n  Running Phase 5 (basic) agent first...")
        steps_basic, elapsed_basic = run_basic_agent(args.goal)
        display_results(steps_basic, elapsed_basic, "PHASE 5 RESULTS")

        print("\n  Now running Phase 6 (autonomous) agent...")
        steps_auto, elapsed_auto = run_autonomous_agent(args.goal)
        display_results(steps_auto, elapsed_auto, "PHASE 6 RESULTS")

        display_comparison(steps_basic, elapsed_basic, steps_auto, elapsed_auto)

    else:
        steps, elapsed = run_autonomous_agent(args.goal)
        display_results(steps, elapsed)


if __name__ == "__main__":
    main()
