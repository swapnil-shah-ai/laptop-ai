"""
Laptop AI — Phase 5: Goal-driven agent

What this does:
    You give it a goal like "prepare me for tomorrow's meeting."
    It breaks that into steps, picks the right tool for each step,
    runs them in order, handles failures, and gives you one result.

What you learn:
    - Task decomposition (breaking a goal into steps)
    - State management (tracking what's done and what's next)
    - Tool selection (picking the right phase for each step)
    - Error recovery (what happens when a step fails)
    - Planning loop (agent adapts based on results so far)

Think of it like a project manager:
    You say "prepare for the meeting." The PM doesn't do the work.
    He looks at his team (Phase 1 for search, Phase 2 for writing,
    Phase 4 for predictions), breaks the job into tasks, assigns
    each task to the right person, collects output, and gives you
    one clean briefing.

Tools available:
    SEARCH   → Phase 1 RAG (find information in your documents)
    WRITE    → Phase 2 fine-tuned model (write in your voice)
    PREDICT  → Phase 4 prediction engine (predict which files you need)
    SUMMARIZE → condense long text into key points

Usage:
    python agent.py                                         # interactive mode
    python agent.py "prepare me for tomorrow's meeting"     # single goal
    python agent.py "find everything about pricing"         # search goal
    python agent.py "what will I need this Monday morning"  # predict goal
"""

import sys
import os
import time
import argparse
from datetime import datetime
from pathlib import Path

# -------------------------------------------------------------------
# CONFIG — what models to use and where things live
# -------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434"

# Phi3 is the planner brain — smart enough to plan, small enough to be fast
PLANNER_MODEL = "phi3:mini"

# Your fine-tuned model — writes in your voice
WRITER_MODEL = "swapnil-tinyllama:latest"

# Embedding model — turns text into numbers for search
EMBED_MODEL = "nomic-embed-text:latest"

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(PROJECT_ROOT, "chroma_db")


# -------------------------------------------------------------------
# HELPER — talk to Ollama
# -------------------------------------------------------------------

def call_ollama(model: str, prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
    """
    Send a prompt to Ollama and get a response.
    This is the basic building block — every tool uses this.
    """
    import requests

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
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
            return "[ERROR] Model timed out. Prompt may be too long."
        else:
            return f"[ERROR] {str(e)}"


def check_ollama() -> bool:
    """Check if Ollama is running before we start."""
    import requests

    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# -------------------------------------------------------------------
# CONCEPT 1: TASK DECOMPOSITION
# Breaking "prepare me for tomorrow" into concrete steps
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


def decompose_goal(goal: str) -> list:
    """
    CONCEPT 1: Task decomposition.

    Takes a vague goal → sends it to Phi3 → gets back structured steps.

    The prompt is carefully designed to force a specific format.
    We then parse that format with simple string splitting.
    No fancy JSON parsing needed — keep it robust.
    """
    print(f"\n  Planning...")

    prompt = PLAN_PROMPT.format(goal=goal)
    response = call_ollama(PLANNER_MODEL, prompt, max_tokens=300, temperature=0.1)

    if response.startswith("[ERROR]"):
        print(f"  {response}")
        return []

    # Parse the response into steps
    steps = []
    for line in response.strip().split("\n"):
        line = line.strip()

        # Only process lines that start with STEP
        if not line.upper().startswith("STEP"):
            continue

        # Need a pipe separator between description and tool
        if "|" not in line:
            continue

        parts = line.split("|", 1)

        # Left side: "STEP 1: do something"
        desc_part = parts[0].strip()
        if ":" in desc_part:
            # Split on first colon to remove "STEP 1"
            description = desc_part.split(":", 1)[1].strip()
        else:
            description = desc_part

        # Right side: "TOOL: SEARCH"
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
        })

    if not steps:
        print(f"  Could not parse plan from model response.")
        print(f"  Raw response:\n  {response[:300]}")

    return steps


# -------------------------------------------------------------------
# CONCEPT 2: STATE MANAGEMENT
# The agent's checklist — what's done, what's next
# -------------------------------------------------------------------

class AgentState:
    """
    Simple checklist that tracks every step.

    Each step has a status:
        pending  → hasn't started yet
        running  → currently executing
        done     → completed successfully
        failed   → something went wrong

    The agent always knows where it is.
    """

    def __init__(self, goal: str):
        self.goal = goal
        self.steps = []
        self.started_at = datetime.now()

    def set_steps(self, steps: list):
        """Load the plan into state."""
        self.steps = steps

    def next_step(self):
        """Get the next pending step, or None if all done."""
        for step in self.steps:
            if step["status"] == "pending":
                return step
        return None

    def mark_running(self, step: dict):
        step["status"] = "running"

    def mark_done(self, step: dict, result: str):
        step["status"] = "done"
        step["result"] = result

    def mark_failed(self, step: dict, error: str):
        step["status"] = "failed"
        step["result"] = error

    def completed_steps(self) -> list:
        """Steps that finished successfully."""
        return [s for s in self.steps if s["status"] == "done"]

    def failed_steps(self) -> list:
        """Steps that failed."""
        return [s for s in self.steps if s["status"] == "failed"]

    def is_finished(self) -> bool:
        """Are all steps done (success or failure)?"""
        return all(s["status"] in ("done", "failed") for s in self.steps)

    def elapsed(self) -> float:
        """Seconds since the agent started."""
        return (datetime.now() - self.started_at).total_seconds()


# -------------------------------------------------------------------
# CONCEPT 3: TOOL SELECTION
# Each tool maps to a phase of the project
# -------------------------------------------------------------------

def tool_search(query: str) -> str:
    """
    SEARCH tool → Phase 1 RAG.
    Finds relevant documents and answers from them.
    """
    try:
        import chromadb
        import requests

        # Connect to the vector database from Phase 1
        client = chromadb.PersistentClient(path=CHROMA_PATH)

        # Try to find the collection (Phase 1 created it)
        collections = client.list_collections()
        if not collections:
            return "No document collection found. Run Phase 1 scan.py first."

        collection = collections[0]  # use the first available collection

        # Turn the query into an embedding (numbers)
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": query},
            timeout=30,
        )
        resp.raise_for_status()
        embedding = resp.json()["embedding"]

        # Find the 3 most relevant chunks
        results = collection.query(
            query_embeddings=[embedding],
            n_results=3,
        )

        if not results["documents"][0]:
            return "No relevant documents found for this query."

        # Combine chunks into context
        chunks = "\n---\n".join(results["documents"][0])

        # Ask the model to answer from those chunks
        answer = call_ollama(
            PLANNER_MODEL,
            f"Answer this question using ONLY the documents below. "
            f"Be concise.\n\n"
            f"Question: {query}\n\n"
            f"Documents:\n{chunks}\n\n"
            f"Answer:",
            max_tokens=300,
        )

        return answer

    except Exception as e:
        return f"[TOOL_ERROR] Search failed: {str(e)}"


def tool_write(instruction: str) -> str:
    """
    WRITE tool → Phase 2 fine-tuned model.
    Writes in your voice using the model trained on your writing.
    """
    try:
        result = call_ollama(WRITER_MODEL, instruction, max_tokens=400)
        if not result or result.startswith("[ERROR]"):
            return result or "[TOOL_ERROR] Write returned empty response."
        return result
    except Exception as e:
        return f"[TOOL_ERROR] Write failed: {str(e)}"


def tool_predict(context: str) -> str:
    """
    PREDICT tool → Phase 4 prediction engine.
    Predicts which files you'll need based on day/time patterns.
    """
    try:
        # Import Phase 4's prediction functions
        predict_dir = os.path.join(PROJECT_ROOT, "phase4_predict")
        if predict_dir not in sys.path:
            sys.path.insert(0, predict_dir)

        from predict import load_models, predict_now, DAY_NAMES

        models, vectorizer, meta = load_models()
        if models is None:
            return ("Prediction model not trained yet. "
                    "Run: cd phase4_predict && python predict.py --retrain")

        now = datetime.now()
        predictions = predict_now(
            models, vectorizer,
            day=now.weekday(), hour=now.hour,
        )

        if not predictions:
            return "No predictions available for this time."

        lines = [f"Predicted files for {DAY_NAMES[now.weekday()]} {now.hour}:00:"]
        for p in predictions:
            pct = p["probability"] * 100
            lines.append(f"  - {p['filename']} ({pct:.0f}% likely)")

        return "\n".join(lines)

    except Exception as e:
        return f"[TOOL_ERROR] Predict failed: {str(e)}"


def tool_summarize(text: str) -> str:
    """
    SUMMARIZE tool → condense text into key points.
    Uses the planner model since it's good at following instructions.
    """
    try:
        result = call_ollama(
            PLANNER_MODEL,
            f"Summarize this concisely in 3-4 bullet points:\n\n{text}",
            max_tokens=200,
        )
        if not result or result.startswith("[ERROR]"):
            return result or "[TOOL_ERROR] Summarize returned empty response."
        return result
    except Exception as e:
        return f"[TOOL_ERROR] Summarize failed: {str(e)}"


# Tool registry — maps names to functions
TOOLS = {
    "SEARCH": tool_search,
    "WRITE": tool_write,
    "PREDICT": tool_predict,
    "SUMMARIZE": tool_summarize,
}


# -------------------------------------------------------------------
# CONCEPT 4: ERROR RECOVERY
# When a step fails, the agent doesn't crash — it moves on
# -------------------------------------------------------------------
# (Built into the execute loop below)


# -------------------------------------------------------------------
# CONCEPT 5: PLANNING LOOP
# After each step, pass results forward so later steps have context
# -------------------------------------------------------------------

def build_tool_input(step: dict, previous_results: list) -> str:
    """
    CONCEPT 5: Planning loop.

    For WRITE and SUMMARIZE, we feed results from earlier steps
    as context. This is how the agent chains — step 2 knows
    what step 1 found.

    For SEARCH and PREDICT, we just use the step description.
    """
    description = step["description"]

    # SEARCH and PREDICT don't need context from previous steps
    if step["tool"] in ("SEARCH", "PREDICT"):
        return description

    # WRITE and SUMMARIZE benefit from knowing what earlier steps found
    if not previous_results:
        return description

    context_parts = []
    for prev in previous_results:
        if prev.get("result"):
            # Keep context short — only first 500 chars per step
            snippet = prev["result"][:500]
            context_parts.append(f"[{prev['tool']}] {snippet}")

    context = "\n\n".join(context_parts)
    return f"{description}\n\nContext from previous steps:\n{context}"


# -------------------------------------------------------------------
# THE MAIN LOOP — where all 5 concepts come together
# -------------------------------------------------------------------

def run_agent(goal: str) -> AgentState:
    """
    The agent loop. This is where it all connects:

    1. DECOMPOSE the goal into steps        (concept 1)
    2. CREATE state to track progress        (concept 2)
    3. For each step, SELECT the tool        (concept 3)
    4. If a step fails, RECOVER and continue (concept 4)
    5. PASS results forward to next steps    (concept 5)
    """

    print()
    print("=" * 60)
    print(f"  LAPTOP AI AGENT")
    print(f"  Goal: {goal}")
    print("=" * 60)

    # ---- CONCEPT 1: Decompose goal into steps ----
    steps = decompose_goal(goal)

    state = AgentState(goal)

    if not steps:
        print("\n  Could not create a plan. Try rephrasing your goal.")
        return state

    # ---- CONCEPT 2: Load steps into state tracker ----
    state.set_steps(steps)

    print(f"\n  Plan ({len(steps)} steps):")
    for s in steps:
        print(f"    {s['step_num']}. {s['description']}  [{s['tool']}]")
    print()

    # ---- Execute each step ----
    for step in steps:
        step_num = step["step_num"]
        total = len(steps)

        print(f"  Step {step_num}/{total}: {step['description']}")
        print(f"    Tool: {step['tool']}")

        # ---- CONCEPT 2: Update state ----
        state.mark_running(step)

        # ---- CONCEPT 5: Build input with context from previous steps ----
        previous = state.completed_steps()
        tool_input = build_tool_input(step, previous)

        # ---- CONCEPT 3: Select and run the right tool ----
        tool_name = step["tool"]
        if tool_name not in TOOLS:
            state.mark_failed(step, f"Unknown tool: {tool_name}")
            print(f"    x Failed: unknown tool '{tool_name}'")
            print()
            continue

        start = time.time()
        result = TOOLS[tool_name](tool_input)
        elapsed = time.time() - start

        # ---- CONCEPT 4: Error recovery ----
        if "[ERROR]" in result or "[TOOL_ERROR]" in result:
            state.mark_failed(step, result)
            print(f"    x Failed ({elapsed:.1f}s): {result[:100]}")
        else:
            state.mark_done(step, result)
            print(f"    Done ({elapsed:.1f}s)")
            # Show a short preview
            preview = result[:150].replace("\n", " ")
            print(f"    > {preview}...")

        print()

    return state


# -------------------------------------------------------------------
# DISPLAY — show the final output
# -------------------------------------------------------------------

def display_results(state: AgentState):
    """Pretty-print everything the agent found."""

    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)

    successes = state.completed_steps()
    failures = state.failed_steps()

    if not successes and not failures:
        print("\n  No steps were executed.")
        print()
        return

    if successes:
        for s in successes:
            print(f"\n  [{s['tool']}] {s['description']}")
            print(f"  {'.' * 50}")
            for line in s["result"].split("\n"):
                print(f"    {line}")

    if failures:
        print(f"\n  Warning: {len(failures)} step(s) failed:")
        for f in failures:
            print(f"    - {f['description']}: {f['result'][:100]}")

    elapsed = state.elapsed()
    print(f"\n  Completed in {elapsed:.1f}s")
    print(f"  {len(successes)} succeeded, {len(failures)} failed")
    print()


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Laptop AI Phase 5 — goal-driven agent"
    )
    parser.add_argument(
        "goal", nargs="?", default=None,
        help='Your goal in quotes (e.g., "find everything about pricing")'
    )
    args = parser.parse_args()

    # Check Ollama is running
    if not check_ollama():
        print("\n  Ollama is not running.")
        print("  Start it with: ollama serve")
        print("  Then try again.")
        sys.exit(1)

    # Interactive mode if no goal provided
    if args.goal is None:
        print()
        print("  LAPTOP AI AGENT")
        print("  Type a goal, or 'quit' to exit.")
        print("  Examples:")
        print('    "find everything about pricing and summarize it"')
        print('    "prepare me for tomorrow\'s meeting"')
        print('    "what will I need this Monday morning?"')
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

            state = run_agent(goal)
            display_results(state)
    else:
        state = run_agent(args.goal)
        display_results(state)


if __name__ == "__main__":
    main()
