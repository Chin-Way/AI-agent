"""
mini_agent — a tiny command-line AI agent built on Claude + tool use.
====================================================================

This is a deliberately small, heavily-commented example of how an "agent"
works under the hood. An agent is just a large language model (here, Claude)
running in a loop, where on each turn it can either talk to you or ask to run
a "tool" (a normal Python function). We run the tool, hand the result back,
and let it decide what to do next — until the goal is done.

THE AGENT LOOP (the heart of this program), step by step:

  1. You type a goal. We add it to the conversation history as a user message.
  2. We send the WHOLE history (plus the list of tools) to Claude.
  3. Claude replies with a list of "content blocks", each of which is either:
       - a text block      -> something Claude wants to say to you
       - a tool_use block  -> a request to run one of our tools, with arguments
  4. If there are NO tool_use blocks, Claude is finished. We print its text
     and stop — that text is the final answer.
  5. If there ARE tool_use blocks, we run each requested tool locally, collect
     the results, append them to the history as a "tool_result", and go back
     to step 2.
  6. A step counter (MAX_STEPS) caps the loop so it can never run forever.

A key thing to notice: the Messages API is *stateless*. Claude doesn't
remember anything between calls — so we resend the entire growing list of
messages every single time. That list IS the agent's memory.

How to run:  see README.md  (short version: set ANTHROPIC_API_KEY, then
`python agent.py`).
"""

import json
import os
import subprocess
import sys

import anthropic


# ---------------------------------------------------------------------------
# Configuration — all the knobs in one place so they're easy to find and tweak.
# ---------------------------------------------------------------------------

# The Claude model to use. Swap this one string to try a different model, e.g.
# "claude-opus-4-8" (most capable) or "claude-haiku-4-5" (fastest/cheapest).
MODEL = "claude-sonnet-4-6"

# Maximum tokens Claude may generate per turn. 4096 is plenty for this demo.
MAX_TOKENS = 4096

# Safety cap: the most tool-using turns we'll allow for a single goal before
# giving up. This is what guarantees the loop can never run forever.
MAX_STEPS = 25

# How long (seconds) a single shell command may run before we abort it.
COMMAND_TIMEOUT = 60

# When showing tool results in the terminal, truncate anything longer than this
# so we don't flood the screen. (The FULL result is still sent to Claude.)
DISPLAY_LIMIT = 1000

# The system prompt sets the agent's role and ground rules for every turn.
SYSTEM_PROMPT = """\
You are a helpful command-line assistant running on the user's local machine.
You have tools to read files, write files, list directories, and run shell
commands. Use them to accomplish the user's goal.

Work one step at a time: call a tool, look at the result, then decide the next
step. Prefer relative paths. Be careful with shell commands that modify or
delete files. When the goal is complete, stop calling tools and reply with a
short, clear summary of what you did."""


# ---------------------------------------------------------------------------
# The tools — the real Python functions the agent can ask us to run.
#
# Each function just does its job and returns a string. If something goes
# wrong it raises a normal exception; execute_tool() (below) catches those and
# turns them into an error message for Claude, so a broken tool can never crash
# the whole program.
# ---------------------------------------------------------------------------

def read_file(path):
    """Return the full text contents of the file at `path`."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path, content):
    """Create or overwrite the file at `path` with `content`."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} characters to {path}."


def list_directory(path):
    """List the entries in directory `path` (folders get a trailing slash)."""
    entries = sorted(os.listdir(path))
    if not entries:
        return "(empty directory)"
    lines = []
    for name in entries:
        full = os.path.join(path, name)
        lines.append(name + "/" if os.path.isdir(full) else name)
    return "\n".join(lines)


def run_command(command):
    """Run `command` in a shell and return its exit code, stdout, and stderr."""
    # Required safety touch: always show the command before running it, so the
    # user can see exactly what is about to happen on their machine.
    print(f"  ⚠  about to run shell command: {command}")

    completed = subprocess.run(
        command,
        shell=True,                 # run through the shell so pipes/globs work
        capture_output=True,        # collect stdout and stderr
        text=True,                  # give us strings, not bytes
        timeout=COMMAND_TIMEOUT,    # never hang forever
    )

    parts = [f"exit code: {completed.returncode}"]
    if completed.stdout:
        parts.append("stdout:\n" + completed.stdout)
    if completed.stderr:
        parts.append("stderr:\n" + completed.stderr)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tool schemas — how we DESCRIBE the tools to Claude.
#
# Claude never sees our Python code; it only sees these JSON descriptions. The
# "description" text and the property descriptions are how Claude decides when
# and how to call each tool, so they're worth writing clearly.
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "read_file",
        "description": (
            "Read the entire contents of a text file and return it as a "
            "string. Use this to inspect a file before editing it or to "
            "gather information you need for the goal."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (relative or absolute).",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Create a new file or overwrite an existing one with the given "
            "content. Overwrites without asking, so read the file first if you "
            "need to preserve what's already there."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path of the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "The full text to write into the file.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_directory",
        "description": (
            "List the files and sub-folders inside a directory. Use this to "
            "explore the project. Folders are shown with a trailing slash."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to list. Use '.' for the current "
                    "directory.",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "run_command",
        "description": (
            "Run a shell command on the user's machine and return its exit "
            "code, stdout, and stderr. Use this for things the other tools "
            "can't do, like running tests or git commands. The command runs in "
            "a real shell, so be careful with anything destructive."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run, e.g. 'ls -la'.",
                }
            },
            "required": ["command"],
        },
    },
]

# Map each tool name to the Python function that implements it. The names here
# MUST match the "name" fields in TOOLS above.
TOOL_FUNCTIONS = {
    "read_file": read_file,
    "write_file": write_file,
    "list_directory": list_directory,
    "run_command": run_command,
}


# ---------------------------------------------------------------------------
# Tool dispatcher — runs one tool by name, with error handling.
# ---------------------------------------------------------------------------

def execute_tool(name, tool_input):
    """Run the tool called `name` with the arguments in `tool_input`.

    Returns a (result_text, is_error) tuple. We catch *every* exception here
    and convert it into an error string, so even a buggy tool or bad arguments
    from the model can't crash the program — Claude just sees the error and can
    try something else.
    """
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return f"Error: unknown tool '{name}'.", True

    try:
        # tool_input is a dict like {"path": "foo.txt"}; ** unpacks it into the
        # function's keyword arguments (path="foo.txt").
        result = str(func(**tool_input))
    except Exception as exc:  # noqa: BLE001 — we deliberately catch everything
        return f"Error while running '{name}': {exc}", True

    # The API rejects empty tool results, so substitute a placeholder.
    if result == "":
        result = "(the tool produced no output)"
    return result, False


# ---------------------------------------------------------------------------
# Small display helpers (purely for nice terminal output).
# ---------------------------------------------------------------------------

def _short(text, limit):
    """Collapse to one line and truncate, for compact display of tool args."""
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


def _preview(text):
    """Truncate a long tool result for display (full result still goes to Claude)."""
    if len(text) <= DISPLAY_LIMIT:
        return text
    return (
        text[:DISPLAY_LIMIT]
        + f"\n... (truncated; {len(text)} chars total, full result sent to Claude)"
    )


def _indent(text, prefix="    "):
    """Indent every line of `text` so results visually nest under their call."""
    return "\n".join(prefix + line for line in text.splitlines())


# ---------------------------------------------------------------------------
# The agent loop.
# ---------------------------------------------------------------------------

def run_agent(client, messages):
    """Drive one goal to completion.

    `messages` is the running conversation history (a list). It already ends
    with the user's new goal when this is called. We mutate it in place: each
    turn we append Claude's reply and any tool results, so the history (and the
    agent's memory) grows as we go.
    """
    for step in range(1, MAX_STEPS + 1):
        print(f"\n--- step {step} ---")

        # STEP 2: send the full history + tool list to Claude.
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Record Claude's reply in the history. We append the raw content
        # blocks unchanged — that's important, because tool_use blocks must be
        # preserved exactly so they can be matched with their results.
        messages.append({"role": "assistant", "content": response.content})

        # STEP 3: walk Claude's reply. Print text; collect tool requests.
        tool_uses = []
        for block in response.content:
            if block.type == "text":
                if block.text.strip():
                    print(f"\nClaude: {block.text.strip()}")
            elif block.type == "tool_use":
                tool_uses.append(block)

        # STEP 4: no tool requests means Claude is done. The text we just
        # printed was its final answer, so we return.
        if not tool_uses:
            return

        # STEP 5: run each requested tool and gather the results.
        tool_results = []
        for block in tool_uses:
            print(f"\n  → {block.name}({_short(json.dumps(block.input), 200)})")

            result_text, is_error = execute_tool(block.name, block.input)

            shown = _preview(result_text)
            print(_indent(("[error] " + shown) if is_error else shown))

            # Each result must reference the id of the tool_use it answers.
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_text,
                "is_error": is_error,
            })

        # Hand the results back as a user turn, then loop to STEP 2.
        messages.append({"role": "user", "content": tool_results})

    # STEP 6: if we fall out of the for-loop, we hit the safety cap.
    print(f"\n[stopped] Reached the step limit ({MAX_STEPS}); the goal may be unfinished.")


# ---------------------------------------------------------------------------
# The REPL — read a goal, run the agent, repeat.
# ---------------------------------------------------------------------------

def main():
    # The SDK reads ANTHROPIC_API_KEY from the environment automatically, but
    # we check first so we can give a friendly message instead of a stack trace.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set.")
        print("  Set it first, e.g.:  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    client = anthropic.Anthropic()  # picks up the key from the environment

    print("Mini Claude Agent — type a goal and watch it work.")
    print(f"Model: {MODEL}")
    print("Commands: 'reset' clears memory, 'exit' or 'quit' leaves.")

    # One shared history for the whole session, so follow-up goals have context.
    # Type 'reset' to start fresh.
    messages = []

    while True:
        try:
            goal = input("\nGoal> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not goal:
            continue
        if goal.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if goal.lower() == "reset":
            messages.clear()
            print("(conversation history cleared)")
            continue

        # STEP 1: add the goal to the history, then run the loop.
        messages.append({"role": "user", "content": goal})

        try:
            run_agent(client, messages)
        except anthropic.APIError as exc:
            # Network / auth / rate-limit problems: report and keep going. The
            # error happens during a create() call, before we append that turn,
            # so the history is still valid for the next goal.
            print(f"\n[API error] {exc}")
        except KeyboardInterrupt:
            # Ctrl-C aborted a goal mid-run. If we stopped right after Claude
            # asked for tools but before we replied with results, the history
            # would end with an unanswered tool_use — which the API rejects.
            # Drop that dangling turn so the next goal still works.
            if messages and messages[-1]["role"] == "assistant":
                last = messages[-1]["content"]
                if isinstance(last, list) and any(
                    getattr(b, "type", None) == "tool_use" for b in last
                ):
                    messages.pop()
            print("\n[interrupted] You can type a new goal.")


if __name__ == "__main__":
    main()
