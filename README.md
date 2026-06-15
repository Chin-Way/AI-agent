# Mini Claude Agent

A tiny command-line AI agent in one file (`agent.py`), built to **learn from**.
It's Claude running in a loop, able to use a few local tools to accomplish a
goal you give it, stopping when the goal is done.

It's intentionally small and heavily commented — the point is that you can read
the whole thing in a sitting and understand exactly how an agent works.

## What an "agent" actually is

An agent is just a large language model in a loop with tools. There's no magic:

1. **You give it a goal.** It's added to the conversation history.
2. **We send the whole history + a list of tools to Claude.**
3. **Claude replies.** Its reply is a list of "content blocks":
   - *text* — something it wants to say to you, or
   - *tool_use* — a request to run one of our tools, with arguments.
4. **No tool requests?** Claude is done — we print its answer and stop.
5. **Tool requests?** We run each tool locally, send the results back to
   Claude as a `tool_result`, and **go back to step 2**.
6. A **step limit** (`MAX_STEPS`) makes sure the loop can never run forever.

```
   you ──goal──▶ history ──▶ Claude ──▶ tool_use? ──no──▶ final answer ──▶ you
                    ▲                        │
                    │                       yes
                    └──── tool_result ◀── run tools locally
```

One important detail: **Claude is stateless.** It doesn't remember anything
between API calls, so we resend the entire growing message list every time.
That list *is* the agent's memory.

## The tools

The agent starts with four local tools (defined in `agent.py`):

| Tool             | What it does                                            |
| ---------------- | ------------------------------------------------------- |
| `read_file`      | Return a file's contents.                               |
| `write_file`     | Create or overwrite a file.                             |
| `list_directory` | List files and folders.                                 |
| `run_command`    | Run a shell command and return stdout/stderr/exit code. |

Each tool is two things working together: a **JSON schema** (how we *describe*
the tool to Claude) and a **Python function** (what actually runs). If a tool
fails, the error is caught and sent back to Claude as text — it never crashes
the program.

> ⚠️ **Safety note:** `run_command` runs real shell commands on your machine.
> The agent prints each command before running it (`⚠ about to run shell
> command: ...`). Only give goals you trust, and read what it's doing — this is
> a learning tool, not a sandbox.

## Setup

You need Python 3.8+ and an Anthropic API key.

```bash
# 1. (recommended) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install the dependency
pip install -r requirements.txt

# 3. set your API key (get one at https://console.anthropic.com/)
export ANTHROPIC_API_KEY=sk-ant-...   # Windows: set ANTHROPIC_API_KEY=sk-ant-...
```

## Run it

```bash
python agent.py
```

You'll get a `Goal>` prompt. Type a goal in plain English and watch each step:

```
Goal> Create a file hello.txt containing a haiku, then read it back to me.

--- step 1 ---
Claude: I'll write the haiku to hello.txt.
  → write_file({"path": "hello.txt", "content": "..."})
    Wrote 57 characters to hello.txt.

--- step 2 ---
  → read_file({"path": "hello.txt"})
    ...the haiku...

--- step 3 ---
Claude: Done — I created hello.txt with a haiku and read it back above.
```

REPL commands:

- **any text** — a goal for the agent
- `reset` — clear the conversation memory and start fresh
- `exit` / `quit` (or Ctrl-D) — leave

The conversation persists between goals during a session, so you can give
follow-ups like *"now add a second haiku to that file."* Use `reset` to forget.

## Swapping the model

The model is a single clearly-marked constant near the top of `agent.py`:

```python
MODEL = "claude-sonnet-4-6"
```

Change it to `"claude-opus-4-8"` (more capable) or `"claude-haiku-4-5"`
(faster/cheaper) — that's the only edit needed. Other handy knobs live right
beside it: `MAX_TOKENS`, `MAX_STEPS`, `COMMAND_TIMEOUT`.

## Where to go next

A few small, satisfying ways to extend this:

1. **Add a web-search tool.** Write a `web_search(query)` function (e.g. using
   an HTTP API), add its JSON schema to `TOOLS`, and register it in
   `TOOL_FUNCTIONS`. The loop already handles everything else — this is the
   best way to see how adding a tool "just works."
2. **Add a confirmation prompt for `run_command`.** Before running, ask the
   user `Run this? [y/N]` and return "user declined" if they say no. A first
   taste of human-in-the-loop safety.
3. **Stream the responses.** Swap `client.messages.create(...)` for
   `client.messages.stream(...)` so Claude's text appears token-by-token
   instead of all at once.

Other ideas once those feel easy: persist the conversation to a JSON file so it
survives restarts, or add a `MAX_TOKENS`-aware summarizer when the history gets
long.
