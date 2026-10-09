"""RegesARC Terminal Chat Client.

Interactive conversation client that connects to the serve API (localhost:8080).
Supports model selection, system prompts, temperature/top_p/max_tokens controls,
conversation history management, and /commands for quick actions.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

# Add parent to path for imports when run directly
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reges.catalog import CATALOG, by_id


# ── ANSI color helpers (cross-platform) ────────────────────────────────

def _supports_color() -> bool:
    """Detect terminal color support."""
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-12)  # STD_OUTPUT_HANDLE
            mode = ctypes.c_ulong()
            return bool(kernel32.GetConsoleMode(handle, ctypes.byref(mode)))
        except Exception:
            return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


COLORS = _supports_color()

if COLORS:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
else:
    RESET = BOLD = DIM = RED = GREEN = YELLOW = BLUE = CYAN = MAGENTA = ""


# ── Settings / State ───────────────────────────────────────────────────

class ChatSettings:
    """Persistent chat configuration."""

    def __init__(self):
        self.model_id: str = "regescore-1.0-35"  # default flagship
        self.system_prompt: str = ""
        self.temperature: float = 0.7
        self.top_p: float = 0.9
        self.max_tokens: int = 2048
        self.context_window: int = 16  # max messages to keep in context

    def save(self, path: Path) -> None:
        data = {
            "model_id": self.model_id,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "context_window": self.context_window,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> "ChatSettings":
        obj = cls()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                for key in ("model_id", "system_prompt"):
                    if key in data:
                        setattr(obj, key, data[key])
                for key in ("temperature", "top_p", "max_tokens", "context_window"):
                    if key in data and isinstance(data[key], (int, float)):
                        setattr(obj, key, data[key])
            except Exception:
                pass  # fall back to defaults
        return obj


# ── API Client ─────────────────────────────────────────────────────────

class ServeClient:
    """HTTP client for the RegesARC serve layer."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")

    def chat_completion(self, messages: list[dict], model_id: str,
                        temperature: float = 0.7, top_p: float = 0.9,
                        max_tokens: int = 2048) -> dict:
        """Send a non-streaming chat completion request."""
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise ConnectionError(f"Cannot reach serve layer at {self.base_url}: {e}")

    def chat_completion_stream(self, messages: list[dict], model_id: str,
                               temperature: float = 0.7, top_p: float = 0.9,
                               max_tokens: int = 2048) -> "StreamIterator":
        """Send a streaming chat completion request (SSE)."""
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return StreamIterator(req, self.base_url)

    def health_check(self) -> bool:
        """Check if the serve layer is running."""
        try:
            req = urllib.request.Request(f"{self.base_url}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False


class StreamIterator:
    """Iterate over SSE stream events from the serve layer."""

    def __init__(self, req: urllib.request.Request, base_url: str):
        self.req = req
        self.base_url = base_url
        self._buffer = ""
        self._done = False

    def __iter__(self) -> "StreamIterator":
        return self

    def __next__(self) -> dict:
        if self._done:
            raise StopIteration

        try:
            resp = urllib.request.urlopen(self.req, timeout=300)
        except Exception as e:
            self._done = True
            raise ConnectionError(f"Stream failed: {e}")

        while not self._done:
            chunk = resp.read(1).decode("utf-8", errors="replace")
            if not chunk:
                break
            self._buffer += chunk

            # Process complete SSE lines
            while "\n" in self._buffer:
                line, _, self._buffer = self._buffer.partition("\n")
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]  # strip "data: " prefix
                if data_str == "[DONE]":
                    self._done = True
                    raise StopIteration
                try:
                    event = json.loads(data_str)
                    yield event
                except json.JSONDecodeError:
                    continue

        self._done = True
        raise StopIteration


# ── Model Selection Helpers ────────────────────────────────────────────

def auto_pick_model(settings: ChatSettings) -> str:
    """Pick the best model for current hardware (heuristic)."""
    # Simple heuristic: prefer smaller models if VRAM is likely limited.
    # In practice, the serve layer handles loading; this just picks a sensible default.
    return settings.model_id


def display_catalog() -> None:
    """Print the full model catalog with verdicts."""
    print(f"\n{BOLD}RegesARC Model Catalog{RESET}")
    print("=" * 70)
    header = f"{'ID':<24} {'Label':<28} {'Params':<16}"
    print(header)
    print("-" * 70)
    for row in CATALOG:
        marker = " ★" if row["id"] == "regescore-1.0-35" else ""
        print(f"{row['id']:<24} {row['label']:<28} {row['params']}{marker}")
    print("-" * 70)
    print(f"\n{DIM}★ = Default / Flagship model{RESET}\n")


# ── Chat Engine ────────────────────────────────────────────────────────

class ChatEngine:
    """Manages conversation state and drives the interactive loop."""

    def __init__(self, client: ServeClient, settings: ChatSettings):
        self.client = client
        self.settings = settings
        self.messages: list[dict] = []  # full history (unbounded)
        self.context_messages: list[dict] = []  # trimmed for API calls

    def _build_api_messages(self) -> list[dict]:
        """Build the message list sent to the API, respecting context window."""
        msgs = []
        if self.settings.system_prompt:
            msgs.append({"role": "system", "content": self.settings.system_prompt})
        # Keep last N messages (excluding system prompt from count)
        recent = self.messages[-self.settings.context_window:]
        msgs.extend(recent)
        return msgs

    def send_message(self, text: str, streaming: bool = False) -> Optional[str]:
        """Send a user message and get the assistant response."""
        if not text.strip():
            return None

        self.messages.append({"role": "user", "content": text})

        api_messages = self._build_api_messages()

        try:
            if streaming:
                full_response = ""
                for event in self.client.chat_completion_stream(
                    api_messages,
                    model_id=self.settings.model_id,
                    temperature=self.settings.temperature,
                    top_p=self.settings.top_p,
                    max_tokens=self.settings.max_tokens,
                ):
                    choices = event.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_response += content
                            print(content, end="", flush=True)
                print()  # newline after stream
            else:
                result = self.client.chat_completion(
                    api_messages,
                    model_id=self.settings.model_id,
                    temperature=self.settings.temperature,
                    top_p=self.settings.top_p,
                    max_tokens=self.settings.max_tokens,
                )
                full_response = result["choices"][0]["message"]["content"]

            if full_response:
                self.messages.append({"role": "assistant", "content": full_response})
                return full_response
            else:
                print(f"{RED}No response received.{RESET}")
                return None

        except ConnectionError as e:
            print(f"\n{RED}{e}{RESET}")
            return None
        except Exception as e:
            print(f"\n{RED}Error: {e}{RESET}")
            return None

    def show_history(self) -> None:
        """Display conversation history."""
        if not self.messages:
            print(DIM + "No messages yet." + RESET)
            return

        for i, msg in enumerate(self.messages):
            role = msg["role"].upper()
            color = CYAN if role == "USER" else GREEN
            content = msg.get("content", "")[:200]  # truncate long messages
            print(f"{color}[{i+1}] {role}:{RESET} {content}")

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
        self.context_messages.clear()
        print(DIM + "Conversation cleared." + RESET)


# ── Interactive Loop ───────────────────────────────────────────────────

def print_banner(settings: ChatSettings) -> None:
    """Print the chat startup banner."""
    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║     RegesARC Terminal Chat Client        ║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════╝{RESET}")
    print()
    print(f"  Model:   {GREEN}{settings.model_id}{RESET}")
    print(f"  Temp:    {settings.temperature}")
    print(f"  Top-P:   {settings.top_p}")
    print(f"  MaxTok:  {settings.max_tokens}")
    if settings.system_prompt:
        print(f"  System:  {DIM}(set){RESET}")
    else:
        print(f"  System:  {DIM}(none){RESET}")
    print()
    print(f"  Type /help for commands, Ctrl+C to exit.")
    print()


def run_interactive(client: ServeClient, settings: ChatSettings) -> None:
    """Run the interactive chat loop."""
    engine = ChatEngine(client, settings)

    # Check connectivity
    if not client.health_check():
        print(f"\n{RED}Warning: Cannot reach serve layer at {client.base_url}{RESET}")
        print(f"  Start it with: python reges/serve.py --port 8080")
        print()

    print_banner(settings)

    # Graceful shutdown handler
    def _shutdown(sig, frame):
        print(f"\n\n{DIM}Shutting down...{RESET}")
        settings.save(_settings_path())
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)

    while True:
        try:
            user_input = input(f"{BOLD}{BLUE}> {RESET}").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            _shutdown(None, None)

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = handle_command(user_input, engine, settings)
            if cmd == "exit":
                settings.save(_settings_path())
                print(f"\n{DIM}Goodbye.{RESET}")
                break
            elif cmd == "help":
                show_help()
            continue

        # Regular message — try streaming first, fall back to non-streaming
        response = engine.send_message(user_input, streaming=True)
        if not response:
            # Retry without streaming on failure
            print(f"{DIM}(retrying without stream...){RESET}")
            engine.send_message(user_input, streaming=False)


def handle_command(cmd_str: str, engine: ChatEngine, settings: ChatSettings) -> Optional[str]:
    """Handle a /command. Returns 'exit' to quit, None otherwise."""
    parts = cmd_str.split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if command == "/help":
        show_help()
        return None

    elif command == "/model":
        if not args:
            print(f"  Current model: {GREEN}{settings.model_id}{RESET}")
            display_catalog()
        else:
            # Try to match by id or label substring
            matched = False
            for row in CATALOG:
                if args.lower() in row["id"].lower() or args.lower() in row["label"].lower():
                    settings.model_id = row["id"]
                    print(f"  Switched to {GREEN}{row['id']}{RESET} ({row['label']})")
                    matched = True
                    break
            if not matched:
                print(f"{RED}Unknown model: {args}{RESET}")
        return None

    elif command == "/clear":
        engine.clear()
        return None

    elif command == "/history":
        engine.show_history()
        return None

    elif command == "/settings":
        if not args:
            print(f"  Model:     {settings.model_id}")
            print(f"  Temp:      {settings.temperature}")
            print(f"  Top-P:     {settings.top_p}")
            print(f"  Max Tokens:{settings.max_tokens}")
            print(f"  Context:   {settings.context_window} messages")
            sys_prompt = settings.system_prompt or "(none)"
            print(f"  System:    {sys_prompt[:60]}{'...' if len(sys_prompt) > 60 else ''}")
        else:
            # Parse key=value pairs
            for pair in args.split():
                if "=" not in pair:
                    continue
                key, _, val = pair.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key == "temperature":
                    try:
                        settings.temperature = float(val)
                        print(f"  Temperature → {settings.temperature}")
                    except ValueError:
                        print(f"{RED}Invalid number: {val}{RESET}")
                elif key == "top_p":
                    try:
                        settings.top_p = float(val)
                        print(f"  Top-P → {settings.top_p}")
                    except ValueError:
                        print(f"{RED}Invalid number: {val}{RESET}")
                elif key == "max_tokens":
                    try:
                        settings.max_tokens = int(val)
                        print(f"  Max Tokens → {settings.max_tokens}")
                    except ValueError:
                        print(f"{RED}Invalid integer: {val}{RESET}")
                elif key == "context_window":
                    try:
                        settings.context_window = int(val)
                        print(f"  Context Window → {settings.context_window}")
                    except ValueError:
                        print(f"{RED}Invalid integer: {val}{RESET}")
                elif key == "system_prompt":
                    # Remove the key= prefix from val
                    settings.system_prompt = val
                    short = val[:50] + ("..." if len(val) > 50 else "")
                    print(f"  System Prompt → {short}")
        return None

    elif command == "/exit" or command == "/quit":
        return "exit"

    elif command == "/catalog":
        display_catalog()
        return None

    else:
        print(f"{RED}Unknown command: {command}. Type /help for options.{RESET}")
        return None


def show_help() -> None:
    """Print help text."""
    print(f"""
{BOLD}Commands:{RESET}
  /model [id|name]   Show catalog or switch model
  /clear             Clear conversation history
  /history           Show full message history
  /settings          Show or set settings (key=value pairs)
  /catalog           Display the full model catalog
  /exit              Save settings and exit

{BOLD}Settings examples:{RESET}
  /settings temperature=0.5 max_tokens=4096
  /settings system_prompt="You are a helpful assistant."

{BOLD}Tips:{RESET}
  - Press Enter to send, Shift+Enter for newline (in most terminals)
  - Conversation history is kept in memory; use /clear to reset
  - Settings persist across sessions (saved to ~/.regesarc_chat.json)
""")


def _settings_path() -> Path:
    """Return the path to persistent settings."""
    home = Path.home()
    return home / ".regesarc_chat.json"


# ── Main Entry Point ───────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="chat.py",
        description="RegesARC Terminal Chat Client",
    )
    parser.add_argument("--port", type=int, default=8080, help="Serve layer port (default: 8080)")
    parser.add_argument("--model", type=str, default=None, help="Override default model")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming responses")
    args = parser.parse_args()

    # Load or create settings
    settings_path = _settings_path()
    settings = ChatSettings.load(settings_path)

    if args.model:
        settings.model_id = args.model

    client = ServeClient(f"http://localhost:{args.port}")

    run_interactive(client, settings)


if __name__ == "__main__":
    main()
