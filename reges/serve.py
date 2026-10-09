"""RegesARC Serve Layer.

OpenAI + Anthropic compatible API on one port.
Handles both chat completions (OpenAI) and messages (Anthropic).
Routes to the correct engine adapter based on model family.

Supports:
  - Non-streaming responses (default)
  - Streaming responses via SSE (stream=true in OpenAI endpoint)
"""

from __future__ import annotations

import json
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional

# Add parent to path for imports when run directly
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engines import get_adapter, list_families  # noqa: E402


class RegesServeHandler(BaseHTTPRequestHandler):
    """HTTP handler for OpenAI + Anthropic compatible API."""
    
    def do_POST(self):
        """Handle POST requests (chat completions / messages)."""
        
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        
        try:
            data = json.loads(body.decode('utf-8'))
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return
        
        # Determine endpoint type
        if self.path == "/v1/chat/completions":
            stream = data.get("stream", False)
            if stream:
                self._handle_openai_stream(data)
            else:
                self._handle_openai(data)
        elif self.path == "/v1/messages":
            self._handle_anthropic(data)
        else:
            self.send_error(404, "Not Found")
    
    def _handle_openai(self, data: dict):
        """Handle OpenAI-style chat completions (non-streaming)."""
        
        model_id = data.get("model", "")
        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens", 256)
        
        # Get the last user message as prompt
        prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break
        
        if not prompt:
            self._send_json({"error": "No user message found"}, 400)
            return
        
        # Load and run model
        adapter = get_adapter(model_id)
        if not adapter:
            self._send_json({
                "error": f"Unknown model family: {model_id}",
                "supported": list_families()
            }, 404)
            return
        
        try:
            # Load model from RegesModels directory
            store = Path(__file__).resolve().parent.parent / "RegesModels"
            model_path = store / model_id
            
            if not model_path.exists():
                self._send_json({
                    "error": f"Model {model_id} not installed. Run: python reges/install.py",
                    "hint": "Install models first with the installer."
                }, 404)
                return
            
            adapter.load_model(model_path)
            response = adapter.generate(prompt, max_tokens)
            
            # Format as OpenAI response
            result = {
                "id": f"chatcmpl-{model_id}",
                "object": "chat.completion",
                "created": 1728000000,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(response.split()),
                    "total_tokens": len(prompt.split()) + len(response.split())
                }
            }
            
            self._send_json(result)
        
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_openai_stream(self, data: dict):
        """Handle OpenAI-style streaming chat completions via SSE."""
        
        model_id = data.get("model", "")
        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens", 256)

        # Get the last user message as prompt
        prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break
        
        if not prompt:
            self._send_json({"error": "No user message found"}, 400)
            return

        # Load and run model
        adapter = get_adapter(model_id)
        if not adapter:
            self._send_error_sse(f"Unknown model family: {model_id}", 404)
            return
        
        try:
            store = Path(__file__).resolve().parent.parent / "RegesModels"
            model_path = store / model_id
            
            if not model_path.exists():
                self._send_error_sse(f"Model {model_id} not installed.", 404)
                return
            
            adapter.load_model(model_path)
            response = adapter.generate(prompt, max_tokens)

            # Send SSE stream: yield the full response in chunks to simulate streaming.
            # If the adapter supports incremental generation (e.g., a generator), use it.
            # Otherwise chunk the final string for SSE delivery.
            self._send_sse_header()

            if hasattr(adapter, 'generate_stream'):
                # Adapter provides token-by-token streaming
                first = True
                for token in adapter.generate_stream(prompt, max_tokens):
                    event = {
                        "id": f"chatcmpl-{model_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": token} if first else {},
                            "finish_reason": None
                        }]
                    }
                    self._send_sse_event(event)
                    first = False

                # Send final chunk with finish_reason
                final_event = {
                    "id": f"chatcmpl-{model_id}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                self._send_sse_event(final_event)
            else:
                # No incremental API — chunk the full response for SSE delivery.
                # This gives the same wire format as streaming clients expect,
                # just with larger chunks rather than true token-by-token.
                words = response.split()
                if not words:
                    # Empty response — send a minimal event
                    self._send_sse_event({
                        "id": f"chatcmpl-{model_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": ""},
                            "finish_reason": None
                        }]
                    })
                else:
                    # Send first word with index=0 delta
                    self._send_sse_event({
                        "id": f"chatcmpl-{model_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": words[0]},
                            "finish_reason": None
                        }]
                    })

                    # Send remaining words as subsequent chunks
                    for word in words[1:]:
                        self._send_sse_event({
                            "id": f"chatcmpl-{model_id}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_id,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": word},
                                "finish_reason": None
                            }]
                        })

                    # Send final chunk with finish_reason=stop
                    self._send_sse_event({
                        "id": f"chatcmpl-{model_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    })

            # Send [DONE] terminator
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        except Exception as e:
            error_event = {
                "error": {"message": str(e), "type": "server_error", "code": 500}
            }
            self._send_sse_event(error_event)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    def _handle_anthropic(self, data: dict):
        """Handle Anthropic-style messages."""
        
        model_id = data.get("model", "")
        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens", 256)
        
        # Get the last user message as prompt
        prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
                break
        
        if not prompt:
            self._send_json({"error": "No user message found"}, 400)
            return
        
        # Load and run model
        adapter = get_adapter(model_id)
        if not adapter:
            self._send_json({
                "error": f"Unknown model family: {model_id}",
                "supported": list_families()
            }, 404)
            return
        
        try:
            store = Path(__file__).resolve().parent.parent / "RegesModels"
            model_path = store / model_id
            
            if not model_path.exists():
                self._send_json({
                    "error": f"Model {model_id} not installed."
                }, 404)
                return
            
            adapter.load_model(model_path)
            response = adapter.generate(prompt, max_tokens)
            
            # Format as Anthropic response
            result = {
                "id": f"msg_{model_id}",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": response
                    }
                ],
                "stop_reason": "end_turn",
                "model": model_id
            }
            
            self._send_json(result)
        
        except Exception as e:
            self._send_json({"error": str(e)}, 500)
    
    def do_GET(self):
        """Handle GET requests (health check, status)."""
        
        if self.path == "/health" or self.path == "/" or self.path == "/index.html":
            result = {
                "status": "ok",
                "version": "0.1.0",
                "supported_models": list_families(),
                "endpoints": ["/v1/chat/completions", "/v1/messages"],
                "streaming_supported": True,
            }
            self._send_json(result)
        else:
            self.send_error(404, "Not Found")
    
    def _send_json(self, data: dict, status: int = 200):
        """Send JSON response."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))

    # ── SSE helpers ────────────────────────────────────────────────────

    def _send_sse_header(self):
        """Send SSE response headers."""
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def _send_sse_event(self, data: dict):
        """Send a single SSE event."""
        payload = json.dumps(data)
        self.wfile.write(f"data: {payload}\n\n".encode('utf-8'))
        self.wfile.flush()

    def _send_error_sse(self, message: str, status: int = 400):
        """Send an SSE error event and terminate."""
        self._send_sse_header()
        error_event = {
            "error": {"message": message, "type": "invalid_request_error", "code": status}
        }
        self._send_sse_event(error_event)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format, *args):
        """Suppress default logging or customize."""
        print(f"[serve] {format % args}")


def serve(port: int = 8080) -> None:
    """Start the HTTP server on the given port."""
    server = HTTPServer(('0.0.0.0', port), RegesServeHandler)
    
    print(f"\nRegesARC Serve Layer")
    print("=" * 60)
    print(f"Listening on http://localhost:{port}")
    print(f"Endpoints:")
    print(f"  GET  /              - Health check")
    print(f"  POST /v1/chat/completions  - OpenAI compatible")
    print(f"  POST /v1/messages        - Anthropic compatible")
    print(f"\nSupported models: {', '.join(list_families())}")
    print("=" * 60)
    print("\nPress Ctrl+C to stop.\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(prog="serve.py")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    args = parser.parse_args()
    
    serve(args.port)
