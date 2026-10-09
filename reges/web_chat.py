"""RegesARC Web Chat UI — single-file, no build step.

Serves a self-contained HTML page (embedded CSS + JS) that connects to the
serve layer at localhost:8080 via the OpenAI-compatible /v1/chat/completions
endpoint with stream=true for SSE token-by-token display.

Usage:
    python reges/web_chat.py [--port 8080] [--host 0.0.0.0]

The HTML page is served at http://<host>:<web_port>/ — default web port is
9090 so it doesn't conflict with the serve layer's 8080.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path


# ── Embedded HTML (single file, no external deps) ──────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RegesARC Chat</title>
<style>
  :root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-tertiary: #21262d;
    --border: #30363d;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --accent: #58a6ff;
    --accent-hover: #79c0ff;
    --user-bubble: #1f2d3d;
    --assistant-bubble: #161b22;
    --system-color: #f0883e;
    --error-color: #f85149;
    --success-color: #3fb950;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* Header */
  .header {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    padding: 12px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-shrink: 0;
  }

  .header h1 {
    font-size: 18px;
    font-weight: 600;
    color: var(--accent);
    white-space: nowrap;
  }

  .header-controls {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }

  .header-controls select,
  .header-controls button {
    background: var(--bg-tertiary);
    color: var(--text-primary);
    border: 1px solid var(--border);
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .header-controls select:hover,
  .header-controls button:hover {
    border-color: var(--accent);
  }

  .header-controls button.active {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }

  /* Settings Panel */
  .settings-panel {
    display: none;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    padding: 16px 20px;
    gap: 16px;
    flex-wrap: wrap;
    align-items: end;
  }

  .settings-panel.open { display: flex; }

  .setting-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .setting-group label {
    font-size: 12px;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .setting-group input,
  .setting-group textarea {
    background: var(--bg-tertiary);
    color: var(--text-primary);
    border: 1px solid var(--border);
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 13px;
    width: 200px;
    max-width: 300px;
  }

  .setting-group textarea { width: 100%; min-height: 60px; resize: vertical; }

  /* Chat Area */
  .chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .message {
    max-width: 85%;
    padding: 12px 16px;
    border-radius: 12px;
    line-height: 1.6;
    font-size: 14px;
    word-wrap: break-word;
    white-space: pre-wrap;
  }

  .message.user {
    align-self: flex-end;
    background: var(--user-bubble);
    border: 1px solid var(--border);
    border-bottom-right-radius: 4px;
  }

  .message.assistant {
    align-self: flex-start;
    background: var(--assistant-bubble);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
  }

  .message.system {
    align-self: center;
    background: transparent;
    color: var(--system-color);
    font-size: 12px;
    padding: 8px;
    text-align: center;
  }

  .message .role-label {
    font-size: 11px;
    color: var(--text-secondary);
    margin-bottom: 4px;
    font-weight: 600;
    text-transform: uppercase;
  }

  /* Markdown rendering (basic) */
  .message.assistant code {
    background: var(--bg-tertiary);
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'Fira Code', 'Consolas', monospace;
    font-size: 13px;
  }

  .message.assistant pre {
    background: var(--bg-tertiary);
    padding: 12px;
    border-radius: 8px;
    overflow-x: auto;
    margin: 8px 0;
  }

  .message.assistant pre code {
    background: none;
    padding: 0;
  }

  /* Input Area */
  .input-area {
    background: var(--bg-secondary);
    border-top: 1px solid var(--border);
    padding: 16px 20px;
    flex-shrink: 0;
  }

  .input-wrapper {
    display: flex;
    gap: 12px;
    align-items: end;
    max-width: 900px;
    margin: 0 auto;
  }

  .input-wrapper textarea {
    flex: 1;
    background: var(--bg-tertiary);
    color: var(--text-primary);
    border: 1px solid var(--border);
    padding: 12px 16px;
    border-radius: 10px;
    font-size: 14px;
    font-family: inherit;
    resize: none;
    min-height: 44px;
    max-height: 200px;
    line-height: 1.5;
    transition: border-color 0.2s;
  }

  .input-wrapper textarea:focus {
    outline: none;
    border-color: var(--accent);
  }

  .send-btn {
    background: var(--accent);
    color: #fff;
    border: none;
    padding: 12px 20px;
    border-radius: 10px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
  }

  .send-btn:hover { background: var(--accent-hover); }
  .send-btn:disabled { opacity: 0.5; cursor: not-allowed; }

  /* Status bar */
  .status-bar {
    text-align: center;
    padding: 4px;
    font-size: 11px;
    color: var(--text-secondary);
    background: var(--bg-primary);
  }

  .status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
  }

  .status-dot.online { background: var(--success-color); }
  .status-dot.offline { background: var(--error-color); }

  /* Typing indicator */
  .typing-indicator {
    display: none;
    align-self: flex-start;
    padding: 12px 16px;
    background: var(--assistant-bubble);
    border: 1px solid var(--border);
    border-radius: 12px;
    border-bottom-left-radius: 4px;
  }

  .typing-indicator.visible { display: flex; gap: 4px; align-items: center; }

  .typing-dot {
    width: 8px;
    height: 8px;
    background: var(--text-secondary);
    border-radius: 50%;
    animation: typing 1.4s infinite;
  }

  .typing-dot:nth-child(2) { animation-delay: 0.2s; }
  .typing-dot:nth-child(3) { animation-delay: 0.4s; }

  @keyframes typing {
    0%, 60%, 100% { opacity: 0.3; transform: translateY(0); }
    30% { opacity: 1; transform: translateY(-4px); }
  }

  /* Responsive */
  @media (max-width: 768px) {
    .header { flex-direction: column; align-items: stretch; }
    .header-controls { justify-content: center; }
    .message { max-width: 95%; }
    .setting-group input,
    .setting-group textarea { width: 100%; max-width: none; }
    .settings-panel { flex-direction: column; }
  }

  /* Scrollbar styling */
  ::-webkit-scrollbar { width: 8px; }
  ::-webkit-scrollbar-track { background: var(--bg-primary); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }
</style>
</head>
<body>

<div class="header">
  <h1>⚡ RegesARC Chat</h1>
  <div class="header-controls">
    <select id="modelSelect"></select>
    <button id="settingsBtn" onclick="toggleSettings()">⚙ Settings</button>
    <button id="clearBtn" onclick="clearChat()">🗑 Clear</button>
  </div>
</div>

<div class="settings-panel" id="settingsPanel">
  <div class="setting-group">
    <label>Temperature</label>
    <input type="number" id="tempInput" min="0" max="2" step="0.1" value="0.7">
  </div>
  <div class="setting-group">
    <label>Top P</label>
    <input type="number" id="topPInput" min="0" max="1" step="0.05" value="0.9">
  </div>
  <div class="setting-group">
    <label>Max Tokens</label>
    <input type="number" id="maxTokensInput" min="64" max="8192" step="256" value="2048">
  </div>
  <div class="setting-group">
    <label>System Prompt</label>
    <textarea id="systemPromptInput" placeholder="Optional system prompt..."></textarea>
  </div>
</div>

<div class="chat-area" id="chatArea">
  <div class="message system">Welcome to RegesARC Chat. Select a model and start chatting.</div>
  <div class="typing-indicator" id="typingIndicator">
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
    <div class="typing-dot"></div>
  </div>
</div>

<div class="input-area">
  <div class="input-wrapper">
    <textarea id="userInput" placeholder="Type your message..." rows="1"
              onkeydown="handleKeyDown(event)"></textarea>
    <button class="send-btn" id="sendBtn" onclick="sendMessage()">Send</button>
  </div>
</div>

<div class="status-bar">
  <span class="status-dot offline" id="statusDot"></span>
  <span id="statusText">Checking connection...</span>
</div>

<script>
// ── State ──────────────────────────────────────────────────────────────
const API_BASE = 'http://localhost:8080';
let messages = []; // {role, content}
let isStreaming = false;
let currentStreamMsg = null;

const CATALOG = [
  { id: 'regescore-1.0-35', label: 'RegesCore 1.0 35B (Flagship)' },
  { id: 'qwen3.8-flash-next', label: 'Qwen3.8-Flash-Next' },
  { id: 'qwen3.6', label: 'Qwen3.6-35B-A3B' },
  { id: 'deepseek-v4-flash', label: 'DeepSeek V4 Flash' },
  { id: 'deepseek-v4.1-flash', label: 'DeepSeek V4.1 Flash' },
  { id: 'glm-5.3-flash', label: 'GLM-5.3-Flash' },
  { id: 'glm-5.2', label: 'GLM-5.2' },
  { id: 'inkling', label: 'Inkling' },
  { id: 'kimi-k3', label: 'Kimi K3' },
  { id: 'olmoe', label: 'OLMoE-7B-7B' },
  { id: 'laya', label: 'Laya' },
];

// ── Init ───────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  populateModelSelect();
  checkHealth();
  setInterval(checkHealth, 30000); // poll every 30s
});

function populateModelSelect() {
  const sel = document.getElementById('modelSelect');
  CATALOG.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = m.label;
    if (m.id === 'regescore-1.0-35') opt.selected = true;
    sel.appendChild(opt);
  });
}

async function checkHealth() {
  try {
    const resp = await fetch(`${API_BASE}/health`, { method: 'GET' });
    if (resp.ok) {
      document.getElementById('statusDot').className = 'status-dot online';
      document.getElementById('statusText').textContent = 'Connected to serve layer';
    } else {
      throw new Error('Not OK');
    }
  } catch {
    document.getElementById('statusDot').className = 'status-dot offline';
    document.getElementById('statusText').textContent = 'Serve layer not reachable';
  }
}

// ── Chat Functions ─────────────────────────────────────────────────────
function toggleSettings() {
  const panel = document.getElementById('settingsPanel');
  const btn = document.getElementById('settingsBtn');
  panel.classList.toggle('open');
  btn.classList.toggle('active');
}

function clearChat() {
  messages = [];
  const chatArea = document.getElementById('chatArea');
  // Remove all message elements (keep typing indicator)
  chatArea.querySelectorAll('.message').forEach(el => el.remove());
  addSystemMessage('Conversation cleared.');
}

function addMessage(role, content) {
  const chatArea = document.getElementById('chatArea');
  const typingEl = document.getElementById('typingIndicator');

  const div = document.createElement('div');
  div.className = `message ${role}`;

  if (role !== 'system') {
    const label = document.createElement('div');
    label.className = 'role-label';
    label.textContent = role === 'user' ? 'You' : 'Assistant';
    div.appendChild(label);
  }

  const contentDiv = document.createElement('div');
  if (role === 'assistant') {
    // Basic markdown rendering
    contentDiv.innerHTML = renderMarkdown(content);
  } else {
    contentDiv.textContent = content;
  }
  div.appendChild(contentDiv);

  chatArea.insertBefore(div, typingEl);
  scrollToBottom();
}

function addSystemMessage(text) {
  const chatArea = document.getElementById('chatArea');
  const typingEl = document.getElementById('typingIndicator');

  const div = document.createElement('div');
  div.className = 'message system';
  div.textContent = text;
  chatArea.insertBefore(div, typingEl);
  scrollToBottom();
}

function renderMarkdown(text) {
  // Basic markdown: code blocks, inline code, bold, italic
  let html = escapeHtml(text);

  // Code blocks (``` ... ```)
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
    return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
  });

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Bold
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

  // Italic
  html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

  // Line breaks
  html = html.replace(/\n/g, '<br>');

  return html;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function scrollToBottom() {
  const chatArea = document.getElementById('chatArea');
  chatArea.scrollTop = chatArea.scrollHeight;
}

async function sendMessage() {
  if (isStreaming) return;

  const input = document.getElementById('userInput');
  const text = input.value.trim();
  if (!text) return;

  // Add user message to UI and state
  addMessage('user', text);
  messages.push({ role: 'user', content: text });
  input.value = '';
  input.style.height = 'auto';

  // Build API messages (with system prompt + history)
  const apiMessages = [];
  const sysPrompt = document.getElementById('systemPromptInput').value.trim();
  if (sysPrompt) {
    apiMessages.push({ role: 'system', content: sysPrompt });
  }
  apiMessages.push(...messages);

  // Show typing indicator
  isStreaming = true;
  updateSendButton(true);
  document.getElementById('typingIndicator').classList.add('visible');

  try {
    const modelId = document.getElementById('modelSelect').value;
    const temperature = parseFloat(document.getElementById('tempInput').value) || 0.7;
    const topP = parseFloat(document.getElementById('topPInput').value) || 0.9;
    const maxTokens = parseInt(document.getElementById('maxTokensInput').value) || 2048;

    // Try streaming first
    await streamResponse(apiMessages, modelId, temperature, topP, maxTokens);
  } catch (err) {
    addSystemMessage(`Error: ${err.message}`);
  } finally {
    isStreaming = false;
    updateSendButton(false);
    document.getElementById('typingIndicator').classList.remove('visible');
  }
}

async function streamResponse(apiMessages, modelId, temperature, topP, maxTokens) {
  const resp = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: modelId,
      messages: apiMessages,
      temperature,
      top_p: topP,
      max_tokens: maxTokens,
      stream: true,
    }),
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ error: 'HTTP ' + resp.status }));
    throw new Error(err.error || `Request failed (${resp.status})`);
  }

  // Create assistant message element for streaming
  const chatArea = document.getElementById('chatArea');
  const typingEl = document.getElementById('typingIndicator');

  const div = document.createElement('div');
  div.className = 'message assistant';
  const label = document.createElement('div');
  label.className = 'role-label';
  label.textContent = 'Assistant';
  div.appendChild(label);
  const contentDiv = document.createElement('div');
  div.appendChild(contentDiv);
  chatArea.insertBefore(div, typingEl);

  let fullText = '';
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || ''; // keep incomplete line in buffer

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data: ')) continue;
      const dataStr = trimmed.slice(6);
      if (dataStr === '[DONE]') return;

      try {
        const event = JSON.parse(dataStr);
        const delta = event.choices?.[0]?.delta?.content;
        if (delta) {
          fullText += delta;
          contentDiv.innerHTML = renderMarkdown(fullText);
          scrollToBottom();
        }
      } catch {}
    }
  }

  // Finalize: add to messages state
  messages.push({ role: 'assistant', content: fullText });
}

async function fallbackNonStream(apiMessages, modelId, temperature, topP, maxTokens) {
  const resp = await fetch(`${API_BASE}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: modelId,
      messages: apiMessages,
      temperature,
      top_p: topP,
      max_tokens: maxTokens,
    }),
  });

  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const data = await resp.json();
  return data.choices[0].message.content;
}

function updateSendButton(streaming) {
  const btn = document.getElementById('sendBtn');
  btn.disabled = streaming;
  btn.textContent = streaming ? '...' : 'Send';
}

function handleKeyDown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  } else if (e.key === 'Enter' && e.shiftKey) {
    // Allow newline in textarea
    setTimeout(() => {
      const ta = document.getElementById('userInput');
      ta.style.height = 'auto';
      ta.style.height = ta.scrollHeight + 'px';
    }, 0);
  }

  // Auto-resize textarea
  const ta = e.target;
  ta.style.height = 'auto';
  ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
}
</script>
</body>
</html>"""


# ── HTTP Server for the HTML page ──────────────────────────────────────

class WebChatHandler(BaseHTTPRequestHandler):
    """Serves the embedded HTML chat UI."""

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        else:
            self.send_error(404, "Not Found")

    def log_message(self, format, *args):
        print(f"[web_chat] {format % args}")


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="web_chat.py",
        description="RegesARC Web Chat UI (single HTML file, no build step)",
    )
    parser.add_argument("--port", type=int, default=9090, help="Web UI port (default: 9090)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind address")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), WebChatHandler)

    print(f"\nRegesARC Web Chat UI")
    print("=" * 60)
    print(f"  Open in browser: http://localhost:{args.port}")
    print(f"  API endpoint:    http://localhost:8080/v1/chat/completions")
    print(f"  Bind address:    {args.host}:{args.port}")
    print("=" * 60)
    print("\nPress Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
