// RepoAgent frontend: chat interface with SSE streaming, tool call visualization.

const API = '/api';
let currentSessionId = null;
let isStreaming = false;

// ── DOM elements ──
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('chatInput');
const btnSend = document.getElementById('btnSend');
const btnNewChat = document.getElementById('btnNewChat');
const healthDot = document.getElementById('healthDot');
const healthText = document.getElementById('healthText');
const modelName = document.getElementById('modelName');
const repoPathText = document.getElementById('repoPathText');
const indexStatus = document.getElementById('indexStatus');
const fileTree = document.getElementById('fileTree');
const sessionsEl = document.getElementById('sessions');

// ── Input handling ──
inputEl.addEventListener('input', () => {
    inputEl.style.height = 'auto';
    inputEl.style.height = inputEl.scrollHeight + 'px';
});

inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

btnSend.addEventListener('click', sendMessage);
btnNewChat.addEventListener('click', () => {
    currentSessionId = null;
    messagesEl.innerHTML = renderWelcome();
    loadSessions();
});

function sendMessage() {
    const text = inputEl.value.trim();
    if (!text || isStreaming) return;

    // Remove welcome if present
    const welcome = messagesEl.querySelector('.welcome');
    if (welcome) welcome.remove();

    // Render user message
    appendUserMessage(text);

    // Clear input
    inputEl.value = '';
    inputEl.style.height = 'auto';

    // Stream response
    streamChat(text);
}

// ── SSE chat streaming ──
async function streamChat(message) {
    isStreaming = true;
    btnSend.disabled = true;

    // Create assistant message container with thinking area
    const msgDiv = document.createElement('div');
    msgDiv.className = 'msg msg-assistant';
    const roleEl = document.createElement('div');
    roleEl.className = 'msg-role';
    roleEl.textContent = 'RepoAgent';
    const contentEl = document.createElement('div');
    contentEl.className = 'msg-content';
    msgDiv.appendChild(roleEl);
    msgDiv.appendChild(contentEl);
    messagesEl.appendChild(msgDiv);

    // Typing indicator
    const typing = document.createElement('div');
    typing.className = 'typing-indicator';
    typing.innerHTML = '<span></span><span></span><span></span>';
    contentEl.appendChild(typing);

    scrollToBottom();

    try {
        const body = JSON.stringify({ message, session_id: currentSessionId });
        const resp = await fetch(`${API}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body,
        });

        // Capture session ID from headers
        const sessionId = resp.headers.get('X-Session-Id');
        if (sessionId) currentSessionId = sessionId;

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let finalText = '';
        let hasContent = false;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split('\n');
            buffer = lines.pop(); // keep incomplete line

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const data = line.slice(6).trim();
                if (!data) continue;

                let event;
                try { event = JSON.parse(data); } catch { continue; }

                // Remove typing indicator on first real content
                if (typing.parentNode && (event.type === 'text' || event.type === 'tool_call'
                    || event.type === 'thinking' || event.type === 'tool_result')) {
                    typing.remove();
                    hasContent = true;
                }

                handleEvent(event, contentEl);
                if (event.type === 'done' || event.type === 'text') {
                    if (event.content) finalText += event.content;
                }
            }
        }

        // If nothing was shown, remove the empty message
        if (!hasContent && !finalText) {
            contentEl.innerHTML = '<em>No response received.</em>';
        }

        // Highlight code blocks
        contentEl.querySelectorAll('pre code').forEach(block => {
            if (window.hljs) hljs.highlightElement(block);
        });

    } catch (err) {
        typing.remove();
        contentEl.innerHTML = `<span style="color: var(--red);">Error: ${err.message}</span>`;
    }

    isStreaming = false;
    btnSend.disabled = false;
    scrollToBottom();
    loadSessions();
}

function handleEvent(event, container) {
    switch (event.type) {
        case 'thinking': {
            const block = document.createElement('div');
            block.className = 'thinking-block';
            const header = document.createElement('div');
            header.className = 'thinking-header';
            header.innerHTML = `<span class="thinking-icon">&#128161;</span> Reasoning (step ${event.iteration || ''})`;
            const content = document.createElement('div');
            content.className = 'thinking-content';
            content.textContent = event.content;
            content.style.display = 'none';
            header.addEventListener('click', () => {
                content.style.display = content.style.display === 'none' ? 'block' : 'none';
            });
            block.appendChild(header);
            block.appendChild(content);
            container.appendChild(block);
            break;
        }
        case 'tool_call': {
            const block = document.createElement('div');
            block.className = 'tool-call-block';
            const nameEl = document.createElement('div');
            nameEl.className = 'tool-call-name';
            nameEl.textContent = '\u25B8 ' + event.content;
            if (event.arguments && Object.keys(event.arguments).length) {
                const argsEl = document.createElement('div');
                argsEl.className = 'tool-call-args';
                argsEl.textContent = JSON.stringify(event.arguments);
                block.appendChild(nameEl);
                block.appendChild(argsEl);
            } else {
                block.appendChild(nameEl);
            }
            container.appendChild(block);
            break;
        }
        case 'tool_result': {
            // Find the last tool-call-block and append result
            const lastTool = container.querySelector('.tool-call-block:last-child');
            if (lastTool) {
                const result = document.createElement('div');
                result.className = 'tool-result';
                result.textContent = event.content;
                result.style.display = 'none';
                // Click to toggle
                lastTool.addEventListener('click', () => {
                    result.style.display = result.style.display === 'none' ? 'block' : 'none';
                });
                lastTool.style.cursor = 'pointer';
                lastTool.appendChild(result);
            }
            break;
        }
        case 'text': {
            const textEl = document.createElement('div');
            textEl.innerHTML = renderMarkdown(event.content);
            container.appendChild(textEl);
            break;
        }
        case 'done': {
            if (event.content) {
                // Final text already handled by 'text' event
            }
            break;
        }
        case 'error': {
            const errEl = document.createElement('div');
            errEl.style.color = 'var(--red)';
            errEl.textContent = event.content;
            container.appendChild(errEl);
            break;
        }
    }
    scrollToBottom();
}

// ── Rendering helpers ──
function appendUserMessage(text) {
    const msg = document.createElement('div');
    msg.className = 'msg msg-user';
    const content = document.createElement('div');
    content.className = 'msg-content';
    content.textContent = text;
    msg.appendChild(content);
    messagesEl.appendChild(msg);
    scrollToBottom();
}

function renderWelcome() {
    return `<div class="welcome">
        <div class="welcome-icon">&#128269;</div>
        <h2>RepoAgent</h2>
        <p>Privacy-first local code intelligence. Ask anything about this codebase.</p>
    </div>`;
}

function renderMarkdown(text) {
    // Simple markdown: code blocks, inline code, bold, paragraphs
    let html = text;

    // Code blocks (```...```)
    html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (m, lang, code) => {
        const langClass = lang ? ` class="language-${lang}"` : '';
        return `<pre><code${langClass}>${escapeHtml(code.trim())}</code></pre>`;
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Paragraphs
    html = html.split('\n\n').map(p => `<p>${p.replace(/\n/g, '<br>')}</p>`).join('');

    return html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ── Health check ──
async function checkHealth() {
    try {
        const resp = await fetch(`${API}/health`);
        const data = await resp.json();

        if (data.llm_available) {
            healthDot.className = 'status-dot ok';
            healthText.textContent = 'LLM Online';
        } else {
            healthDot.className = 'status-dot error';
            healthText.textContent = 'LLM Offline';
        }

        if (data.gpu_name) {
            modelName.textContent = `${data.llm_model} | ${data.gpu_name}`;
        } else {
            modelName.textContent = data.llm_model || '';
        }

        indexStatus.textContent = data.chunk_count > 0
            ? `${data.chunk_count} chunks indexed`
            : 'Not indexed';

        if (data.indexing) {
            healthDot.className = 'status-dot warn';
            healthText.textContent = 'Indexing...';
        }
    } catch {
        healthDot.className = 'status-dot error';
        healthText.textContent = 'Server Offline';
    }
}

// ── File browser ──
async function loadFileTree(path = '.') {
    try {
        const resp = await fetch(`${API}/workspace/tree?path=${encodeURIComponent(path)}`);
        const data = await resp.json();
        fileTree.innerHTML = '';

        data.entries.forEach(entry => {
            const el = document.createElement('div');
            el.className = 'file-entry' + (entry.is_dir ? ' dir' : '');
            el.textContent = (entry.is_dir ? '\u{1F4C1} ' : '\u{1F4C4} ') + entry.name;
            if (entry.is_dir) {
                el.addEventListener('click', () => {
                    const newPath = path === '.' ? entry.name : `${path}/${entry.name}`;
                    loadFileTree(newPath);
                });
            }
            fileTree.appendChild(el);
        });

        if (data.entries.length === 0) {
            fileTree.innerHTML = '<div class="file-entry" style="opacity:0.5">Empty</div>';
        }
    } catch {
        fileTree.innerHTML = '<div class="file-entry" style="opacity:0.5">Unavailable</div>';
    }
}

// ── Sessions ──
async function loadSessions() {
    try {
        const resp = await fetch(`${API}/sessions`);
        const data = await resp.json();
        sessionsEl.innerHTML = '';

        data.sessions.forEach(s => {
            const el = document.createElement('div');
            el.className = 'session-item';
            if (s.session_id === currentSessionId) el.classList.add('active');
            el.textContent = s.title || s.session_id;
            el.addEventListener('click', () => loadSession(s.session_id));
            sessionsEl.appendChild(el);
        });

        if (data.sessions.length === 0) {
            sessionsEl.innerHTML = '<div class="file-entry" style="opacity:0.5">No sessions</div>';
        }
    } catch {
        // Silent fail
    }
}

async function loadSession(sessionId) {
    currentSessionId = sessionId;
    try {
        const resp = await fetch(`${API}/sessions/${sessionId}`);
        const data = await resp.json();
        messagesEl.innerHTML = '';

        data.messages.forEach(msg => {
            if (msg.role === 'user') {
                appendUserMessage(msg.content);
            } else if (msg.role === 'assistant') {
                const div = document.createElement('div');
                div.className = 'msg msg-assistant';
                const role = document.createElement('div');
                role.className = 'msg-role';
                role.textContent = 'RepoAgent';
                const content = document.createElement('div');
                content.className = 'msg-content';
                content.innerHTML = renderMarkdown(msg.content);
                div.appendChild(role);
                div.appendChild(content);
                messagesEl.appendChild(div);
            }
        });

        messagesEl.querySelectorAll('pre code').forEach(block => {
            if (window.hljs) hljs.highlightElement(block);
        });

        loadSessions();
        scrollToBottom();
    } catch {
        // Silent fail
    }
}

// ── Init ──
checkHealth();
loadSessions();
setInterval(checkHealth, 15000);
