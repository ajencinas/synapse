#!/usr/bin/env python3
"""
Synapse Chatbot — internet-connected inference via ngrok.
Auto-downloads the latest checkpoint + tokenizer from Google Drive.
Requires GPU (CUDA).

SFT prompting (train == inference):
  tools ON  -> system = CANONICAL_TOOL_SYSTEM (the prompt every tool_use /
               tool_negative record was trained with); <|tool_call|> turns are
               executed by sft/tool_loop.py (python sandbox) and fed back.
  tools OFF -> no system prompt (every prose source was trained with none).
No other system prompt is ever sent: the model has seen exactly these two.

pip install torch flask tokenizers

Usage:
  python sparky_chatbot.py                                    # auto-download + ngrok
  python sparky_chatbot.py --ckpt model.pth --tokenizer tok.json
  python sparky_chatbot.py --no-ngrok --no-download
"""
import json
import os
import re
import sys
import time
import signal
import socket
import argparse
import subprocess
import gc
import threading

import torch
from flask import Flask, request, Response, stream_with_context, jsonify

from sparky_model import SynapseInfer, VOCAB_SIZE, BLOCK_SIZE
from sparky_chat_template import build_sft_prompt, sft_stop_token_ids

# Tool loop + the ONE tool system prompt live in sft/ (shared with the eval).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sft"))
from tool_loop import run_tool_loop, execute as tool_execute
from tools_runtime import CANONICAL_TOOL_SYSTEM

# ── forced-search assist ────────────────────────────────────────────────────
# The v3 model searches on short factoid questions (its whole training
# distribution) but NOT on news/current-events phrasing or "search for X"
# instructions — zero such training examples. Until v3.1 adds that data, the
# CHATBOT detects the intent, issues the search itself, and injects the tool
# turn in the trained wire format; the model then continues from the snippet,
# which is 100% in-distribution (same trick as the generator's force_first_tool).
NEWS_INTENT_PAT = re.compile(
    r"\b(news|headlines?|latest|current events?|recently|this (week|month|year)|today)\b"
    r"|^search\b|\bsearch (for|the web|online)\b|\blook up\b|\bgoogle\b", re.IGNORECASE)


def derive_search_query(text):
    """Strip instruction filler so 'search for X' / 'tell me about X' -> 'X'."""
    q = text.strip()
    q = re.sub(r"^(please\s+)?(can you\s+|could you\s+)?"
               r"(search( the web| online)?( for)?|look up|google|tell me( about)?|"
               r"find( out)?( about)?|give me|show me)\s*", "", q, flags=re.IGNORECASE)
    q = q.strip(" ?.!\"'")
    return q or text.strip()

# ── tokenizer ──────────────────────────────────────────────────────────────

from tokenizers import Tokenizer


def load_tokenizer(path):
    tok = Tokenizer.from_file(path)
    eot_id = tok.token_to_id("<|endoftext|>")
    if eot_id is None:
        eot_id = 0
    return tok, eot_id


# ── ngrok ───────────────────────────────────────────────────────────────────

NGROK_BIN = "/usr/local/bin/ngrok"

def start_ngrok(port):
    """Launch ngrok and return the public URL."""
    cmd = [NGROK_BIN, "http", str(port), "--log=stdout"]
    if NGROK_AUTHTOKEN:
        cmd += ["--authtoken", NGROK_AUTHTOKEN]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    for _ in range(15):
        try:
            import urllib.request
            resp = urllib.request.urlopen(f"http://127.0.0.1:4040/api/tunnels", timeout=2)
            data = json.loads(resp.read())
            for tun in data.get("tunnels", []):
                url = tun.get("public_url", "")
                if url.startswith("https"):
                    return proc, url
        except Exception:
            time.sleep(1)
    print("[ngrok] WARNING: could not detect public URL (check NGROK_AUTHTOKEN in .env)")
    return proc, None

# ── web UI ──────────────────────────────────────────────────────────────────

HTML_PAGE = r'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover, interactive-widget=resizes-content">
    <meta name="theme-color" content="#0f0c29">
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="Synapse">
    <title>Synapse Chat</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
        html, body { height: 100%; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            height: 100vh; height: 100dvh; overflow: hidden;
            display: flex; flex-direction: column; align-items: center;
            padding: 20px;
            padding-top: max(20px, env(safe-area-inset-top));
        }
        h1 { color: #e94560; margin-bottom: 16px; text-shadow: 0 0 20px rgba(233,69,96,0.5); font-size: clamp(1.3rem, 5vw, 2rem); }
        .chat-container {
            width: 100%; max-width: 800px; background: rgba(255,255,255,0.05);
            border-radius: 20px; backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
            overflow: hidden; display: flex; flex-direction: column;
            flex: 1; min-height: 0;
        }
        #settings-toggle {
            display: none; width: 100%; padding: 12px; border: none;
            background: rgba(0,0,0,0.2); color: #e94560; font-size: 15px; cursor: pointer;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }
        .messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 15px; }
        .message {
            position: relative; padding: 15px 28px 15px 20px; border-radius: 15px; max-width: 80%; word-wrap: break-word;
            animation: fadeIn 0.3s ease; line-height: 1.5;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .message.user { background: linear-gradient(135deg, #e94560, #c23a51); color: #fff; align-self: flex-end; border-bottom-right-radius: 5px; }
        .message.assistant { background: rgba(255,255,255,0.1); color: #e0e0e0; align-self: flex-start; border-bottom-left-radius: 5px; }
        .message.assistant .cursor { display: inline-block; width: 8px; height: 18px; background: #e94560; margin-left: 2px; animation: blink 0.7s infinite; vertical-align: text-bottom; }
        @keyframes blink { 0%,50% { opacity: 1; } 51%,100% { opacity: 0; } }
        .message.assistant pre { background: rgba(0,0,0,0.3); padding: 10px; border-radius: 8px; overflow-x: auto; margin: 8px 0; font-size: 13px; }
        .message.assistant code { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; }
        .message.assistant .tool { border-left: 3px solid #e94560; margin: 8px 0; padding: 6px 10px; background: rgba(0,0,0,0.25); border-radius: 6px; font-size: 13px; }
        .message.assistant .tool .tool-hdr { color: #e94560; font-weight: 600; margin-bottom: 4px; }
        .message.assistant .tool pre { margin: 4px 0; white-space: pre-wrap; }
        .speak-btn {
            position: absolute; top: 4px; right: 4px;
            background: transparent; border: none; cursor: pointer;
            font-size: 13px; line-height: 1; padding: 3px 4px; border-radius: 6px;
            opacity: 0.45; transition: opacity 0.2s, transform 0.2s; color: inherit;
            touch-action: manipulation;
        }
        .speak-btn:hover { opacity: 1; transform: scale(1.15); }
        .speak-btn.speaking { opacity: 1; color: #e94560; animation: pulse 1.5s infinite; }
        .message.user .speak-btn { color: rgba(255,255,255,0.85); }
        .message.user .speak-btn.speaking { color: #fff; }
        .input-area { padding: 20px; background: rgba(0,0,0,0.2); display: flex; gap: 10px; align-items: center; }
        #send-btn, #mic-btn, #new-chat-btn { touch-action: manipulation; }
        #user-input { flex: 1; padding: 15px 20px; border: none; border-radius: 25px; background: rgba(255,255,255,0.1); color: #fff; font-size: 16px; outline: none; transition: background 0.3s; }
        #user-input:focus { background: rgba(255,255,255,0.15); }
        #user-input::placeholder { color: rgba(255,255,255,0.5); }
        #send-btn { padding: 15px 30px; border: none; border-radius: 25px; background: linear-gradient(135deg, #e94560, #c23a51); color: #fff; font-size: 16px; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; }
        #send-btn:hover { transform: scale(1.05); box-shadow: 0 5px 20px rgba(233,69,96,0.4); }
        #send-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .audio-btns { display: flex; gap: 6px; }
        #mic-btn {
            padding: 12px 16px; border: none; border-radius: 25px; font-size: 18px; cursor: pointer;
            transition: transform 0.2s, background 0.3s; background: rgba(255,255,255,0.1); color: #aaa;
        }
        #mic-btn:hover { transform: scale(1.1); background: rgba(255,255,255,0.2); }
        #mic-btn.active { background: #e94560; color: #fff; animation: pulse 1.5s infinite; }
        @keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(233,69,96,0.6); } 50% { box-shadow: 0 0 0 10px rgba(233,69,96,0); } }
        .settings { padding: 10px 20px; background: rgba(0,0,0,0.2); display: flex; gap: 20px; flex-wrap: wrap; justify-content: center; align-items: center; font-size: 14px; color: #aaa; }
        .settings label { display: flex; align-items: center; gap: 8px; }
        #new-chat-btn { padding: 8px 16px; border: none; border-radius: 15px; background: rgba(255,255,255,0.1); color: #e94560; font-size: 14px; cursor: pointer; transition: background 0.3s, transform 0.2s; border: 1px solid #e94560; }
        #new-chat-btn:hover { background: rgba(233,69,96,0.2); transform: scale(1.05); }
        .settings input[type="range"] { width: 100px; accent-color: #e94560; }
        .settings input[type="number"] { width: 60px; padding: 5px; border: none; border-radius: 5px; background: rgba(255,255,255,0.1); color: #fff; text-align: center; }
        .settings select { padding: 5px 10px; border: none; border-radius: 5px; background: rgba(255,255,255,0.1); color: #fff; font-size: 13px; cursor: pointer; outline: none; }
        .settings select option { background: #302b63; color: #fff; }
        #status { color: #aaa; font-size: 14px; text-align: center; padding: 5px; }
        #status.loading { color: #e94560; animation: pulse 1.5s infinite; }
        #status.error { color: #ff4444; }

        /* ── Mobile ──────────────────────────────────────────────── */
        @media (max-width: 640px) {
            body { padding: 0; padding-top: env(safe-area-inset-top); }
            h1 { margin: 10px 0 8px; }
            .chat-container { border-radius: 0; border-left: none; border-right: none; }
            #settings-toggle { display: block; }
            .settings { display: none; flex-direction: column; align-items: stretch; gap: 14px; padding: 16px; }
            .settings.open { display: flex; }
            .settings label { justify-content: space-between; font-size: 15px; }
            .settings input[type="range"] { width: 55%; height: 24px; }
            .settings input[type="number"] { width: 72px; padding: 8px; font-size: 15px; }
            .settings select { flex: 1; padding: 8px 10px; font-size: 15px; }
            #new-chat-btn { padding: 12px; font-size: 15px; }
            .messages { padding: 14px; gap: 12px; }
            .message { max-width: 88%; padding: 12px 15px; }
            .input-area {
                padding: 12px;
                padding-bottom: calc(12px + env(safe-area-inset-bottom));
                gap: 6px; flex-wrap: wrap;
            }
            #user-input { padding: 14px 16px; min-width: 0; flex: 1 1 100%; order: 1; }
            .input-area > #send-btn { order: 3; flex: 1; }
            .audio-btns { order: 2; gap: 6px; }
            #send-btn { padding: 13px 18px; }
            #mic-btn { padding: 11px 16px; font-size: 20px; }
            .message { padding: 12px 26px 12px 15px; }
            .speak-btn { font-size: 15px; padding: 4px 5px; opacity: 0.6; }
        }
        /* Larger touch targets / no hover-scale jank on touch devices */
        @media (hover: none) {
            #send-btn:hover, #mic-btn:hover, #new-chat-btn:hover, .speak-btn:hover { transform: none; box-shadow: none; }
            #send-btn:active, #mic-btn:active, .speak-btn:active { transform: scale(0.96); }
            .speak-btn { opacity: 0.6; }
        }
    </style>
</head>
<body>
    <h1>Synapse Chat</h1>
    <div id="status">model ready</div>
    <div class="chat-container">
        <button id="settings-toggle" aria-expanded="false">⚙️ Settings</button>
        <div class="settings" id="settings">
            <label>Model: <select id="model-select">
                <option value="pretrain">Pretrain</option>
                <option value="sft">SFT</option>
            </select></label>
            <button id="new-chat-btn">New Chat</button>
            <label title="SFT only: canonical tool prompt + python sandbox"><input type="checkbox" id="tools-toggle" checked> Tools (python)</label>
            <label>Temp: <span id="temp-value">0.8</span> <input type="range" id="temperature" min="0.1" max="2.0" step="0.05" value="0.8"></label>
            <label>Max Tokens: <input type="number" id="max-tokens" min="10" max="1024" value="128"></label>
            <label>Top-K: <input type="number" id="top-k" min="1" max="200" value="50"></label>
            <label>Top-P: <span id="topp-value">0.9</span> <input type="range" id="top-p" min="0.1" max="1.0" step="0.05" value="0.9"></label>
            <label>Rep.Pen: <span id="reppen-value">1.15</span> <input type="range" id="repetition-penalty" min="1.0" max="2.0" step="0.05" value="1.15"></label>
        </div>
        <div class="messages" id="messages">
            <div class="message assistant">Hello! I'm Synapse, a 2B-parameter language model. How can I help you?</div>
        </div>
        <div class="input-area">
            <input type="text" id="user-input" placeholder="Type your message..." autofocus>
            <button id="send-btn">Send</button>
            <div class="audio-btns">
                <button id="mic-btn" title="Voice input">🎙️</button>
            </div>
        </div>
    </div>
    <script>
        const messagesDiv = document.getElementById('messages');
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const micBtn = document.getElementById('mic-btn');
        const tempSlider = document.getElementById('temperature');
        const tempValue = document.getElementById('temp-value');
        const topPSlider = document.getElementById('top-p');
        const topPValue = document.getElementById('topp-value');
        const repPenSlider = document.getElementById('repetition-penalty');
        const repPenValue = document.getElementById('reppen-value');

        /* ── Settings panel toggle (mobile) ───────────────────────── */
        const settingsToggle = document.getElementById('settings-toggle');
        const settingsPanel = document.getElementById('settings');
        settingsToggle.addEventListener('click', () => {
            const open = settingsPanel.classList.toggle('open');
            settingsToggle.setAttribute('aria-expanded', open);
            settingsToggle.textContent = open ? '⚙️ Hide Settings' : '⚙️ Settings';
        });

        tempSlider.addEventListener('input', () => tempValue.textContent = tempSlider.value);
        topPSlider.addEventListener('input', () => topPValue.textContent = topPSlider.value);
        repPenSlider.addEventListener('input', () => repPenValue.textContent = repPenSlider.value);

        let conversation = [];   // [{role:'user'|'assistant', content:str}, ...]

        /* ── Audio Output (TTS) — per-message speaker icon ────────── */
        // getVoices() is populated asynchronously in Chrome — cache it and
        // refresh on the voiceschanged event so the first reply still has a voice.
        let voices = [];
        function loadVoices() { voices = window.speechSynthesis.getVoices(); }
        loadVoices();
        if (window.speechSynthesis.onvoiceschanged !== undefined) {
            window.speechSynthesis.onvoiceschanged = loadVoices;
        }

        // Chrome stalls/truncates a single utterance after ~15s; this keep-alive
        // timer pumps the queue so long replies finish.
        let ttsKeepAlive = null;
        function startKeepAlive() {
            stopKeepAlive();
            ttsKeepAlive = setInterval(() => {
                if (window.speechSynthesis.speaking) {
                    window.speechSynthesis.pause();
                    window.speechSynthesis.resume();
                } else {
                    stopKeepAlive();
                }
            }, 10000);
        }
        function stopKeepAlive() {
            if (ttsKeepAlive) { clearInterval(ttsKeepAlive); ttsKeepAlive = null; }
        }

        // Only one message speaks at a time — track the active button so a second
        // click stops playback and a click on another message swaps to it.
        let activeSpeakBtn = null;
        function stopSpeaking() {
            window.speechSynthesis.cancel();
            stopKeepAlive();
            if (activeSpeakBtn) { activeSpeakBtn.classList.remove('speaking'); activeSpeakBtn.textContent = '🔊'; activeSpeakBtn = null; }
        }

        // Split into sentence-sized chunks (further capped by length) so Chrome
        // queues several short utterances instead of one long one it would truncate.
        function chunkText(text) {
            const sentences = text.replace(/\s+/g, ' ').trim()
                .match(/[^.!?]+[.!?]+|\S[^.!?]*$/g) || [text];
            const out = [];
            for (const s of sentences) {
                let chunk = s.trim();
                if (!chunk) continue;
                while (chunk.length > 200) {
                    let cut = chunk.lastIndexOf(' ', 200);
                    if (cut <= 0) cut = 200;
                    out.push(chunk.slice(0, cut).trim());
                    chunk = chunk.slice(cut).trim();
                }
                if (chunk) out.push(chunk);
            }
            return out;
        }

        function speakText(text, btn) {
            if (!text) return;
            if (activeSpeakBtn === btn) { stopSpeaking(); return; }   // toggle off
            stopSpeaking();
            const eng = voices.find(v => v.lang && v.lang.startsWith('en'));
            for (const chunk of chunkText(text)) {
                const utterance = new SpeechSynthesisUtterance(chunk);
                utterance.rate = 1.0;
                utterance.pitch = 1.0;
                if (eng) utterance.voice = eng;
                window.speechSynthesis.speak(utterance);
            }
            activeSpeakBtn = btn;
            btn.classList.add('speaking');
            btn.textContent = '⏸️';
            // Reset the button when playback actually ends.
            const done = () => {
                btn.classList.remove('speaking');
                btn.textContent = '🔊';
                stopKeepAlive();
                if (activeSpeakBtn === btn) activeSpeakBtn = null;
            };
            // There is no single utterance-end for a queued batch, so poll.
            startKeepAlive();
            const poll = setInterval(() => {
                if (!window.speechSynthesis.speaking) { clearInterval(poll); done(); }
            }, 250);
        }

        // Attach a speaker icon to a finished message. `getText` lets us read the
        // current (possibly streaming) text at click time without re-storing it.
        function attachSpeakButton(msgEl, getText) {
            const btn = document.createElement('button');
            btn.className = 'speak-btn';
            btn.textContent = '🔊';
            btn.title = 'Read aloud';
            btn.type = 'button';
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                speakText(getText(), btn);
            });
            msgEl.appendChild(btn);
            return btn;
        }

        /* ── Audio Input (Microphone) ─────────────────────────────── */
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        let recognition = null;
        let listening = false;

        if (SpeechRecognition) {
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-US';

            recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript.trim();
                if (transcript) {
                    userInput.value = transcript;
                    sendMessage();
                }
                stopListening();
            };

            recognition.onerror = (event) => {
                console.log('Mic error:', event.error);
                stopListening();
            };

            recognition.onend = () => { stopListening(); };
        }

        function startListening() {
            if (!recognition) return;
            listening = true;
            micBtn.classList.add('active');
            micBtn.textContent = '🔴';
            userInput.placeholder = 'Listening...';
            recognition.start();
        }

        function stopListening() {
            listening = false;
            micBtn.classList.remove('active');
            micBtn.textContent = '🎙️';
            userInput.placeholder = 'Type your message...';
        }

        micBtn.addEventListener('click', () => {
            if (!recognition) { alert('Speech recognition not supported in this browser. Use Chrome.'); return; }
            if (listening) { recognition.stop(); stopListening(); }
            else { startListening(); }
        });

        /* ── Chat ─────────────────────────────────────────────────── */
        function addMessage(text, isUser) {
            const msg = document.createElement('div');
            msg.className = 'message ' + (isUser ? 'user' : 'assistant');
            msg.textContent = text;
            attachSpeakButton(msg, () => text);
            messagesDiv.appendChild(msg);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            return msg;
        }

        function createStreamingMessage() {
            const msg = document.createElement('div');
            msg.className = 'message assistant';
            msg.innerHTML = '<span class="cursor"></span>';
            messagesDiv.appendChild(msg);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            return msg;
        }

        async function sendMessage() {
            const text = userInput.value.trim();
            if (!text) return;
            addMessage(text, true);
            userInput.value = '';
            sendBtn.disabled = true;

            conversation.push({ role: 'user', content: text });

            const assistantMsg = createStreamingMessage();
            let responseText = '';     // prose of the CURRENT segment (between tool blocks)
            let newTurns = null;       // server-side trace (assistant/tool/.../assistant)
            const esc = (t) => t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
            let segment = document.createElement('span');
            assistantMsg.innerHTML = '';
            assistantMsg.appendChild(segment);
            const cursor = document.createElement('span'); cursor.className = 'cursor';
            assistantMsg.appendChild(cursor);
            let pendingTool = null;   // header of the tool block awaiting its result
            function toolBlock(hdr, body) {
                const d = document.createElement('div'); d.className = 'tool';
                d.innerHTML = '<div class="tool-hdr">' + esc(hdr) + '</div><pre>' + esc(body) + '</pre>';
                assistantMsg.insertBefore(d, cursor);
                segment = document.createElement('span');
                assistantMsg.insertBefore(segment, cursor);
                responseText = '';
                return d;
            }

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        messages: conversation,
                        tools: document.getElementById('tools-toggle').checked,
                        max_tokens: parseInt(document.getElementById('max-tokens').value),
                        temperature: parseFloat(tempSlider.value),
                        top_k: parseInt(document.getElementById('top-k').value),
                        top_p: parseFloat(topPSlider.value),
                        repetition_penalty: parseFloat(repPenSlider.value)
                    })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop();
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                if (data.token) {
                                    responseText += data.token;
                                    segment.textContent = responseText;
                                    messagesDiv.scrollTop = messagesDiv.scrollHeight;
                                }
                                if (data.tool_call) {
                                    const c = data.tool_call;
                                    let hdr, body;
                                    if (c.tool === 'search') { hdr = '\u{1F50D} search' + (data.forced ? ' \u00b7 auto' : ''); body = c.query || JSON.stringify(c); }
                                    else if (c.tool === 'python') { hdr = '\u{1F40D} python'; body = c.code || JSON.stringify(c); }
                                    else { hdr = '\u26A0\uFE0F ' + (c.tool || 'malformed tool call'); body = c.code || c.query || c.raw || JSON.stringify(c); }
                                    const d = toolBlock(hdr + '  \u2026running', body);
                                    pendingTool = d.querySelector('.tool-hdr');
                                    pendingTool._base = hdr;
                                }
                                if (data.tool_result !== undefined) {
                                    if (pendingTool) { pendingTool.textContent = pendingTool._base + '  \u2713'; pendingTool = null; }
                                    toolBlock('\u2192 result', data.tool_result);
                                }
                                if (data.error) {
                                    toolBlock('error', data.error);
                                }
                                if (data.done) {
                                    cursor.remove();
                                    newTurns = data.messages || null;
                                    const spoken = responseText;
                                    if (spoken) attachSpeakButton(assistantMsg, () => spoken);
                                }
                            } catch (e) {}
                        }
                    }
                }
                if (newTurns && newTurns.length) newTurns.forEach(t => conversation.push(t));
                else if (responseText) conversation.push({ role: 'assistant', content: responseText });
                else assistantMsg.innerHTML = '<i>(no response)</i>';
            } catch (error) {
                assistantMsg.innerHTML = 'Error: ' + error.message;
                conversation.pop();  // drop the user turn that never got a reply
            }
            sendBtn.disabled = false;
            userInput.focus();
        }

        sendBtn.addEventListener('click', sendMessage);
        userInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });
        document.getElementById('new-chat-btn').addEventListener('click', () => {
            messagesDiv.innerHTML = '';
            const welcome = document.createElement('div');
            welcome.className = 'message assistant';
            welcome.textContent = "Hello! I'm Synapse, a 2B-parameter language model. How can I help you?";
            attachSpeakButton(welcome, () => welcome.textContent);
            messagesDiv.appendChild(welcome);
            conversation = [];
        });

        /* ── Model Switch ──────────────────────────────────────────── */
        const modelSelect = document.getElementById('model-select');
        const statusDiv = document.getElementById('status');
        modelSelect.addEventListener('change', async () => {
            const variant = modelSelect.value;
            statusDiv.textContent = `Loading ${variant} model...`;
            statusDiv.className = 'loading';
            sendBtn.disabled = true;
            try {
                const resp = await fetch('/switch_model', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ variant }),
                });
                const data = await resp.json();
                if (data.ok) {
                    statusDiv.textContent = `${variant} model ready`;
                    statusDiv.className = '';
                    conversation = [];
                    messagesDiv.innerHTML = '';
                    const welcome = document.createElement('div');
                    welcome.className = 'message assistant';
                    welcome.textContent = `Switched to ${variant} model. How can I help you?`;
                    attachSpeakButton(welcome, () => welcome.textContent);
                    messagesDiv.appendChild(welcome);
                } else {
                    statusDiv.textContent = `Error: ${data.error}`;
                    statusDiv.className = 'error';
                }
            } catch (e) {
                statusDiv.textContent = `Error: ${e.message}`;
                statusDiv.className = 'error';
            }
            sendBtn.disabled = false;
        });
    </script>
    <script>
        (function() {
            // Decorate the static welcome message with a speaker button.
            document.querySelectorAll('.message').forEach(m => {
                if (!m.querySelector('.speak-btn')) {
                    attachSpeakButton(m, () => m.textContent);
                }
            });
            const v = '__VARIANT__';
            document.getElementById('model-select').value = v;
            document.getElementById('status').textContent = v === 'sft' ? 'SFT model ready' : 'pretrain model ready';
        })();
    </script>
</body>
</html>
'''

# ── Flask app ──────────────────────────────────────────────────────────────

app = Flask(__name__)
model = None
tokenizer = None
eot_id = 0
device = None
use_chatml = False
current_variant = None
data_dir = None

# Serializes generation vs. model-switch on the single GPU (Flask is threaded).
gen_lock = threading.Lock()


@app.route("/")
def index():
    variant = current_variant or "pretrain"
    return HTML_PAGE.replace("__VARIANT__", variant)


@app.route("/switch_model", methods=["POST"])
def switch_model():
    global model, tokenizer, eot_id, use_chatml, current_variant
    try:
        variant = request.json.get("variant")
        if variant not in DRIVE_PATHS:
            return jsonify({"ok": False, "error": f"unknown variant: {variant}"}), 400
        if variant == current_variant:
            return jsonify({"ok": True, "variant": variant})

        drive_path = DRIVE_PATHS[variant]
        local_ckpt = os.path.join(data_dir, os.path.basename(drive_path))

        if not os.path.exists(local_ckpt):
            try:
                _download(drive_path, local_ckpt)
            except Exception as e:
                return jsonify({"ok": False, "error": f"download failed: {e}"}), 500

        # Load the new checkpoint into a temp var BEFORE touching the live model,
        # and hold the gen lock so no /chat is mid-generation during the swap.
        # If loading fails the old model stays usable (no half-deleted state).
        with gen_lock:
            old_model = model
            del model
            gc.collect()
            if device and device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[switch] loading {variant} from {local_ckpt}")
            try:
                new_model, info = SynapseInfer.from_checkpoint(local_ckpt, device=device)
            except Exception as e:
                model = old_model  # restore — server stays alive
                print(f"[switch] load failed, kept {current_variant}: {e}")
                return jsonify({"ok": False, "error": f"load failed: {e}"}), 500

            del old_model
            gc.collect()
            if device and device.type == "cuda":
                torch.cuda.empty_cache()
            model = new_model
            current_variant = variant
            use_chatml = bool(info.get("is_sft")) if info else False
        print(f"[switch] {variant} model loaded (chatml={use_chatml})")
        return jsonify({"ok": True, "variant": variant, "chatml": use_chatml})
    except Exception as e:
        print(f"[switch] error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    # Preferred: full conversation history; fall back to single prompt.
    messages = data.get("messages")
    if not messages:
        messages = [{"role": "user", "content": data.get("prompt", "")}]
    max_tokens = min(int(data.get("max_tokens", 128)), 1024)
    temperature = max(float(data.get("temperature", 0.8)), 1e-3)
    top_k = int(data.get("top_k", 50))
    top_p = float(data.get("top_p", 0.9))
    repetition_penalty = float(data.get("repetition_penalty", 1.15))
    tools = bool(data.get("tools", True))
    max_prompt = max(BLOCK_SIZE - max_tokens, 1)
    stop_tokens = sft_stop_token_ids(tokenizer, eot_id)

    def sse(obj):
        return f"data: {json.dumps(obj)}\n\n"

    def generate_ids(prompt_ids):
        """Contract for tool_loop: yield generated ids; stop tokens are consumed."""
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        yield from model.generate(idx, max_tokens, temperature=temperature, top_k=top_k,
                                  top_p=top_p, repetition_penalty=repetition_penalty,
                                  eot_id=eot_id, stop_tokens=stop_tokens)

    def generate():
        try:
            yield from _generate()
        except Exception as e:
            # A dead SSE stream renders as a cryptic browser "error in input
            # stream" — surface the real exception in the UI instead.
            import traceback
            traceback.print_exc()
            yield sse({"error": f"server error: {type(e).__name__}: {e}"})
            yield sse({"done": True, "status": "error", "messages": []})

    def _generate():
        # gen_lock is held across the WHOLE tool loop: a model switch or a second
        # request must never interleave with a multi-round trace.
        with gen_lock:
            if use_chatml:
                system = CANONICAL_TOOL_SYSTEM if tools else ""
                # Fit the context by dropping the OLDEST turns whole (never slice a
                # prompt mid-turn — that breaks the trained wire format).
                hist = list(messages)
                while len(hist) > 1 and len(tokenizer.encode(
                        build_sft_prompt(hist, system=system or None),
                        add_special_tokens=False).ids) > max_prompt:
                    hist = hist[1:]
                    while hist and hist[0]["role"] != "user":   # history must open on a user turn
                        hist = hist[1:]
                # Forced-search assist: news/"search for" intent -> the chatbot
                # searches and seeds the trained tool-turn; the model continues.
                seeded = []
                last_user = hist[-1].get("content", "") if hist and hist[-1]["role"] == "user" else ""
                if tools and last_user and NEWS_INTENT_PAT.search(last_user) \
                        and os.environ.get("BRAVE_API_KEY"):
                    call = {"tool": "search", "query": derive_search_query(last_user)}
                    yield sse({"tool_call": call, "forced": True})
                    result = tool_execute(call)
                    yield sse({"tool_result": result})
                    seeded = [{"role": "assistant", "content": "", "tool_call": call},
                              {"role": "tool", "content": result}]
                    hist = hist + seeded
                for ev in run_tool_loop(generate_ids, tokenizer, hist, system,
                                        max_prompt_tokens=max_prompt):
                    if ev["type"] == "token":
                        yield sse({"token": ev["text"]})
                    elif ev["type"] == "tool_call":
                        yield sse({"tool_call": ev["call"]})
                    elif ev["type"] == "tool_result":
                        yield sse({"tool_result": ev["text"]})
                    elif ev["type"] == "final":
                        if ev["status"] != "answered":
                            yield sse({"error": f"tool loop stopped: {ev['status']}"})
                        yield sse({"done": True, "status": ev["status"],
                                   "messages": seeded + ev["messages"]})
                return

            # Pretrain has no chat structure — continue the latest user text.
            input_ids = tokenizer.encode(messages[-1].get("content", "")).ids
            if len(input_ids) > max_prompt:
                input_ids = input_ids[-max_prompt:]
            # Incremental decode: emit only the new suffix; skip trailing partial
            # UTF-8 (renders as U+FFFD) until the next token completes the character.
            gen_ids, printed = [], 0
            for token_id in generate_ids(input_ids):
                gen_ids.append(token_id)
                text = tokenizer.decode(gen_ids)
                if text.endswith("�"):
                    continue
                if len(text) > printed:
                    yield sse({"token": text[printed:]})
                    printed = len(text)
            final = tokenizer.decode(gen_ids)
            if not final.endswith("�") and len(final) > printed:
                yield sse({"token": final[printed:]})
            yield sse({"done": True, "messages": [{"role": "assistant", "content": final}]})

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── helpers ─────────────────────────────────────────────────────────────────

DRIVE_PATHS = {
    "pretrain": "synapse/checkpoints/synapse_2b_d2560_l28.pth",
    "sft": "synapse/sft_checkpoints/v3_15source/sft_best.pth",  # v3 SHIPPED 2026-09-05 (python+search tools; v2 archive: v2_12source/)
}
DRIVE_TOKENIZER = "synapse/tokenizer_out/tokenizer.json"

# ── .env loading ──────────────────────────────────────────────────────────────

def _load_env():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return {}
    env = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env

_env = _load_env()
NGROK_AUTHTOKEN = _env.get("NGROK_AUTHTOKEN", "")


def _download(path, dest):
    """Download a file from Google Drive via rclone."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"[download] pulling {path} ...")
    subprocess.run(["rclone", "copyto", f"gdrive:{path}", dest,
                    "--progress", "--drive-chunk-size", "64M",
                    "--transfers", "4"], check=True)
    print(f"[download] done: {dest} ({os.path.getsize(dest) / 1e9:.2f} GB)")


def _get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ── main ───────────────────────────────────────────────────────────────────

def main():
    global model, tokenizer, eot_id, device, use_chatml, current_variant, data_dir

    parser = argparse.ArgumentParser(description="Synapse Chatbot")
    parser.add_argument("--ckpt", default=None, help="Override checkpoint .pth path")
    parser.add_argument("--tokenizer", default=None, help="Override tokenizer.json path")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip auto-download (use if files already local)")
    parser.add_argument("--port", type=int, default=5000, help="Flask port (default: 5000)")
    parser.add_argument("--no-ngrok", action="store_true", help="Skip ngrok tunnel")
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip torch.compile (more stable)")
    parser.add_argument("--variant", choices=["pretrain", "sft"], default="pretrain",
                        help="Which checkpoint to load (default: pretrain)")
    args = parser.parse_args()

    # ── require GPU ──
    if not torch.cuda.is_available():
        print("[error] CUDA GPU required but not available")
        sys.exit(1)
    device = torch.device("cuda")
    print(f"[gpu] {torch.cuda.get_device_name(0)} | "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── paths ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "sparky_data")
    local_ckpt = (args.ckpt
                  or os.path.join(data_dir, os.path.basename(DRIVE_PATHS[args.variant])))
    local_tok = args.tokenizer or os.path.join(data_dir, "tokenizer.json")

    # ── auto-download from Drive (unless overridden) ──
    if not args.no_download:
        if args.ckpt is None and (not os.path.exists(local_ckpt)):
            _download(DRIVE_PATHS[args.variant], local_ckpt)
        if args.tokenizer is None and (not os.path.exists(local_tok)):
            _download(DRIVE_TOKENIZER, local_tok)

    if not os.path.exists(local_ckpt):
        print(f"[error] checkpoint not found: {local_ckpt}")
        print(f"  ensure rclone is configured and drive is accessible")
        sys.exit(1)
    if not os.path.exists(local_tok):
        print(f"[error] tokenizer not found: {local_tok}")
        sys.exit(1)

    # ── tokenizer ──
    print(f"[tokenizer] loading {local_tok}")
    # tools_runtime.truncate_tokens lazy-loads ITS OWN tokenizer via SYNAPSE_DIR /
    # SYNAPSE_TOKENIZER — neither exists on a Colab box with Drive unmounted, so the
    # first tool result would crash the SSE stream. Point it at the same file.
    os.environ.setdefault("SYNAPSE_TOKENIZER", os.path.abspath(local_tok))
    tokenizer, eot_id = load_tokenizer(local_tok)
    print(f"[tokenizer] vocab={tokenizer.get_vocab_size()} eot_id={eot_id}")

    # ── model ──
    print(f"[model] loading {local_ckpt} ({os.path.getsize(local_ckpt) / 1e9:.2f} GB)")
    model, info = SynapseInfer.from_checkpoint(local_ckpt, device=device)
    use_chatml = bool(info.get("is_sft")) if info else False
    current_variant = "sft" if use_chatml else "pretrain"
    if info:
        print(f"[model] training step: {info.get('step')}")
        if info.get("eval_history"):
            print(f"[model] last eval loss: {info['eval_history'][-1].get('overall') or info['eval_history'][-1]['loss']:.4f}")
        if use_chatml:
            print(f"[model] SFT checkpoint detected — ChatML + tool loop (python sandbox)")
    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"[model] VRAM used: {allocated:.2f} GB")

    # ── ngrok ──
    ngrok_proc = None
    public_url = None
    if not args.no_ngrok and os.path.exists(NGROK_BIN):
        print("[ngrok] starting tunnel...")
        ngrok_proc, public_url = start_ngrok(args.port)
        if public_url:
            print(f"[ngrok] public URL: {public_url}")

    # ── launch Flask in background thread ──
    local_ip = _get_local_ip()
    print()
    print("=" * 60)
    print("  Synapse Chatbot ready")
    print(f"  Local:   http://localhost:{args.port}")
    print(f"  LAN:     http://{local_ip}:{args.port}")
    if public_url:
        print(f"  Public:  {public_url}")
    print("=" * 60)
    print()
    print("  Press Ctrl+C to stop")
    print()

    stop_event = threading.Event()

    def run_flask():
        app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    def cleanup(*_):
        if ngrok_proc:
            ngrok_proc.terminate()
        stop_event.set()

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        stop_event.wait()
    finally:
        cleanup()
        os._exit(0)


if __name__ == "__main__":
    main()
