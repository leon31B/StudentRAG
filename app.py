"""
Web Application for NVIDIA Nemotron-3.5-Embed-1B RAG Environment.
Serves a sleek dark-themed dashboard for Chat, Ingestion, and Index Inspection.
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from config import config, BASE_DIR
from rag_pipeline import NVIDIARagPipeline, MissingAPIKeyError

app = FastAPI(title="NVIDIA Nemotron-3.5-Embed-1B RAG")
pipeline = NVIDIARagPipeline()


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 4


class KeyUpdateRequest(BaseModel):
    api_key: str


class EmbedTestRequest(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
@app.get("/landing", response_class=HTMLResponse)
async def get_landing():
    """Serves the StudentRAG.com landing page as the primary entrance."""
    landing_file = BASE_DIR / "landing page" / "index.html"
    if landing_file.exists():
        return HTMLResponse(landing_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Landing page not found</h1>", status_code=404)


@app.get("/app", response_class=HTMLResponse)
@app.get("/chat", response_class=HTMLResponse)
async def get_dashboard():
    """Serves the rich modern web dashboard."""
    stats = pipeline.get_index_stats()
    has_key = config.is_api_key_valid()

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome to StudentRAG.com</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        :root {{
            --bg-canvas: linear-gradient(135deg, #f7f3ec 0%, #ede5da 45%, #f6efe6 100%);
            --bg-card: #fcf9f5;
            --card-border: rgba(255, 255, 255, 0.75);
            --clay-sage: #95b3a2;
            --clay-sage-dark: #7a9789;
            --clay-blue: #9fc0c8;
            --clay-blue-dark: #81a4ac;
            --clay-peach: #e8bfa8;
            --clay-peach-dark: #d2a48d;
            --clay-cream: #fcf9f5;
            --text-primary: #283339;
            --text-secondary: #586771;
            --text-muted: #8897a2;
            --accent-sage: #95b3a2;
            --clay-shadow-panel: 0 24px 48px -12px rgba(125, 110, 95, 0.22), 0 10px 24px -6px rgba(125, 110, 95, 0.12), inset 3px 3px 6px rgba(255, 255, 255, 0.95), inset -4px -4px 9px rgba(175, 160, 145, 0.25);
            --clay-shadow-btn: 0 12px 24px -6px rgba(95, 115, 105, 0.35), inset 2px 2px 5px rgba(255, 255, 255, 0.8), inset -3px -3px 7px rgba(60, 80, 70, 0.25);
            --clay-shadow-inset: inset 3px 3px 6px rgba(135, 120, 105, 0.22), inset -2px -2px 5px rgba(255, 255, 255, 0.95);
            --clay-radius: 28px;
            --pill-radius: 24px;
            --glass-bg: rgba(255, 255, 255, 0.55);
            --glass-border: rgba(255, 255, 255, 0.75);
            --glass-shadow: 0 8px 24px -4px rgba(125, 110, 95, 0.12), inset 1px 1px 2px rgba(255, 255, 255, 0.8);
            --glass-text: #283339;
            --glass-placeholder: rgba(40, 51, 57, 0.55);
        }}

        [data-theme="dark"] {{
            --bg-canvas: radial-gradient(circle at 50% 0%, #151b24 0%, #0a0c10 100%);
            --bg-card: rgba(22, 27, 34, 0.85);
            --card-border: rgba(255, 255, 255, 0.12);
            --clay-sage: #76b900;
            --clay-sage-dark: #588a00;
            --clay-blue: #38bdf8;
            --clay-blue-dark: #0284c7;
            --clay-peach: #f59e0b;
            --clay-peach-dark: #d97706;
            --clay-cream: rgba(22, 27, 34, 0.75);
            --text-primary: #f0f6fc;
            --text-secondary: #c9d1d9;
            --text-muted: #8b949e;
            --accent-sage: #76b900;
            --clay-shadow-panel: 0 20px 40px -10px rgba(0, 0, 0, 0.6), inset 1px 1px 2px rgba(255, 255, 255, 0.08);
            --clay-shadow-btn: 0 8px 20px -4px rgba(118, 185, 0, 0.35), inset 1px 1px 3px rgba(255, 255, 255, 0.3);
            --clay-shadow-inset: inset 0 2px 4px rgba(0, 0, 0, 0.3);
            --glass-bg: rgba(255, 255, 255, 0.06);
            --glass-border: rgba(255, 255, 255, 0.16);
            --glass-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37), inset 0 1px 0 rgba(255, 255, 255, 0.12);
            --glass-text: #ffffff;
            --glass-placeholder: rgba(255, 255, 255, 0.85);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            transition: background 0.25s ease, color 0.2s ease, border-color 0.2s ease, box-shadow 0.25s ease;
        }}

        body {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-canvas);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            line-height: 1.6;
            overflow-x: hidden;
            position: relative;
        }}

        /* Floating Foliage Accent Layer */
        .foliage-layer {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            pointer-events: none;
            z-index: 0;
            overflow: hidden;
        }}

        .leaf {{
            position: absolute;
            pointer-events: none;
            opacity: 0.85;
            filter: drop-shadow(0 20px 30px rgba(100, 115, 105, 0.22));
            animation: antigravityFloat 9s ease-in-out infinite;
        }}

        .leaf-tl {{
            top: -30px;
            left: -35px;
            width: 220px;
            animation-duration: 10s;
        }}

        .leaf-tr {{
            top: -20px;
            right: -25px;
            width: 200px;
            animation-duration: 12s;
            animation-delay: -3s;
        }}

        .leaf-bl {{
            bottom: -35px;
            left: -30px;
            width: 210px;
            animation-duration: 11s;
            animation-delay: -5s;
        }}

        .leaf-br {{
            bottom: -30px;
            right: -30px;
            width: 230px;
            animation-duration: 13s;
            animation-delay: -2s;
        }}

        @keyframes antigravityFloat {{
            0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
            50% {{ transform: translateY(-14px) rotate(2deg); }}
        }}

        /* Header */
        header {{
            max-width: 1240px;
            width: 92%;
            margin: 1.25rem auto 0 auto;
            background: var(--bg-card);
            border: 1px solid var(--card-border);
            border-radius: var(--pill-radius);
            box-shadow: var(--clay-shadow-panel);
            padding: 0.85rem 1.75rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: relative;
            z-index: 20;
            backdrop-filter: blur(12px);
        }}

        .brand {{
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }}

        .logo-badge {{
            background: var(--clay-sage);
            color: #ffffff;
            font-weight: 800;
            font-size: 0.85rem;
            padding: 0.45rem 0.85rem;
            border-radius: 16px;
            letter-spacing: 0.6px;
            box-shadow: var(--clay-shadow-btn);
        }}

        .brand h1 {{
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.3px;
        }}

        .brand span {{
            color: var(--clay-sage);
        }}

        .header-actions {{
            display: flex;
            align-items: center;
            gap: 1rem;
        }}

        .theme-toggle-btn {{
            background: var(--bg-card);
            border: 1px solid var(--card-border);
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.85rem;
            font-weight: 700;
            padding: 0.45rem 1rem;
            border-radius: 20px;
            cursor: pointer;
            box-shadow: var(--clay-shadow-panel);
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
        }}

        .theme-toggle-btn:active {{
            transform: translateY(2px) scale(0.97);
        }}

        .status-badge {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.85rem;
            font-weight: 600;
            background: var(--bg-card);
            color: var(--text-secondary);
            padding: 0.45rem 1rem;
            border-radius: 20px;
            border: 1px solid var(--card-border);
            box-shadow: var(--clay-shadow-panel);
        }}

        .status-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background-color: {'#95b3a2' if has_key else '#e8bfa8'};
            box-shadow: 0 0 8px {'rgba(149, 179, 162, 0.7)' if has_key else 'rgba(232, 191, 168, 0.7)'};
        }}

        /* Main Container */
        .main-container {{
            max-width: 1240px;
            width: 92%;
            margin: 1.5rem auto 3rem auto;
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            position: relative;
            z-index: 10;
        }}

        /* Tabs */
        .tabs {{
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
        }}

        .tab-btn {{
            background: var(--bg-card);
            border: 1px solid var(--card-border);
            color: var(--text-secondary);
            font-family: inherit;
            font-size: 0.95rem;
            font-weight: 700;
            padding: 0.65rem 1.4rem;
            border-radius: var(--pill-radius);
            cursor: pointer;
            box-shadow: var(--clay-shadow-panel);
            transition: all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .tab-btn:hover {{
            color: var(--text-primary);
            transform: translateY(-2px);
        }}

        .tab-btn.active {{
            background: var(--clay-sage);
            color: #ffffff;
            box-shadow: var(--clay-shadow-btn);
            transform: translateY(-1px);
        }}

        .tab-content {{
            display: none;
        }}

        .tab-content.active {{
            display: block;
        }}

        /* Clay Cards */
        .card {{
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: var(--clay-radius);
            padding: 1.75rem;
            box-shadow: var(--clay-shadow-panel);
            position: relative;
        }}

        .chat-box {{
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
            min-height: 520px;
        }}

        /* Frosted Blur Glass Dial / Top-K Control */
        .topk-control-bar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.75rem 1.35rem;
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 20px;
            box-shadow: var(--glass-shadow);
            font-size: 0.88rem;
            font-weight: 600;
            color: var(--glass-text);
        }}

        .dial-knob {{
            display: inline-flex;
            align-items: center;
            gap: 0.65rem;
        }}

        .dial-knob span {{
            color: var(--glass-text);
            font-weight: 600;
            letter-spacing: 0.2px;
        }}

        .concentric-circle {{
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: var(--clay-blue);
            box-shadow: 
                0 4px 8px rgba(120, 150, 160, 0.35),
                inset 2px 2px 4px rgba(255, 255, 255, 0.8),
                inset -2px -2px 4px rgba(70, 95, 105, 0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }}

        .concentric-circle::after {{
            content: '';
            width: 14px;
            height: 14px;
            border-radius: 50%;
            background: var(--clay-cream);
            box-shadow: inset 1px 1px 2px rgba(100, 120, 125, 0.3);
        }}

        .messages-container {{
            flex: 1;
            overflow-y: auto;
            max-height: 520px;
            display: flex;
            flex-direction: column;
            gap: 1.15rem;
            padding-right: 0.5rem;
        }}

        .message {{
            padding: 1.15rem 1.4rem;
            border-radius: 22px;
            max-width: 88%;
            line-height: 1.65;
            position: relative;
        }}

        .message.user {{
            background: var(--clay-blue);
            color: #1e293b;
            align-self: flex-end;
            box-shadow: 
                0 14px 28px -6px rgba(120, 155, 165, 0.32),
                inset 2px 2px 5px rgba(255, 255, 255, 0.85),
                inset -3px -3px 6px rgba(90, 125, 135, 0.28);
            border: 1px solid rgba(255, 255, 255, 0.5);
            font-weight: 500;
        }}

        .message.bot {{
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            color: var(--glass-text);
            align-self: flex-start;
            border: 1px solid var(--glass-border);
            box-shadow: var(--glass-shadow);
            border-left: 5px solid var(--clay-sage);
        }}

        .bot-response {{
            font-size: 0.96rem;
            line-height: 1.7;
            color: var(--glass-text);
        }}

        .bot-response p {{
            margin-bottom: 0.85rem;
        }}

        .bot-response p:last-child {{
            margin-bottom: 0;
        }}

        .bot-response ul, .bot-response ol {{
            margin: 0.75rem 0 1rem 0;
            padding-left: 1.5rem;
        }}

        .bot-response li {{
            margin-bottom: 0.55rem;
            line-height: 1.65;
            color: var(--glass-text);
            opacity: 0.95;
        }}

        .bot-response li::marker {{
            color: var(--clay-sage);
        }}

        .bot-response strong {{
            color: var(--text-primary);
            font-weight: 700;
        }}

        .bot-response h1, .bot-response h2, .bot-response h3, .bot-response h4 {{
            color: var(--text-primary);
            font-weight: 700;
            margin: 1.25rem 0 0.5rem 0;
            line-height: 1.4;
        }}

        .bot-response code {{
            background: rgba(149, 179, 162, 0.18);
            padding: 0.2rem 0.45rem;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85em;
            color: var(--clay-sage-dark);
            font-weight: 600;
        }}

        .bot-response pre {{
            background: rgba(240, 233, 223, 0.7);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 1rem;
            overflow-x: auto;
            margin: 0.85rem 0;
            box-shadow: var(--clay-shadow-inset);
        }}

        .bot-response pre code {{
            background: transparent;
            padding: 0;
            color: var(--text-primary);
        }}

        .citation-chip {{
            display: inline-flex;
            align-items: center;
            background: rgba(149, 179, 162, 0.2);
            color: var(--clay-sage-dark);
            border: 1px solid rgba(149, 179, 162, 0.4);
            font-size: 0.74rem;
            padding: 0.1rem 0.5rem;
            border-radius: 12px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            vertical-align: baseline;
            margin: 0 0.25rem;
            box-shadow: 0 2px 6px rgba(149, 179, 162, 0.15);
        }}

        .sources-card {{
            margin-top: 1rem;
            padding-top: 0.85rem;
            border-top: 1px dashed rgba(140, 155, 145, 0.3);
            font-size: 0.85rem;
        }}

        .sources-title {{
            font-weight: 700;
            color: var(--clay-sage);
            margin-bottom: 0.45rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }}

        .source-pill {{
            background: var(--glass-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 0.35rem 0.8rem;
            border-radius: 14px;
            display: inline-block;
            margin: 0.25rem 0.4rem 0.25rem 0;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: var(--glass-text);
            border: 1px solid var(--glass-border);
            box-shadow: var(--glass-shadow);
        }}

        /* Input Bar */
        .input-bar {{
            display: flex;
            gap: 0.85rem;
            margin-top: auto;
        }}

        .input-bar input {{
            flex: 1;
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            color: var(--glass-text);
            padding: 0.95rem 1.4rem;
            border-radius: var(--pill-radius);
            font-size: 0.95rem;
            font-family: inherit;
            outline: none;
            box-shadow: var(--glass-shadow);
            transition: all 0.25s ease;
        }}

        .input-bar input::placeholder {{
            color: var(--glass-placeholder);
            opacity: 1;
        }}

        .input-bar input:focus {{
            border-color: var(--clay-sage);
            background: rgba(255, 255, 255, 0.1);
            box-shadow: 0 0 0 3px rgba(118, 185, 0, 0.25), var(--glass-shadow);
        }}

        .glass-input {{
            width: 100%;
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            color: var(--glass-text);
            padding: 0.85rem 1.2rem;
            border-radius: 16px;
            margin-bottom: 0.85rem;
            box-shadow: var(--glass-shadow);
            outline: none;
            font-family: inherit;
            font-size: 0.95rem;
            transition: all 0.25s ease;
        }}

        .glass-input::placeholder {{
            color: var(--glass-placeholder);
            opacity: 1;
        }}

        .glass-input:focus {{
            border-color: var(--clay-sage);
            background: rgba(255, 255, 255, 0.1);
            box-shadow: 0 0 0 3px rgba(118, 185, 0, 0.25), var(--glass-shadow);
        }}

        .glass-result {{
            margin-top: 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            max-height: 160px;
            overflow-y: auto;
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            color: var(--glass-text);
            padding: 0.85rem;
            border-radius: 14px;
            box-shadow: var(--glass-shadow);
            display: none;
        }}

        .btn {{
            background: var(--clay-sage);
            color: #ffffff;
            font-family: inherit;
            font-weight: 700;
            border: none;
            padding: 0.95rem 1.6rem;
            border-radius: var(--pill-radius);
            cursor: pointer;
            box-shadow: var(--clay-shadow-btn);
            transition: all 0.15s cubic-bezier(0.34, 1.56, 0.64, 1);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }}

        .btn:hover {{
            transform: translateY(-2px);
        }}

        .btn:active {{
            transform: translateY(2px) scale(0.97);
        }}

        .btn-peach {{
            background: var(--clay-peach);
            color: #283339;
            box-shadow: 0 10px 22px -6px rgba(232, 191, 168, 0.45), inset 2px 2px 5px rgba(255, 255, 255, 0.8), inset -3px -3px 7px rgba(160, 125, 105, 0.25);
        }}

        .btn-secondary {{
            background: var(--bg-card);
            color: var(--text-primary);
            border: 1px solid var(--card-border);
            box-shadow: var(--clay-shadow-panel);
        }}

        .grid-2 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1.5rem;
        }}

        .drop-zone {{
            border: 2px dashed rgba(118, 185, 0, 0.5);
            border-radius: var(--clay-radius);
            padding: 2.75rem 1.5rem;
            text-align: center;
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            cursor: pointer;
            box-shadow: var(--glass-shadow);
            transition: all 0.25s;
            color: var(--glass-text);
            font-weight: 600;
        }}

        .drop-zone:hover {{
            border-color: var(--clay-sage);
            background: rgba(118, 185, 0, 0.12);
            color: #ffffff;
            box-shadow: 0 12px 32px rgba(118, 185, 0, 0.2);
            transform: scale(1.01);
        }}

        .stat-item {{
            display: flex;
            justify-content: space-between;
            padding: 0.85rem 0;
            border-bottom: 1px solid rgba(140, 155, 145, 0.2);
            font-size: 0.92rem;
        }}

        .stat-label {{
            color: var(--text-secondary);
            font-weight: 500;
        }}

        .stat-val {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            color: var(--text-primary);
        }}

        .banner-warning {{
            background: rgba(232, 191, 168, 0.25);
            border: 1px solid rgba(232, 191, 168, 0.6);
            border-radius: 20px;
            padding: 1rem 1.4rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1.5rem;
            box-shadow: var(--clay-shadow-panel);
        }}

        .banner-warning strong {{
            color: #b45309;
        }}
    </style>
</head>
<body>
    <!-- Floating Organic Foliage -->
    <div class="foliage-layer" aria-hidden="true">
        <!-- Top Left Monstera -->
        <svg class="leaf leaf-tl" viewBox="0 0 200 240" fill="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="monstera1" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#95b3a2"/>
                    <stop offset="100%" stop-color="#6e8f7e"/>
                </linearGradient>
            </defs>
            <path d="M100 230 C98 170 102 110 98 10 C125 25 180 75 190 135 C198 175 180 215 145 230 C125 238 110 235 100 230 Z" fill="url(#monstera1)"/>
            <path d="M100 230 C102 170 98 110 102 10 C75 25 20 75 10 135 C2 175 20 215 55 230 C75 238 90 235 100 230 Z" fill="url(#monstera1)"/>
            <ellipse cx="140" cy="110" rx="10" ry="25" transform="rotate(35 140 110)" fill="#f6efe6"/>
            <ellipse cx="60" cy="110" rx="10" ry="25" transform="rotate(-35 60 110)" fill="#f6efe6"/>
            <ellipse cx="150" cy="165" rx="8" ry="20" transform="rotate(50 150 165)" fill="#f6efe6"/>
            <ellipse cx="50" cy="165" rx="8" ry="20" transform="rotate(-50 50 165)" fill="#f6efe6"/>
        </svg>

        <!-- Top Right Palm Branch -->
        <svg class="leaf leaf-tr" viewBox="0 0 200 220" fill="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="palm1" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#a4c2b2"/>
                    <stop offset="100%" stop-color="#779986"/>
                </linearGradient>
            </defs>
            <path d="M190 10 C120 70 60 140 10 210" stroke="#779986" stroke-width="3" stroke-linecap="round"/>
            <path d="M170 30 C130 30 80 50 50 85 C80 75 130 65 170 30 Z" fill="url(#palm1)"/>
            <path d="M140 65 C100 70 55 95 30 130 C60 115 105 100 140 65 Z" fill="url(#palm1)"/>
            <path d="M110 105 C75 115 35 145 15 185 C45 165 85 145 110 105 Z" fill="url(#palm1)"/>
            <path d="M185 20 C160 55 130 110 105 150 C125 110 155 60 185 20 Z" fill="url(#palm1)"/>
        </svg>

        <!-- Bottom Left Palm Branch -->
        <svg class="leaf leaf-bl" viewBox="0 0 200 220" fill="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="palm2" x1="0%" y1="100%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="#95b3a2"/>
                    <stop offset="100%" stop-color="#709180"/>
                </linearGradient>
            </defs>
            <path d="M10 210 C70 140 130 70 190 10" stroke="#709180" stroke-width="3" stroke-linecap="round"/>
            <path d="M30 180 C60 180 110 160 140 125 C110 135 60 145 30 180 Z" fill="url(#palm2)"/>
            <path d="M60 145 C95 140 140 115 165 80 C135 95 90 110 60 145 Z" fill="url(#palm2)"/>
            <path d="M90 105 C120 95 160 65 180 25 C150 45 110 65 90 105 Z" fill="url(#palm2)"/>
        </svg>

        <!-- Bottom Right Monstera -->
        <svg class="leaf leaf-br" viewBox="0 0 200 240" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M100 230 C98 170 102 110 98 10 C125 25 180 75 190 135 C198 175 180 215 145 230 C125 238 110 235 100 230 Z" fill="url(#monstera1)"/>
            <path d="M100 230 C102 170 98 110 102 10 C75 25 20 75 10 135 C2 175 20 215 55 230 C75 238 90 235 100 230 Z" fill="url(#monstera1)"/>
            <ellipse cx="140" cy="110" rx="10" ry="25" transform="rotate(35 140 110)" fill="#f6efe6"/>
            <ellipse cx="60" cy="110" rx="10" ry="25" transform="rotate(-35 60 110)" fill="#f6efe6"/>
        </svg>
    </div>

    <!-- Header Bar -->
    <header>
        <div class="brand">
            <div class="logo-badge">NVIDIA</div>
            <h1>Welcome to <span>StudentRAG.com</span></h1>
        </div>
        <div class="header-actions">
            <a href="/" class="theme-toggle-btn" style="text-decoration: none;" title="Back to Landing Page">
                <span>←</span> <span>Landing Page</span>
            </a>
            <button class="theme-toggle-btn" onclick="toggleTheme()" title="Toggle Claymorphic / Dark mode">
                <span id="theme-icon">🌿</span> <span id="theme-label">Pastel Clay</span>
            </button>
            <div class="status-badge">
                <div class="status-dot"></div>
                <span>{'API Connected' if has_key else 'API Key Required'}</span>
            </div>
        </div>
    </header>

    <div class="main-container">
        {'' if has_key else '''
        <div class="banner-warning">
            <div>
                <strong>NVIDIA API Key is missing:</strong> Enter your free API key from <a href="https://build.nvidia.com" target="_blank" style="color: #95b3a2; font-weight: 700;">build.nvidia.com</a> below to activate live embeddings.
            </div>
            <button class="btn btn-peach" onclick="switchTab('settings')">Configure Key</button>
        </div>
        '''}

        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('chat')">💬 Q&A Chat</button>
            <button class="tab-btn" onclick="switchTab('ingest')">📥 Document Ingestion</button>
            <button class="tab-btn" onclick="switchTab('index')">📊 Vector Index Explorer</button>
        </div>

        <!-- Chat Tab -->
        <div id="tab-chat" class="tab-content active">
            <div class="card chat-box">
                <!-- Top-K Tactile Dial Bar -->
                <div class="topk-control-bar">
                    <div class="dial-knob">
                        <div class="concentric-circle"></div>
                        <span>Retrieval Depth (Top-K Chunks)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <input type="range" id="topk-slider" min="1" max="10" value="4" oninput="document.getElementById('topk-val').innerText = this.value" style="accent-color: var(--clay-sage); cursor: pointer;">
                        <span id="topk-val" style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: var(--clay-sage-dark);">4</span>
                    </div>
                </div>

                <div class="messages-container" id="chat-messages">
                    <div class="message bot">
                        <div class="bot-response">
                            <p>👋 Welcome to the <strong>NVIDIA Nemotron-3.5-Embed-1B</strong> Claymorphic RAG Studio!</p>
                            <p>This pipeline embeds document passages with <code>nvidia/nemotron-3.5-embed-1b</code> into dense FAISS vectors (2048 dimensions via MRL), performs asymmetric query retrieval, and synthesizes answers with the <code>nvidia/nemotron-3.5-lightning-30b-a3b</code> LLM with verified source citations.</p>
                            <p><strong>Suggested questions to try:</strong></p>
                            <ul>
                                <li><strong>Model Architecture:</strong> What is the parameter size and context length of Nemotron-3.5-Embed-1B?</li>
                                <li><strong>Retrieval Strategy:</strong> How does asymmetric retrieval work between query and passage modes?</li>
                                <li><strong>RAG Guidelines:</strong> What are the recommended chunking and overlap settings?</li>
                            </ul>
                        </div>
                    </div>
                </div>
                <div class="input-bar">
                    <input type="text" id="query-input" placeholder="Ask a question about your indexed knowledge base..." onkeydown="if(event.key==='Enter') sendQuery()">
                    <button class="btn" id="send-btn" onclick="sendQuery()">Ask RAG</button>
                </div>
            </div>
        </div>

        <!-- Ingest Tab -->
        <div id="tab-ingest" class="tab-content">
            <div class="grid-2">
                <div class="card">
                    <h3 style="margin-bottom: 0.75rem; color: var(--text-primary); font-weight: 700;">Built-in Knowledge Base</h3>
                    <p style="color: var(--text-secondary); margin-bottom: 1.25rem; font-size: 0.92rem;">
                        Index pre-packaged technical guides (Nemotron-3.5 architecture, multimodal retrieval, and RAG best practices).
                    </p>
                    <button class="btn" onclick="ingestBuiltin()">⚡ Ingest Sample Knowledge Base</button>
                    <div id="ingest-status" style="margin-top: 1rem; font-size: 0.88rem; font-weight: 600;"></div>
                </div>

                <div class="card">
                    <h3 style="margin-bottom: 0.75rem; color: var(--text-primary); font-weight: 700;">Upload Custom Documents</h3>
                    <p style="color: var(--text-secondary); margin-bottom: 1rem; font-size: 0.92rem;">
                        Upload PDF, Markdown, TXT, or JSON files to chunk and index with Nemotron-3.5-Embed-1B.
                    </p>
                    <div class="drop-zone" onclick="document.getElementById('file-upload').click()">
                        📁 Click or Drag & Drop file here (PDF, MD, TXT, JSON)
                        <input type="file" id="file-upload" style="display: none;" onchange="uploadFile(this.files[0])">
                    </div>
                    <div id="upload-status" style="margin-top: 1rem; font-size: 0.88rem; font-weight: 600;"></div>
                </div>
            </div>
        </div>

        <!-- Index Explorer Tab -->
        <div id="tab-index" class="tab-content">
            <div class="grid-2">
                <div class="card">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
                        <h3 style="color: var(--text-primary); font-weight: 700;">FAISS Vector Store Status</h3>
                        <div class="concentric-circle" title="Active Vector Index"></div>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Index Status</span>
                        <span class="stat-val" style="color: var(--clay-sage-dark);">{'Active' if stats.get('indexed') else 'Empty'}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Total Vectors</span>
                        <span class="stat-val">{stats.get('total_vectors', 0)}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Embedding Dimensions</span>
                        <span class="stat-val">{stats.get('dimensions', 2048)}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Embedding Model</span>
                        <span class="stat-val">{stats.get('embedding_model', 'nvidia/nemotron-3.5-embed-1b')}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Index Directory</span>
                        <span class="stat-val" style="font-size: 0.75rem;">{stats.get('index_dir')}</span>
                    </div>
                    <div style="margin-top: 1.25rem;">
                        <span class="stat-label">Indexed Documents:</span>
                        <div style="margin-top: 0.5rem;">
                            {''.join([f'<span class="source-pill">{f}</span>' for f in stats.get('source_files', [])]) or '<span style="color: var(--text-muted); font-size: 0.85rem;">No documents indexed yet.</span>'}
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h3 style="margin-bottom: 1rem; color: var(--text-primary); font-weight: 700;">Direct Embedding Tester</h3>
                    <p style="color: var(--text-secondary); margin-bottom: 1rem; font-size: 0.92rem;">
                        Test <code>nvidia/nemotron-3.5-embed-1b</code> directly and view the raw vector response.
                    </p>
                    <input type="text" id="embed-test-input" class="glass-input" placeholder="Type test phrase...">
                    <button class="btn btn-secondary" onclick="testEmbed()">Generate Vector</button>
                    <div id="embed-result" class="glass-result"></div>
                </div>
            </div>
        </div>

    </div>

    <script>
        // Theme Toggle - Always start in Light Theme (Pastel Clay) 1st on load
        function initTheme() {{
            document.body.removeAttribute('data-theme');
            const icon = document.getElementById('theme-icon');
            const label = document.getElementById('theme-label');
            if (icon) icon.innerText = '🌿';
            if (label) label.innerText = 'Pastel Clay';
            localStorage.setItem('nemotron_theme', 'light');
        }}

        function toggleTheme() {{
            const isDark = document.body.getAttribute('data-theme') === 'dark';
            if (isDark) {{
                document.body.removeAttribute('data-theme');
                localStorage.setItem('nemotron_theme', 'light');
                document.getElementById('theme-icon').innerText = '🌿';
                document.getElementById('theme-label').innerText = 'Pastel Clay';
            }} else {{
                document.body.setAttribute('data-theme', 'dark');
                localStorage.setItem('nemotron_theme', 'dark');
                document.getElementById('theme-icon').innerText = '🌙';
                document.getElementById('theme-label').innerText = 'Cyber Dark';
            }}
        }}

        window.addEventListener('DOMContentLoaded', initTheme);

        function switchTab(tabId) {{
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('tab-' + tabId).classList.add('active');
        }}

        async function sendQuery() {{
            const input = document.getElementById('query-input');
            const q = input.value.trim();
            if (!q) return;

            const topk = parseInt(document.getElementById('topk-slider').value) || 4;
            const chatMessages = document.getElementById('chat-messages');
            chatMessages.innerHTML += `<div class="message user">${{escapeHtml(q)}}</div>`;
            input.value = '';

            const loadingId = 'loading-' + Date.now();
            chatMessages.innerHTML += `<div class="message bot" id="${{loadingId}}"><span style="color: var(--text-muted);">Retrieving with Nemotron-3.5-Embed-1B & Synthesizing...</span></div>`;
            chatMessages.scrollTop = chatMessages.scrollHeight;

            try {{
                const res = await fetch('/api/query', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{query: q, top_k: topk}})
                }});

                const data = await res.json();
                const loadEl = document.getElementById(loadingId);

                if (!res.ok) {{
                    loadEl.innerHTML = `<span style="color: #ef4444;">Error: ${{data.detail || 'Query failed'}}</span>`;
                    return;
                }}

                let sourcesHtml = '';
                if (data.sources && data.sources.length > 0) {{
                    sourcesHtml = '<div class="sources-card"><div class="sources-title">Verified Sources:</div>';
                    data.sources.forEach((s, idx) => {{
                        sourcesHtml += `<span class="source-pill" title="${{escapeHtml(s.snippet)}}">Source ${{idx+1}}: ${{escapeHtml(s.source_file)}} (Chunk ${{s.chunk_id}})</span>`;
                    }});
                    sourcesHtml += '</div>';
                }}

                const formattedAnswer = formatBotAnswer(data.answer);
                loadEl.innerHTML = `<div class="bot-response">${{formattedAnswer}}</div>${{sourcesHtml}}`;
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }} catch (err) {{
                document.getElementById(loadingId).innerHTML = `<span style="color: #ef4444;">Network error: ${{err.message}}</span>`;
            }}
        }}

        async function ingestBuiltin() {{
            const status = document.getElementById('ingest-status');
            status.innerHTML = '<span style="color: var(--clay-blue-dark);">Ingesting sample documents with Nemotron-3.5-Embed-1B...</span>';
            try {{
                const res = await fetch('/api/ingest-sample', {{method: 'POST'}});
                const data = await res.json();
                if (res.ok) {{
                    status.innerHTML = `<span style="color: var(--clay-sage-dark);">✔ Successfully ingested ${{data.num_documents}} documents into ${{data.num_chunks}} vector passages!</span>`;
                }} else {{
                    status.innerHTML = `<span style="color: #ef4444;">✘ ${{data.detail}}</span>`;
                }}
            }} catch (e) {{
                status.innerHTML = `<span style="color: #ef4444;">✘ Error: ${{e.message}}</span>`;
            }}
        }}

        async function uploadFile(file) {{
            if (!file) return;
            const status = document.getElementById('upload-status');
            status.innerHTML = `<span style="color: var(--clay-blue-dark);">Uploading & indexing ${{file.name}}...</span>`;

            const formData = new FormData();
            formData.append('file', file);

            try {{
                const res = await fetch('/api/upload', {{
                    method: 'POST',
                    body: formData
                }});
                const data = await res.json();
                if (res.ok) {{
                    status.innerHTML = `<span style="color: var(--clay-sage-dark);">✔ Ingested ${{file.name}} into ${{data.num_chunks}} vector chunks!</span>`;
                }} else {{
                    status.innerHTML = `<span style="color: #ef4444;">✘ ${{data.detail}}</span>`;
                }}
            }} catch (e) {{
                status.innerHTML = `<span style="color: #ef4444;">✘ Error: ${{e.message}}</span>`;
            }}
        }}

        async function testEmbed() {{
            const text = document.getElementById('embed-test-input').value.trim() || 'NVIDIA Nemotron';
            const resEl = document.getElementById('embed-result');
            resEl.style.display = 'block';
            resEl.innerHTML = '<span style="color: var(--text-muted);">Generating embedding vector...</span>';

            try {{
                const res = await fetch('/api/test-embed', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{text: text}})
                }});
                const data = await res.json();
                if (res.ok) {{
                    resEl.innerHTML = `<strong>Model:</strong> ${{data.model}}<br><strong>Dimensions:</strong> ${{data.dimensions}}<br><strong>Vector Sample:</strong><br>[${{data.sample.join(', ')}}]`;
                }} else {{
                    resEl.innerHTML = `<span style="color: #ef4444;">Error: ${{data.detail}}</span>`;
                }}
            }} catch (e) {{
                resEl.innerHTML = `<span style="color: #ef4444;">Error: ${{e.message}}</span>`;
            }}
        }}

        async function saveKey() {{
            const key = document.getElementById('api-key-input').value.trim();
            const status = document.getElementById('settings-status');
            try {{
                const res = await fetch('/api/set-key', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{api_key: key}})
                }});
                const data = await res.json();
                if (res.ok) {{
                    status.innerHTML = '<span style="color: var(--clay-sage-dark);">✔ API Key saved successfully! Reloading...</span>';
                    setTimeout(() => location.reload(), 1000);
                }} else {{
                    status.innerHTML = `<span style="color: #ef4444;">✘ ${{data.detail}}</span>`;
                }}
            }} catch (e) {{
                status.innerHTML = `<span style="color: #ef4444;">✘ Error: ${{e.message}}</span>`;
            }}
        }}

        function formatBotAnswer(rawText) {{
            if (!rawText) return '';
            let text = rawText.trim();

            // Strip robotic negative disclaimers if generated
            text = text.replace(/\\n*No other [^\\n\\.]*(appear|mentioned|found)[^\\n\\.]*\\.?/gi, '');
            text = text.replace(/\\n*No additional [^\\n\\.]*(appear|mentioned|found)[^\\n\\.]*\\.?/gi, '');

            let html = '';
            if (window.marked && typeof marked.parse === 'function') {{
                try {{
                    html = marked.parse(text, {{ breaks: true, gfm: true }});
                }} catch (e) {{
                    html = fallbackMarkdown(text);
                }}
            }} else {{
                html = fallbackMarkdown(text);
            }}

            // Convert [Source X] to citation chip badges
            html = html.replace(/\\[Source\\s*([0-9,\\s]+)\\]/gi, function(match, p1) {{
                return '<span class="citation-chip">Source ' + p1.trim() + '</span>';
            }});

            return html;
        }}

        function fallbackMarkdown(text) {{
            const lines = text.split('\\n');
            let out = '';
            let inList = false;

            for (let i = 0; i < lines.length; i++) {{
                const line = lines[i].trim();
                if (!line) {{
                    if (inList) {{ out += '</ul>'; inList = false; }}
                    continue;
                }}

                if (line.startsWith('- ') || line.startsWith('* ') || line.startsWith('• ')) {{
                    if (!inList) {{ out += '<ul>'; inList = true; }}
                    const item = line.replace(/^[-*•]\\s+/, '');
                    out += '<li>' + formatInline(item) + '</li>';
                }} else {{
                    if (inList) {{ out += '</ul>'; inList = false; }}
                    if (line.startsWith('### ')) {{
                        out += '<h3>' + formatInline(line.substring(4)) + '</h3>';
                    }} else if (line.startsWith('## ')) {{
                        out += '<h2>' + formatInline(line.substring(3)) + '</h2>';
                    }} else if (line.startsWith('# ')) {{
                        out += '<h1>' + formatInline(line.substring(2)) + '</h1>';
                    }} else {{
                        out += '<p>' + formatInline(line) + '</p>';
                    }}
                }}
            }}
            if (inList) {{ out += '</ul>'; }}
            return out;
        }}

        function formatInline(str) {{
            return escapeHtml(str)
                .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
                .replace(/\\*(.+?)\\*/g, '<em>$1</em>')
                .replace(/`([^`]+)`/g, '<code>$1</code>');
        }}

        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


@app.post("/api/query")
async def api_query(req: QueryRequest):
    """Executes RAG inference with Nemotron-3.5-Embed-1B."""
    try:
        result = pipeline.answer_query(req.query, top_k=req.top_k)
        return result
    except MissingAPIKeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest-sample")
async def api_ingest_sample():
    """Ingests the sample documents in data/sample_documents."""
    try:
        docs = pipeline.load_directory(config.sample_docs_dir)
        num_docs, num_chunks = pipeline.ingest_documents(docs, persist=True)
        return {"num_documents": num_docs, "num_chunks": num_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    """Uploads and ingests a custom document."""
    try:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)

        docs = pipeline.load_file(tmp_path)
        # Update metadata to original filename
        for d in docs:
            d.metadata["source_file"] = file.filename

        num_docs, num_chunks = pipeline.ingest_documents(docs, persist=True)
        tmp_path.unlink(missing_ok=True)

        return {"filename": file.filename, "num_chunks": num_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/test-embed")
async def api_test_embed(req: EmbedTestRequest):
    """Generates direct embedding vector with Nemotron-3.5-Embed-1B."""
    try:
        vec = pipeline.embedder.embed_query(req.text)
        return {
            "model": config.embedding_model,
            "dimensions": len(vec),
            "sample": [round(v, 5) for v in vec[:8]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/set-key")
async def api_set_key(req: KeyUpdateRequest):
    """Updates API key in memory and .env file."""
    key = req.api_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="API Key cannot be empty.")

    config.nvidia_api_key = key
    pipeline.api_key = key
    pipeline._embedder = None
    pipeline._llm = None

    # Update .env file
    env_file = BASE_DIR / ".env"
    lines = []
    if env_file.exists():
        lines = env_file.read_text(encoding="utf-8").splitlines()

    updated = False
    new_lines = []
    for line in lines:
        if line.startswith("NVIDIA_API_KEY="):
            new_lines.append(f"NVIDIA_API_KEY={key}")
            updated = True
        else:
            new_lines.append(line)
    if not updated:
        new_lines.append(f"NVIDIA_API_KEY={key}")

    env_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return {"status": "success"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\nStarting NVIDIA Nemotron RAG Studio on http://0.0.0.0:{port} ...")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
