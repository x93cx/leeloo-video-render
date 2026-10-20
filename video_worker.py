"""Виробництво відео за сценарієм з контент-плану у віральному форматі — безкоштовно.

Монтаж: сцени ріжуться на плани по ~2 с (сток Pexels, AI-картинки або типографіка) з наїздами камери,
субтитри підсвічують слово, коли воно звучить (таймінги слів edge-tts), великий хук у перші ~2 с,
музика з приглушенням під голос. Параметри монтажу бере з ДНК формату ролика-донора, якщо вона є.

Сценарій (шаблон short-form-scripts: таблиця «Время | На экране | Говорит | B-roll») розбирається на сцени.
Тег на початку колонки B-roll задає тип плану: [stock], [screen], [face] — сток; [text] — типографіка; [ai] — AI;
«— en: …» — запит для стоку, «— cut: …» — перебивка (вставка ~0,7 с поверх голосу).
Між сценами — переходи (свайп, спалах, глітч, ривок зуму) зі звуком «вжух»; удар на хуку, «поп» на перебивках.
Для кожного плану: сток Pexels → AI-кліп/картинка (FreeLLMAPI або Pollinations) → фон-типографіка.
Голос: edge-tts (uk-UA / ru-RU) з межами слів. Текст «На экране» — плашкою зверху або великим по центру.
Усе склеюється у вертикальне відео 9:16; субтитри накладає ffmpeg з libass.

Режими:
  python video_worker.py --watch            # брати чергу з дашборду (статус «У виробництві») кожні 30 с
  python video_worker.py --once             # обробити чергу один раз
  python video_worker.py --file script.md   # локальний тест без дашборду, результат у --out

Змінні середовища:
  DASHBOARD_URL, DASHBOARD_TOKEN   — адреса дашборду й токен імпорту (той самий, що для run_all.py)
  Медіа (одне з):
    FREELLMAPI_URL (http://localhost:3001), FREELLMAPI_KEY — локальний FreeLLMAPI
    POLLINATIONS_KEY (sk_…), POLLINATIONS_IMAGE_MODEL (tongyi-mai/z-image-turbo),
      POLLINATIONS_VIDEO_MODEL (порожньо = без кліпів; напр. alibaba/wan-2.2-fast — платні Pollen)
  План кадрів: FreeLLMAPI, або LLM_URL/LLM_KEY/LLM_MODEL, або Claude через дашборд (DASHBOARD_URL + DASHBOARD_TOKEN)
  Сток: PEXELS_KEY — безкоштовні вертикальні відео з pexels.com/api для кожного плану
  Музика: MUSIC_DIR або папка music/ поруч (mp3/wav/m4a); без них — згенерований біт без авторських прав
  Аватар: налаштування HeyGen (ключ, аватар, голос, режим) приходять з черги дашборду

Залежності: pip install requests pillow imageio-ffmpeg edge-tts
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

# Логи GitHub Actions у публічному репозиторії бачать усі: туди не пишемо тексти сценаріїв,
# промпти й відповіді сервісів (у них буває промпт). Локально лог повний.
PUBLIC_LOG = os.environ.get("GITHUB_ACTIONS") == "true"


def safe(text: object) -> str:
    return "…" if PUBLIC_LOG else str(text)


def safe_err(e: Exception) -> str:
    return type(e).__name__ if PUBLIC_LOG else str(e)

FPS = 30
VOICES = {"uk": "uk-UA-OstapNeural", "ru": "ru-RU-DmitryNeural", "en": "en-US-GuyNeural"}
BG_DARK = (17, 17, 17)
ACCENT = (255, 206, 0)  # жовтий акцент на темному фоні — як у креативах Leeloo


# ---------- сцени зі сценарію ----------

@dataclass
class Scene:
    start: float
    end: float
    on_screen: str
    voice: str
    broll: str
    prompt: str = ""
    clip: Path | None = None
    clip_kind: str = ""
    audio: Path | None = None
    voice_engine: str = ""
    dur: float = 0.0
    notes: list[str] = field(default_factory=list)
    tag: str = ""                      # [stock] / [text] / [ai] / [screen] / [face] на початку колонки B-roll
    queries: list[str] = field(default_factory=list)
    words: list[tuple[float, float, str]] = field(default_factory=list)
    cutaway: str = ""                  # запит стоку для короткої вставки-перебивки поверх голосу
    per: list[int] = field(default_factory=list)   # кадри кожного плану сцени
    cut_at: int = -1                   # номер плану-перебивки або -1


def _clean(cell: str) -> str:
    cell = re.sub(r"\*\*|__|`", "", cell).strip()
    return "" if cell in ("-", "–", "—") else cell


def detect_language(md: str) -> str:
    m = re.search(r"ЯЗ[ЫИ]К:\s*\**\s*(UA|RU|EN)", md, re.I)
    if m:
        return {"UA": "uk", "RU": "ru", "EN": "en"}[m.group(1).upper()]
    return "ru" if re.search(r"[ыэъё]", md) else "uk"


def parse_scenes(md: str) -> list[Scene]:
    """Перша таблиця сценарію: колонки час / на екрані / говорить / візуал (RU або UA заголовки)."""
    part = md.split("СЦЕНАРИЙ", 1)[-1] if "СЦЕНАРИЙ" in md else md
    lines = [l.strip() for l in part.splitlines()]
    scenes: list[Scene] = []
    header: list[str] | None = None
    for line in lines:
        if not line.startswith("|"):
            if header and scenes:
                break  # таблиця закінчилась — беремо лише першу (перший сценарій, якщо їх кілька)
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        low = [c.lower() for c in cells]
        if header is None:
            if any(k in " ".join(low) for k in ("время", "час")) and any(k in " ".join(low) for k in ("говор", "озвуч")):
                header = low
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        col = lambda *keys: next((i for i, h in enumerate(header) if any(k in h for k in keys)), None)  # noqa: E731
        it, isc, iv, ib = col("время", "час"), col("экран", "екран"), col("говор", "озвуч"), col("b-roll", "визуал", "візуал")
        get = lambda i: _clean(cells[i]) if i is not None and i < len(cells) else ""  # noqa: E731
        nums = [float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", get(it))]
        start, end = (nums[0], nums[1]) if len(nums) >= 2 else ((scenes[-1].end if scenes else 0.0), (scenes[-1].end if scenes else 0.0) + 4)
        broll = get(ib)
        tag = re.match(r"\[(stock|text|ai|screen|face|card)\]\s*", broll, re.I)
        scenes.append(Scene(start, max(end, start + 1), get(isc), get(iv), broll[tag.end():] if tag else broll,
                            tag=("text" if tag.group(1).lower() == "card" else tag.group(1).lower()) if tag else ""))
    return scenes


# ---------- FreeLLMAPI ----------

class FreeLLM:
    def __init__(self, url: str | None, key: str | None, use_video: bool = True):
        self.url = (url or "").rstrip("/")
        self.key = key or ""
        self.use_video = use_video
        self.ok = False
        if self.url:
            try:
                r = requests.get(f"{self.url}/v1/models", headers=self._h(), params={"available": "true"}, timeout=8)
                self.ok = r.status_code == 200
            except requests.RequestException:
                self.ok = False

    def _h(self) -> dict:
        return {"Authorization": f"Bearer {self.key}"} if self.key else {}

    def chat_json(self, prompt: str) -> object | None:
        if not self.ok:
            return None
        try:
            r = requests.post(f"{self.url}/v1/chat/completions", headers=self._h(), timeout=180, json={
                "model": "auto", "temperature": 0.4,
                "messages": [{"role": "user", "content": prompt}],
            })
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
            return json.loads(m.group(1)) if m else None
        except Exception as e:  # noqa: BLE001
            print(f"  chat: {safe_err(e)}", file=sys.stderr)
            return None

    def video(self, prompt: str, seconds: float, dest: Path) -> str | None:
        if not (self.ok and self.use_video):
            return None
        try:
            r = requests.post(f"{self.url}/v1/videos/generations", headers=self._h(), timeout=320, json={
                "model": "auto", "prompt": prompt, "duration": int(max(3, min(6, round(seconds)))), "aspect_ratio": "9:16",
            })
            if r.status_code != 200 or "video" not in r.headers.get("Content-Type", "video"):
                print(f"  video: HTTP {r.status_code} {safe(r.text[:160])}", file=sys.stderr)
                return None
            dest.write_bytes(r.content)
            return f"video:{r.headers.get('X-Provider', '?')}/{r.headers.get('X-Model', '?')}"
        except requests.RequestException as e:
            print(f"  video: {safe_err(e)}", file=sys.stderr)
            return None

    def image(self, prompt: str, dest: Path) -> str | None:
        if not self.ok:
            return None
        try:
            r = requests.post(f"{self.url}/v1/images/generations", headers=self._h(), timeout=180, json={
                "model": "auto", "prompt": prompt, "size": "1024x1792", "n": 1,
            })
            if r.status_code != 200:
                print(f"  image: HTTP {r.status_code} {safe(r.text[:160])}", file=sys.stderr)
                return None
            if r.headers.get("Content-Type", "").startswith("image/"):
                dest.write_bytes(r.content)
            else:
                item = r.json()["data"][0]
                if item.get("b64_json"):
                    dest.write_bytes(base64.b64decode(item["b64_json"]))
                else:
                    dest.write_bytes(requests.get(item["url"], timeout=60).content)
            return f"image:{r.headers.get('X-Provider', '?')}/{r.headers.get('X-Model', '?')}"
        except Exception as e:  # noqa: BLE001
            print(f"  image: {safe_err(e)}", file=sys.stderr)
            return None

    def speech(self, text: str, dest: Path) -> str | None:
        if not self.ok:
            return None
        try:
            r = requests.post(f"{self.url}/v1/audio/speech", headers=self._h(), timeout=120, json={
                "model": "auto", "input": text, "voice": "alloy", "response_format": "mp3",
            })
            if r.status_code != 200 or not r.content:
                return None
            dest.write_bytes(r.content)
            return f"tts:{r.headers.get('X-Provider', '?')}/{r.headers.get('X-Model', '?')}"
        except requests.RequestException:
            return None


class Pollinations:
    """Pollinations напряму (gen.pollinations.ai) — для GitHub Actions і будь-де без FreeLLMAPI.

    Потрібен секретний ключ sk_ з enter.pollinations.ai. Картинки (z-image-turbo, flux.2-klein-4b) йдуть і за
    безкоштовні Quest Pollen; відеомоделі — лише за платні Pollen, тому кліпи вмикаються тільки якщо задано
    POLLINATIONS_VIDEO_MODEL (напр. alibaba/wan-2.2-fast ≈ 0.01 Pollen/с). Голос робить edge-tts.
    """
    BASE = "https://gen.pollinations.ai"

    def __init__(self, key: str, image_model: str, video_model: str = "", video_seconds: int = 5):
        self.key = key
        self.image_model = image_model
        self.video_model = video_model
        self.video_seconds = video_seconds
        self.ok = bool(key)
        self.name = "pollinations"

    def _h(self) -> dict:
        return {"Authorization": f"Bearer {self.key}"}

    def video(self, prompt: str, seconds: float, dest: Path) -> str | None:
        if not (self.ok and self.video_model):
            return None
        try:
            from urllib.parse import quote
            r = requests.get(f"{self.BASE}/video/{quote(prompt)}", headers=self._h(), timeout=600,
                             params={"model": self.video_model, "duration": self.video_seconds})
            if r.status_code != 200 or not r.headers.get("Content-Type", "").startswith("video/"):
                print(f"  video: HTTP {r.status_code} {safe(r.text[:160])}", file=sys.stderr)
                return None
            dest.write_bytes(r.content)
            return f"video:pollinations/{self.video_model}"
        except requests.RequestException as e:
            print(f"  video: {safe_err(e)}", file=sys.stderr)
            return None

    def image(self, prompt: str, dest: Path) -> str | None:
        if not self.ok:
            return None
        try:
            r = requests.post(f"{self.BASE}/v1/images/generations", headers=self._h(), timeout=180, json={
                "model": self.image_model, "prompt": prompt, "size": "1024x1792", "response_format": "b64_json",
            })
            if r.status_code != 200:
                print(f"  image: HTTP {r.status_code} {safe(r.text[:160])}", file=sys.stderr)
                return None
            item = r.json()["data"][0]
            if item.get("b64_json"):
                dest.write_bytes(base64.b64decode(item["b64_json"]))
            else:
                dest.write_bytes(requests.get(item["url"], timeout=60).content)
            return f"image:pollinations/{self.image_model}"
        except Exception as e:  # noqa: BLE001
            print(f"  image: {safe_err(e)}", file=sys.stderr)
            return None

    def speech(self, text: str, dest: Path) -> str | None:
        return None  # голос — безкоштовний edge-tts


class OpenAIChat:
    """Будь-яка OpenAI-сумісна текстова модель для промптів. У GitHub Actions — безкоштовні GitHub Models
    (https://models.github.ai/inference, ключ = GITHUB_TOKEN з дозволом models: read)."""

    def __init__(self, base: str, key: str, model: str):
        self.base = base.rstrip("/")
        self.key = key
        self.model = model

    def chat_json(self, prompt: str) -> object | None:
        r = None
        try:
            r = requests.post(f"{self.base}/chat/completions", timeout=120,
                              headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                                       "Accept": "application/json", "User-Agent": "insta-viral-render/1.0"},
                              json={"model": self.model, "temperature": 0.4, "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
            try:
                return json.loads(text)
            except ValueError:
                m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
                return json.loads(m.group(1)) if m else None
        except Exception as e:  # noqa: BLE001
            body = " ".join(r.text[:200].split()) if r is not None else ""
            print(f"  chat ({self.model}): {safe_err(e)} · HTTP {getattr(r, 'status_code', '-')} · {safe(body)}", file=sys.stderr)
            return None


class DashboardChat:
    """Текстова модель через дашборд (Claude з ключем у його Налаштуваннях) — для воркера на GitHub Actions."""

    def __init__(self, url: str, token: str):
        self.base = url.rstrip("/") + "/api/index.php"
        self.h = {"X-Api-Token": token, "User-Agent": "insta-viral-render/1.0"}

    def chat_json(self, prompt: str) -> object | None:
        r = None
        try:
            r = requests.post(self.base, params={"action": "render_llm"}, headers=self.h, timeout=150,
                              json={"prompt": prompt, "max_tokens": 3000})
            data = r.json()
            if not data.get("ok"):
                raise RuntimeError(data.get("error") or f"HTTP {r.status_code}")
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", data["text"].strip())
            try:
                return json.loads(text)
            except ValueError:
                m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
                return json.loads(m.group(1)) if m else None
        except Exception as e:  # noqa: BLE001
            print(f"  chat (дашборд): {safe_err(e)} · HTTP {getattr(r, 'status_code', '-')}", file=sys.stderr)
            return None


def pick_providers(use_video: bool) -> tuple[object | None, object]:
    """Хто пише промпти (chat) і хто робить кліпи й картинки (media) — за тим, що доступно в оточенні."""
    free = FreeLLM(os.environ.get("FREELLMAPI_URL", "" if os.environ.get("GITHUB_ACTIONS") else "http://localhost:3001"),
                   os.environ.get("FREELLMAPI_KEY"), use_video=use_video)
    free.name = "freellmapi"
    pkey = os.environ.get("POLLINATIONS_KEY", "")
    if os.environ.get("MEDIA_PROVIDER") == "pollinations" or (pkey and not free.ok):
        media: object = Pollinations(pkey, os.environ.get("POLLINATIONS_IMAGE_MODEL", "tongyi-mai/z-image-turbo"),
                                     os.environ.get("POLLINATIONS_VIDEO_MODEL", "") if use_video else "",
                                     int(os.environ.get("POLLINATIONS_VIDEO_SECONDS", "5")))
    else:
        media = free
    if free.ok:
        chat: object | None = free
    elif os.environ.get("LLM_URL"):
        chat = OpenAIChat(os.environ["LLM_URL"], os.environ.get("LLM_KEY", ""), os.environ.get("LLM_MODEL", "auto"))
    elif os.environ.get("DASHBOARD_URL") and os.environ.get("DASHBOARD_TOKEN"):
        # GitHub Models з жовтня 2026 відповідає заглушкою «OK», тож план кадрів робить Claude через дашборд.
        chat = DashboardChat(os.environ["DASHBOARD_URL"], os.environ["DASHBOARD_TOKEN"])
    else:
        chat = None
    return chat, media


def visual_plan(llm: object | None, scenes: list[Scene], lang: str) -> None:
    """Для кожної сцени: англійський запит для стоку Pexels, запасний запит, промпт для AI-картинки й тип кадру."""
    for i, s in enumerate(scenes):
        c = re.search(r"(?:^|\s)cut:\s*(.+)$", s.broll)  # «… — cut: man facepalm» — перебивка зі сценарію
        if c:
            s.cutaway = c.group(1).strip()
            s.broll = s.broll[:c.start()].rstrip(" —-–")
        m = re.search(r"(?:^|\s)en:\s*(.+)$", s.broll)
        s.queries = [m.group(1).strip()] if m else []
        if m:
            s.broll = s.broll[:m.start()].rstrip(" —-–")
        s.prompt = f"cinematic vertical shot, modern small business, {s.queries[0] if s.queries else s.broll or s.on_screen}"
    if llm is None or all(s.queries and s.cutaway for s in scenes):
        return
    payload = [{"i": i, "visual": s.broll, "on_screen": s.on_screen, "voice": s.voice} for i, s in enumerate(scenes)]
    res = llm.chat_json(
        "You plan visuals for a vertical 9:16 short-form ad about Leeloo, an AI sales and marketing automation platform "
        "for small businesses. For every scene below return:\n"
        '- "query": 2-4 English words to search stock footage on Pexels: concrete, filmable objects and actions '
        '(e.g. "woman typing laptop", "phone notifications", "cash register"), never abstract ideas;\n'
        '- "alt": a different, more generic stock query for the same scene;\n'
        '- "prompt": a concise English prompt (max 35 words) for an AI image of the same moment, realistic and '
        "cinematic, no on-screen text, no logos, no real people's names;\n"
        '- "kind": "text" if the scene is best shown as big typography only (a number, a list item, a punchline), '
        'otherwise "stock";\n'
        '- "cutaway": 2-4 English words for a 0.7-second stock insert that punctuates the line — a reaction or a '
        'telling close-up (e.g. "man facepalm", "money falling", "clock ticking", "phone buzzing"), or "" if none.\n'
        'Return only JSON: [{"i": 0, "query": "...", "alt": "...", "prompt": "...", "kind": "stock", "cutaway": "..."}].\n\n'
        + json.dumps(payload, ensure_ascii=False))
    got = {int(x["i"]): x for x in res if isinstance(x, dict) and "i" in x} if isinstance(res, list) else {}
    for i, s in enumerate(scenes):
        x = got.get(i) or {}
        s.queries = s.queries + [str(q) for q in (x.get("query"), x.get("alt")) if q and q not in s.queries]
        if x.get("prompt"):
            s.prompt = str(x["prompt"])
        if not s.tag and x.get("kind") == "text":
            s.tag = "text"
        if x.get("cutaway") and not s.cutaway:
            s.cutaway = str(x["cutaway"])



# ---------- медіа ----------

_FFMPEG: str | None = None


def ffmpeg() -> str:
    """ffmpeg із libass (субтитри ASS): FFMPEG_BINARY, вбудований у imageio-ffmpeg або системний."""
    global _FFMPEG
    if _FFMPEG:
        return _FFMPEG
    import shutil
    import imageio_ffmpeg
    cands = [os.environ.get("FFMPEG_BINARY"), imageio_ffmpeg.get_ffmpeg_exe(), shutil.which("ffmpeg")]
    for c in [c for c in cands if c]:
        p = subprocess.run([c, "-hide_banner", "-filters"], capture_output=True, text=True, errors="replace")
        if " ass " in p.stdout:
            _FFMPEG = c
            return c
    raise RuntimeError("Немає ffmpeg з libass (субтитри) — встанови системний ffmpeg або задай FFMPEG_BINARY.")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=cwd)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg: " + p.stderr[-800:])


def duration_of(path: Path) -> float:
    p = subprocess.run([ffmpeg(), "-i", str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", p.stderr)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0


def find_font(bold: bool = True) -> str:
    names = ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"]
    dirs = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts", Path("/usr/share/fonts/truetype/dejavu"), Path("/Library/Fonts")]
    for d in dirs:
        for n in names:
            if (d / n).exists():
                return str(d / n)
    raise RuntimeError("Не знайдено шрифт з кирилицею — передай --font шлях до .ttf")


# ---------- стиль монтажу (ДНК формату) ----------

def _hex(v: object) -> tuple[int, int, int] | None:
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", str(v or "").strip())
    return tuple(int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4)) if m else None  # type: ignore[return-value]


@dataclass
class Style:
    """Як монтувати. За замовчуванням — типовий віральний рілс; ДНК ролика-донора уточнює параметри."""
    shot_len: float = 2.2              # середня довжина плану, с: довгі сцени ріжемо на кілька планів
    captions: str = "word"             # word — слово підсвічується, коли звучить; phrase — фразами; none
    caption_y: float = 0.64            # центр субтитрів, частка висоти кадру
    caption_upper: bool = True
    caption_words: int = 3
    highlight: tuple = ACCENT
    hook: bool = True                  # великий текст хука в перші ~1,8 с
    top_text: bool = True              # плашка «На экране» зверху
    zoom: str = "punch"                # punch — різкий наїзд на склейці; slow — плавний; none
    music: str = "upbeat"              # upbeat | calm | none
    card_share: float = 0.0            # частка планів-типографіки без відео
    screen_share: float = 0.0          # частка планів «екран телефону/ноутбука»
    transitions: str = "mixed"         # mixed — по черзі свайп/спалах/глітч/зум; whip | flash | glitch | zoom | cut
    sfx: bool = True                   # «вжух» на переходах, удар на хуку, «поп» на перебивках
    cutaways: float = 0.5              # частка сцен з короткою вставкою-перебивкою поверх голосу
    source: str = "за замовчуванням"

    @classmethod
    def from_dna(cls, dna: dict | None) -> "Style":
        st = cls()
        if not isinstance(dna, dict) or not dna:
            return st
        st.source = "ДНК ролика-донора"
        m = dna.get("metrics") or {}
        v = dna.get("visual") or {}
        avg = m.get("avg_shot_s")
        if isinstance(avg, (int, float)) and avg > 0:
            st.shot_len = min(4.0, max(1.2, float(avg)))
        cap = v.get("captions") or {}
        st.captions = {"word_by_word": "word", "phrase": "phrase", "none": "none"}.get(cap.get("style"), st.captions)
        st.caption_y = {"top": 0.30, "middle": 0.52, "bottom": 0.70}.get(cap.get("position"), st.caption_y)
        if cap.get("case") in ("upper", "normal"):
            st.caption_upper = cap["case"] == "upper"
        if isinstance(cap.get("words_per_group"), (int, float)):
            st.caption_words = int(min(5, max(1, cap["words_per_group"])))
        st.highlight = _hex(cap.get("highlight_color")) or st.highlight
        ed = v.get("editing") or {}
        if isinstance(ed.get("zoom_punch_ins"), bool):
            st.zoom = "punch" if ed["zoom_punch_ins"] else "slow"
        st.transitions = {"hard_cut": "cut", "whip": "whip", "fade": "flash", "mixed": "mixed",
                          "glitch": "glitch", "zoom": "zoom"}.get(ed.get("transitions"), st.transitions)
        if isinstance(ed.get("sound_effects"), bool):
            st.sfx = ed["sound_effects"]
        st.cutaways = {"none": 0.0, "some": 0.4, "many": 0.8}.get(ed.get("cutaways"), st.cutaways)
        if isinstance(v.get("hook_text_on_screen"), bool):
            st.hook = v["hook_text_on_screen"]
        mus = v.get("music")
        if mus in ("none", "upbeat", "calm", "dramatic", "lofi"):
            st.music = {"dramatic": "upbeat", "lofi": "calm"}.get(mus, mus)
        mix = v.get("shot_types") or {}
        try:
            st.card_share = min(0.6, float(mix.get("text_card", 0)))
            st.screen_share = min(0.6, float(mix.get("screen_recording", 0)))
        except (TypeError, ValueError):
            pass
        return st

    def describe(self) -> str:
        cap = {"word": "субтитри по словах", "phrase": "субтитри фразами", "none": "без субтитрів"}[self.captions]
        tr = {"mixed": "мікс", "whip": "свайп", "flash": "спалах", "glitch": "глітч", "zoom": "зум", "cut": "жорсткі склейки"}
        return (f"монтаж: {self.source}, план ~{self.shot_len:.1f} с, {cap}, зум: {self.zoom}, "
                f"переходи: {tr.get(self.transitions, self.transitions)}{', звуки' if self.sfx else ''}, "
                f"музика: {'без музики' if self.music == 'none' else self.music}")


# ---------- голос із таймінгами слів ----------

def tts_words(text: str, voice: str, dest: Path) -> list[tuple[float, float, str]] | None:
    """edge-tts з межами слів: [(початок, кінець, слово)] — для субтитрів, що підсвічують слово, коли воно звучить."""
    try:
        import edge_tts as et

        async def go() -> list[tuple[float, float, str]]:
            try:
                comm = et.Communicate(text, voice, rate="+8%", boundary="WordBoundary")
            except TypeError:  # старі edge-tts віддають межі слів без параметра
                comm = et.Communicate(text, voice, rate="+8%")
            words = []
            with dest.open("wb") as f:
                async for ch in comm.stream():
                    if ch["type"] == "audio":
                        f.write(ch["data"])
                    elif ch["type"] == "WordBoundary":
                        words.append((ch["offset"] / 1e7, (ch["offset"] + ch["duration"]) / 1e7, ch["text"]))
            return words

        words = asyncio.run(go())
        return words if dest.exists() and dest.stat().st_size > 0 else None
    except Exception as e:  # noqa: BLE001
        print(f"  edge-tts: {safe_err(e)}", file=sys.stderr)
        return None


def estimate_words(text: str, dur: float) -> list[tuple[float, float, str]]:
    """Голос без таймінгів (інший TTS) — розкладаємо слова пропорційно довжині."""
    words = text.split()
    total = sum(len(w) + 1 for w in words) or 1
    out, t = [], 0.0
    for w in words:
        span = dur * (len(w) + 1) / total
        out.append((t, t + span * 0.9, w))
        t += span
    return out


# ---------- сток Pexels ----------

class Pexels:
    """Безкоштовні стокові відео (pexels.com/api): ключ у PEXELS_KEY, ліміт 200 запитів/год і 20 000/міс."""
    API = "https://api.pexels.com/videos/search"
    REACTIONS = ["surprised woman face", "man facepalm", "thumbs up closeup", "clock ticking closeup",
                 "money cash closeup", "phone notification closeup", "shocked man laptop", "high five office"]
    GENERIC = ["business laptop", "smartphone in hands", "office team meeting", "city night traffic",
               "typing keyboard closeup", "coffee shop owner", "online shopping phone", "notifications phone"]

    def __init__(self, key: str, locale: str):
        self.key = key
        self.ok = bool(key)
        self.locale = locale
        self.used: set[int] = set()
        self.cache: dict[str, list[dict]] = {}

    def search(self, query: str, local: bool = False) -> list[dict]:
        ck = f"{local}:{query}"
        if ck not in self.cache:
            params = {"query": query, "orientation": "portrait", "size": "medium", "per_page": 15}
            if local:
                params["locale"] = self.locale
            try:
                r = requests.get(self.API, headers={"Authorization": self.key}, params=params, timeout=30)
                self.cache[ck] = r.json().get("videos", []) if r.status_code == 200 else []
                if r.status_code != 200:
                    print(f"  pexels: HTTP {r.status_code}", file=sys.stderr)
            except (requests.RequestException, ValueError) as e:
                print(f"  pexels: {safe_err(e)}", file=sys.stderr)
                self.cache[ck] = []
        return self.cache[ck]

    @staticmethod
    def best_file(v: dict) -> dict | None:
        files = [f for f in v.get("video_files", []) if f.get("file_type") == "video/mp4" and f.get("width") and f.get("height")]
        portrait = [f for f in files if f["height"] >= f["width"]] or files
        good = sorted((f for f in portrait if f["width"] >= 700), key=lambda f: f["width"])
        return good[0] if good else (max(portrait, key=lambda f: f["width"]) if portrait else None)

    def clip(self, queries: list[str], local_query: str, dest: Path, min_dur: float, seed: int) -> bool:
        if not self.ok:
            return False
        tries = [(q, False) for q in queries] + ([(local_query, True)] if local_query else []) \
            + [(self.GENERIC[(seed + k) % len(self.GENERIC)], False) for k in range(2)]
        for q, local in tries:
            vids = [v for v in self.search(q, local) if v.get("id") not in self.used and (v.get("duration") or 0) >= min(min_dur, 4)]
            for v in vids[seed % 3:] + vids[:seed % 3]:  # трохи різноманіття між сценами
                f = self.best_file(v)
                if not f:
                    continue
                try:
                    with requests.get(f["link"], stream=True, timeout=120) as r:
                        if r.status_code != 200:
                            continue
                        size = 0
                        with dest.open("wb") as out:
                            for chunk in r.iter_content(1 << 16):
                                size += len(chunk)
                                if size > 80 * 1024 * 1024:
                                    break
                                out.write(chunk)
                    self.used.add(v["id"])
                    return dest.stat().st_size > 0
                except requests.RequestException as e:
                    print(f"  pexels: {safe_err(e)}", file=sys.stderr)
        return False


# ---------- плани (кадри) ----------

def brand_card(size: tuple[int, int], dest: Path, variant: int = 0) -> None:
    """Фон для планів-типографіки: темний градієнт з акцентною смугою; текст дають субтитри й плашки."""
    from PIL import ImageChops, ImageFilter
    W, H = size
    tops = [(14, 14, 18), (24, 16, 44), (8, 30, 40), (38, 14, 20)]
    glows = [(40, 90, 255), (120, 90, 255), (0, 200, 200), (255, 90, 60)]
    a, glow = tops[variant % len(tops)], glows[variant % len(glows)]
    img = Image.new("RGB", size, a)
    d = ImageDraw.Draw(img)
    for y in range(H):  # вертикальний градієнт до майже чорного
        k = y / H
        d.line([(0, y), (W, y)], fill=tuple(int(a[i] * (1 - 0.6 * k)) for i in range(3)))
    layer = Image.new("RGB", size, (0, 0, 0))
    ld = ImageDraw.Draw(layer)
    cx, cy = (W * (0.25 + 0.5 * (variant % 2)), H * (0.3 + 0.15 * (variant % 3)))
    r = W * 0.55
    ld.ellipse([cx - r, cy - r, cx + r, cy + r], fill=tuple(int(c * 0.4) for c in glow))
    layer = layer.filter(ImageFilter.GaussianBlur(W * 0.18))
    img = ImageChops.add(img, layer)
    d = ImageDraw.Draw(img, "RGBA")
    step = int(W / 9)
    for x in range(0, W, step):
        d.line([(x, 0), (x, H)], fill=(255, 255, 255, 10))
    for y in range(0, H, step):
        d.line([(0, y), (W, y)], fill=(255, 255, 255, 10))
    d.polygon([(W * 0.55, 0), (W * 0.75, 0), (W * 0.15, H), (-W * 0.05, H)], fill=(255, 255, 255, 8))
    img.save(dest)


def media_length(path: Path) -> float:
    return duration_of(path) or 0.0


def render_shot(src: Path, is_video: bool, frames: int, size: tuple[int, int], dest: Path, zoom: str, n: int,
                offset: float = 0.0, intro: str = "", loop: bool = True) -> None:
    """Один план рівно на `frames` кадрів: кроп під 9:16, рух камери і, на початку сцени, перехід
    (whip — свайп з розмиттям, flash — спалах, glitch — RGB-зсув і шум, zoom — ривок зуму)."""
    W, H = size
    k = 1.15 if is_video else 1.6  # запас по розміру, щоб zoompan не «тремтів»
    pw, ph = int(W * k) // 2 * 2, int(H * k) // 2 * 2
    if zoom == "punch":
        z = (f"if(lt(on,6),1.18-0.18*on/6,1+0.05*(on-6)/{frames})" if n % 2 == 0
             else f"1.08-0.07*on/{frames}")
    elif zoom == "slow":
        z = f"1+0.08*on/{frames}"
    else:
        z = "1"
    x = "iw/2-(iw/zoom/2)"
    if intro == "whip":  # кадр «влітає» справа: збільшення дає запас для зсуву
        z = f"if(lt(on,6),1.35,{z})"
        x = f"if(lt(on,6),(iw-iw/zoom)/2*(1+(6-on)/6),{x})"
    elif intro == "zoom":
        z = f"if(lt(on,7),1.6-0.6*on/7,{z})"
    inp = ((["-stream_loop", "-1"] if loop else []) + ["-ss", f"{offset:.3f}", "-i", str(src)] if is_video
           else ["-loop", "1", "-framerate", str(FPS), "-i", str(src)])
    hold = "" if loop or not is_video else "tpad=stop_mode=clone:stop_duration=3,"  # кінець кліпу — тримаємо кадр
    vf = (f"{hold}scale={pw}:{ph}:force_original_aspect_ratio=increase,crop={pw}:{ph},fps={FPS},"
          f"zoompan=z='{z}':x='{x}':y='ih/2-(ih/zoom/2)':d=1:s={W}x{H}:fps={FPS},setsar=1")
    vf += {"whip": ",gblur=sigma=26:steps=2:enable='lt(n,5)'",
           "flash": ",fade=t=in:st=0:d=0.2:color=white",
           "glitch": ",rgbashift=rh=-16:bh=16:enable='lt(t,0.2)',noise=alls=45:allf=t:enable='lt(t,0.2)'",
           "zoom": ",gblur=sigma=12:steps=1:enable='lt(n,4)'"}.get(intro, "")
    vf += ",format=yuv420p"
    run([ffmpeg(), "-y", *inp, "-vf", vf, "-an", "-frames:v", str(frames), "-r", str(FPS),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-g", str(FPS), str(dest)])


# ---------- музика ----------

def music_track(style: Style, dur: float, dest: Path, seed: int) -> str:
    """Фонова музика: свій трек з MUSIC_DIR або music/ (mp3/wav/m4a), інакше згенерований біт без авторських прав."""
    if style.music == "none":
        return ""
    dirs = [Path(os.environ["MUSIC_DIR"])] if os.environ.get("MUSIC_DIR") else []
    dirs.append(Path(__file__).with_name("music"))
    tracks = sorted(p for d in dirs if d.is_dir() for p in d.iterdir() if p.suffix.lower() in (".mp3", ".wav", ".m4a", ".ogg"))
    fade = f"afade=t=in:d=0.6,afade=t=out:st={max(0.0, dur - 1.5):.2f}:d=1.5"
    if tracks:
        t = tracks[seed % len(tracks)]
        run([ffmpeg(), "-y", "-stream_loop", "-1", "-i", str(t), "-t", f"{dur:.2f}", "-af", f"{fade},aresample=44100",
             "-ac", "2", str(dest)])
        return f"трек {t.stem}"
    bpm = 118 if style.music == "upbeat" else 84
    B = 60 / bpm
    bar = 4 * B
    # Am – F – C – G: тризвуки (Гц) і бас на октаву нижче.
    chords = [(220.0, 261.63, 329.63), (174.61, 220.0, 261.63), (196.0, 261.63, 329.63), (196.0, 246.94, 293.66)]
    i = f"mod(floor(t/{bar:.4f}),4)"
    sel = lambda j: "if(eq({i},0),{a},if(eq({i},1),{b},if(eq({i},2),{c},{d})))".format(  # noqa: E731
        i=i, a=chords[0][j], b=chords[1][j], c=chords[2][j], d=chords[3][j])
    pump = f"(1-0.55*exp(-7*mod(t,{B:.4f})))"
    pad = f"0.09*(sin(2*PI*{sel(0)}*t)+sin(2*PI*{sel(1)}*t)+0.8*sin(2*PI*{sel(2)}*t))*{pump}"
    bass = f"0.22*sin(2*PI*{sel(0)}/2*t)*{pump}"
    if style.music == "upbeat":
        kick = f"0.55*sin(2*PI*(48+95*exp(-28*mod(t,{B:.4f})))*mod(t,{B:.4f}))*exp(-10*mod(t,{B:.4f}))"
        hat = f"0.05*(2*random(0)-1)*exp(-80*mod(t+{B / 2:.4f},{B:.4f}))"
        expr = f"{pad}+{bass}+{kick}+{hat}"
    else:
        expr = f"{pad}+{bass}"
    run([ffmpeg(), "-y", "-f", "lavfi", "-i", f"aevalsrc=exprs='{expr}':s=44100:d={dur:.2f}",
         "-af", f"lowpass=f=3200,aecho=0.6:0.4:90:0.18,{fade},volume=0.9", "-ac", "2", str(dest)])
    return f"згенерований біт {bpm} bpm"


SFX = {  # синтезовані звуки без авторських прав: (тривалість, вираз aevalsrc, фільтри)
    "whoosh": (0.5, "(2*random(0)-1)*pow(sin(PI*t/0.5),2)", "highpass=f=500,lowpass=f=5200,aphaser=type=t:speed=2,volume=0.9"),
    "impact": (0.8, "0.9*sin(2*PI*(55+120*exp(-8*t))*t)*exp(-4*t)+0.4*(2*random(0)-1)*exp(-25*t)", "lowpass=f=2500"),
    "pop": (0.14, "0.8*sin(2*PI*(1100-3500*t)*t)*exp(-28*t)", "highpass=f=200"),
}


def sfx_track(events: list[tuple[float, str]], total: float, work: Path) -> Path | None:
    """Звукові ефекти на таймлайні: [(секунда, назва)] → один WAV на всю довжину ролика."""
    if not events:
        return None
    for name, (d, expr, filt) in SFX.items():
        run([ffmpeg(), "-y", "-f", "lavfi", "-i", f"aevalsrc=exprs='{expr}':s=44100:d={d}", "-af", filt, "-ac", "2",
             str(work / f"sfx_{name}.wav")])
    events = sorted((max(0.0, a), n) for a, n in events)[:40]
    inputs, chain = [], []
    for k, (a, n) in enumerate(events):
        inputs += ["-i", str(work / f"sfx_{n}.wav")]
        ms = int(a * 1000)
        chain.append(f"[{k}:a]adelay={ms}|{ms}[s{k}]")
    chain.append("".join(f"[s{k}]" for k in range(len(events))) + f"amix=inputs={len(events)}:normalize=0,"
                 f"apad,atrim=0:{total:.3f}[a]")
    out = work / "sfx.wav"
    run([ffmpeg(), "-y", *inputs, "-filter_complex", ";".join(chain), "-map", "[a]", "-ac", "2", "-ar", "44100", str(out)])
    return out


# ---------- субтитри, хук і плашки (ASS) ----------

def ass_color(rgb: tuple, alpha: int = 0) -> str:
    return f"&H{alpha:02X}{rgb[2]:02X}{rgb[1]:02X}{rgb[0]:02X}"


def ass_time(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def ass_text(s: str) -> str:
    return s.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ").strip()


def caption_groups(words: list[tuple[float, float, str]], k: int) -> list[list[tuple[float, float, str]]]:
    groups, cur = [], []
    for w in words:
        if cur and (len(cur) >= k or w[0] - cur[-1][1] > 0.35 or sum(len(x[2]) for x in cur) + len(w[2]) > 18):
            groups.append(cur)
            cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    return groups


def build_ass(scenes: list[Scene], offsets: list[float], total: float, size: tuple[int, int], style: Style,
              font_name: str, dest: Path, hook_y: float = 0.40) -> None:
    W, H = size
    cap_size, top_size, hook_size = int(W * 0.082), int(W * 0.052), int(W * 0.098)
    ol = max(3, int(W * 0.007))
    hl = ass_color(style.highlight)
    top_y = 8 if style.caption_y > 0.45 else 2  # плашка зверху, якщо субтитри не зверху
    head = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 0\nScaledBorderAndShadow: yes\n"
        f"PlayResX: {W}\nPlayResY: {H}\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, "
        "MarginV, Encoding\n"
        f"Style: Cap,{font_name},{cap_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,1,0,1,{ol},2,5,"
        f"{int(W * 0.07)},{int(W * 0.07)},0,1\n"
        f"Style: Top,{font_name},{top_size},{ass_color(ACCENT)},&H00FFFFFF,&H50000000,&H50000000,0,0,0,0,100,100,0,0,3,"
        f"{int(W * 0.02)},0,{top_y},{int(W * 0.08)},{int(W * 0.08)},{int(H * (0.11 if top_y == 8 else 0.2))},1\n"
        f"Style: Hook,{font_name},{hook_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,0,0,0,0,100,100,1,0,1,"
        f"{ol + 2},3,5,{int(W * 0.06)},{int(W * 0.06)},0,1\n"
        f"Style: Big,{font_name},{int(W * 0.085)},{ass_color(ACCENT)},&H00FFFFFF,&H00000000,&H90000000,0,0,0,0,100,100,1,0,1,"
        f"{ol + 1},3,5,{int(W * 0.08)},{int(W * 0.08)},0,1\n"
        f"Style: Mark,{font_name},{int(W * 0.03)},&H70FFFFFF,&H00FFFFFF,&H90000000,&H00000000,0,0,0,0,100,100,1,0,1,1,0,9,"
        f"{int(W * 0.05)},{int(W * 0.05)},{int(H * 0.035)},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    ev: list[str] = []
    add = lambda layer, a, b, st, txt: ev.append(f"Dialogue: {layer},{ass_time(a)},{ass_time(b)},{st},,0,0,0,,{txt}")  # noqa: E731
    up = (lambda s: s.upper()) if style.caption_upper else (lambda s: s)

    hook_end = 0.0
    if style.hook and scenes:
        hook = scenes[0].on_screen or " ".join(scenes[0].voice.split()[:6])
        if hook:
            hook_end = min(1.9, scenes[0].dur)
            add(3, 0, hook_end, "Hook", f"{{\\pos({W // 2},{int(H * hook_y)})\\fscx135\\fscy135\\t(0,180,\\fscx100\\fscy100)"
                f"\\c{hl}}}{ass_text(up(hook))}")
    for s, off in zip(scenes, offsets):
        if style.top_text and s.on_screen:
            a = max(off, hook_end) if s is scenes[0] else off
            if a < off + s.dur - 0.3 and s.clip_kind == "card":
                y = int(H * (0.36 if style.caption_y > 0.45 else 0.62))
                add(1, a, off + s.dur, "Big", f"{{\\pos({W // 2},{y})\\fscx90\\fscy90\\t(0,140,\\fscx100\\fscy100)}}"
                    + ass_text(s.on_screen))
            elif a < off + s.dur - 0.3:
                add(1, a, off + s.dur, "Top", ass_text(s.on_screen))
        if style.captions == "none" or not s.words:
            continue
        pos = f"\\pos({W // 2},{int(H * style.caption_y)})"
        groups = caption_groups(s.words, style.caption_words)
        for gi, g in enumerate(groups):
            nxt = groups[gi + 1][0][0] if gi + 1 < len(groups) else None
            g_end = nxt if nxt is not None and nxt - g[-1][1] < 0.7 else min(g[-1][1] + 0.3, s.dur)
            texts = [ass_text(up(w[2])) for w in g]
            if style.captions == "phrase":
                if off + g_end > hook_end + 0.2:  # поки на екрані хук, субтитри не дублюють його
                    add(2, max(off + g[0][0], hook_end), off + g_end, "Cap", f"{{{pos}}}" + " ".join(texts))
                continue
            for j, w in enumerate(g):
                a = off + w[0]
                b = off + (g[j + 1][0] if j + 1 < len(g) else g_end)
                if b <= a or (hook_end and b <= hook_end):
                    continue
                pop = "\\fscx88\\fscy88\\t(0,90,\\fscx100\\fscy100)" if j == 0 else ""
                line = " ".join(f"{{\\c{hl}\\fscx106\\fscy106}}{t}{{\\r}}" if q == j else t for q, t in enumerate(texts))
                add(2, max(a, hook_end), b, "Cap", f"{{{pos}{pop}}}{line}")
    add(0, 0, total, "Mark", "Leeloo.ai")
    dest.write_text(head + "\n".join(ev) + "\n", encoding="utf-8")


def caption_font(font: str | None) -> Path:
    """Жирний шрифт з кирилицею для субтитрів: Montserrat (якщо встановлений), Arial Black, DejaVu Sans Bold."""
    if font:
        return Path(font)
    roots = [Path("/usr/share/fonts"), Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts", Path("/Library/Fonts")]
    for pat in ("Montserrat-Black.*", "Montserrat-ExtraBold.*", "ariblk.ttf", "Arial Black.ttf", "DejaVuSans-Bold.ttf"):
        for r in roots:
            if r.is_dir():
                hit = next(iter(sorted(r.rglob(pat))), None)
                if hit:
                    return hit
    return Path(find_font(True))


# ---------- аватар HeyGen ----------

class HeyGen:
    """Аватар і голос HeyGen (API v3). Налаштування приходять із черги дашборду: key, avatar_id, voice_id, engine, mode.

    Голос: якщо задано voice_id (напр. клон голосу основателя) — озвучка HeyGen з таймінгами слів, інакше edge-tts.
    Відео: для кожної сцени з аватаром завантажуємо рівно її озвучку й просимо HeyGen синхронізувати губи —
    платимо лише за секунди, де аватар у кадрі, а звук у ролику збігається з рухом губ."""
    API = "https://api.heygen.com"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.h = {"x-api-key": cfg["key"], "User-Agent": "insta-viral-render/1.0"}

    def speech(self, text: str, lang: str, dest: Path) -> list[tuple[float, float, str]] | None:
        if not self.cfg.get("voice_id"):
            return None
        try:
            r = requests.post(f"{self.API}/v3/voices/speech", headers=self.h, timeout=120,
                              json={"text": text[:5000], "voice_id": self.cfg["voice_id"], "language": lang})
            if r.status_code != 200:
                print(f"  heygen speech: HTTP {r.status_code}", file=sys.stderr)
                return None
            d = r.json()["data"]
            dest.write_bytes(requests.get(d["audio_url"], timeout=120).content)
            ts = d.get("word_timestamps") or []
            words = [(float(w["start"]), float(w["end"]), str(w["word"])) for w in ts if w.get("word")]
            return words or estimate_words(text, float(d.get("duration") or media_length(dest)))
        except Exception as e:  # noqa: BLE001
            print(f"  heygen speech: {safe_err(e)}", file=sys.stderr)
            return None

    def start(self, audio: Path) -> str | None:
        """Завантажує озвучку сцени й ставить у HeyGen відео аватара під неї. Повертає video_id."""
        try:
            with audio.open("rb") as f:
                r = requests.post(f"{self.API}/v3/assets", headers=self.h, timeout=120,
                                  files={"file": (audio.name, f, "audio/mpeg")})
            if r.status_code != 200:
                print(f"  heygen upload: HTTP {r.status_code}", file=sys.stderr)
                return None
            asset = r.json()["data"]["asset_id"]
            body = {"type": "avatar", "avatar_id": self.cfg["avatar_id"], "audio_asset_id": asset,
                    "aspect_ratio": "9:16", "resolution": "720p", "fit": "cover", "title": "Leeloo render"}
            if self.cfg.get("engine"):
                body["engine"] = {"type": self.cfg["engine"]}
            r = requests.post(f"{self.API}/v3/videos", headers=self.h, timeout=60, json=body)
            if r.status_code != 200:
                err = (r.json().get("error") or {}) if r.headers.get("Content-Type", "").startswith("application/json") else {}
                print(f"  heygen video: HTTP {r.status_code} {err.get('code', '')}", file=sys.stderr)
                return None
            return r.json()["data"]["video_id"]
        except Exception as e:  # noqa: BLE001
            print(f"  heygen video: {safe_err(e)}", file=sys.stderr)
            return None

    def wait(self, jobs: dict[int, str], work: Path, report, timeout: float = 1800) -> dict[int, Path]:
        """Чекає готовності всіх відео й качає їх. Повертає {номер сцени: mp4}."""
        done: dict[int, Path] = {}
        pending = dict(jobs)
        t0 = time.time()
        while pending and time.time() - t0 < timeout:
            for i, vid in list(pending.items()):
                try:
                    r = requests.get(f"{self.API}/v3/videos/{vid}", headers=self.h, timeout=30)
                    d = r.json().get("data") or {} if r.status_code == 200 else {}
                except (requests.RequestException, ValueError):
                    continue
                if d.get("status") == "completed" and d.get("video_url"):
                    dest = work / f"avatar_{i:02d}.mp4"
                    dest.write_bytes(requests.get(d["video_url"], timeout=300).content)
                    done[i] = dest
                    del pending[i]
                elif d.get("status") == "failed":
                    print(f"  heygen: сцена {i + 1} — {d.get('failure_code') or 'failed'}", file=sys.stderr)
                    del pending[i]
            if pending:
                report(22 + int(20 * len(done) / max(1, len(jobs))), f"Аватар HeyGen: {len(done)} з {len(jobs)} сцен")
                time.sleep(10)
        for i in pending:
            print(f"  heygen: сцена {i + 1} не встигла за {int(timeout / 60)} хв — буде сток", file=sys.stderr)
        return done


def avatar_scenes(scenes: list[Scene], cfg: dict | None, style: Style, dna: dict | None) -> set[int]:
    """Де в кадрі аватар: auto — хук, заклик і сцени [face] (або кожна третя, якщо в донора багато «людини в кадрі»)."""
    if not cfg or not scenes:
        return set()
    face = {i for i, s in enumerate(scenes) if s.tag == "face"}
    if cfg.get("mode") == "face":
        return face
    talking = float(((dna or {}).get("visual") or {}).get("shot_types", {}).get("talking_head", 0) or 0)
    extra = {i for i in range(len(scenes)) if i % 3 == 0} if talking >= 0.3 and not face else set()
    return face | extra | {0, len(scenes) - 1}


# ---------- збірка ----------

def produce(md: str, out_path: Path, chat: object | None, media: object, size: tuple[int, int], font: str | None,
            on_progress=None, dna: dict | None = None, avatar: dict | None = None) -> dict:
    """on_progress(відсоток, етап) — для смуги прогресу в дашборді: озвучка до 20 %, аватар до 42 %, плани до 85 %,
    звук і склейка до 95 %. avatar — налаштування HeyGen із черги (None — без аватара)."""
    report = on_progress or (lambda pct, stage: None)
    started = time.time()
    scenes = parse_scenes(md)
    if not scenes:
        raise RuntimeError("У сценарії не знайдено таблицю «Время | На экране | Говорит | B-roll».")
    lang = detect_language(md)
    voice = VOICES.get(lang, VOICES["uk"])
    style = Style.from_dna(dna)
    pexels = Pexels(os.environ.get("PEXELS_KEY", ""), {"uk": "uk-UA", "ru": "ru-RU"}.get(lang, "en-US"))
    W, H = size
    report(2, "Планую кадри")
    visual_plan(chat, scenes, lang)
    font_path = caption_font(font)
    fam, sub = ImageFont.truetype(str(font_path), 20).getname()
    font_name = fam if (sub or "Regular") in ("Regular", "Bold", "Normal") or sub in fam else f"{fam} {sub}"
    kinds: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="leeloo-video-") as tmp:
        work = Path(tmp)
        (work / "fonts").mkdir()
        (work / "fonts" / font_path.name).write_bytes(font_path.read_bytes())
        shots: list[Path] = []
        offsets: list[float] = []
        t = 0.0
        shot_no = 0
        sfx_events: list[tuple[float, str]] = []
        heygen = HeyGen(avatar) if avatar else None
        # Прохід 1: озвучка й тривалість кожної сцени.
        for i, s in enumerate(scenes):
            report(3 + int(17 * i / len(scenes)), f"Озвучка: сцена {i + 1} з {len(scenes)}")
            if s.voice:
                a = work / f"voice_{i:02d}.mp3"
                words = heygen.speech(s.voice, lang, a) if heygen else None
                if words is not None:
                    s.voice_engine, s.audio, s.words = "heygen", a, words
                else:
                    words = tts_words(s.voice, voice, a)
                    if words is not None:
                        s.voice_engine, s.audio, s.words = "edge-tts:" + voice, a, words
                    elif media.speech(s.voice, a):
                        s.voice_engine, s.audio = "tts", a
                        s.words = estimate_words(s.voice, media_length(a))
            vlen = media_length(s.audio) if s.audio else 0.0
            # Темп віральних роликів: сцена триває рівно стільки, скільки звучить голос.
            want = vlen + 0.2 if vlen else max(s.end - s.start, 1.5)
            n = max(1, round(want / style.shot_len))
            frames_total = max(int(round(want * FPS)), n * 12)
            s.per = [frames_total // n + (1 if k < frames_total % n else 0) for k in range(n)]
            # Перебивка: другий план сцени стає вставкою ~0,7 с (реакція, деталь); решту кадрів віддаємо першому.
            s.cut_at = -1
            if pexels.ok and style.cutaways and n >= 2 and int((i + 1) * style.cutaways) > int(i * style.cutaways) \
                    and s.per[1] > 26:
                s.cut_at, short = 1, 21
                s.per[0] += s.per[1] - short
                s.per[1] = short
            s.dur = frames_total / FPS
            offsets.append(t)
            t += s.dur

        # Аватар: відео HeyGen під озвучку обраних сцен (паралельно), поки чекаємо — нічого не рендеримо.
        avatar_clips: dict[int, Path] = {}
        want_avatar = avatar_scenes(scenes, avatar, style, dna)
        if heygen and want_avatar:
            report(21, f"Аватар HeyGen: {len(want_avatar)} сцен")
            jobs = {i: vid for i in sorted(want_avatar) if scenes[i].audio
                    for vid in [heygen.start(scenes[i].audio)] if vid}
            avatar_clips = heygen.wait(jobs, work, report)
            print(f"  аватар: {len(avatar_clips)} з {len(want_avatar)} сцен")

        # Прохід 2: плани.
        for i, s in enumerate(scenes):
            print(f"  сцена {i + 1}/{len(scenes)}: {safe(s.on_screen[:50] or s.voice[:50])}")
            report(43 + int(42 * i / len(scenes)), f"Сцена {i + 1} з {len(scenes)}")
            per, cut_at, start = s.per, s.cut_at, offsets[i]
            # Візуал: кілька планів на сцену, джерела за тегом сцени, ДНК і доступністю.
            clips: list[tuple[Path, bool, str]] = []
            for j, fr in enumerate(per):
                kind = s.tag or "stock"
                intro = ""
                if j == 0 and i > 0 and style.transitions != "cut":
                    intro = (["whip", "flash", "glitch", "zoom"][i % 4] if style.transitions == "mixed" else style.transitions)
                    sfx_events.append((start - 0.15, "whoosh"))
                if j == cut_at:
                    p = work / f"cut_{i:02d}.mp4"
                    rx = [Pexels.REACTIONS[(shot_no + k) % len(Pexels.REACTIONS)] for k in range(2)]
                    if pexels.clip(([s.cutaway] if s.cutaway else []) + rx, "", p, fr / FPS, seed=shot_no + 7):
                        dest = work / f"shot_{shot_no:03d}.mp4"
                        render_shot(p, True, fr, size, dest, "punch", 0, 0.5, intro="glitch" if i % 2 else "flash")
                        shots.append(dest)
                        kinds["cutaway"] = kinds.get("cutaway", 0) + 1
                        sfx_events.append((start + sum(per[:j]) / FPS, "pop"))
                        shot_no += 1
                        continue
                if i in avatar_clips:
                    # Аватар: той самий фрагмент озвучки, що звучить у цей момент, — губи збігаються з голосом.
                    dest = work / f"shot_{shot_no:03d}.mp4"
                    render_shot(avatar_clips[i], True, fr, size, dest, style.zoom, j, sum(per[:j]) / FPS,
                                intro=intro, loop=False)
                    shots.append(dest)
                    kinds["avatar"] = kinds.get("avatar", 0) + 1
                    clips.append((avatar_clips[i], True, "avatar"))
                    shot_no += 1
                    continue
                if not s.tag and style.card_share and (shot_no % max(2, round(1 / style.card_share))) == 1:
                    kind = "text"
                src, is_video, label = None, False, ""
                if kind in ("stock", "screen", "face"):
                    reuse = [c for c in clips if c[1]]
                    if j >= 2 and reuse:  # третій і далі план сцени — інший фрагмент уже скачаного кліпу
                        src, is_video, label = reuse[j % len(reuse)]
                    else:
                        qs = list(s.queries)
                        if kind == "screen" or (style.screen_share and shot_no % max(2, round(1 / style.screen_share)) == 0):
                            qs = [f"{q} phone screen" for q in qs[:1]] + ["smartphone app screen", "laptop dashboard screen"] + qs
                        p = work / f"stock_{i:02d}_{j}.mp4"
                        if pexels.clip(qs, re.sub(r"\[[^]]*\]", "", s.broll)[:60], p, fr / FPS, seed=shot_no):
                            src, is_video, label = p, True, "stock"
                            clips.append((p, True, "stock"))
                if src is None and kind != "text":
                    p = work / f"ai_{i:02d}_{j}.mp4"
                    got = media.video(s.prompt, fr / FPS, p) if j == 0 else None
                    if got:
                        src, is_video, label = p, True, "ai-video"
                    else:
                        p = work / f"ai_{i:02d}_{j}.png"
                        if j == 0 and media.image(s.prompt, p):
                            src, is_video, label = p, False, "ai-image"
                            clips.append((p, False, "ai-image"))
                        elif clips:
                            src, is_video, label = clips[j % len(clips)]
                if src is None:
                    src = work / f"card_{i:02d}_{j}.png"
                    brand_card(size, src, shot_no)
                    is_video, label = False, "card"
                offset = 0.0
                if is_video:
                    ln = media_length(src)
                    offset = ((shot_no * 2.7) % max(0.1, ln - fr / FPS)) if ln > fr / FPS + 0.5 else 0.0
                dest = work / f"shot_{shot_no:03d}.mp4"
                render_shot(src, is_video, fr, size, dest, style.zoom, shot_no, offset, intro=intro)
                shots.append(dest)
                kinds[label] = kinds.get(label, 0) + 1
                shot_no += 1
            s.clip_kind = ",".join(sorted({c[2] for c in clips})) or "card"
        total = t

        report(86, "Зводжу звук і музику")
        voice_parts = []
        for i, s in enumerate(scenes):
            part = work / f"a_{i:02d}.wav"
            src = ["-i", str(s.audio)] if s.audio else ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]
            run([ffmpeg(), "-y", *src, "-af", f"aresample=44100,apad,atrim=0:{s.dur:.3f}", "-ac", "2", "-ar", "44100",
                 "-c:a", "pcm_s16le", str(part)])
            voice_parts.append(part)
        (work / "a.txt").write_text("".join(f"file '{p.as_posix()}'\n" for p in voice_parts), encoding="utf-8")
        run([ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", str(work / "a.txt"), "-c", "copy", str(work / "voice.wav")])
        music = work / "music.wav"
        music_label = music_track(style, total, music, seed=len(md))
        mix = work / "mix.m4a"
        if style.sfx and style.hook:
            sfx_events.append((0.0, "impact"))
        sfx = sfx_track(sfx_events, total, work) if style.sfx else None
        inputs, chain, parts = ["-i", str(work / "voice.wav")], [], []
        if music_label:
            inputs += ["-i", str(music)]
            chain += ["[0:a]asplit=2[v1][v2]", "[1:a]volume=0.32[m]",
                      "[m][v1]sidechaincompress=threshold=0.02:ratio=10:attack=10:release=380[md]"]
            parts += ["[v2]", "[md]"]
        else:
            parts.append("[0:a]")
        if sfx:
            inputs += ["-i", str(sfx)]
            chain.append(f"[{len(inputs) // 2 - 1}:a]volume=0.5[fx]")
            parts.append("[fx]")
        chain.append("".join(parts) + f"amix=inputs={len(parts)}:duration=first:normalize=0,alimiter=limit=0.95[a]")
        run([ffmpeg(), "-y", *inputs, "-filter_complex", ";".join(chain), "-map", "[a]", "-c:a", "aac", "-b:a", "160k", str(mix)])

        report(90, "Склеюю відео й субтитри")
        (work / "v.txt").write_text("".join(f"file '{p.as_posix()}'\n" for p in shots), encoding="utf-8")
        run([ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", str(work / "v.txt"), "-c", "copy", str(work / "visual.mp4")])
        # Якщо хук говорить аватар — текст хука вище, щоб не закривати обличчя.
        build_ass(scenes, offsets, total, size, style, font_name, work / "captions.ass",
                  hook_y=0.2 if 0 in avatar_clips else 0.40)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        run([ffmpeg(), "-y", "-i", "visual.mp4", "-i", "mix.m4a", "-vf", "ass=captions.ass:fontsdir=fonts",
             "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
             "-maxrate", "3500k", "-bufsize", "7000k", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest",
             "-movflags", "+faststart", str(out_path.resolve())], cwd=work)
    info = {
        "duration_s": round(total, 1), "size": f"{W}x{H}", "language": lang,
        "media": media.name if media.ok else "", "prompts": type(chat).__name__ if chat else "",
        "stock": "pexels" if pexels.ok else "", "shots": kinds, "music": music_label, "style": style.describe(),
        "dna": bool(dna), "font": font_name, "avatar": bool(avatar_clips),
        "scenes": [{"on_screen": s.on_screen[:80], "visual": s.clip_kind, "voice": s.voice_engine, "dur": round(s.dur, 2)}
                   for s in scenes],
    }
    info["render_sec"] = round(time.time() - started, 1)
    info["summary"] = summarize(info)
    return info


def summarize(info: dict) -> str:
    sh = info.get("shots", {})
    names = {"stock": "сток", "ai-video": "AI-кліпи", "ai-image": "AI-картинки", "card": "типографіка", "cutaway": "перебивки",
             "avatar": "аватар"}
    parts = ", ".join(f"{names.get(k, k)} {v}" for k, v in sh.items())
    return (f"{len(info['scenes'])} сцен, {sum(sh.values())} планів ({parts}), {info['duration_s']} с · {info['style']}"
            + (f" · {info['music']}" if info.get("music") else "")
            + ("" if info.get("stock") else " · без PEXELS_KEY — стоку немає"))


# ---------- дашборд ----------

class Dashboard:
    def __init__(self, url: str, token: str):
        self.base = url.rstrip("/") + "/api/index.php"
        # Фаєрвол хостингу (Fornex) відповідає 447 на стандартний User-Agent python-requests.
        self.h = {"X-Api-Token": token, "User-Agent": "insta-viral-render/1.0"}

    def _check(self, r: requests.Response) -> dict:
        try:
            data = r.json()
        except ValueError:
            raise RuntimeError(f"Дашборд відповів {r.status_code} не JSON")
        if not data.get("ok"):
            raise RuntimeError(data.get("error") or f"HTTP {r.status_code}")
        return data

    def queue(self) -> list[dict]:
        return self._check(requests.get(self.base, params={"action": "video_queue"}, headers=self.h, timeout=30))["items"]

    def status(self, sid: int, status: str, error: str = "", progress: int | None = None, stage: str = "") -> None:
        body = {"id": sid, "video_status": status, "error": error}
        if progress is not None:
            body.update(progress=progress, stage=stage)
        self._check(requests.post(self.base, params={"action": "video_status"}, headers=self.h, timeout=30, json=body))

    def progress(self, sid: int, pct: int, stage: str) -> None:
        """Прогрес не критичний: якщо хостинг не відповів, рендер триває."""
        try:
            self.status(sid, "rendering", progress=pct, stage=stage)
        except Exception as e:  # noqa: BLE001
            print(f"  (прогрес не надіслано: {e})", file=sys.stderr)

    def upload(self, sid: int, path: Path, info: dict) -> None:
        with path.open("rb") as f:
            self._check(requests.post(self.base, params={"action": "video_upload"}, headers=self.h, timeout=600,
                                      data={"id": str(sid), "info": json.dumps(info, ensure_ascii=False)},
                                      files={"file": (path.name, f, "video/mp4")}))


def process_queue(dash: Dashboard, chat: object | None, media: object, out_dir: Path, size: tuple[int, int], font: str | None) -> int:
    items = dash.queue()
    for it in items:
        sid = it["id"]
        print(f"Сценарій #{sid}: {safe(it.get('title') or it.get('hook_text') or '')}")
        if it.get("avatar_note"):
            print(f"  аватар: {it['avatar_note']}")
        try:
            dash.status(sid, "rendering", progress=0, stage="Старт рендера")
            path = out_dir / f"script-{sid}.mp4"
            info = produce(it["content"], path, chat, media, size, font, lambda pct, stage: dash.progress(sid, pct, stage),
                           dna=it.get("format_dna"), avatar=it.get("avatar"))
            dash.progress(sid, 96, "Завантажую на дашборд")
            dash.upload(sid, path, info)
            print(f"  готово: {info['summary']}")
        except Exception as e:  # noqa: BLE001
            print(f"  помилка: {safe_err(e)} (повний текст — у дашборді)", file=sys.stderr)
            try:
                dash.status(sid, "error", str(e)[:900])
            except Exception:  # noqa: BLE001
                pass
    return len(items)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--watch", action="store_true", help="постійно брати чергу з дашборду")
    mode.add_argument("--once", action="store_true", help="обробити чергу один раз")
    mode.add_argument("--file", help="локальний тест: шлях до .md зі сценарієм")
    ap.add_argument("--out", default="videos", help="куди класти готові відео")
    ap.add_argument("--size", default="720x1280", help="розмір кадру, напр. 1080x1920")
    ap.add_argument("--no-video-model", action="store_true", help="не генерувати кліпи, лише картинки/картки (швидше)")
    ap.add_argument("--font", help="шлях до .ttf з кирилицею")
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--dna", help="локальний тест: JSON з ДНК формату ролика-донора")
    ap.add_argument("--probe-llm", action="store_true", help="перевірити текстову модель тестовим запитом і вийти")
    args = ap.parse_args()

    if args.probe_llm:  # нейтральний запит — відповідь можна показувати навіть у публічному лозі
        chat, _ = pick_providers(use_video=False)
        if isinstance(chat, OpenAIChat):
            r = requests.post(f"{chat.base}/chat/completions", timeout=60,
                              headers={"Authorization": f"Bearer {chat.key}", "Content-Type": "application/json"},
                              json={"model": chat.model, "max_tokens": 5, "messages": [{"role": "user", "content": "Say hi"}]})
            print(f"LLM {chat.base} → HTTP {r.status_code} {r.headers.get('Content-Type')} · {' '.join(r.text[:120].split())}")
        else:
            print(f"LLM: {type(chat).__name__ if chat else 'немає'}")
        return

    W, H = (int(x) for x in args.size.lower().split("x"))
    chat, media = pick_providers(use_video=not args.no_video_model)
    print(f"Медіа: {media.name if media.ok else 'недоступні — будуть фірмові картки'} · промпти: "
          f"{type(chat).__name__ if chat else 'без LLM (опис сцени як є)'} · голос: edge-tts")
    out_dir = Path(args.out)

    if args.file:
        md = Path(args.file).read_text(encoding="utf-8")
        path = out_dir / (Path(args.file).stem + ".mp4")
        dna = json.loads(Path(args.dna).read_text(encoding="utf-8")) if args.dna else None
        info = produce(md, path, chat, media, (W, H), args.font, dna=dna)
        print(f"Готово: {path} · {info['summary']}")
        return

    url, token = os.environ.get("DASHBOARD_URL"), os.environ.get("DASHBOARD_TOKEN")
    if not url or not token:
        sys.exit("Потрібні DASHBOARD_URL і DASHBOARD_TOKEN (токен імпорту з config.php дашборду).")
    dash = Dashboard(url, token)
    while True:
        n = process_queue(dash, chat, media, out_dir, (W, H), args.font)
        if args.once:
            print(f"Оброблено: {n}")
            return
        if not n:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
