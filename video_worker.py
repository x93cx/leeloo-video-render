"""Виробництво відео за сценарієм з контент-плану — безкоштовними моделями через FreeLLMAPI.

Сценарій (шаблон short-form-scripts: таблиця «Время | На экране | Говорит | B-roll») розбирається на сцени.
Для кожної сцени:
  - кліп: FreeLLMAPI /v1/videos/generations (Pollinations / HuggingFace) → якщо ні, картинка
    /v1/images/generations з плавним наближенням → якщо ні, фірмова картка з текстом;
  - голос: FreeLLMAPI /v1/audio/speech → якщо ні, безкоштовний edge-tts (голоси Microsoft, є uk-UA і ru-RU);
  - текст «На экране» зверху і субтитри мови знизу.
Сцени склеюються у вертикальне відео 9:16 (ffmpeg з пакета imageio-ffmpeg).

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
  Промпти: FreeLLMAPI, або LLM_URL/LLM_KEY/LLM_MODEL, або в GitHub Actions — GitHub Models через GITHUB_TOKEN

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
        scenes.append(Scene(start, max(end, start + 1), get(isc), get(iv), get(ib)))
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
    elif os.environ.get("GITHUB_ACTIONS") and os.environ.get("GITHUB_TOKEN"):
        chat = OpenAIChat("https://models.github.ai/inference", os.environ["GITHUB_TOKEN"],
                          os.environ.get("LLM_MODEL", "openai/gpt-4.1-mini"))
    else:
        chat = None
    return chat, media


def visual_prompts(llm: object | None, scenes: list[Scene], lang: str) -> None:
    """Описи візуалу (UA/RU) → короткі англійські промпти для відео/картинок, одним запитом."""
    payload = [{"i": i, "visual": s.broll, "on_screen": s.on_screen, "voice": s.voice} for i, s in enumerate(scenes)]
    if llm is None:
        for s in scenes:
            s.prompt = f"cinematic vertical b-roll, modern small business office, {s.broll or s.on_screen}"
        return
    res = llm.chat_json(
        "You write prompts for a text-to-video model that makes vertical 9:16 B-roll clips for a short ad about "
        "Leeloo, an AI sales and marketing automation platform for small businesses. For each scene below write one "
        "concise English prompt (max 40 words) describing a realistic, cinematic shot that matches the scene's visual "
        "note. No on-screen text, no logos, no real people's names, no faces of specific people. "
        'Return only JSON: [{"i": 0, "prompt": "..."}].\n\n' + json.dumps(payload, ensure_ascii=False))
    got = {int(x["i"]): str(x["prompt"]) for x in res if isinstance(x, dict) and "i" in x and "prompt" in x} if isinstance(res, list) else {}
    for i, s in enumerate(scenes):
        s.prompt = got.get(i) or f"cinematic vertical b-roll, modern small business office, {s.broll or s.on_screen}"


# ---------- медіа ----------

def ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError("ffmpeg: " + p.stderr[-800:])


def duration_of(path: Path) -> float:
    p = subprocess.run([ffmpeg(), "-i", str(path)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", p.stderr)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0


def edge_tts(text: str, voice: str, dest: Path) -> bool:
    try:
        import edge_tts as et
        asyncio.run(et.Communicate(text, voice, rate="+8%").save(str(dest)))
        return dest.exists() and dest.stat().st_size > 0
    except Exception as e:  # noqa: BLE001
        print(f"  edge-tts: {safe_err(e)}", file=sys.stderr)
        return False


def find_font(bold: bool = True) -> str:
    names = ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold else ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"]
    dirs = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts", Path("/usr/share/fonts/truetype/dejavu"), Path("/Library/Fonts")]
    for d in dirs:
        for n in names:
            if (d / n).exists():
                return str(d / n)
    raise RuntimeError("Не знайдено шрифт з кирилицею — передай --font шлях до .ttf")


def wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def base_overlay(s: Scene, size: tuple[int, int], dest: Path, fonts: tuple[str, str]) -> None:
    """Прозорий шар на всю сцену: плашка «На экране» зверху і жовта смуга Leeloo.ai знизу."""
    W, H = size
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pad = int(W * 0.06)
    if s.on_screen:
        f = ImageFont.truetype(fonts[0], int(W * 0.068))
        lines = wrap(d, s.on_screen, f, W - 4 * pad)[:4]
        lh = int(f.size * 1.25)
        box_h = lh * len(lines) + pad
        y0 = int(H * 0.12)
        d.rounded_rectangle([pad, y0, W - pad, y0 + box_h], radius=int(W * 0.03), fill=(0, 0, 0, 170))
        for i, ln in enumerate(lines):
            d.text((W / 2, y0 + pad / 2 + i * lh), ln, font=f, fill=ACCENT, anchor="ma")
    bar = int(H * 0.045)
    d.rectangle([0, H - bar, W, H], fill=ACCENT)
    fb = ImageFont.truetype(fonts[0], int(bar * 0.55))
    d.text((W / 2, H - bar / 2), "Leeloo.ai", font=fb, fill=BG_DARK, anchor="mm")
    img.save(dest)


def subtitle_chunks(s: Scene, size: tuple[int, int], work: Path, idx: int, font: str) -> list[tuple[Path, float, float]]:
    """Субтитри мови шматками по 2 рядки, кожен показується пропорційно своїй довжині тексту."""
    if not s.voice:
        return []
    W, H = size
    pad = int(W * 0.06)
    f = ImageFont.truetype(font, int(W * 0.045))
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    lines = wrap(probe, s.voice, f, W - 4 * pad)
    groups = [lines[i:i + 2] for i in range(0, len(lines), 2)]
    total = sum(len(" ".join(g)) for g in groups) or 1
    out, t = [], 0.0
    lh = int(f.size * 1.3)
    for gi, g in enumerate(groups):
        span = s.dur * len(" ".join(g)) / total
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        y0 = int(H * 0.80) - lh * len(g)
        for i, ln in enumerate(g):
            d.text((W / 2, y0 + i * lh), ln, font=f, fill="white", anchor="ma", stroke_width=3, stroke_fill="black")
        path = work / f"sub_{idx:02d}_{gi:02d}.png"
        img.save(path)
        out.append((path, round(t, 2), round(t + span, 2)))
        t += span
    return out


def brand_card(s: Scene, size: tuple[int, int], dest: Path, font: str) -> None:
    W, H = size
    img = Image.new("RGB", size, BG_DARK)
    d = ImageDraw.Draw(img)
    for y in range(H):  # м'який градієнт, щоб картка не була пласкою
        c = int(17 + 30 * (y / H))
        d.line([(0, y), (W, y)], fill=(c, c, c + 8))
    img.save(dest)


def render_scene(s: Scene, size: tuple[int, int], work: Path, idx: int, fonts: tuple[str, str]) -> Path:
    W, H = size
    out = work / f"scene_{idx:02d}.mp4"
    overlay = work / f"overlay_{idx:02d}.png"
    base_overlay(s, size, overlay, fonts)
    subs = subtitle_chunks(s, size, work, idx, fonts[1])
    scale = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1"
    if s.clip_kind.startswith("video"):
        vin = ["-stream_loop", "-1", "-i", str(s.clip)]
        vf = f"[0:v]{scale},fps={FPS}[bg]"
    else:  # картинка або фірмова картка — повільне наближення
        frames = int(s.dur * FPS) + 1
        vin = ["-loop", "1", "-i", str(s.clip)]
        vf = (f"[0:v]scale={W * 2}:{H * 2}:force_original_aspect_ratio=increase,crop={W * 2}:{H * 2},"
              f"zoompan=z='min(zoom+0.0008,1.12)':d={frames}:s={W}x{H}:fps={FPS},setsar=1[bg]")
    inputs = [*vin, "-loop", "1", "-i", str(overlay)]
    chain = [vf, "[bg][1:v]overlay=0:0:shortest=1[v0]"]
    for j, (png, a, b) in enumerate(subs):
        inputs += ["-loop", "1", "-i", str(png)]
        chain.append(f"[v{j}][{j + 2}:v]overlay=0:0:shortest=1:enable='between(t,{a},{b})'[v{j + 1}]")
    last = f"[v{len(subs)}]"
    a_idx = len(subs) + 2
    inputs += ["-i", str(s.audio)] if s.audio else ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]
    chain.append(f"[{a_idx}:a]aresample=44100,apad[a]")
    run([ffmpeg(), "-y", *inputs, "-filter_complex", ";".join(chain),
         "-map", last, "-map", "[a]", "-t", f"{s.dur:.2f}", "-r", str(FPS),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2", str(out)])
    return out


def produce(md: str, out_path: Path, chat: object | None, media: object, size: tuple[int, int], font: str | None,
            on_progress=None) -> dict:
    """on_progress(відсоток, етап) — для смуги прогресу в дашборді: промпти 5 %, сцени до 90 %, склейка 95 %."""
    report = on_progress or (lambda pct, stage: None)
    started = time.time()
    scenes = parse_scenes(md)
    if not scenes:
        raise RuntimeError("У сценарії не знайдено таблицю «Время | На экране | Говорит | B-roll».")
    lang = detect_language(md)
    voice = VOICES.get(lang, VOICES["uk"])
    fonts = (font or find_font(True), font or find_font(False))
    report(2, "Готую промпти для картинок")
    visual_prompts(chat, scenes, lang)
    with tempfile.TemporaryDirectory(prefix="leeloo-video-") as tmp:
        work = Path(tmp)
        parts = []
        for i, s in enumerate(scenes):
            print(f"  сцена {i + 1}/{len(scenes)}: {safe(s.on_screen[:50] or s.voice[:50])}")
            report(5 + int(85 * i / len(scenes)), f"Сцена {i + 1} з {len(scenes)}")
            if s.voice:
                a = work / f"voice_{i:02d}.mp3"
                s.voice_engine = media.speech(s.voice, a) or ("edge-tts:" + voice if edge_tts(s.voice, voice, a) else "")
                s.audio = a if s.voice_engine else None
            vlen = duration_of(s.audio) if s.audio else 0.0
            s.dur = round(max(s.end - s.start, vlen + 0.3, 1.5), 2)
            clip = work / f"clip_{i:02d}.mp4"
            kind = media.video(s.prompt, s.dur, clip)
            if not kind:
                clip = work / f"clip_{i:02d}.png"
                kind = media.image(s.prompt, clip)
            if not kind:
                brand_card(s, size, clip, fonts[0])
                kind = "card"
            s.clip, s.clip_kind = clip, kind
            parts.append(render_scene(s, size, work, i, fonts))
        report(92, "Склеюю відео")
        concat = work / "concat.txt"
        concat.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        run([ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", "-movflags", "+faststart", str(out_path)])
    info = {
        "duration_s": round(sum(s.dur for s in scenes), 1), "size": f"{size[0]}x{size[1]}", "language": lang,
        "media": media.name if media.ok else "", "prompts": type(chat).__name__ if chat else "",
        "scenes": [{"on_screen": s.on_screen[:80], "visual": s.clip_kind, "voice": s.voice_engine, "dur": s.dur} for s in scenes],
    }
    info["render_sec"] = round(time.time() - started, 1)
    info["summary"] = summarize(info)
    return info


def summarize(info: dict) -> str:
    kinds = [x["visual"].split(":")[0] for x in info["scenes"]]
    return (f"{len(kinds)} сцен, {info['duration_s']} с · кліпи: {kinds.count('video')} · картинки: {kinds.count('image')} · "
            f"картки: {kinds.count('card')}" + (f" · {info['media']}" if info["media"] else " · без генеративних моделей — лише картки й edge-tts"))


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
        try:
            dash.status(sid, "rendering", progress=0, stage="Старт рендера")
            path = out_dir / f"script-{sid}.mp4"
            info = produce(it["content"], path, chat, media, size, font, lambda pct, stage: dash.progress(sid, pct, stage))
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
    args = ap.parse_args()

    W, H = (int(x) for x in args.size.lower().split("x"))
    chat, media = pick_providers(use_video=not args.no_video_model)
    print(f"Медіа: {media.name if media.ok else 'недоступні — будуть фірмові картки'} · промпти: "
          f"{type(chat).__name__ if chat else 'без LLM (опис сцени як є)'} · голос: edge-tts")
    out_dir = Path(args.out)

    if args.file:
        md = Path(args.file).read_text(encoding="utf-8")
        path = out_dir / (Path(args.file).stem + ".mp4")
        info = produce(md, path, chat, media, (W, H), args.font)
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
