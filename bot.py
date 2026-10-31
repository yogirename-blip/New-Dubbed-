import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from math import gcd

import httpx
import librosa
import numpy as np
import soundfile as sf
import torch
from aiohttp import web
from pyrogram import Client, filters
from pyrogram.types import Message
from scipy.signal import resample_poly

API_ID = int(os.environ["API_ID"])
API_HASH = os.environ["API_HASH"]
BOT_TOKEN = os.environ["BOT_TOKEN"]
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
SOURCE_LANG = os.environ.get("SOURCE_LANG") or None   # ja / hi / en ... (காலியா விட்டா auto)
NUM_STEP = int(os.environ.get("NUM_STEP", "32"))
BG_GAIN = float(os.environ.get("BG_GAIN", "0.8"))     # BGM சத்தம்
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHARS_PER_SEC = 11.0                                  # தமிழ் பேச்சு வேகம் (எழுத்து/வினாடி)

queue_lock = asyncio.Semaphore(1)
app = Client("tamil_dub_bot", api_id=API_ID, api_hash=API_HASH,
             bot_token=BOT_TOKEN, in_memory=True)

WHISPER = None
OMNI = None


# ---------------------------------------------------------------- models
def load_models():
    global WHISPER, OMNI
    from faster_whisper import WhisperModel
    from omnivoice import OmniVoice

    try:
        if DEVICE != "cuda":
            raise RuntimeError("GPU இல்ல")
        WHISPER = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    except Exception as e:
        print("Whisper GPU load failed, CPU-க்கு மாறுறேன்:", e)
        WHISPER = WhisperModel("small", device="cpu", compute_type="int8")

    if DEVICE == "cuda":
        OMNI = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0",
                                         dtype=torch.float16)
    else:
        OMNI = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cpu",
                                         dtype=torch.float32)
    print("Models loaded on", DEVICE)


# ---------------------------------------------------------------- helpers
async def safe_edit(msg: Message, text: str):
    try:
        await msg.edit_text(text)
    except Exception:
        pass


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} பிழை: {p.stderr[-500:]}")
    return p


def probe(path):
    p = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height:format=duration",
             "-of", "json", path])
    j = json.loads(p.stdout)
    st = (j.get("streams") or [{}])[0]
    return float(j["format"]["duration"]), int(st.get("width") or 0), int(st.get("height") or 0)


def separate(audio_path, workdir):
    """demucs: குரல் + BGM பிரிக்கும். (vocals_path, background_path) தரும்."""
    outdir = os.path.join(workdir, "sep")
    run([sys.executable, "-m", "demucs.separate", "--two-stems=vocals",
         "-n", "htdemucs", "-d", DEVICE, "-o", outdir, audio_path])
    base = os.path.splitext(os.path.basename(audio_path))[0]
    folder = os.path.join(outdir, "htdemucs", base)
    vocals = os.path.join(folder, "vocals.wav")
    bg = os.path.join(folder, "no_vocals.wav")
    if not (os.path.exists(vocals) and os.path.exists(bg)):
        raise RuntimeError("demucs output கிடைக்கல")
    return vocals, bg


def transcribe(vocals_path):
    segments, info = WHISPER.transcribe(
        vocals_path, language=SOURCE_LANG, vad_filter=True, beam_size=5,
        word_timestamps=True, condition_on_previous_text=False)
    segs, words = [], []
    for s in segments:
        text = (s.text or "").strip()
        if not text:
            continue
        segs.append({"start": float(s.start), "end": float(s.end), "text": text})
        for w in (s.words or []):
            words.append((float(w.start), float(w.end), w.word))
    return segs, words, info.language


def add_slots(segs, total):
    """ஒவ்வொரு வசனத்துக்கும் அதிகபட்சம் எவ்வளவு நேரம் பேசலாம்னு கணக்கு."""
    for i, s in enumerate(segs):
        dur = max(0.3, s["end"] - s["start"])
        nxt = segs[i + 1]["start"] if i + 1 < len(segs) else total
        avail = nxt - s["start"] - 0.06
        slot = min(avail, dur * 1.6 + 0.4)
        slot = max(slot, min(dur, avail))
        s["slot"] = max(0.4, slot)


# ---------------------------------------------------------------- translate
def gemini_batch(items, src_lang):
    payload = [{"id": k, "text": s["text"],
                "max_chars": max(10, int(s["slot"] * CHARS_PER_SEC))}
               for k, s in enumerate(items)]
    prompt = (
        "You are a professional Tamil dubbing writer. Translate each line "
        f"(source language: {src_lang or 'auto'}) into natural spoken Tamil (Tamil script) "
        "for dubbing a video.\n"
        "Rules: keep meaning and emotion; casual spoken style that fits the scene; "
        "each translation MUST be at most max_chars characters (shorten the wording if "
        "needed, never exceed it); write names in Tamil script; write numbers as Tamil "
        "words; no English letters, no emojis, no explanations.\n"
        "Return ONLY a JSON array of strings, same order and same count as the input.\n\n"
        "INPUT:\n" + json.dumps(payload, ensure_ascii=False))
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}")
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3,
                                 "responseMimeType": "application/json"}}
    r = httpx.post(url, json=body, timeout=180)
    r.raise_for_status()
    txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    arr = json.loads(txt)
    if not isinstance(arr, list) or len(arr) != len(items):
        raise ValueError("Gemini count mismatch")
    return [str(x).strip() for x in arr]


def fallback_translate(text):
    from deep_translator import GoogleTranslator
    return GoogleTranslator(source="auto", target="ta").translate(text) or text


def translate_all(segs, src_lang):
    out = []
    B = 25
    for i in range(0, len(segs), B):
        chunk = segs[i:i + B]
        res = None
        if GEMINI_KEY:
            for attempt in range(3):
                try:
                    res = gemini_batch(chunk, src_lang)
                    break
                except Exception as e:
                    print("Gemini retry:", e)
                    time.sleep(2 + attempt * 3)
        if res is None:
            res = []
            for s in chunk:
                try:
                    res.append(fallback_translate(s["text"]))
                except Exception:
                    res.append("")
        out.extend(res)
    return out


# ---------------------------------------------------------------- voice
def make_ref(i, s, voc, vsr, words, total, refdir):
    """அந்த வசனத்தோட original குரலிலிருந்து 3-10 வினாடி reference எடுக்கும்."""
    ws, we = s["start"], s["end"]
    if we - ws < 3.0:
        need = 3.0 - (we - ws)
        ws, we = ws - need / 2, we + need / 2
        if ws < 0:
            we, ws = we - ws, 0.0
        if we > total:
            ws, we = max(0.0, ws - (we - total)), total
    if we - ws > 10.0:
        we = ws + 10.0
    a, b = int(ws * vsr), int(we * vsr)
    clip = voc[a:b]
    if len(clip) < vsr * 0.5:
        return None, None
    path = os.path.join(refdir, f"ref_{i}.wav")
    sf.write(path, clip, vsr)
    text = "".join(w for (st, en, w) in words if st >= ws - 0.02 and en <= we + 0.02).strip()
    return path, (text or None)


def synth(text, ref_path, ref_text, duration=None):
    kw = dict(text=text, language="ta", num_step=NUM_STEP, pad_duration=0.0)
    if ref_path:
        kw["ref_audio"] = ref_path
        if ref_text:
            kw["ref_text"] = ref_text
    if duration:
        kw["duration"] = float(duration)
    audio = OMNI.generate(**kw)
    return np.asarray(audio[0], dtype=np.float32)


def fit_to_slot(wav, sr, slot, text, ref_path, ref_text):
    L = len(wav) / sr
    if L > slot * 1.05:
        try:
            w2 = synth(text, ref_path, ref_text, duration=slot)
            if len(w2) > 0:
                wav = w2
        except Exception as e:
            print("duration regen failed:", e)
        L = len(wav) / sr
        if L > slot:
            rate = min(L / slot, 1.35)
            wav = librosa.effects.time_stretch(wav, rate=rate)
    maxlen = int(slot * sr)
    if len(wav) > maxlen:
        wav = wav[:maxlen].copy()
        f = min(int(0.04 * sr), len(wav))
        wav[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
    # சத்தம் சமநிலை
    rms = float(np.sqrt(np.mean(wav ** 2))) + 1e-8
    wav = wav * min(4.0, 0.08 / rms)
    peak = float(np.max(np.abs(wav))) if len(wav) else 0
    if peak > 0.95:
        wav = wav * (0.95 / peak)
    return wav


# ---------------------------------------------------------------- pipeline
def pipeline(src, workdir, out_path, progress):
    total, width, height = probe(src)

    audio = os.path.join(workdir, "audio.wav")
    run(["ffmpeg", "-y", "-i", src, "-vn", "-ac", "2", "-ar", "44100", audio])

    progress("🎚️ குரல் / BGM பிரிக்கிறேன்...")
    try:
        vocals, bg = separate(audio, workdir)
    except Exception as e:
        print("demucs failed:", e)
        vocals, bg = audio, None

    progress("📝 பேச்சை எழுத்தாக மாற்றுறேன்...")
    segs, words, lang = transcribe(vocals)
    if not segs:
        raise RuntimeError("வீடியோவில் பேச்சு எதுவும் கிடைக்கல")
    add_slots(segs, total)

    progress(f"🌐 தமிழுக்கு மொழிபெயர்க்கிறேன்... ({len(segs)} வசனம்)")
    texts = translate_all(segs, lang)

    voc, vsr = sf.read(vocals, always_2d=True)
    voc = voc.mean(axis=1).astype(np.float32)

    refdir = os.path.join(workdir, "refs")
    os.makedirs(refdir, exist_ok=True)
    sr = OMNI.sampling_rate
    track = np.zeros(int((total + 2) * sr), dtype=np.float32)
    failed = 0

    for i, s in enumerate(segs):
        progress(f"🎙️ தமிழ் குரல் உருவாக்குறேன்... {i + 1}/{len(segs)}")
        text = (texts[i] or "").strip()
        if not text:
            failed += 1
            continue
        try:
            ref_path, ref_text = make_ref(i, s, voc, vsr, words, total, refdir)
            wav = synth(text, ref_path, ref_text)
            wav = fit_to_slot(wav, sr, s["slot"], text, ref_path, ref_text)
            a = int(s["start"] * sr)
            b = min(a + len(wav), len(track))
            track[a:b] += wav[:b - a]
        except Exception as e:
            failed += 1
            print(f"segment {i} failed:", e)

    progress("🎛️ குரலையும் BGM-ஐயும் சேர்க்கிறேன்...")
    base_path = bg if bg else audio
    base, bsr = sf.read(base_path, always_2d=True)
    base = base.astype(np.float32)
    g = gcd(bsr, sr)
    dub = resample_poly(track, bsr // g, sr // g).astype(np.float32)
    n = base.shape[0]
    if len(dub) < n:
        dub = np.pad(dub, (0, n - len(dub)))
    dub = dub[:n]
    gain = BG_GAIN if bg else 0.12
    mix = base * gain + dub[:, None]
    peak = float(np.max(np.abs(mix)))
    if peak > 0.98:
        mix = mix * (0.98 / peak)
    mix_path = os.path.join(workdir, "mix.wav")
    sf.write(mix_path, mix, bsr)

    progress("🎬 வீடியோவோடு இணைக்கிறேன்...")
    try:
        run(["ffmpeg", "-y", "-i", src, "-i", mix_path, "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-movflags", "+faststart", out_path])
    except Exception:
        run(["ffmpeg", "-y", "-i", src, "-i", mix_path, "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path])

    return dict(lang=lang, n=len(segs), failed=failed,
                duration=int(total), width=width, height=height)


# ---------------------------------------------------------------- telegram
@app.on_message(filters.command("start") & filters.private)
async def start_cmd(_, m: Message):
    await m.reply_text(
        "வணக்கம்! 🎬\nவீடியோவை அனுப்பு. அதே கேரக்டர் குரலில், அதே நேரத்தில் "
        "தமிழ் டப்பிங் பண்ணி தருகிறேன்.\nநேரம் ஆகலாம், பொறுமையா காத்திரு.")


@app.on_message(filters.private & (filters.video | filters.document))
async def handle_video(_, m: Message):
    if m.document and not (m.document.mime_type or "").startswith("video/"):
        return await m.reply_text("வீடியோ கோப்பு மட்டும் அனுப்பு.")

    status = await m.reply_text("⏳ வரிசையில் இருக்கு...")
    workdir = tempfile.mkdtemp(prefix="dub_")
    src = os.path.join(workdir, "input.mp4")
    out = os.path.join(workdir, "tamil.mp4")
    loop = asyncio.get_running_loop()
    last = {"t": 0.0, "txt": ""}

    def progress(text):
        now = time.time()
        if text != last["txt"] and now - last["t"] > 3:
            last["t"], last["txt"] = now, text
            asyncio.run_coroutine_threadsafe(safe_edit(status, text), loop)

    try:
        async with queue_lock:
            await safe_edit(status, "📥 வீடியோ எடுக்கிறேன்...")
            await m.download(file_name=src)
            info = await asyncio.to_thread(pipeline, src, workdir, out, progress)
            await safe_edit(status, "📤 உனக்கு அனுப்புறேன்...")
            cap = f"✅ தமிழ் டப்பிங் ரெடி!\n(மூல மொழி: {info['lang']}, {info['n']} வசனம்"
            cap += f", {info['failed']} தவறியது)" if info["failed"] else ")"
            await m.reply_video(out, caption=cap, supports_streaming=True,
                                duration=info["duration"], width=info["width"],
                                height=info["height"])
            await status.delete()
    except Exception as e:
        await safe_edit(status, f"❌ பிழை: {str(e)[:500]}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


async def health(_):
    return web.Response(text="ok")


async def main():
    await asyncio.to_thread(load_models)
    web_app = web.Application()
    web_app.router.add_get("/", health)
    runner = web.AppRunner(web_app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", int(os.environ.get("PORT", 8080))).start()
    await app.start()
    print("Bot started")
    await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
