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
from pyrogram.types import InlineKeyboardButton, InlineKeyboardMarkup, Message
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
CHARS_PER_SEC = 11.0                                  # பேச்சு வேகம் (எழுத்து/வினாடி)
# code: (தமிழ் பெயர், English name, ISO-639-2, script, chars/sec)
LANGS = {
    "ta": ("தமிழ்", "Tamil", "tam", "Tamil", 11.0),
    "te": ("தெலுங்கு", "Telugu", "tel", "Telugu", 11.0),
    "ml": ("மலையாளம்", "Malayalam", "mal", "Malayalam", 10.0),
    "kn": ("கன்னடம்", "Kannada", "kan", "Kannada", 11.0),
    "hi": ("ஹிந்தி", "Hindi", "hin", "Devanagari", 14.0),
}

_T0 = time.time()


def log(*a):
    print(f"[{time.time() - _T0:7.1f}s]", *a, flush=True)


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
        log("Whisper GPU load failed, CPU-க்கு மாறுறேன்:", e)
        WHISPER = WhisperModel("small", device="cpu", compute_type="int8")

    if DEVICE == "cuda":
        OMNI = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0",
                                         dtype=torch.float16)
    else:
        OMNI = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cpu",
                                         dtype=torch.float32)
    log("Models loaded on", DEVICE)


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


def count_audio(path):
    p = run(["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", path])
    return len([x for x in p.stdout.split() if x.strip()])


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


# ---------------------------------------------------------------- subtitles
def _srt_time(t):
    t = max(0.0, t)
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    sec, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def english_subs(vocals_path, lang, segs):
    """English subtitle வரிகள் [(start, end, text)]. மூலம் English இல்லன்னா whisper translate."""
    if lang == "en":
        return [(s["start"], s["end"], s["text"]) for s in segs]
    try:
        segments, _ = WHISPER.transcribe(
            vocals_path, language=lang, task="translate", vad_filter=True,
            beam_size=5, condition_on_previous_text=False)
        out = []
        for s in segments:
            t = (s.text or "").strip()
            if t:
                out.append((float(s.start), float(s.end), t))
        return out
    except Exception as e:
        log("English subtitle failed:", repr(e)[:150])
        return []


def write_srt(subs, path):
    lines = []
    for i, (a, b, t) in enumerate(subs, 1):
        lines.append(f"{i}\n{_srt_time(a)} --> {_srt_time(b)}\n{t}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return len(subs)


# ---------------------------------------------------------------- translate
_GEM_MODEL = None
_GEM_BASE = "https://generativelanguage.googleapis.com/v1beta"
_SKIP_CODES = (400, 404, 429, 500, 502, 503, 504)   # இந்த model வேண்டாம், அடுத்ததை try செய்


def _gem_headers():
    return {"x-goog-api-key": GEMINI_KEY, "Content-Type": "application/json"}


def _gem_candidates():
    """env model -> சில known models -> API-ல் இருக்கும் flash models."""
    names = []
    if os.environ.get("GEMINI_MODEL"):
        names.append(os.environ["GEMINI_MODEL"])
    names += ["gemini-flash-latest", "gemini-2.5-flash",
              "gemini-2.5-flash-lite", "gemini-2.0-flash"]
    try:
        r = httpx.get(f"{_GEM_BASE}/models", headers=_gem_headers(),
                      params={"pageSize": 200}, timeout=60)
        r.raise_for_status()
        listed = []
        for m in r.json().get("models", []):
            n = m.get("name", "").replace("models/", "")
            if ("generateContent" in m.get("supportedGenerationMethods", [])
                    and "flash" in n
                    and not any(x in n for x in ("image", "tts", "live", "audio",
                                                 "embedding", "robotics", "thinking"))):
                listed.append(n)
        names += sorted(listed, reverse=True)
        log("Gemini models available:", listed[:8])
    except Exception as e:
        log("Gemini model list failed:", str(e)[:150])
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _gem_prompt(items, src_lang, tgt="ta"):
    _, name, _, script, cps = LANGS[tgt]
    payload = [{"id": k, "text": s_["text"],
                "max_chars": max(10, int(s_["slot"] * cps))}
               for k, s_ in enumerate(items)]
    return (
        f"You are a professional {name} dubbing writer. Translate each line "
        f"(source language: {src_lang or 'auto'}) into natural spoken {name} ({script} script) "
        "for dubbing a video.\n"
        "Rules: keep meaning and emotion; casual spoken style that fits the scene; "
        "each translation MUST be at most max_chars characters (shorten the wording if "
        f"needed, never exceed it); write names in {script} script; write numbers as "
        f"{name} words; no English letters, no emojis, no explanations.\n"
        "Return ONLY a JSON array of strings, same order and same count as the input.\n\n"
        "INPUT:\n" + json.dumps(payload, ensure_ascii=False))


def gemini_batch(items, src_lang, tgt="ta"):
    global _GEM_MODEL
    body = {"contents": [{"parts": [{"text": _gem_prompt(items, src_lang, tgt)}]}],
            "generationConfig": {"temperature": 0.3,
                                 "responseMimeType": "application/json"}}
    models = [_GEM_MODEL] if _GEM_MODEL else _gem_candidates()
    last = "model இல்ல"
    for m in models:
        try:
            r = httpx.post(f"{_GEM_BASE}/models/{m}:generateContent",
                           headers=_gem_headers(), json=body, timeout=90)
        except httpx.HTTPError as e:           # timeout / network
            last = f"{m}: {type(e).__name__}"
            log("Gemini skip", last)
            _GEM_MODEL = None
            continue
        if r.status_code in _SKIP_CODES:
            last = f"{m}: HTTP {r.status_code} {r.text[:100]}"
            log("Gemini skip", last)
            _GEM_MODEL = None
            continue
        if r.status_code != 200:
            raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:200]}")
        try:
            txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
            arr = json.loads(txt)
            if not isinstance(arr, list) or len(arr) != len(items):
                raise ValueError("Gemini count mismatch")
        except Exception as e:
            last = f"{m}: {str(e)[:100]}"
            log("Gemini bad reply", last)
            _GEM_MODEL = None
            continue
        _GEM_MODEL = m
        log("Gemini model:", m)
        return [str(x).strip() for x in arr]
    raise RuntimeError("Gemini வேலை செய்யல: " + last)


def fallback_translate(text, tgt="ta"):
    from deep_translator import GoogleTranslator
    last = None
    for k in range(4):
        try:
            return GoogleTranslator(source="auto", target=tgt).translate(text) or ""
        except Exception as e:
            last = e
            time.sleep(2 + 3 * k)
    raise last


def translate_all(segs, src_lang, tgt="ta"):
    out = []
    B = 25
    for i in range(0, len(segs), B):
        chunk = segs[i:i + B]
        res = None
        if GEMINI_KEY:
            for attempt in range(3):
                try:
                    res = gemini_batch(chunk, src_lang, tgt)
                    break
                except Exception as e:
                    log("Gemini retry:", str(e).replace(GEMINI_KEY, "***")[:250])
                    time.sleep(2 + attempt * 3)
        if res is None:
            log("Google Translate fallback பயன்படுத்துறேன்")
            res = []
            for s_ in chunk:
                try:
                    res.append(fallback_translate(s_["text"], tgt))
                except Exception as e2:
                    log("Google Translate failed:", repr(e2)[:150])
                    res.append("")
                time.sleep(0.5)
        log("translated sample:", [x[:40] for x in res[:2]])
        out.extend(res)
    if not any((t or "").strip() for t in out):
        raise RuntimeError(f"{LANGS[tgt][0]} மொழிபெயர்ப்பு fail ஆச்சு")
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


def synth(text, ref_path, ref_text, duration=None, lang="ta"):
    kw = dict(text=text, language=lang, num_step=NUM_STEP, pad_duration=0.0)
    if ref_path:
        kw["ref_audio"] = ref_path
        if ref_text:
            kw["ref_text"] = ref_text
    if duration:
        kw["duration"] = float(duration)
    audio = OMNI.generate(**kw)
    return np.asarray(audio[0], dtype=np.float32)


def fit_to_slot(wav, sr, slot, text, ref_path, ref_text, lang="ta"):
    L = len(wav) / sr
    if L > slot * 1.05:
        try:
            w2 = synth(text, ref_path, ref_text, duration=slot, lang=lang)
            if len(w2) > 0:
                wav = w2
        except Exception as e:
            log("duration regen failed:", repr(e))
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
def pipeline(src, workdir, out_path, progress, opts):
    langs = [c for c in LANGS if opts.get("langs", {}).get(c)] or ["ta"]
    total, width, height = probe(src)

    progress("🎵 ஆடியோ எடுக்கிறேன்...", 2)
    audio = os.path.join(workdir, "audio.wav")
    run(["ffmpeg", "-y", "-i", src, "-vn", "-ac", "2", "-ar", "44100", audio])

    progress("🎚️ குரல் / BGM பிரிக்கிறேன்...", 5)
    try:
        vocals, bg = separate(audio, workdir)
    except Exception as e:
        log("demucs failed:", repr(e))
        vocals, bg = audio, None

    progress("📝 பேச்சை எழுத்தாக மாற்றுறேன்...", 25)
    segs, words, src_lang = transcribe(vocals)
    if not segs:
        raise RuntimeError("வீடியோவில் பேச்சு எதுவும் கிடைக்கல")
    add_slots(segs, total)
    log("whisper done:", len(segs), "segments, lang", src_lang)

    progress("💬 English subtitle தயார் பண்றேன்...", 38)
    srt_path = os.path.join(workdir, "english.srt")
    n_subs = write_srt(english_subs(vocals, src_lang, segs), srt_path)

    voc, vsr = sf.read(vocals, always_2d=True)
    voc = voc.mean(axis=1).astype(np.float32)
    refdir = os.path.join(workdir, "refs")
    os.makedirs(refdir, exist_ok=True)
    refs = [make_ref(i, s, voc, vsr, words, total, refdir) for i, s in enumerate(segs)]

    sr = OMNI.sampling_rate
    base_path = bg if bg else audio
    base, bsr = sf.read(base_path, always_2d=True)
    base = base.astype(np.float32)
    g = gcd(bsr, sr)
    gain = BG_GAIN if bg else 0.12
    n_seg = len(segs)

    results = []          # [{code, mix, dub, failed}]
    skipped = []
    span = 48.0 / len(langs)
    for li, code in enumerate(langs):
        lname = LANGS[code][0]
        p0 = 42 + span * li
        progress(f"🌐 {lname} மொழிபெயர்க்கிறேன்... ({n_seg} வசனம்)", int(p0))
        try:
            texts = translate_all(segs, src_lang, code)
        except Exception as e:
            log(f"{code} translate failed:", repr(e)[:150])
            skipped.append(code)
            continue

        track = np.zeros(int((total + 2) * sr), dtype=np.float32)
        failed = 0
        for i, s in enumerate(segs):
            progress(f"🎙️ {lname} குரல் உருவாக்குறேன்... {i + 1}/{n_seg}",
                     int(p0 + span * (0.1 + 0.9 * (i + 1) / n_seg)))
            text = (texts[i] or "").strip()
            if not text:
                log(f"{code} segment {i}: text காலி")
                failed += 1
                continue
            try:
                ref_path, ref_text = refs[i]
                wav = synth(text, ref_path, ref_text, lang=code)
                wav = fit_to_slot(wav, sr, s["slot"], text, ref_path, ref_text, lang=code)
                a = int(s["start"] * sr)
                b = min(a + len(wav), len(track))
                track[a:b] += wav[:b - a]
            except Exception as e:
                failed += 1
                log(f"{code} segment {i} failed:", repr(e))

        dub = resample_poly(track, bsr // g, sr // g).astype(np.float32)
        n = base.shape[0]
        if len(dub) < n:
            dub = np.pad(dub, (0, n - len(dub)))
        dub = dub[:n]
        mix = base * gain + dub[:, None]
        peak = float(np.max(np.abs(mix)))
        if peak > 0.98:
            mix = mix * (0.98 / peak)
        mix_path = os.path.join(workdir, f"mix_{code}.wav")
        sf.write(mix_path, mix, bsr)
        results.append(dict(code=code, mix=mix_path, dub=dub, failed=failed))

    if not results:
        raise RuntimeError("எந்த மொழியிலும் மொழிபெயர்ப்பு நடக்கல, கொஞ்ச நேரம் கழித்து மறுபடி முயற்சி செய்")

    tracks = []
    mkv_path = None
    if opts.get("doc"):
        progress("🎬 MKV-ஆ இணைக்கிறேன்...", 94)
        inputs = [src]
        maps = ["-map", "0:v:0"]
        meta = []
        aidx = 0

        def add_input(path):
            inputs.append(path)
            return len(inputs) - 1

        def add_audio(path, lng, title, label):
            nonlocal aidx
            maps.extend(["-map", f"{add_input(path)}:a:0"])
            meta.extend([f"-metadata:s:a:{aidx}", f"language={lng}",
                         f"-metadata:s:a:{aidx}", f"title={title}"])
            tracks.append(label)
            aidx += 1

        for r in results:
            _, ename, iso, _, _ = LANGS[r["code"]]
            add_audio(r["mix"], iso, f"{ename} Dub", LANGS[r["code"]][0])
        if opts.get("voice"):
            for r in results:
                _, ename, iso, _, _ = LANGS[r["code"]]
                vpath = os.path.join(workdir, f"voice_{r['code']}.wav")
                vo = r["dub"]
                pk = float(np.max(np.abs(vo))) if len(vo) else 0.0
                if pk > 0.98:
                    vo = vo * (0.98 / pk)
                sf.write(vpath, vo, bsr)
                add_audio(vpath, iso, f"{ename} Voice Only", f"{LANGS[r['code']][0]} குரல் மட்டும்")
        if opts.get("bg"):
            if bg:
                add_audio(bg, "und", "BGM Only", "BGM மட்டும்")
            else:
                log("BGM track skip: demucs இல்ல")
        if opts.get("orig"):
            n_orig = count_audio(src)
            for k in range(n_orig):
                maps.extend(["-map", f"0:a:{k}"])
                meta.extend([f"-metadata:s:a:{aidx}", "language=und",
                             f"-metadata:s:a:{aidx}",
                             f"title={'Original' if n_orig == 1 else 'Original ' + str(k + 1)}"])
                aidx += 1
            if n_orig:
                tracks.append("அசல் ஆடியோ")
        if n_subs > 0:
            maps.extend(["-map", f"{add_input(srt_path)}:0"])
            meta.extend(["-c:s", "srt", "-metadata:s:s:0", "language=eng",
                         "-metadata:s:s:0", "title=English", "-disposition:s:0", "default"])
        disp = ["-disposition:a:0", "default"]
        for k in range(1, aidx):
            disp += [f"-disposition:a:{k}", "0"]
        cmd = ["ffmpeg", "-y"]
        for pth in inputs:
            cmd += ["-i", pth]
        try:
            run(cmd + maps + ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k"] + meta + disp + [out_path])
        except Exception as e:
            log("mkv copy failed, re-encode:", repr(e)[:150])
            run(cmd + maps + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                              "-c:a", "aac", "-b:a", "192k"] + meta + disp + [out_path])
        mkv_path = out_path

    mp4s = []
    if opts.get("video"):
        for k, r in enumerate(results):
            code = r["code"]
            progress(f"🎬 {LANGS[code][0]} MP4 தயார் பண்றேன்...", 96 + k * 3 // len(results))
            mp4 = os.path.join(os.path.dirname(out_path), f"Dubbed_{LANGS[code][1]}.mp4")
            base_cmd = ["ffmpeg", "-y", "-i", src, "-i", r["mix"],
                        "-map", "0:v:0", "-map", "1:a:0"]
            try:
                run(base_cmd + ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                                "-movflags", "+faststart", mp4])
            except Exception as e:
                log("mp4 copy failed, re-encode:", repr(e)[:150])
                run(base_cmd + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", mp4])
            mp4s.append((code, mp4))
            if not opts.get("doc"):
                tracks.append(LANGS[code][0])

    progress("✅ முடிந்தது", 100)

    return dict(lang=src_lang, n=n_seg, failed=sum(r["failed"] for r in results),
                subs=n_subs, tracks=tracks, mkv=mkv_path, mp4s=mp4s,
                skipped=[LANGS[c][0] for c in skipped],
                duration=int(total), width=width, height=height)


# ---------------------------------------------------------------- telegram
@app.on_message(filters.command("start") & filters.private)
async def start_cmd(_, m: Message):
    await m.reply_text(
        "வணக்கம்! 🎬\nவீடியோவை அனுப்பு. அதே கேரக்டர் குரலில், அதே நேரத்தில் "
        "தமிழ் டப்பிங் பண்ணி தருகிறேன்.\nநேரம் ஆகலாம், பொறுமையா காத்திரு.")


PENDING = {}
TASKS = set()


def sel_markup(key, opts):
    def mark(on):
        return "✅ " if on else "⬜ "

    def lang_btn(c):
        return InlineKeyboardButton(mark(opts["langs"][c]) + LANGS[c][0],
                                    callback_data=f"sel:{key}:{c}")

    def extra_btn(label, k):
        return [InlineKeyboardButton(mark(opts[k]) + label, callback_data=f"sel:{key}:{k}")]

    return InlineKeyboardMarkup([
        [lang_btn("ta"), lang_btn("te"), lang_btn("ml")],
        [lang_btn("kn"), lang_btn("hi")],
        extra_btn("குரல் மட்டும் tracks (BGM இல்லாம)", "voice"),
        extra_btn("அசல் ஆடியோ", "orig"),
        extra_btn("BGM மட்டும்", "bg"),
        [InlineKeyboardButton(mark(opts["doc"]) + "📄 Document (MKV)",
                              callback_data=f"sel:{key}:doc"),
         InlineKeyboardButton(mark(opts["video"]) + "🎬 Video (MP4)",
                              callback_data=f"sel:{key}:video")],
        [InlineKeyboardButton("▶️ தொடங்கு", callback_data=f"sel:{key}:go")],
    ])


SEL_TEXT = ("🎧 எந்தெந்த மொழியில் டப் வேணும்? (பல மொழி தேர்வு செய்யலாம்)\n"
            "ஒவ்வொரு மொழியும் ஒரு audio track-ஆ MKV-ல் வரும். "
            "மொழி கூடக்கூட நேரமும் கூடும்.\n"
            "📄 Document (MKV) = எல்லா audio + subtitle ஒரே file. "
            "🎬 Video (MP4) = Telegram-லயே play ஆகும், ஒரு மொழிக்கு ஒரு video.\n"
            "தேர்வு பண்ணிட்டு ▶️ தொடங்கு அழுத்து.")


@app.on_message(filters.private & (filters.video | filters.document))
async def handle_video(_, m: Message):
    if m.document and not (m.document.mime_type or "").startswith("video/"):
        return await m.reply_text("வீடியோ கோப்பு மட்டும் அனுப்பு.")
    key = f"{m.chat.id}_{m.id}"
    opts = {"langs": {c: c == "ta" for c in LANGS},
            "voice": False, "orig": False, "bg": False,
            "doc": True, "video": False}
    PENDING[key] = {"m": m, "opts": opts}
    await m.reply_text(SEL_TEXT, reply_markup=sel_markup(key, opts), quote=True)


@app.on_callback_query(filters.regex(r"^sel:"))
async def on_select(_, q):
    _, key, act = q.data.split(":")
    st = PENDING.get(key)
    if not st:
        return await q.answer("இந்த request காலாவதி ஆச்சு, வீடியோவை மறுபடி அனுப்பு", show_alert=True)
    opts = st["opts"]
    if act in LANGS:
        opts["langs"][act] = not opts["langs"][act]
    elif act in ("voice", "orig", "bg", "doc", "video"):
        opts[act] = not opts[act]
    elif act == "go":
        if not any(opts["langs"].values()):
            return await q.answer("குறைந்தது ஒரு மொழி தேர்வு பண்ணு", show_alert=True)
        if not (opts["doc"] or opts["video"]):
            return await q.answer("Document அல்லது Video-ல ஒண்ணு தேர்வு பண்ணு", show_alert=True)
        PENDING.pop(key, None)
        await q.answer("தொடங்குறேன்...")
        await safe_edit(q.message, "⏳ வரிசையில் இருக்கு...")
        task = asyncio.create_task(process_video(st["m"], q.message, opts))
        TASKS.add(task)
        task.add_done_callback(TASKS.discard)
        return
    try:
        await q.edit_message_reply_markup(sel_markup(key, opts))
    except Exception:
        pass
    await q.answer()


async def process_video(m: Message, status: Message, opts: dict):
    workdir = tempfile.mkdtemp(prefix="dub_")
    src = os.path.join(workdir, "input.mp4")
    out = os.path.join(workdir, "Dubbed.mkv")
    loop = asyncio.get_running_loop()
    last = {"t": 0.0, "txt": "", "pct": 0}

    def progress(text, pct=None):
        log(text, f"{pct}%" if pct is not None else "")
        if pct is not None:
            last["pct"] = pct
        p = last["pct"]
        msg = f"{'▓' * (p // 10)}{'░' * (10 - p // 10)} {p}%\n{text}"
        now = time.time()
        if msg != last["txt"] and (now - last["t"] > 3 or p >= 100):
            last["t"], last["txt"] = now, msg
            asyncio.run_coroutine_threadsafe(safe_edit(status, msg), loop)

    try:
        async with queue_lock:
            await safe_edit(status, "░░░░░░░░░░ 0%\n📥 வீடியோ எடுக்கிறேன்...")
            await m.download(file_name=src)
            info = await asyncio.to_thread(pipeline, src, workdir, out, progress, opts)
            await safe_edit(status, "▓▓▓▓▓▓▓▓▓▓ 100%\n📤 உனக்கு அனுப்புறேன்...")
            cap = f"✅ டப்பிங் ரெடி!\n(மூல மொழி: {info['lang']}, {info['n']} வசனம்"
            cap += f", {info['failed']} தவறியது)" if info["failed"] else ")"
            cap += "\n🎧 Audio: " + ", ".join(info["tracks"])
            if info["skipped"]:
                cap += "\n⚠️ தவறிய மொழி: " + ", ".join(info["skipped"])
            if info["subs"]:
                cap += "\n💬 English subtitle உள்ளது"
            if info["mkv"]:
                cap += "\n(VLC / MX Player-ல் Audio & Subtitle track மாத்திக்கலாம்)"
                await m.reply_document(info["mkv"], caption=cap, force_document=True)
            for k, (code, pth) in enumerate(info["mp4s"]):
                vcap = f"🎬 {LANGS[code][0]} டப்"
                if not info["mkv"] and k == 0:
                    vcap = cap + "\n" + vcap
                await m.reply_video(pth, caption=vcap, supports_streaming=True,
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
    log("Bot started")
    await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
