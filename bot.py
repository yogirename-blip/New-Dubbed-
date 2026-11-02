import asyncio
import json
import os

# transformers TensorFlow/JAX-ஐ தொடவே வேணாம் (numpy 1.26 உடன் jax உடையுது)
for _k, _v in (("USE_TF", "0"), ("USE_JAX", "0"), ("USE_FLAX", "0"),
               ("TRANSFORMERS_NO_TF", "1"), ("TRANSFORMERS_NO_FLAX", "1"),
               ("USE_TORCH", "1")):
    os.environ.setdefault(_k, _v)
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
NUM_STEP = int(os.environ.get("NUM_STEP", "32"))       # குறைச்சா வேகம், கூட்டினா தரம்
BG_GAIN = float(os.environ.get("BG_GAIN", "0.8"))     # BGM சத்தம்
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHARS_PER_SEC = 11.0                                  # பேச்சு வேகம் (எழுத்து/வினாடி)
# code: (தமிழ் பெயர், English name, ISO-639-2, script, chars/sec)
BOT_VERSION = "2026-11-02-r21 (IndicF5 no-TF/JAX fix)"
NATURAL_TAMIL_MODE = os.environ.get("NATURAL_TAMIL_MODE", "cinema_anime_character").strip().lower()
REVIEW_TIMEOUT = int(os.environ.get("REVIEW_TIMEOUT", "1200"))
MANUAL_REVIEW = os.environ.get("MANUAL_REVIEW", "0") == "1"   # 1 போட்டா மட்டும் கையால் review
TTS_BACKEND = os.environ.get("TTS_BACKEND", "indicf5").lower()   # indicf5 (தமிழ் specialist) | omnivoice
ACTIVE_TTS = None                                                # உண்மையில் ஏற்றப்பட்டது
OMNI_FP32 = os.environ.get("OMNI_FP32", "0") == "1"          # 1 = கரகரப்பு போகலன்னா float32
DEMUCS_MODEL = os.environ.get("DEMUCS_MODEL", "htdemucs_ft")   # சுத்தமான குரல் பிரிப்பு (htdemucs = வேகம்)

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

# ---- Kaggle background-run lifecycle ----
IDLE_EXIT_MIN = float(os.environ.get("IDLE_EXIT_MIN", "30"))   # இத்தனை நிமிஷம் வேலை இல்லன்னா bot நிற்கும் (GPU நேரம் மிச்சம்). 0 = நிற்காது
MAX_RUN_HOURS = float(os.environ.get("MAX_RUN_HOURS", "11"))   # Kaggle session வரம்புக்கு முன் புது வீடியோ நிறுத்தம் (மணி)
OWNER_ID = int(os.environ.get("OWNER_ID", "0") or 0)           # உன் Telegram user id: bot தொடங்கும்/நிற்கும் போது message வரும்
LAST_ACTIVE = time.time()
KNOWN_CHATS = set()
ACCEPTING = True
app = Client("tamil_dub_bot", api_id=API_ID, api_hash=API_HASH,
             bot_token=BOT_TOKEN, in_memory=True)

WHISPER = None
OMNI = None


# ---------------------------------------------------------------- models
class _IndicF5:
    """ai4bharat/IndicF5 wrapper: zero-shot voice clone (Tamil உட்பட 11 மொழி)."""
    sampling_rate = 24000

    def __init__(self, model):
        self.model = model

    def __call__(self, text, ref_path, ref_text):
        a = self.model(text, ref_audio_path=ref_path, ref_text=ref_text or "")
        if hasattr(a, "detach"):
            a = a.detach().cpu().numpy()
        a = np.asarray(a)
        if a.dtype == np.int16:
            a = a.astype(np.float32) / 32768.0
        a = a.astype(np.float32)
        return a.reshape(-1) if a.ndim > 1 else a


def load_models():
    global WHISPER, OMNI, ACTIVE_TTS
    from faster_whisper import WhisperModel

    try:
        if DEVICE != "cuda":
            raise RuntimeError("GPU இல்ல")
        WHISPER = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    except Exception as e:
        log("Whisper GPU load failed, CPU-க்கு மாறுறேன்:", e)
        WHISPER = WhisperModel("small", device="cpu", compute_type="int8")

    if TTS_BACKEND == "indicf5":
        try:
            import transformers
            from transformers import AutoModel
            log("transformers version:", transformers.__version__)
            m = None
            # 'meta device' பிழை வராம இருக்க முதல்ல low_cpu_mem_usage=False
            for kw in ({"low_cpu_mem_usage": False}, {}):
                try:
                    m = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True, **kw)
                    break
                except Exception as e:
                    log("IndicF5 load try failed", kw, repr(e)[:200])
            if m is None:
                raise RuntimeError("IndicF5 ஏத்த முடியல (transformers version சரியில்லையா?)")
            if DEVICE == "cuda":
                try:
                    m = m.to("cuda")
                except Exception as e:
                    log("IndicF5 .to(cuda) skip:", repr(e)[:100])
            OMNI = _IndicF5(m)
            ACTIVE_TTS = "indicf5"
            log("IndicF5 loaded")
        except Exception as e:
            log("IndicF5 load FAILED -> OmniVoice-க்கு மாறுறேன:", repr(e)[:300])

    if ACTIVE_TTS is None:
        from omnivoice import OmniVoice
        if DEVICE == "cuda":
            OMNI = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0",
                                             dtype=torch.float32 if OMNI_FP32 else torch.float16)
        else:
            OMNI = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cpu",
                                             dtype=torch.float32)
        ACTIVE_TTS = "omnivoice"
    log("Models loaded on", DEVICE, "| TTS:", ACTIVE_TTS)


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
         "-n", DEMUCS_MODEL, "-d", DEVICE, "-o", outdir, audio_path])
    base = os.path.splitext(os.path.basename(audio_path))[0]
    folder = os.path.join(outdir, DEMUCS_MODEL, base)
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
        st, en = float(s.start), float(s.end)
        if s.words:                           # வார்த்தை நேரம் இன்னும் துல்லியம்
            st, en = float(s.words[0].start), float(s.words[-1].end)
        segs.append({"start": st, "end": max(en, st + 0.2), "text": text})
        for w in (s.words or []):
            words.append((float(w.start), float(w.end), w.word))
    return segs, words, info.language


# ---------------------------------------------------------------- speakers / voice analysis
SR16 = 16000
FX_GAIN = float(os.environ.get("FX_GAIN", "0.85"))     # அசல் சிரிப்பு/அலறல் சத்தம்
SPK_THRESH = float(os.environ.get("SPK_THRESH", "20"))   # கூட்டினா = குறைவான ஆட்கள்
SPK_MAX = int(os.environ.get("SPK_MAX", "3"))           # ஒரு பாலினத்துக்கு max ஆட்கள்
PITCH_FIX = os.environ.get("PITCH_FIX", "0") == "1"
LEAD = 0.05                                            # டப் குரல் கொஞ்சம் முன்னாடியே தொடங்க


def to16(x, sr):
    g = gcd(SR16, sr)
    return resample_poly(x, SR16 // g, sr // g).astype(np.float32)


def voiced_f0(clip, sr=SR16, fmin=70, fmax=420):
    """குரலோட pitch (Hz) median; குரல் இல்லன்னா None."""
    if len(clip) < sr * 0.3:
        return None
    try:
        f = librosa.yin(clip, fmin=fmin, fmax=fmax, sr=sr, frame_length=1024, hop_length=256)
        r = librosa.feature.rms(y=clip, frame_length=1024, hop_length=256)[0]
    except Exception:
        return None
    n = min(len(f), len(r))
    f, r = f[:n], r[:n]
    ok = (r > np.percentile(r, 45)) & (f > fmin * 1.05) & (f < fmax * 0.95)
    if int(ok.sum()) < 6:
        return None
    return float(np.median(f[ok]))


_SPK_MODEL = None
SPK_SIM = float(os.environ.get("SPK_SIM", "0.86"))        # இதுக்கு மேல ஒத்திருந்தா = அதே ஆள்
SPK_MAX_TOTAL = int(os.environ.get("SPK_MAX_TOTAL", "8"))


def _load_spk_model():
    """உண்மையான speaker-embedding model (WavLM x-vector) - ஆள் குரல் தன்மையை பிடிக்கும்."""
    global _SPK_MODEL
    if _SPK_MODEL is None:
        from transformers import AutoFeatureExtractor, WavLMForXVector
        name = "microsoft/wavlm-base-plus-sv"
        fe = AutoFeatureExtractor.from_pretrained(name)
        m = WavLMForXVector.from_pretrained(name).to(DEVICE).float().eval()
        _SPK_MODEL = (fe, m)
    return _SPK_MODEL


def seg_embeddings(segs, voc16):
    fe, m = _load_spk_model()
    embs = []
    for s in segs:
        clip = voc16[int(s["start"] * SR16):int(min(s["end"], s["start"] + 8.0) * SR16)]
        if len(clip) < SR16 * 0.6:
            embs.append(None)
            continue
        inp = fe(clip, sampling_rate=SR16, return_tensors="pt", padding=True)
        with torch.no_grad():
            e = m(**{k: v.to(DEVICE) for k, v in inp.items()}).embeddings
        e = torch.nn.functional.normalize(e, dim=-1)[0].float().cpu().numpy()
        embs.append(e)
    return embs


def cluster_embeddings(embs, thr):
    """cosine ஒத்துமை அடிப்படையில ஆட்களை பிரிக்குது (எத்தனை பேர்னு முன்னாடியே தேவையில்ல)."""
    idx = [i for i, e in enumerate(embs) if e is not None]
    members = [[i] for i in idx]
    cent = [embs[i].copy() for i in idx]
    best_pair_sim = None
    while len(members) > 1:
        C = np.stack([c / (np.linalg.norm(c) + 1e-9) for c in cent])
        S = C @ C.T
        np.fill_diagonal(S, -1.0)
        x, y = np.unravel_index(int(np.argmax(S)), S.shape)
        best_pair_sim = float(S[x, y])
        if best_pair_sim < thr and len(members) <= SPK_MAX_TOTAL:
            break
        if x > y:
            x, y = y, x
        members[x] += members[y]
        cent[x] = np.mean([embs[i] for i in members[x]], axis=0)
        del members[y], cent[y]
    return members, best_pair_sim


_PYA = None


def pyannote_speakers(segs, vocals_path, gender):
    """pyannote (HF_TOKEN இருந்தா) - யார் எப்போ பேசுறாங்கன்னு துல்லியமா."""
    tok = os.environ.get("HF_TOKEN")
    if not tok or not vocals_path:
        return None
    global _PYA
    if _PYA is None:
        from pyannote.audio import Pipeline
        name = "pyannote/speaker-diarization-3.1"
        try:
            pipe = Pipeline.from_pretrained(name, token=tok)
        except TypeError:
            pipe = Pipeline.from_pretrained(name, use_auth_token=tok)
        if pipe is None:
            raise RuntimeError("pyannote model access இல்ல (HF-ல நிபந்தனை ஏத்துக்கணும்)")
        pipe.to(torch.device(DEVICE))
        _PYA = pipe
    out = _PYA(vocals_path)
    ann = getattr(out, "speaker_diarization", out)
    turns = [(t.start, t.end, lab) for t, _, lab in ann.itertracks(yield_label=True)]
    if not turns:
        return None
    label = {}
    for i, sg in enumerate(segs):
        ov = {}
        for a, b, lab in turns:
            o = min(b, sg["end"]) - max(a, sg["start"])
            if o > 0:
                ov[lab] = ov.get(lab, 0.0) + o
        if ov:
            label[i] = max(ov, key=ov.get)
    if not label:
        return None
    known = sorted(label)
    for i in range(len(segs)):
        if i not in label:
            label[i] = label[min(known, key=lambda j: abs(j - i))]
    spk, count, names = [None] * len(segs), {"M": 0, "F": 0}, {}
    for lab in sorted(set(label.values())):
        gs = [gender[i] for i in range(len(segs)) if label[i] == lab]
        g = "F" if gs.count("female") > len(gs) / 2 else "M"
        names[lab] = f"{g}{count[g]}"
        count[g] += 1
    for i in range(len(segs)):
        spk[i] = names[label[i]]
        gender[i] = "female" if spk[i][0] == "F" else "male"
    log(f"pyannote: {len(names)} ஆட்கள்")
    return spk


def embed_speakers(segs, voc16, gender):
    embs = seg_embeddings(segs, voc16)
    members, nxt = cluster_embeddings(embs, SPK_SIM)
    if not members:
        return None
    label = {}
    for k, mem in enumerate(members):
        for i in mem:
            label[i] = k
    known = sorted(label)
    for i in range(len(segs)):               # chinna வசனங்களுக்கு பக்கத்துல இருக்கிறவங்க label
        if i not in label:
            label[i] = label[min(known, key=lambda j: abs(j - i))]
    spk, count = [None] * len(segs), {"M": 0, "F": 0}
    names = {}
    for k in sorted(set(label.values())):
        gs = [gender[i] for i in range(len(segs)) if label[i] == k]
        g = "F" if gs.count("female") > len(gs) / 2 else "M"
        names[k] = f"{g}{count[g]}"
        count[g] += 1
    for i in range(len(segs)):
        spk[i] = names[label[i]]
        gender[i] = "female" if spk[i][0] == "F" else "male"
    log(f"speaker embeddings: {len(members)} ஆட்கள் (next-closest sim {nxt if nxt is None else round(nxt, 3)})")
    return spk


def assign_speakers(segs, voc16, vocals_path=None):
    """ஆண்/பெண் + ஆளை (speaker) கண்டுபிடிக்கும் (pitch + MFCC clustering)."""
    mfs, f0s = [], []
    for s in segs:
        clip = voc16[int(s["start"] * SR16):int(s["end"] * SR16)]
        f0 = voiced_f0(clip)
        f0s.append(f0)
        mf = None
        if len(clip) > SR16 * 0.4:
            try:
                mf = librosa.feature.mfcc(y=clip, sr=SR16, n_mfcc=13).mean(axis=1)
            except Exception:
                mf = None
        mfs.append(mf)
        s["f0s"] = [f0] if f0 else []

    known = [f for f in f0s if f]
    thr = 165.0
    if len(known) >= 6:                       # ரெண்டு குழு தெளிவா இருந்தா threshold-ஐ பொருத்து
        lo, hi = np.percentile(known, 20), np.percentile(known, 80)
        if hi / max(lo, 1) > 1.45:
            thr = float(np.sqrt(lo * hi))
    gender = []
    last = "male"
    for f in f0s:
        if f:
            last = "female" if f >= thr else "male"
            gender.append(last)
        else:
            gender.append(None)
    for i, g in enumerate(gender):            # தெரியாததுக்கு பக்கத்து வசனத்தோட
        if g is None:
            near = [gender[j] for j in range(len(gender)) if gender[j] and abs(j - i) <= 3]
            gender[i] = near[0] if near else "male"

    spk = None
    try:
        spk = pyannote_speakers(segs, vocals_path, gender)
    except Exception as e:
        log("pyannote failed, WavLM-க்கு மாறுறேன்:", repr(e)[:150])
    try:
        if spk is None:
            spk = embed_speakers(segs, voc16, gender)
    except Exception as e:
        log("speaker embedding failed, MFCC fallback:", repr(e)[:150])
    if spk is None:
        spk = [None] * len(segs)
        for g in ("male", "female"):
            idx = [i for i in range(len(segs)) if gender[i] == g]
            labels = {i: 0 for i in idx}
            withmf = [i for i in idx if mfs[i] is not None]
            if len(withmf) >= 4:
                try:
                    from sklearn.cluster import AgglomerativeClustering
                    X = np.array([mfs[i] for i in withmf])
                    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
                    cl = AgglomerativeClustering(n_clusters=None, distance_threshold=SPK_THRESH,
                                                 linkage="ward").fit(X)
                    if cl.n_clusters_ > SPK_MAX:
                        cl = AgglomerativeClustering(n_clusters=SPK_MAX, linkage="ward").fit(X)
                    for i, lab in zip(withmf, cl.labels_):
                        labels[i] = int(lab)
                    for i in idx:                  # MFCC இல்லாததுக்கு பக்கத்து label
                        if mfs[i] is None:
                            nb = [labels[j] for j in withmf if abs(j - i) <= 3]
                            labels[i] = nb[0] if nb else 0
                except Exception as e:
                    log("speaker clustering skip:", repr(e)[:100])
            for i in idx:
                spk[i] = f"{'F' if g == 'female' else 'M'}{labels[i]}"
    for i, s in enumerate(segs):
        s["speaker"], s["gender"] = spk[i], gender[i]
    log("speakers:", sorted(set(spk)))


def finalize_segs(segs, voc16):
    """merge-க்கு அப்புறம்: pitch, சத்தம் (rms), கத்துறாரா/மெதுவா பேசுறாரான்னு."""
    spk_f0, spk_rms = {}, {}
    for s in segs:
        clip = voc16[int(s["start"] * SR16):int(s["end"] * SR16)]
        s["rms"] = float(np.sqrt(np.mean(clip ** 2))) if len(clip) else 0.0
        spk_rms.setdefault(s["speaker"], []).append(s["rms"])
        for f in s.get("f0s", []):
            spk_f0.setdefault(s["speaker"], []).append(f)
    med_f0 = {k: float(np.median(v)) for k, v in spk_f0.items()}
    med_rms = {k: float(np.median(v)) for k, v in spk_rms.items() if len(v) >= 3}
    for s in segs:
        s["f0"] = float(np.median(s["f0s"])) if s.get("f0s") else med_f0.get(s["speaker"])
        m = med_rms.get(s["speaker"])
        s["delivery"] = None
        if m and m > 1e-4:
            ratio = s["rms"] / m
            if ratio > 1.8:
                s["delivery"] = "shouting"
            elif ratio < 0.4:
                s["delivery"] = "whisper"
    return med_f0


def build_ref(i, segs, voc, vsr, words, refdir, tag="ref"):
    """அதே ஆளோட குரல் மட்டும் வச்சு reference (வேற ஆள் குரல் கலக்காது)."""
    s = segs[i]
    pieces = [(s["start"], min(s["end"], s["start"] + 10.0))]
    dur = pieces[0][1] - pieces[0][0]
    if dur < 2.5:
        others = sorted((j for j in range(len(segs))
                         if j != i and segs[j]["speaker"] == s["speaker"]),
                        key=lambda j: abs(segs[j]["start"] - s["start"]))
        for j in others:
            if dur >= 4.0:
                break
            a, b = segs[j]["start"], min(segs[j]["end"], segs[j]["start"] + 4.0)
            pieces.append((a, b))
            dur += b - a
    clips, texts = [], []
    gap = np.zeros(int(0.12 * vsr), dtype=np.float32)
    for a, b in pieces:
        c = voc[int(a * vsr):int(b * vsr)]
        if len(c):
            clips += [c, gap]
            t = "".join(w for (st, en, w) in words if st >= a - 0.02 and en <= b + 0.02).strip()
            if t:
                texts.append(t)
    if not clips:
        return None, None
    clip = np.concatenate(clips)
    if len(clip) < vsr * 0.5:
        return None, None
    path = os.path.join(refdir, f"{tag}_{i}.wav")
    sf.write(path, clip, vsr)
    return path, (" ".join(texts) or None)


def build_profile(spk, segs, voc, vsr, words, refdir):
    """கேரக்டரோட 'குரல் அடையாளம்': அவரோட தெளிவான ஒரு நீளமான வசனம் (3-9 நொடி) -> ஒரே reference."""
    idx = [i for i, s in enumerate(segs) if s["speaker"] == spk]
    calm = [i for i in idx if not segs[i].get("delivery")]

    def words_in(a, b):
        return "".join(w for (st, en, w) in words if st >= a - 0.02 and en <= b + 0.02).strip()

    def save(clip, text):
        clip = hp_filter(clip, vsr)
        pk = float(np.max(np.abs(clip))) if len(clip) else 0.0
        if pk > 1e-4:
            clip = clip * (0.9 / pk)
        path = os.path.join(refdir, f"profile_{spk}.wav")
        sf.write(path, clip.astype(np.float32), vsr)
        return path, (text or None)

    for pool in (calm, idx):
        pool = sorted(pool, key=lambda i: segs[i]["end"] - segs[i]["start"], reverse=True)
        cands = []                                        # சுத்தமான + அந்த ஆளை பிரதிபலிக்கிற clip
        for i in pool[:25]:
            a, b = segs[i]["start"], min(segs[i]["end"], segs[i]["start"] + 9.0)
            if b - a >= 3.0:
                q = clip_quality(voc[int(a * vsr):int(b * vsr)], vsr) - 0.08 * abs((b - a) - 6.0)
                cands.append([q, a, b, None])
        if cands:
            cands.sort(key=lambda c: c[0], reverse=True)
            top = cands[:6]
            try:
                for c in top:
                    c[3] = voice_emb(voc[int(c[1] * vsr):int(c[2] * vsr)], vsr)
                embs = [c[3] for c in top if c[3] is not None]
                if len(embs) >= 3:
                    cen = np.mean(embs, axis=0)
                    cen = cen / (np.linalg.norm(cen) + 1e-9)
                    for c in top:
                        if c[3] is not None:
                            c[0] += 2.0 * float(np.dot(c[3], cen))
            except Exception as e:
                log("profile centrality skip:", repr(e)[:100])
            best = max(top, key=lambda c: c[0])
            a, b = best[1], best[2]
            return save(voc[int(a * vsr):int(b * vsr)], words_in(a, b))
        for i in pool:                                    # ஒரே நீளமான clip
            a, b = segs[i]["start"], min(segs[i]["end"], segs[i]["start"] + 9.0)
            if b - a >= 2.5:
                return save(voc[int(a * vsr):int(b * vsr)], words_in(a, b))
        clips, texts, total = [], [], 0.0                 # இல்லன்னா சின்னதுகளை சேர்த்து
        gap = np.zeros(int(0.15 * vsr), dtype=np.float32)
        for i in pool:
            a, b = segs[i]["start"], min(segs[i]["end"], segs[i]["start"] + 5.0)
            if b - a < 0.6:
                continue
            clips += [voc[int(a * vsr):int(b * vsr)], gap]
            total += b - a
            t = words_in(a, b)
            if t:
                texts.append(t)
            if total >= 5.0:
                break
        if total >= 1.0:
            return save(np.concatenate(clips), " ".join(texts))
    return None, None


def loudnorm_file(path, sr):
    """எல்லா track-உம் ஒரே சத்த அளவுல (ஒலிம்பிக் மாதிரி: -16 LUFS)."""
    tmp = path + ".ln.wav"
    try:
        run(["ffmpeg", "-y", "-i", path, "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
             "-ar", str(sr), tmp])
        os.replace(tmp, path)
    except Exception as e:
        log("loudnorm skip:", repr(e)[:100])


def pitch_check(wav, sr, target_f0, tol_semi=3.5):
    """உருவாக்கின குரல் pitch, அசல் ஆளோட pitch-க்கு பக்கத்துல இருக்கான்னு (ஆண்/பெண் மாறிடுச்சான்னு)."""
    if not target_f0:
        return 0.0
    f = voiced_f0(to16(wav, sr))
    if not f:
        return 0.0
    return float(12 * np.log2(target_f0 / f))


def fix_pitch(wav, sr, semi, tol_semi=3.5):
    if abs(semi) <= tol_semi:
        return wav
    shift = float(np.clip(semi, -12, 12))
    try:
        return librosa.effects.pitch_shift(wav, sr=sr, n_steps=shift).astype(np.float32)
    except Exception as e:
        log("pitch shift failed:", repr(e)[:100])
        return wav


def trim_silence(wav, sr):
    try:
        w, _ = librosa.effects.trim(wav, top_db=38)
        if len(w) > sr * 0.15:
            wav = w
    except Exception:
        pass
    f = min(int(0.01 * sr), len(wav) // 4)
    if f > 0:
        wav = wav.copy()
        wav[:f] *= np.linspace(0, 1, f, dtype=np.float32)
    return wav



def merge_segments(segs, max_gap=0.7, max_dur=7.0, max_chars=170):
    """சின்ன சின்ன வசனங்களை ஒண்ணா சேர்த்து, ஒரே ஓட்டத்துல (flow) பேச வைக்க."""
    out = []
    for s in segs:
        if out:
            p = out[-1]
            gap = s["start"] - p["end"]
            ends_sentence = p["text"].rstrip()[-1:] in ".?!…。"
            if (gap <= max_gap and s.get("speaker") == p.get("speaker")
                    and (s["end"] - p["start"]) <= max_dur
                    and len(p["text"]) + len(s["text"]) + 1 <= max_chars
                    and (not ends_sentence or gap <= 0.35)):
                p["end"] = s["end"]
                p["text"] = p["text"].rstrip() + " " + s["text"].lstrip()
                p["f0s"] = p.get("f0s", []) + s.get("f0s", [])
                continue
        out.append({**s, "f0s": list(s.get("f0s", []))})
    return out


def add_slots(segs, total):
    """ஒவ்வொரு வசனத்துக்கும் அதிகபட்சம் எவ்வளவு நேரம் பேசலாம்னு கணக்கு."""
    for i, s in enumerate(segs):
        dur = max(0.3, s["end"] - s["start"])
        nxt = segs[i + 1]["start"] if i + 1 < len(segs) else total
        avail = nxt - s["start"] - 0.06
        slot = min(avail, dur * 1.6 + 0.4)
        slot = max(slot, min(dur, avail))
        s["slot"] = max(0.4, slot)
        s["hard"] = max(s["slot"], avail)      # அடுத்த வசனம் தொடங்குற வரை overflow ஆகலாம்


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


STYLE = {
    "ta": ("Write like a modern Tamil FILM DIALOGUE writer: everyday COLLOQUIAL spoken Tamil "
           "(பேச்சு வழக்கு, Chennai/Madurai street-casual), the way friends talk. "
           "NEVER write literary/formal Tamil (செந்தமிழ்): no endings like -கிறேன், -கிறோம், -கிறாய், "
           "-கிறீர்கள், -கிறார்கள், -கிறது; no நாங்கள், நீங்கள், அவர்கள், இங்கு, அங்கு, எங்கு, "
           "இப்போது, மிகவும், வேண்டும், இல்லை. Use நாங்க, நீங்க, அவங்க, இங்க, அங்க, எங்க, "
           "இப்போ, ரொம்ப, வேணும், இல்ல instead. Use short punchy sentences, natural fillers "
           "(டா, டி, ப்பா, சரி, அட, ஏய், பாரு, விடு) when the tone fits. "
           "Style examples (original, in the spirit of Tamil cinema):\n"
           "[casual] 'We have come here.' -> 'நாங்க இங்க வந்திருக்கோம்.'\n"
           "[casual] 'I am coming.' -> 'நான் வந்துட்டே இருக்கேன்.'\n"
           "[casual] 'What are you doing?' -> 'என்ன பண்ற?'\n"
           "[casual] 'Do not worry about it.' -> 'அதப் பத்தி கவலைப்படாத.'\n"
           "[friends] 'I do not know anything.' -> 'எனக்கு ஒண்ணுமே தெரியாதுப்பா.'\n"
           "[friends] 'Why are you shouting at me?' -> 'ஏன் என்மேல கத்துற?'\n"
           "[friends] 'Come on, let us go.' -> 'வா டா, போலாம்.'\n"
           "[angry] 'Get out of my sight right now.' -> 'முதல்ல என் கண்ணு முன்னாடி நிக்காத, போ!'\n"
           "[angry] 'Who gave you the right to decide?' -> 'முடிவு பண்ண உனக்கு யாரு உரிமை குடுத்தா?'\n"
           "[emotional] 'I missed you so much.' -> 'உன்ன ரொம்ப மிஸ் பண்ணேன் டா.'\n"
           "[emotional] 'Please forgive me.' -> 'என்ன மன்னிச்சிடு, ப்ளீஸ்.'\n"
           "[serious] 'We have no time left.' -> 'நம்ம கிட்ட நேரமே இல்ல.'\n"
           "[serious] 'They are coming for us.' -> 'அவங்க நம்மள தேடி வராங்க.'\n"
           "[revenge] 'I hold a grudge against you.' -> 'நான் உனக்கு வஞ்சம் வச்சிருக்கேன்.'\n"
           "[revenge] 'Because I hold a grudge, I am going to destroy you.' -> "
           "'உன்மேல வஞ்சம் வச்சிருக்கிறதால உன்ன அழிக்கப் போறேன்.'\n"
           "[revenge] 'I will not leave you alone.' -> 'உன்ன சும்மா விடமாட்டேன்.'\n"
           "[daily] 'Did you eat?' -> 'சாப்பிட்டியா?'  'I ate.' -> 'சாப்பிட்டேன்.'\n"
           "[daily] 'What am I doing now?' -> 'இப்போ என்ன பண்ணுறேன்?'\n"
           "[daily] 'I have come only here.' -> 'நான் இங்கதான் வந்திருக்கேன்.'\n"
           "Never use bookish forms like வைத்துள்ளேன், வந்துள்ளேன், செய்துள்ளேன், வைத்துக்கொள்கிறேன், "
           "வந்தனர், அழிக்கப் போகிறேன்; say வச்சிருக்கேன், வந்திருக்கேன், பண்ணியிருக்கேன், வச்சுக்கிறேன், "
           "வந்தாங்க, அழிக்கப் போறேன். "
           "Spoken Tamil also DROPS the accusative -ஐ (உலகத்தை -> உலகத்த, மரத்தை -> மரத்த, "
           "துணியை -> துணிய), says -றது not -வது (போவது -> போறது, திரிவது -> சுத்தறது), "
           "uses '-னு' for 'என்று' (வருவேன்னு), says தொறக்க not திறக்க, வேற not வேறு, "
           "adverbs end in -ா (சரியா, சீக்கிரமா), and avoids passive -ப்படும் (use active). "
           "[scene] 'They are trying to open the door to the underworld.' -> "
           "'அவங்க பாதாள உலகத்துக்கு கதவ தொறக்க பாக்குறாங்க.'\n"
           "Do NOT translate sentence by sentence like a textbook. Say it the way a real person says it "
           "in daily life: drop obvious subjects, trail off, keep it short and a bit rough, avoid stiff "
           "full sentences and stiff words like இங்கே/அங்கே (say இங்க/அங்க). "
           "[casual] 'Let them stay there.' -> 'அவங்க அங்கயே இருக்கட்டும் விடு.'\n"
           "[casual] 'There is no need for you to come.' -> 'நீ வர வேணாம் விடு.'\n"
           "[casual] 'Where did you go all this time?' -> 'இவ்ளோ நேரம் எங்க போன?'\n"
           "Always use these casual verb forms: பண்ணுறேன்/பண்ணுற/பண்ணுறீங்க, சாப்பிட்டியா, "
           "வர்றேன், போறேன், சொல்றேன், பார்க்குறேன். Never write செய்கிறேன்/வருகிறேன்/சாப்பிட்டாயா."),
    "te": ("Use everyday colloquial spoken Telugu (వ్యావహారిక భాష) like friends talking, "
           "not bookish/formal Telugu."),
    "ml": ("Use everyday colloquial spoken Malayalam (സംസാരഭാഷ) like friends talking, "
           "not bookish/formal Malayalam."),
    "kn": ("Use everyday colloquial spoken Kannada (ಆಡುಮಾತು) like friends talking, "
           "not bookish/formal Kannada."),
    "hi": ("Use everyday casual spoken Hindi (Hinglish-free, Devanagari) like friends talking, "
           "avoid heavy Sanskritised words."),
}


FLOW_RULES = (
    "The lines are CONSECUTIVE dialogue from one video scene. Translate them as ONE continuous, "
    "flowing conversation, not as isolated sentences: join ideas with natural connectors "
    "(அதான், அப்போ, சரி, அது தான், ஆனா...) so it sounds like real people talking without stiff pauses. "
    "Infer each speaker's character from the lines (age, gender, background, mood, relationship: "
    "kids, elders, street/rowdy, college friends, family, polite officer, narrator) and adapt the "
    "register to that character; even a narrator must sound conversational, never bookish. "
    "Flow example: 'We came all this way, and this is the place.' -> "
    "'இவ்ளோ தூரம் வந்தது இதுக்குத்தான், இதுதான் அந்த இடம்.'"
)


STORY_BRIEF = ""


def set_brief(t):
    global STORY_BRIEF
    STORY_BRIEF = t or ""


def gemini_text(system, user):
    """JSON இல்லாத சாதாரண text பதில் (கதை சுருக்கத்துக்கு)."""
    models = [_GEM_MODEL] if _GEM_MODEL else _gem_candidates()
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.3}}
    for m in models[:8]:
        try:
            r = httpx.post(f"{_GEM_BASE}/models/{m}:generateContent",
                           headers=_gem_headers(), json=body, timeout=90)
        except httpx.HTTPError:
            continue
        if r.status_code in _SKIP_CODES:
            continue
        if r.status_code != 200:
            break
        try:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            continue
    return ""


def story_brief(segs, src_lang):
    """முழு உரையாடலை படிச்சு: கேரக்டர்கள், உறவு, பெயர்கள், tone -> மொழிபெயர்ப்புக்கு வழிகாட்டி."""
    if not GEMINI_KEY:
        return ""
    lines, size = [], 0
    for i, s in enumerate(segs):
        ln = f"{s.get('speaker')}({s.get('gender')}): {s['text']}"
        size += len(ln)
        if size > 14000:
            break
        lines.append(ln)
    system = ("You are a film dubbing director preparing a Tamil dub. Read the dialogue and write a SHORT "
              "brief (max 1100 characters, English): 1) each speaker id with gender, approx age, role and "
              "how they talk (rowdy, polite, elderly, childish, villain...). 2) relationships and how they "
              "address each other (டா, டி, அண்ணா, அம்மா, சார், மச்சான்...). 3) character/place names with "
              "their Tamil-script spelling. 4) genre and overall tone. No preamble.")
    try:
        return gemini_text(system, "SOURCE LANGUAGE: " + (src_lang or "auto") + "\n" + "\n".join(lines))[:1300]
    except Exception as e:
        log("story brief failed:", repr(e)[:100])
        return ""


TA_DIALOGUE = (
    "TAMIL DIALOGUE STYLE (MANDATORY): This is a dubbed movie/anime, NOT a book, news report, "
    "school essay or formal translation. Write the way real Tamil actors speak naturally in daily "
    "conversation. Prefer spoken forms: நாங்க, நீங்க, அவங்க, இங்க, அங்க, இப்போ, அப்போ, இல்ல, "
    "வேணும், போறேன், வர்றேன், சொல்றேன், பண்ணுறேன், தெரியல, புரியல, என்னாச்சு, அப்படியா. "
    "Avoid literary forms: நாங்கள், நீங்கள், அவர்கள், இங்கு, அங்கு, தற்போது, மிகவும், வேண்டும், "
    "இல்லை, செய்கிறேன், கூறினான், என்றான், ஆகையால், இருப்பினும், மேலும். Do not mechanically "
    "swap words; rewrite the whole sentence if needed so it sounds like spoken dialogue. Use short, "
    "punchy, speakable lines, natural pauses and contractions. Preserve meaning; add no plot facts.\n"
    "CHARACTER ACTING: the same speaker id must retain a consistent personality and way of addressing "
    "others. Match age, attitude, relationship and gender. A hot-headed teen can be blunt; a villain "
    "can be cold and sharp; a child can sound simple; an elderly character can be respectful without "
    "sounding literary. For anime, retain energetic reactions, dramatic punch, humour and catchphrases "
    "naturally in Tamil; do not make every character sound alike.\n"
    "Examples: 'What are you doing?' -> 'என்ன பண்ணுற?', 'I don't know' -> 'எனக்குத் தெரியல', "
    "'We have to go now' -> 'நாம இப்போ போகணும்'. Examples guide style, not fixed translations.\n"
)


def _gem_prompt(items, src_lang, tgt="ta"):
    """(system instruction, user text)"""
    _, name, _, script, cps = LANGS[tgt]
    payload = []
    for k, s_ in enumerate(items):
        d = {"id": k, "text": s_["text"], "max_chars": max(10, int(s_["slot"] * cps))}
        if s_.get("speaker"):
            d["speaker"] = s_["speaker"]
            d["speaker_gender"] = s_.get("gender")
        if s_.get("delivery"):
            d["delivery"] = s_["delivery"]
        payload.append(d)
    system = (
        f"You are a professional {name} cinema and anime dubbing dialogue writer. Translate each line "
        f"(source language: {src_lang or 'auto'}) into {name} ({script} script) for dubbing.\n"
        + STYLE.get(tgt, "Use casual spoken style.") + "\n" + FLOW_RULES + "\n"
        "Each input may carry 'speaker' (same id = same person) and 'speaker_gender': keep each "
        "speaker's voice consistent and use the correct gendered forms for the SPEAKER (important in "
        "Hindi, Telugu, Kannada, Malayalam). 'delivery': 'shouting' -> write it as a shout: short, "
        "emphatic, ending with '!'; 'whisper' -> soft and short with '...'. Keep the original "
        "punctuation and emotion marks (! ? …) because they control the voice's emotion.\n"
        + (TA_DIALOGUE if tgt == "ta" else "") +
        "Rules: keep meaning and emotion; each translation MUST be at most max_chars characters "
        f"(shorten the wording if needed, never exceed it); write names in {script} script; write "
        f"numbers as {name} words; no English letters, no emojis, no explanations.\n"
        "Return ONLY a JSON array of strings, same order and same count as the input.")
    if STORY_BRIEF:
        system += ("\n\nSCENE BRIEF (use it so every character keeps a consistent voice, names and "
                   "forms of address):\n" + STORY_BRIEF)
    user = "INPUT:\n" + json.dumps(payload, ensure_ascii=False)
    return system, user


def gemini_batch(items, src_lang, tgt="ta", prompt=None):
    global _GEM_MODEL
    system, user = prompt or _gem_prompt(items, src_lang, tgt)
    body = {"systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.5,
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


_TA = r"(?<![\u0B80-\u0BFF])"
_TA_END = r"(?![\u0B80-\u0BFF])"
_WORDS = {
    "நாங்கள்": "நாங்க", "நீங்கள்": "நீங்க", "அவர்கள்": "அவங்க", "இவர்கள்": "இவங்க",
    "எங்களுக்கு": "எங்களுக்கு", "அவர்களுக்கு": "அவங்களுக்கு", "அவர்களை": "அவங்கள",
    "இங்கு": "இங்க", "அங்கு": "அங்க", "எங்கு": "எங்க", "இப்போது": "இப்போ",
    "அப்போது": "அப்போ", "எப்போது": "எப்போ", "மிகவும்": "ரொம்ப", "மிக": "ரொம்ப",
    "வேண்டும்": "வேணும்", "வேண்டாம்": "வேணாம்", "இல்லை": "இல்ல", "ஆமாம்": "ஆமா",
    "என்னுடைய": "என்னோட", "உன்னுடைய": "உன்னோட", "அவருடைய": "அவரோட",
    "உங்களுடைய": "உங்களோட", "எங்களுடைய": "எங்களோட", "அவளுடைய": "அவளோட",
    "என்ன செய்கிறாய்": "என்ன பண்ணுற", "என்ன செய்கிறீர்கள்": "என்ன பண்ணுறீங்க",
    "செய்கிறேன்": "பண்ணுறேன்", "செய்கிறோம்": "பண்ணுறோம்", "செய்கிறாய்": "பண்ணுற",
    "செய்கிறீர்கள்": "பண்ணுறீங்க", "செய்கிறார்கள்": "பண்ணுறாங்க", "செய்கிறான்": "பண்ணுறான்",
    "போகிறேன்": "போறேன்", "போகிறோம்": "போறோம்", "போகிறாய்": "போற",
    "போகிறீர்கள்": "போறீங்க", "போகிறார்கள்": "போறாங்க", "போகிறான்": "போறான்",
    "வருகிறேன்": "வரேன்", "வருகிறோம்": "வரோம்", "வருகிறாய்": "வர்ற",
    "வருகிறீர்கள்": "வரீங்க", "வருகிறார்கள்": "வராங்க", "வருகிறான்": "வரான்",
    "சொல்கிறேன்": "சொல்றேன்", "சொல்கிறாய்": "சொல்ற", "சொல்கிறீர்கள்": "சொல்றீங்க",
    "சொல்கிறார்கள்": "சொல்றாங்க", "பேசுகிறேன்": "பேசுறேன்", "பேசுகிறாய்": "பேசுற",
    "பேசுகிறீர்கள்": "பேசுறீங்க", "பேசுகிறார்கள்": "பேசுறாங்க",
}
_WORDS.update({
    "இங்கே": "இங்க", "அங்கே": "அங்க", "எங்கே": "எங்க", "இங்கேயே": "இங்கயே",
    "அங்கேயே": "அங்கயே", "எங்கேயும்": "எங்கயும்", "இங்கேயும்": "இங்கயும்",
    "அங்கேயும்": "அங்கயும்", "என்னவென்று": "என்னன்னு", "ஏனெனில்": "ஏன்னா",
    "திறக்க": "தொறக்க", "திறந்து": "தொறந்து", "திறக்கும்": "தொறக்கும்", "திறந்த": "தொறந்த",
    "வேறு": "வேற", "அல்லது": "இல்லனா", "மீண்டும்": "மறுபடியும்", "நிச்சயமாக": "கண்டிப்பா",
    "செய்வது": "பண்றது", "போவது": "போறது", "வருவது": "வர்றது", "சொல்வது": "சொல்றது",
    "பேசுவது": "பேசுறது", "இருப்பது": "இருக்கறது", "எடுப்பது": "எடுக்கறது",
    "திரிவது": "சுத்தறது", "பிடிக்கவில்லை": "பிடிக்கல", "தெரியவில்லை": "தெரியல",
    "முடியவில்லை": "முடியல", "புரியவில்லை": "புரியல",
    "உள்ளது": "இருக்கு", "உள்ளேன்": "இருக்கேன்", "உள்ளோம்": "இருக்கோம்", "உள்ளாய்": "இருக்க",
    "உள்ளீர்கள்": "இருக்கீங்க", "உள்ளார்கள்": "இருக்காங்க", "உள்ளான்": "இருக்கான்",
    "உள்ளாள்": "இருக்கா", "உள்ளார்": "இருக்காரு",
    "செய்துள்ளேன்": "பண்ணியிருக்கேன்", "செய்துள்ளோம்": "பண்ணியிருக்கோம்",
    "செய்துள்ளார்கள்": "பண்ணியிருக்காங்க", "செய்துள்ளான்": "பண்ணியிருக்கான்",
    "செய்துள்ளது": "பண்ணியிருக்கு", "செய்துள்ளாய்": "பண்ணியிருக்க",
    "நாம்": "நாம", "அவ்வளவு": "அவ்ளோ", "இவ்வளவு": "இவ்ளோ", "எவ்வளவு": "எவ்ளோ",
    "ஏனென்றால்": "ஏன்னா", "ஆனால்": "ஆனா", "மற்றும்": "அப்புறம்", "எனவே": "அதனால",
    "தயவுசெய்து": "ப்ளீஸ்", "என்னை": "என்ன", "உன்னை": "உன்ன", "அவனை": "அவன",
    "அவளை": "அவள", "உங்களை": "உங்கள", "எங்களை": "எங்கள", "நம்மை": "நம்மள",
    "அதை": "அத", "இதை": "இத", "எதை": "எத", "அவரை": "அவர", "என்றால்": "னா",
    "கிடையாது": "கிடையாது", "தான்": "தான்", "அல்லவா": "இல்லையா", "பிறகு": "அப்புறம்",
})
_PERF_STEM = {"த்த": "ச்சி", "ந்த": "ந்தி", "ட்ட": "ட்டி"}
_PERF_END = {"ேன்": "கேன்", "ோம்": "கோம்", "ாய்": "க", "ீர்கள்": "கீங்க", "ார்கள்": "காங்க",
             "ான்": "கான்", "ாள்": "கா", "ார்": "காரு", "து": "கு"}
_PERF_RE = re.compile("(த்த|ந்த|ட்ட)ுள்ள(ேன்|ோம்|ாய்|ீர்கள்|ார்கள்|ான்|ாள்|ார்|து)" + _TA_END)
_PLURAL_RE = re.compile("(ந்த|ட்ட|ன்ற|த்த)னர்" + _TA_END)
_REFLEX_RE = re.compile("த்துக்\\s?கொள்கிற(ேன்|ோம்|ாய்|ீர்கள்|ார்கள்|ான்|ாள்|ார்)" + _TA_END)
_VERB_STEM = {"க்": "க்குற", "ு": "ுற", "ல்": "ல்ற", "ள்": "ள்ற", "ய்": "ய்ற"}
_VERB_END = {"ேன்": "ேன்", "ோம்": "ோம்", "ாய்": "", "ீர்கள்": "ீங்க", "ார்கள்": "ாங்க",
             "ான்": "ான்", "ாள்": "ா", "ார்": "ாரு"}
_VERB_RE = re.compile("(க்|ு|ல்|ள்|ய்)கிற(ேன்|ோம்|ாய்|ீர்கள்|ார்கள்|ான்|ாள்|ார்|து)" + _TA_END)


_ACC = [("த்தை", "த்த"), ("ியை", "ிய"), ("ையை", "ைய"), ("ங்களை", "ங்கள"), ("ர்களை", "ர்கள")]
_ACC_RE = [(re.compile(a + "(?=(?:ப்|த்|க்|ச்)?" + _TA_END + ")"), b) for a, b in _ACC]
_ENRU_RE = re.compile("\\s*என்று" + _TA_END)
_ADV_RE = re.compile("(?<=[\u0B80-\u0BFF]{2})ாக" + _TA_END)


def _perf_sub(m):
    return _PERF_STEM[m.group(1)] + "ருக்" + _PERF_END[m.group(2)]


def _reflex_sub(m):
    end = {"ேன்": "ேன்", "ோம்": "ோம்", "ாய்": "", "ீர்கள்": "ீங்க", "ார்கள்": "ாங்க",
           "ான்": "ான்", "ாள்": "ா", "ார்": "ாரு"}[m.group(1)]
    return "ச்சுக்கிற" + end if end else "ச்சுக்கிற"


def _verb_sub(m):
    stem, end = m.group(1), m.group(2)
    if end == "து":
        return {"க்": "க்குது", "ு": "ுது", "ல்": "ல்லுது", "ள்": "ள்ளுது", "ய்": "ய்யுது"}[stem]
    return _VERB_STEM[stem] + _VERB_END[end]


_ENDINGS = [   # வினை முடிவுகள் (suffix)
    ("ட்டாயா", "ட்டியா"), ("ந்தாயா", "ந்தியா"), ("ட்டீர்களா", "ட்டீங்களா"),
    ("னாய்", "ன"), ("ந்தாய்", "ந்த"), ("ட்டாய்", "ட்ட"),
    ("ருக்கிறேன்", "ருக்கேன்"), ("ருக்கிறோம்", "ருக்கோம்"), ("ருக்கிறாய்", "ருக்க"),
    ("ருக்கிறீர்கள்", "ருக்கீங்க"), ("ருக்கிறார்கள்", "ருக்காங்க"),
    ("ருக்கிறார்", "ருக்காரு"), ("ருக்கிறாள்", "ருக்கா"), ("ருக்கிறான்", "ருக்கான்"),
    ("ருக்கிறது", "ருக்கு"), ("ருக்கின்றன", "ருக்கு"),
]
_WORD_RE = re.compile(_TA + "(" + "|".join(sorted(map(re.escape, _WORDS), key=len, reverse=True)) + ")" + _TA_END)


def colloquial_ta(text):
    """செந்தமிழ் சொற்களை பேச்சு வழக்குக்கு மாத்துது (Gemini விட்டுப்போனதையும் பிடிக்கும்)."""
    if not text:
        return text
    t = _WORD_RE.sub(lambda m: _WORDS[m.group(1)], text)
    for a, b in _ENDINGS:
        t = re.sub(a + _TA_END, b, t)
    for rx, b in _ACC_RE:                       # 'உலகத்தை' -> 'உலகத்த' (பேச்சுல -ஐ இல்ல)
        t = rx.sub(b, t)
    t = _ENRU_RE.sub("னு", t)                   # 'வருவேன் என்று' -> 'வருவேன்னு'
    t = _ADV_RE.sub("ா", t)                     # 'சரியாக' -> 'சரியா'
    t = _REFLEX_RE.sub(_reflex_sub, t)
    t = _PERF_RE.sub(_perf_sub, t)
    t = _PLURAL_RE.sub(lambda m: m.group(1) + "ாங்க", t)
    t = re.sub("கின்ற(ேன்|ோம்|ாய்|ீர்கள்|ார்கள்|ான்|ாள்|ார்)", "கிற\\1", t)
    t = _VERB_RE.sub(_verb_sub, t)
    return t


_FORMAL_RE = re.compile(
    "கிற(?:ேன்|ோம்|ாய்|ீர்கள்|ார்கள்|ான்|ாள்|ார்|து)|நாங்கள்|நீங்கள்|இங்கு|அங்கு|வேண்டும்|இல்லை"
    "|அவர்கள்|மிகவும்|ுள்ள(?:ேன்|ோம்|ாய்|ீர்கள்|ார்கள்|ான்|ாள்|ார்|து)|கின்ற|னர்\\b|கொள்கிற|வேண்டிய"
    "|இவ்வாறு|அவ்வாறு|எவ்வாறு|ஆயினும்|மேலும்|இருப்பினும்")
ALWAYS_REWRITE = os.environ.get("ALWAYS_REWRITE", "1") == "1"


def gemini_colloquialize(lines, tgt="ta"):
    """இன்னும் செந்தமிழ்ல இருக்கிற வரிகளை மட்டும் பேச்சு வழக்குக்கு திருப்பி எழுத (ஒரு முறை)."""
    system = ("You are the final dialogue polish editor for a Tamil movie/anime dub. Rewrite every "
              "line into believable everyday spoken Tamil (பேச்சுத் தமிழ்), like trained Tamil dubbing "
              "artists naturally speak, not written or literary Tamil. Preserve meaning, speaker "
              "personality, relationship, gender, emotion and movie/anime dramatic energy. Keep each "
              "speaker's individual voice: blunt, shy, funny, arrogant, childish, villainous, respectful "
              "or elderly as context indicates. Prefer நாங்க, நீங்க, அவங்க, இங்க, அங்க, இப்போ, இல்ல, "
              "வேணும், போறேன், வர்றேன், சொல்றேன், பண்ணுறேன், தெரியல, புரியல. Avoid நாங்கள், நீங்கள், "
              "அவர்கள், இங்கு, அங்கு, தற்போது, மிகவும், வேண்டும், இல்லை, செய்கிறேன், என்றான், ஆகையால், "
              "இருப்பினும். Do not just swap words; rebuild awkward sentences as short, speakable "
              "dialogue. Do not add plot details or forced slang. Keep punctuation and roughly the same "
              "length (never longer). If already natural, keep it. " + STYLE["ta"] +
              "\nReturn ONLY a JSON array of strings, same order and count.")
    user = "INPUT:\n" + json.dumps(lines, ensure_ascii=False)
    items = [{"text": t, "slot": 3.0} for t in lines]
    return gemini_batch(items, "ta", tgt, prompt=(system, user))


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
        if tgt == "ta":
            res = [colloquial_ta(x) for x in res]
            bad = [k for k, x in enumerate(res)
                   if x and (ALWAYS_REWRITE or _FORMAL_RE.search(x))]
            if bad and GEMINI_KEY:
                try:
                    fixed = gemini_colloquialize([res[k] for k in bad], tgt)
                    for k, t in zip(bad, fixed):
                        if t and t.strip():
                            res[k] = colloquial_ta(t.strip())
                    log(f"செந்தமிழ் வரிகள் திருத்தினேன்: {len(bad)}")
                except Exception as e:
                    log("colloquial rewrite skip:", str(e)[:100])
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


_SEED = None


def set_seed(v):
    global _SEED
    _SEED = v


def synth(text, ref_path, ref_text, duration=None, lang="ta"):
    if _SEED is not None:                    # அதே ஆளுக்கு எப்போவும் அதே குரல் வர
        torch.manual_seed(_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_SEED)
    if ACTIVE_TTS == "indicf5":
        # IndicF5-க்கு duration/language parameter இல்ல; நேரம் பொருத்துறத fit_to_slot பாத்துக்கும்
        if not ref_path:
            raise RuntimeError("IndicF5-க்கு reference குரல் வேணும்")
        return OMNI(text, ref_path, ref_text)
    kw = dict(text=text, language=lang, num_step=NUM_STEP, pad_duration=0.0)
    if ref_path:
        kw["ref_audio"] = ref_path
        if ref_text:
            kw["ref_text"] = ref_text
    if duration:
        kw["duration"] = float(duration)
    audio = OMNI.generate(**kw)
    return np.asarray(audio[0], dtype=np.float32)


def fit_to_slot(wav, sr, slot, text, ref_path, ref_text, lang="ta", target_rms=None, hard=None):
    limit = max(slot, hard or slot)
    L = len(wav) / sr
    # முதல்ல வார்த்தையை சுருக்கு (வேகப்படுத்தினா ரோபோ மாதிரி ஆகும்)
    if L > limit * 1.15:
        short = shorten_line(text, int(len(text) * limit / L * 0.95), lang)
        if short:
            try:
                w2 = trim_silence(synth(short, ref_path, ref_text, lang=lang), sr)
                if 0 < len(w2) < len(wav):
                    wav, text = w2, short
                    L = len(wav) / sr
            except Exception as e:
                log("shorten regen failed:", repr(e)[:100])
    if L > limit:
        rate = L / limit
        if rate <= 1.15:
            wav = librosa.effects.time_stretch(wav, rate=rate)
        elif ACTIVE_TTS == "indicf5":              # IndicF5-ல duration இல்ல -> மெதுவா மட்டும் வேகப்படுத்து
            wav = librosa.effects.time_stretch(wav, rate=min(rate, 1.25))
        else:
            try:
                w2 = synth(text, ref_path, ref_text, duration=limit * 0.97, lang=lang)
                if len(w2) > 0:
                    wav = w2
            except Exception as e:
                log("duration regen failed:", repr(e))
            L = len(wav) / sr
            if L > limit:
                wav = librosa.effects.time_stretch(wav, rate=min(L / limit, 1.25))
    slot = limit
    maxlen = int(slot * sr)
    if len(wav) > maxlen:
        wav = wav[:maxlen].copy()
        f = min(int(0.04 * sr), len(wav))
        wav[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
    # சத்தம் சமநிலை
    rms = float(np.sqrt(np.mean(wav ** 2))) + 1e-8
    if target_rms and target_rms > 1e-4:      # அசல் சத்தத்துக்கு ஏத்த மாதிரி (கத்தினா சத்தமா)
        wav = wav * float(np.clip(target_rms / rms, 0.25, 4.0))
    else:
        wav = wav * min(4.0, 0.08 / rms)
    peak = float(np.max(np.abs(wav))) if len(wav) else 0
    if peak > 0.95:
        wav = wav * (0.95 / peak)
    return wav


# ---------------------------------------------------------------- AUTO QA (நீ பார்க்க வேண்டாம், bot-ஏ பாக்கும்)
AUTO_REVIEW = os.environ.get("AUTO_REVIEW", "1") == "1"        # தமிழ் வசனத்தை தானா சரிபார்க்க
VERIFY_TRIES = int(os.environ.get("VERIFY_TRIES", "4"))         # ஒரு வசனத்துக்கு max முயற்சி
VOICE_SIM_MIN = float(os.environ.get("VOICE_SIM_MIN", "0.75"))  # கேரக்டர் குரல் ஒற்றுமை (0-1)
TEXT_SIM_MIN = float(os.environ.get("TEXT_SIM_MIN", "0.50"))    # சொன்னது = எழுதினது ஒற்றுமை (0-1)
_REF_EMB = {}


def auto_review(segs, texts, src_lang, tgt="ta"):
    """Gemini 2-ஆவது முறை: அசல் வசனம் vs தமிழ் வசனம் ஒப்பிட்டு, தப்பு / ஆண்-பெண் வடிவம் / நீளம் திருத்தும்."""
    if not GEMINI_KEY:
        return texts
    _, name, _, script, cps = LANGS[tgt]
    system = (
        f"You are a strict {name} dubbing QA editor. For each item compare 'source' with the current "
        f"'{name.lower()}' line and FIX only real problems: wrong or missing meaning, mistranslated "
        f"words, wrong gender/respect forms for the speaker, formal/bookish {name} (make it natural "
        "spoken style), English letters, or a line longer than max_chars (shorten it). Keep the "
        "speaker's personality and the original punctuation/emotion marks. If the line is already "
        "good, return it UNCHANGED. If it is empty, translate it.\n" + STYLE.get(tgt, "")
        + "\nReturn ONLY a JSON array of strings, same order and same count as the input.")
    if tgt == "ta":
        system += (
            "\nNATURALNESS (very important): the line must sound like a real person talking in a Tamil "
            "film, NOT like an AI assistant or a written essay. Use short, broken, spoken sentences, "
            "natural interjections and fillers (அட, டேய், ஏய், சரி, ஆமா, இல்ல, என்னது), drop needless "
            "pronouns, avoid polite textbook phrases, long complete sentences and bookish connectors. "
            "Match each speaker's age, attitude and emotion, and keep a natural flow with the 'prev' "
            "and 'next' lines (they are context only, never translate them).")
    if STORY_BRIEF:
        system += "\n\nSCENE BRIEF:\n" + STORY_BRIEF
    out = list(texts)
    changed = 0
    B = 20
    for i in range(0, len(segs), B):
        idx = list(range(i, min(i + B, len(segs))))
        payload = []
        for k in idx:
            d = {"id": k, "speaker": segs[k].get("speaker"), "gender": segs[k].get("gender"),
                 "prev": segs[k - 1]["text"] if k > 0 else "",
                 "next": segs[k + 1]["text"] if k + 1 < len(segs) else "",
                 "source": segs[k]["text"], "current": out[k],
                 "max_chars": max(10, int(segs[k]["slot"] * cps))}
            if segs[k].get("delivery"):
                d["delivery"] = segs[k]["delivery"]
            payload.append(d)
        try:
            res = gemini_batch(payload, src_lang, tgt,
                               prompt=(system, "INPUT:\n" + json.dumps(payload, ensure_ascii=False)))
        except Exception as e:
            log("auto_review skip:", str(e).replace(GEMINI_KEY, "***")[:150])
            continue
        for k, t in zip(idx, res):
            t = (t or "").strip()
            if tgt == "ta":
                t = colloquial_ta(t)
            limit = max(10, int(segs[k]["slot"] * cps)) * 1.25
            if t and t != out[k] and len(t) <= limit:
                out[k] = t
                changed += 1
    log(f"auto_review: {changed}/{len(segs)} வசனம் திருத்தினேன்")
    return out


def _norm_txt(t):
    return re.sub(r"[^\u0B80-\u0BFFa-zA-Z0-9]", "", t or "")


def text_sim(a, b):
    import difflib
    a, b = _norm_txt(a), _norm_txt(b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def heard_text(wav, sr, lang):
    """உருவாக்கின தமிழ் ஆடியோவை Whisper-ஆ மறுபடி கேட்டு எழுத்தா மாத்தும்."""
    try:
        segs, _ = WHISPER.transcribe(to16(wav, sr), language=lang, beam_size=1,
                                     vad_filter=False, condition_on_previous_text=False)
        return " ".join((s.text or "").strip() for s in segs).strip()
    except Exception as e:
        log("heard_text skip:", repr(e)[:100])
        return None


def voice_emb(wav, sr):
    fe, m = _load_spk_model()
    x = to16(wav, sr)[:SR16 * 10]
    if len(x) < SR16 * 0.5:
        return None
    inp = fe(x, sampling_rate=SR16, return_tensors="pt", padding=True)
    with torch.no_grad():
        e = m(**{k: v.to(DEVICE) for k, v in inp.items()}).embeddings
    return torch.nn.functional.normalize(e, dim=-1)[0].float().cpu().numpy()


def ref_emb(path):
    if path not in _REF_EMB:
        try:
            w, s = sf.read(path, always_2d=True)
            _REF_EMB[path] = voice_emb(w.mean(axis=1).astype(np.float32), s)
        except Exception as e:
            log("ref_emb skip:", repr(e)[:100])
            _REF_EMB[path] = None
    return _REF_EMB[path]


_SPK_ANCHOR = {}
_VS_LOG = {}      # speaker -> [voice similarity of chosen takes] (diagnostics)
_REF_BUZZ = {}


def hp_filter(x, sr, fc=70.0):
    """கீழ்-அதிர்வு இரைச்சல் நீக்கம் (reference சுத்தமாக)."""
    try:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(2, fc / (sr / 2.0), btype="highpass", output="sos")
        return sosfiltfilt(sos, x).astype(np.float32)
    except Exception:
        return x


def buzz(wav, sr):
    """குரல் உள்ள இடங்கள்ல spectral flatness - அதிகம் = கரகரப்பு / இரைச்சல்."""
    try:
        y = to16(wav, sr)
        S = librosa.feature.spectral_flatness(y=y)[0]
        r = librosa.feature.rms(y=y)[0]
        n = min(len(S), len(r))
        S, r = S[:n], r[:n]
        m = r > np.percentile(r, 50)
        return float(np.mean(S[m])) if m.any() else float(np.mean(S))
    except Exception:
        return None


def ref_buzz(path):
    if path not in _REF_BUZZ:
        try:
            w, s_ = sf.read(path, always_2d=True)
            _REF_BUZZ[path] = buzz(w.mean(axis=1).astype(np.float32), s_)
        except Exception:
            _REF_BUZZ[path] = None
    return _REF_BUZZ[path]


def best_take(text, ref_path, ref_text, code, base_seed, sr, tgt_f0, spk=None):
    """பல முறை உருவாக்கி சிறந்ததை எடுக்கும்:
    (1) சொன்னது = எழுதினது  (2) அந்த கேரக்டரோட குரல் (reference + அந்த கேரக்டரோட முந்தைய வசனங்கள்)
    (3) கரகரப்பு இல்ல  (4) ஆண்/பெண் pitch மாறல."""
    remb = ref_emb(ref_path) if ref_path else None
    rbz = ref_buzz(ref_path) if ref_path else None
    anchor = _SPK_ANCHOR.setdefault(spk, []) if spk else []
    best, best_score, best_semi, best_e, best_vs = None, -9.0, 0.0, None, 0.0
    for t in range(max(1, VERIFY_TRIES)):
        set_seed(base_seed + 17 * t)
        wav = synth(text, ref_path, ref_text, lang=code)
        if len(wav) < sr * 0.15:
            continue
        semi = pitch_check(wav, sr, tgt_f0)
        heard = heard_text(wav, sr, code)
        ts = text_sim(text, heard) if heard is not None else 0.7
        e, parts = None, []
        try:
            e = voice_emb(wav, sr)
            if e is not None and remb is not None:
                parts.append(float(np.dot(e, remb)))
            if e is not None and len(anchor) >= 2:
                cen = np.mean(anchor, axis=0)
                cen = cen / (np.linalg.norm(cen) + 1e-9)
                parts.append(float(np.dot(e, cen)))
        except Exception as ex:
            log("voice_emb skip:", repr(ex)[:100])
        vs = float(np.mean(parts)) if parts else 0.85
        bz = buzz(wav, sr)
        pen_b = 0.0
        if bz is not None and rbz:
            pen_b = float(np.clip(bz / (rbz + 1e-6) - 1.6, 0.0, 3.0)) * 0.1
        score = 0.45 * ts + 0.5 * vs - 0.03 * max(0.0, abs(semi) - 3.0) - pen_b
        if abs(semi) > 7:
            score -= 0.3
        if t:
            log(f"  retry {t}: text {ts:.2f} voice {vs:.2f} buzz-pen {pen_b:.2f} pitch {semi:+.1f}")
        if score > best_score:
            best, best_score, best_semi, best_e, best_vs = wav, score, semi, e, vs
        if ts >= TEXT_SIM_MIN and vs >= VOICE_SIM_MIN and abs(semi) <= 4 and pen_b == 0.0:
            break
    if best is None:
        raise RuntimeError("synth காலி ஆடியோ தந்தது")
    if spk and best_e is not None and best_vs >= VOICE_SIM_MIN * 0.92 and len(anchor) < 12:
        anchor.append(best_e)
    if spk:
        _VS_LOG.setdefault(spk, []).append(best_vs)
    return best, best_semi


def shorten_line(text, max_chars, code="ta"):
    """வசனம் நேரத்துக்கு நீளமா இருந்தா, வேகப்படுத்தாம வார்த்தைகளை சுருக்கும்."""
    if not GEMINI_KEY or max_chars < 6:
        return None
    system = (f"Shorten this {LANGS[code][1]} dubbing line to at most {int(max_chars)} characters. "
              "Keep meaning, emotion, casual spoken style and the punctuation. "
              "Return ONLY a JSON array with one string.")
    try:
        r = gemini_batch([{"text": text, "slot": 3.0}], "ta", code,
                         prompt=(system, json.dumps([text], ensure_ascii=False)))
    except Exception as e:
        log("shorten skip:", repr(e)[:100])
        return None
    t = (r[0] if r else "").strip()
    return t if t and len(t) < len(text) else None


def clip_quality(clip, sr):
    """reference-க்கு சுத்தமான, தொடர்ச்சியான குரல் clip-ஆ (BGM/இரைச்சல் குறைவு) பாக்க."""
    try:
        y = to16(clip, sr)
        flat = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        r = librosa.feature.rms(y=y)[0]
        steady = float(np.std(r) / (np.mean(r) + 1e-6))
        voiced = float(np.mean(r > 0.3 * np.mean(r)))
        return voiced - flat * 4.0 - steady * 0.3
    except Exception:
        return 0.0


# ---------------------------------------------------------------- pipeline
def speaker_report(segs, spk_profile):
    """எத்தனை கேரக்டர் கண்டுபிடிச்சது + ஒவ்வொருத்தர் குரல் எவ்வளவு ஒத்துப்போச்சுன்னு (0-1)."""
    parts = []
    for k in sorted({x["speaker"] for x in segs}):
        mine = [x for x in segs if x["speaker"] == k]
        g = "பெண்" if mine[0].get("gender") == "female" else "ஆண்"
        vs = _VS_LOG.get(k) or []
        sim = f"{float(np.mean(vs)):.2f}" if vs else "-"
        ok = "✓" if spk_profile.get(k, (None,))[0] else "✗ref இல்ல"
        parts.append(f"{k} {g}: {len(mine)} வசனம், குரல் ஒற்றுமை {sim} {ok}")
    return parts


def pipeline(src, workdir, out_path, progress, opts):
    langs = [c for c in LANGS if opts.get("langs", {}).get(c)] or ["ta"]
    _REF_EMB.clear(); _REF_BUZZ.clear(); _SPK_ANCHOR.clear(); _VS_LOG.clear()
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
    log("whisper done:", len(segs), "segments, lang", src_lang)

    progress("💬 English subtitle தயார் பண்றேன்...", 36)
    srt_path = os.path.join(workdir, "english.srt")
    n_subs = write_srt(english_subs(vocals, src_lang, segs), srt_path)

    voc, vsr = sf.read(vocals, always_2d=True)
    voc = voc.mean(axis=1).astype(np.float32)
    voc16 = to16(voc, vsr)
    progress("👥 யார் யார் பேசுறாங்கன்னு (ஆண்/பெண்) பாக்குறேன்...", 38)
    try:
        assign_speakers(segs, voc16, vocals if vocals != audio else None)
    except Exception as e:
        log("speaker analysis failed:", repr(e)[:150])
        for s_ in segs:
            s_["speaker"], s_["gender"], s_["f0s"] = "M0", "male", []
    raw_n = len(segs)
    segs = merge_segments(segs)          # flow-க்காக சேர்த்தல் (அதே ஆள் மட்டும்)
    med_f0 = finalize_segs(segs, voc16)
    add_slots(segs, total)
    progress("🎬 கதையையும் கேரக்டர்களையும் புரிஞ்சுக்கிறேன்...", 40)
    set_brief(story_brief(segs, src_lang))
    log("story brief:", (STORY_BRIEF[:160] or "இல்ல").replace("\n", " "))
    log(f"merged for flow: {raw_n} -> {len(segs)} segments;",
        "shout:", sum(1 for x in segs if x["delivery"] == "shouting"))

    refdir = os.path.join(workdir, "refs")
    os.makedirs(refdir, exist_ok=True)
    refs = [build_ref(i, segs, voc, vsr, words, refdir) for i, s in enumerate(segs)]
    # ஒவ்வொரு கேரக்டருக்கும் ஒரே 'குரல் அடையாளம்' -> எல்லா வசனத்துலயும் அதே குரல்
    spk_profile = {k: build_profile(k, segs, voc, vsr, words, refdir)
                   for k in sorted({x["speaker"] for x in segs})}
    log("voice profiles:", {k: bool(v[0]) for k, v in spk_profile.items()})

    sr = OMNI.sampling_rate
    base_path = bg if bg else audio
    base, bsr = sf.read(base_path, always_2d=True)
    base = base.astype(np.float32)
    g = gcd(bsr, sr)
    gain = BG_GAIN if bg else 0.12
    n_seg = len(segs)

    fx = None   # அசல் சிரிப்பு / அலறல் / மூச்சு / அழுகை சத்தம் (பேச்சு இல்லாத இடங்கள்ல)
    if opts.get("fx", True) and bg:
        mask = np.ones(len(voc), dtype=np.float32)
        for s_ in segs:
            mask[max(0, int((s_["start"] - 0.06) * vsr)):int((s_["end"] + 0.10) * vsr)] = 0.0
        k = max(3, int(0.03 * vsr))
        mask = np.convolve(mask, np.ones(k, dtype=np.float32) / k, mode="same")
        fx_mono = voc * mask * FX_GAIN
        g2 = gcd(bsr, vsr)
        fx_mono = resample_poly(fx_mono, bsr // g2, vsr // g2).astype(np.float32)
        fx = np.zeros(base.shape[0], dtype=np.float32)
        m = min(len(fx), len(fx_mono))
        fx[:m] = fx_mono[:m]

    results = []          # [{code, mix, dub, failed}]
    skipped = []
    span = 48.0 / len(langs)
    for li, code in enumerate(langs):
        lname = LANGS[code][0]
        p0 = 42 + span * li
        progress(f"🌐 {lname} மொழிபெயர்க்கிறேன்... ({n_seg} வசனம்)", int(p0))
        try:
            texts = translate_all(segs, src_lang, code)
            if AUTO_REVIEW:                            # நீ பாக்க வேண்டாம் - bot-ஏ சரிபார்க்கும்
                progress(f"🧐 {lname} வசனங்களை தானா சரிபார்க்கிறேன்...", int(p0 + span * 0.08))
                texts = auto_review(segs, texts, src_lang, code)
            if opts.get("_review"):                    # நீ பார்த்து திருத்த
                try:
                    texts = opts["_review"](code, segs, texts)
                except Exception as e:
                    log("review skip:", repr(e)[:100])
        except Exception as e:
            log(f"{code} translate failed:", repr(e)[:150])
            skipped.append(code)
            continue

        track = np.zeros(int((total + 2) * sr), dtype=np.float32)
        failed = 0
        t_loop = time.time()
        for i, s in enumerate(segs):
            eta = ""
            if i >= 2:
                left = (time.time() - t_loop) / i * (n_seg - i)
                eta = f"  ⏳ ~{int(left // 60) + 1} நிமிடம் மீதம்"
            progress(f"🎙️ {lname} குரல் உருவாக்குறேன்... {i + 1}/{n_seg}{eta}",
                     int(p0 + span * (0.1 + 0.9 * (i + 1) / n_seg)))
            text = (texts[i] or "").strip()
            if not text:
                log(f"{code} segment {i}: text காலி")
                failed += 1
                continue
            try:
                ref_path, ref_text = spk_profile.get(s["speaker"]) or (None, None)
                if not ref_path:
                    ref_path, ref_text = refs[i]
                tgt_f0 = med_f0.get(s["speaker"]) or s.get("f0")
                tgt_rms = s.get("rms")
                base_seed = 1000 + sum(ord(ch) for ch in s["speaker"])
                wav, semi = best_take(text, ref_path, ref_text, code, base_seed, sr, tgt_f0,
                                      spk=s["speaker"])
                if abs(semi) > 4:
                    log(f"{code} seg {i} {s['speaker']}: pitch வித்தியாசம் {semi:.1f} semitone")
                wav = trim_silence(wav, sr)
                if PITCH_FIX:                            # pitch மாத்துனா குரல் ரோபோ ஆகும் - default off
                    wav = fix_pitch(wav, sr, semi)
                wav = fit_to_slot(wav, sr, s["slot"], text, ref_path, ref_text,
                                  lang=code, target_rms=tgt_rms, hard=s.get("hard"))
                a = int(max(0.0, s["start"] - LEAD) * sr)
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
        if fx is not None:
            mix = mix + fx[:, None]
        peak = float(np.max(np.abs(mix)))
        if peak > 0.98:
            mix = mix * (0.98 / peak)
        mix_path = os.path.join(workdir, f"mix_{code}.wav")
        sf.write(mix_path, mix, bsr)
        loudnorm_file(mix_path, bsr)
        results.append(dict(code=code, mix=mix_path, dub=dub, failed=failed, texts=texts))

    if not results:
        raise RuntimeError("எந்த மொழியிலும் மொழிபெயர்ப்பு நடக்கல, கொஞ்ச நேரம் கழித்து மறுபடி முயற்சி செய்")

    # எல்லா மொழி / extra audio tracks இங்க சேரும் -> MKV, MP4 ரெண்டுலயும் ஒரே மாதிரி
    audio_items = []      # (path, iso, title, label)
    for r in results:
        lname, ename, iso, _, _ = LANGS[r["code"]]
        audio_items.append((r["mix"], iso, f"{ename} Dub", lname))
    if opts.get("voice"):
        for r in results:
            lname, ename, iso, _, _ = LANGS[r["code"]]
            vpath = os.path.join(workdir, f"voice_{r['code']}.wav")
            vo = r["dub"]
            pk = float(np.max(np.abs(vo))) if len(vo) else 0.0
            if pk > 0.98:
                vo = vo * (0.98 / pk)
            sf.write(vpath, vo, bsr)
            audio_items.append((vpath, iso, f"{ename} Voice Only", f"{lname} குரல் மட்டும்"))
    if opts.get("bg"):
        if bg:
            audio_items.append((bg, "und", "BGM Only", "BGM மட்டும்"))
        else:
            log("BGM track skip: demucs இல்ல")
    n_orig = count_audio(src) if opts.get("orig") else 0
    tracks = [it[3] for it in audio_items] + (["அசல் ஆடியோ"] if n_orig else [])

    def build(path, sub_codec):
        inputs = [src]
        maps = ["-map", "0:v:0"]
        meta = []
        aidx = 0
        for apath, lng, title, _ in audio_items:
            inputs.append(apath)
            maps += ["-map", f"{len(inputs) - 1}:a:0"]
            meta += [f"-metadata:s:a:{aidx}", f"language={lng}",
                     f"-metadata:s:a:{aidx}", f"title={title}"]
            aidx += 1
        for k in range(n_orig):
            maps += ["-map", f"0:a:{k}"]
            meta += [f"-metadata:s:a:{aidx}", "language=und",
                     f"-metadata:s:a:{aidx}",
                     f"title={'Original' if n_orig == 1 else 'Original ' + str(k + 1)}"]
            aidx += 1
        if n_subs > 0:
            inputs.append(srt_path)
            maps += ["-map", f"{len(inputs) - 1}:0"]
            meta += ["-c:s", sub_codec, "-metadata:s:s:0", "language=eng",
                     "-metadata:s:s:0", "title=English", "-disposition:s:0", "default"]
        disp = ["-disposition:a:0", "default"]
        for k in range(1, aidx):
            disp += [f"-disposition:a:{k}", "0"]
        cmd = ["ffmpeg", "-y"]
        for pth in inputs:
            cmd += ["-i", pth]
        extra = ["-movflags", "+faststart"] if path.endswith(".mp4") else []
        try:
            run(cmd + maps + ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
                + meta + disp + extra + [path])
        except Exception as e:
            log(f"{os.path.basename(path)} copy failed, re-encode:", repr(e)[:150])
            run(cmd + maps + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                              "-c:a", "aac", "-b:a", "192k"] + meta + disp + extra + [path])

    mkv_path = None
    if opts.get("doc"):
        progress("🎬 MKV-ஆ இணைக்கிறேன்...", 94)
        build(out_path, "srt")
        mkv_path = out_path

    mp4_path = None
    if opts.get("video"):
        progress("🎬 MP4-ஆ இணைக்கிறேன்...", 97)
        mp4_path = os.path.join(os.path.dirname(out_path), "Dubbed_AllAudio.mp4")
        build(mp4_path, "mov_text")

    progress("✅ முடிந்தது", 100)

    return dict(lang=src_lang, n=n_seg, failed=sum(r["failed"] for r in results),
                subs=n_subs, tracks=tracks, mkv=mkv_path, mp4=mp4_path,
                skipped=[LANGS[c][0] for c in skipped],
                samples={r["code"]: [t for t in r["texts"] if t][:8] for r in results},
                speakers=speaker_report(segs, spk_profile),
                duration=int(total), width=width, height=height)


# ---------------------------------------------------------------- lifecycle
def touch(chat_id=None):
    global LAST_ACTIVE
    LAST_ACTIVE = time.time()
    if chat_id:
        KNOWN_CHATS.add(chat_id)


async def notify(text):
    targets = set(KNOWN_CHATS)
    if OWNER_ID:
        targets.add(OWNER_ID)
    for cid in targets:
        try:
            await app.send_message(cid, text)
        except Exception as e:
            log("notify failed:", cid, repr(e)[:80])


async def lifecycle_watchdog():
    """வேலை இல்லாம இருந்தா / session முடியப்போனா, bot-ஐ சுத்தமா நிறுத்தும்."""
    global ACCEPTING
    warned = False
    while True:
        await asyncio.sleep(30)
        up = time.time() - _T0
        busy = bool(TASKS) or bool(REVIEWS)
        near_end = up > MAX_RUN_HOURS * 3600
        if near_end and ACCEPTING:
            ACCEPTING = False
            if not warned:
                warned = True
                await notify("⏰ Kaggle session முடியப்போகுது. புது வீடியோ இனி எடுக்க மாட்டேன்; "
                             "இப்போ ஓடுற வேலை முடிஞ்சதும் நிற்கும். மறுபடி தொடங்க Kaggle-ல Save Version அழுத்து.")
        idle = IDLE_EXIT_MIN > 0 and (time.time() - LAST_ACTIVE) > IDLE_EXIT_MIN * 60
        if not busy and (idle or not ACCEPTING):
            why = "session நேரம் முடிஞ்சது" if not ACCEPTING else f"{int(IDLE_EXIT_MIN)} நிமிஷம் வேலை இல்ல"
            await notify(f"😴 Bot நின்னுடுச்சு ({why}). GPU நேரம் மிச்சமாச்சு. "
                         "மறுபடி தொடங்க Kaggle-ல Save Version → Save & Run All அழுத்து.")
            log("exiting:", why)
            try:
                await app.stop()
            except Exception:
                pass
            os._exit(0)


@app.on_message(filters.private, group=-1)
async def _touch_msg(_, m: Message):
    touch(m.chat.id)


@app.on_callback_query(group=-1)
async def _touch_cb(_, q):
    touch(q.message.chat.id if q.message else None)


# ---------------------------------------------------------------- telegram
@app.on_message(filters.command("start") & filters.private)
async def start_cmd(_, m: Message):
    await m.reply_text(
        "வணக்கம்! 🎬\nவீடியோவை அனுப்பு. அதே கேரக்டர் குரலில், அதே நேரத்தில் "
        "தமிழ் டப்பிங் பண்ணி தருகிறேன்.\nநேரம் ஆகலாம், பொறுமையா காத்திரு.\n"
        f"🔖 Version: {BOT_VERSION}")


@app.on_message(filters.command("version") & filters.private)
async def version_cmd(_, m: Message):
    await m.reply_text(f"🔖 Bot version: {BOT_VERSION}")


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
        extra_btn("🎭 அசல் சிரிப்பு/அலறல்/மூச்சு சத்தம் சேர்", "fx"),
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
            "🎬 Video (MP4) = Telegram-ல video-வா வரும். ரெண்டுலயும் எல்லா மொழி audio-வும் ஒரே file-க்குள்ள இருக்கும்.\n"
            "தேர்வு பண்ணிட்டு ▶️ தொடங்கு அழுத்து.")


@app.on_message(filters.private & (filters.video | filters.document))
async def handle_video(_, m: Message):
    if (m.document and (m.document.file_name or "").lower().endswith(".txt")
            and m.chat.id in REVIEWS):
        return await apply_script(m)
    if m.document and not (m.document.mime_type or "").startswith("video/"):
        return await m.reply_text("வீடியோ கோப்பு மட்டும் அனுப்பு.")
    if not ACCEPTING:
        return await m.reply_text("⏰ Kaggle session முடியப்போகுது, இப்போ புது வீடியோ எடுக்க முடியாது. "
                                  "Save Version அழுத்தி bot-ஐ மறுபடி தொடங்கிட்டு அனுப்பு.", quote=True)
    key = f"{m.chat.id}_{m.id}"
    opts = {"langs": {c: c == "ta" for c in LANGS},
            "voice": False, "orig": False, "bg": False,
            "doc": True, "video": False, "fx": True, "review": False}
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
    elif act in ("voice", "orig", "bg", "doc", "video", "fx", "review"):
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


REVIEWS = {}


async def ask_review(m: Message, workdir, code, texts):
    """தமிழ் வசனங்களை அனுப்பி, திருத்தின file வந்தா / சரி-ன்னா தொடரும்."""
    chat = m.chat.id
    path = os.path.join(workdir, f"script_{code}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(f"{i + 1}|{t}" for i, t in enumerate(texts)))
    fut = asyncio.get_running_loop().create_future()
    REVIEWS[chat] = {"fut": fut, "texts": list(texts)}
    await m.reply_document(
        path, caption=(f"📝 {LANGS[code][0]} வசனங்கள். திருத்தணும்னா file-ல 'எண்|வசனம்' வடிவத்துல "
                       "மாத்தி, அதே .txt file-ஐ திருப்பி அனுப்பு. சரின்னா கீழ அழுத்து "
                       f"({REVIEW_TIMEOUT // 60} நிமிஷத்துல பதில் வரலைன்னா தானா தொடரும்)."),
        reply_markup=InlineKeyboardMarkup(
            [[InlineKeyboardButton("✅ சரி, தொடங்கு", callback_data=f"rv:{chat}")]]))
    try:
        return await asyncio.wait_for(fut, REVIEW_TIMEOUT)
    except asyncio.TimeoutError:
        return REVIEWS[chat]["texts"]
    finally:
        REVIEWS.pop(chat, None)


@app.on_callback_query(filters.regex(r"^rv:"))
async def on_review_ok(_, q):
    chat = int(q.data.split(":")[1])
    st = REVIEWS.get(chat)
    if st and not st["fut"].done():
        st["fut"].set_result(st["texts"])
    await q.answer("தொடங்குறேன்...")


async def apply_script(m: Message):
    st = REVIEWS.get(m.chat.id)
    path = await m.download()
    texts = st["texts"]
    n = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if "|" in ln:
                a, b = ln.split("|", 1)
                if a.strip().isdigit() and 1 <= int(a) <= len(texts) and b.strip():
                    texts[int(a) - 1] = b.strip()
                    n += 1
    await m.reply_text(f"✅ {n} வசனம் திருத்தினேன், தொடர்கிறேன்.")
    if not st["fut"].done():
        st["fut"].set_result(texts)


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
            run_opts = dict(opts)
            if MANUAL_REVIEW and opts.get("review"):
                def review_cb(code, segs, texts):
                    fut = asyncio.run_coroutine_threadsafe(
                        ask_review(m, workdir, code, texts), loop)
                    return fut.result(timeout=REVIEW_TIMEOUT + 60)
                run_opts["_review"] = review_cb
            info = await asyncio.to_thread(pipeline, src, workdir, out, progress, run_opts)
            await safe_edit(status, "▓▓▓▓▓▓▓▓▓▓ 100%\n📤 உனக்கு அனுப்புறேன்...")
            cap = f"✅ டப்பிங் ரெடி!\n(மூல மொழி: {info['lang']}, {info['n']} வசனம்"
            cap += f", {info['failed']} தவறியது)" if info["failed"] else ")"
            cap += "\n🎧 Audio: " + ", ".join(info["tracks"])
            if info["skipped"]:
                cap += "\n⚠️ தவறிய மொழி: " + ", ".join(info["skipped"])
            if info.get("speakers"):
                cap += "\n👥 கேரக்டர்கள்:\n" + "\n".join(info["speakers"][:10])
            if info["subs"]:
                cap += "\n💬 English subtitle உள்ளது"
            if info["mkv"]:
                cap_mkv = cap + "\n(VLC / MX Player-ல் Audio & Subtitle track மாத்திக்கலாம்)"
                await m.reply_document(info["mkv"], caption=cap_mkv, force_document=True)
            if info["mp4"]:
                cap_mp4 = cap if not info["mkv"] else "🎬 அதே video, MP4 format-ல்"
                cap_mp4 += "\n(Telegram-ல் முதல் மொழி கேட்கும்; பதிவிறக்கி MX / VLC-ல் Audio மாத்திக்கலாம்)"
                await m.reply_video(info["mp4"], caption=cap_mp4, supports_streaming=True,
                                    duration=info["duration"], width=info["width"],
                                    height=info["height"])
            try:
                lines = []
                for code, arr in info["samples"].items():
                    lines.append(f"📝 {LANGS[code][0]} வசனங்கள் (மாதிரி)  🔖 {BOT_VERSION}:")
                    lines += [f"{k + 1}. {t}" for k, t in enumerate(arr)]
                    lines.append("")
                if lines:
                    await m.reply_text("\n".join(lines)[:3800])
            except Exception as e:
                log("samples send failed:", repr(e)[:100])
            await status.delete()
    except Exception as e:
        await safe_edit(status, f"❌ பிழை: {str(e)[:500]}")
    finally:
        touch()
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
    log("Bot started | version:", BOT_VERSION)
    touch()
    asyncio.create_task(lifecycle_watchdog())
    if OWNER_ID:
        try:
            await app.send_message(OWNER_ID, f"✅ Bot ஆன்லைன்! வீடியோ அனுப்பு. (🔖 {BOT_VERSION})")
        except Exception as e:
            log("owner notice failed:", repr(e)[:80])
    await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
