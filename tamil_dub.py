"""
Tamil Auto-Dubbing Pipeline (Kaggle notebook-ku)
================================================
Video -> BGM pirippu (demucs) -> Gemini: yaar enna pesuranga + pechu Tamil
-> Gemini TTS (character-ku oru nilaiyana kural) -> BGM + Tamil kural + video

SETUP (Kaggle):
 1. Notebook Settings -> Internet = ON  (phone verify pannanum)
 2. Add-ons -> Secrets -> Label: GEMINI_API_KEY, Value: unga key
 3. Video-va Kaggle Dataset-aa add pannunga (or /kaggle/working-la upload)
 4. Idha oru cell-la paste panni run pannunga.

Intermediate files /kaggle/working/work-la save aagum. Mid-la nindrutha
marubadiyum run pannunga - mudincha steps skip aagum.
Tamil text-la tappu irundha work/script.json-la edit pannitu marubadi run
pannunga (dubbed TTS cache-a work/tts folder-a delete pannunga).
"""

import os, sys, re, json, time, glob, shutil, subprocess, wave, uuid
from concurrent.futures import ThreadPoolExecutor

# ----------------------------- CONFIG ---------------------------------------
VIDEO_PATH   = ""        # kaali na /kaggle/input-la irukkura mudhal video edukkum
OUT_DIR      = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
WORK         = os.path.join(OUT_DIR, "work")
OUTPUT_VIDEO = os.path.join(OUT_DIR, "dubbed_tamil.mp4")

TEXT_MODEL   = "gemini-3.8-flash"            # video puriyum + Tamil mozhipeyarpu
TTS_MODEL    = "gemini-3.8-flash-tts"        # nalla quality; fail na lite-ku maarum
TTS_FALLBACK = "gemini-3.8-flash-lite-tts"

CHUNK_MIN    = 8         # video-va ivlo nimisham thundaa pirichu anuppum
PROXY_HEIGHT = 480       # Gemini-ku anuppa chinna video (upload vegam)
MAX_SPEEDUP  = 1.35      # Tamil vari neelamaa irundha max vegappaduthal
TTS_WORKERS  = 3         # ore neraththil TTS call (429 vandha 1-aa kuraikkavum)
BGM_VOLUME   = 0.9
VOICE_VOLUME = 1.0
SR           = 24000     # Gemini TTS output sample rate
# ----------------------------------------------------------------------------


def sh(cmd, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"CMD FAILED: {cmd}\n{r.stderr[-1500:]}")
    return r


def ensure_deps():
    try:
        import google.genai  # noqa
    except ImportError:
        sh(f"{sys.executable} -m pip install -q google-genai")
    try:
        import demucs  # noqa
    except ImportError:
        sh(f"{sys.executable} -m pip install -q demucs", check=False)
    import numpy  # noqa


ensure_deps()
import numpy as np
from google import genai
from google.genai import types


def get_api_key():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GEMINI_API_KEY")
    except Exception:
        k = os.environ.get("GEMINI_API_KEY")
        if not k:
            raise SystemExit("GEMINI_API_KEY illa. Kaggle Secrets-la add pannunga.")
        return k


client = genai.Client(api_key=get_api_key())
os.makedirs(WORK, exist_ok=True)
os.makedirs(os.path.join(WORK, "tts"), exist_ok=True)


# ----------------------------- helpers --------------------------------------
def retry(fn, tries=6, base=4.0, what="call"):
    last = ""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa
            msg = str(e)
            last = msg
            wait = base * (2 ** i)
            print(f"  [{what}] error: {msg[:220].replace(chr(10), ' ')}")
            if "limit: 0" in msg:      # indha model-ku quota-ve illa, wait panni payan illa
                raise RuntimeError(f"{what}: quota 0 -> {msg[:250]}")
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg or "UNAVAILABLE" in msg:
                if i < tries - 1:
                    print(f"  [{what}] {wait:.0f}s kaathirukkirom...")
                    time.sleep(wait)
                continue
            if i == tries - 1:
                raise
            time.sleep(2)
    raise RuntimeError(f"{what} failed: {last[:300]}")


def ts_to_sec(t):
    if isinstance(t, (int, float)):
        return float(t)
    parts = [float(p) for p in str(t).strip().split(":")]
    s = 0.0
    for p in parts:
        s = s * 60 + p
    return s


def duration(path):
    r = sh(f'ffprobe -v error -show_entries format=duration -of csv=p=0 "{path}"')
    return float(r.stdout.strip())


def find_video():
    if VIDEO_PATH and os.path.exists(VIDEO_PATH):
        return VIDEO_PATH
    for root in ("/kaggle/input", "/kaggle/working", "."):
        for ext in ("mp4", "mkv", "mov", "webm", "avi"):
            hits = glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True)
            hits = [h for h in hits if "dubbed_tamil" not in h and "/work/" not in h]
            if hits:
                return hits[0]
    raise SystemExit("Video kidaikala. VIDEO_PATH set pannunga.")


# ----------------------------- STEP 1: BGM ----------------------------------
def step_bgm(video):
    bgm_path = os.path.join(WORK, "bgm.wav")
    if os.path.exists(bgm_path):
        print("[1] BGM cache irukku, skip")
        return bgm_path
    print("[1] Audio eduthu BGM pirikkirom (CPU-la konjam neram aagum)...")
    full = os.path.join(WORK, "full_audio.wav")
    sh(f'ffmpeg -y -i "{video}" -vn -ac 2 -ar 44100 "{full}"')
    r = sh(f'{sys.executable} -m demucs --two-stems=vocals -n htdemucs '
           f'-o "{WORK}/demucs" "{full}"', check=False)
    hits = glob.glob(os.path.join(WORK, "demucs", "**", "no_vocals.wav"), recursive=True)
    if r.returncode == 0 and hits:
        shutil.copy(hits[0], bgm_path)
        voc = glob.glob(os.path.join(WORK, "demucs", "**", "vocals.wav"), recursive=True)
        if voc:
            shutil.copy(voc[0], os.path.join(WORK, "vocals.wav"))
    else:
        print("  !! demucs fail aachu. Fallback: asal audio-va romba kuraichu BGM-aa "
              "use pannuvom (asal kural mella kekkum).")
        print("  ", r.stderr[-400:])
        sh(f'ffmpeg -y -i "{full}" -af volume=0.12 "{bgm_path}"')
    return bgm_path


# ----------------------------- STEP 2: Gemini script ------------------------
SCRIPT_PROMPT = """You are a professional dubbing scriptwriter for Tamil movies and web series. Watch and listen to this video clip.

Task: list EVERY spoken line in this clip, from the very first second to the very last second, and write the
Tamil dubbing line for each. Do NOT skip any dialogue, not even short words like "hey", "no", "what".

Tamil style (VERY IMPORTANT): write the casual SPOKEN Tamil of everyday Chennai conversation, exactly like
dialogue in Tamil-dubbed movies. NEVER literary/formal Tamil (sentamil).
Examples: "என்ன சொல்கிறாய்?" -> "என்ன சொல்ற?",  "நான் வருகிறேன்" -> "நான் வரேன்",
"அது இருக்கிறது" -> "அது இருக்கு",  "எனக்கு வேண்டும்" -> "எனக்கு வேணும்",  "ஆனால்" -> "ஆனா",
"இல்லை" -> "இல்ல",  "நீங்கள்" -> "நீங்க",  "அவர்கள்" -> "அவங்க",  "பிறகு" -> "அப்புறம்".
Never use words like மற்றும், ஏனெனில், எனவே. Use natural interjections (டா, சரி, ஆமா, அய்யோ) only where they fit.
Write in Tamil script. Common loanwords are fine (ஓகே, சாரி, போன்). No Hindi/English sentences.
Keep meaning, emotion and length close to the original so it fits the same time. Short, speakable sentences.

Speakers: use the video (who is on screen, who is speaking) and the voices to decide who speaks.
Use stable IDs like C1, C2, C3. The same person must ALWAYS get the same ID.
{roster}

Timestamps are relative to the START OF THIS CLIP in MM:SS.s format (e.g. 00:12.4).
Skip songs/singing/background noise. Include narrator voice-over as its own character.

Return ONLY JSON:
{{
 "characters": [ {{"id":"C1","gender":"male|female","age":"child|young|adult|elder",
                   "pitch":"low|mid|high","tone":"short English description of voice/personality",
                   "is_new": true}} ],
 "lines": [ {{"start":"00:01.2","end":"00:03.9","speaker":"C1",
              "original":"original words","tamil":"colloquial Tamil",
              "emotion":"neutral|angry|sad|happy|scared|whisper|shout|calm"}} ]
}}
Only list characters that appear in this clip in "characters"."""


POLISH_PROMPT = """You are the top dialogue writer for Tamil-dubbed movies and web series (the kind of dub
people watch on TV/OTT). Rewrite the Tamil of every line below so it sounds like REAL spoken Tamil as people
talk in Chennai, as in a professional movie dub. NOT written / formal / literary Tamil.

Rules:
- Spoken forms: இருக்கிறது->இருக்கு, செய்கிறேன்->பண்றேன்/செய்றேன், சொல்கிறாய்->சொல்ற, வருகிறேன்->வரேன்,
  போகிறேன்->போறேன், இல்லை->இல்ல, வேண்டும்->வேணும், ஆனால்->ஆனா, அதனால்->அதனால, நீங்கள்->நீங்க,
  அவர்கள்->அவங்க, என்னுடைய->என்னோட, எப்படி இருக்கிறாய்->எப்படி இருக்க, பிறகு->அப்புறம்.
- Never use: மற்றும், ஏனெனில், எனவே, ஆகையால், அவ்வாறு.
- Match the speaker: young/angry male can say டா, வாடா, போடா; polite or elder-facing speech uses நீங்க, சார்,
  -ங்க; kids and women talk naturally for their age. Use fillers (சரி, ஆமா, அய்யோ, ஹே) only when they fit the emotion.
- Keep the meaning and emotion of the original, keep length about the same (it must fit the same time).
- Write only Tamil script (loanwords like ஓகே, சாரி, போன் are fine). No explanations.

Input is JSON: each item has id, who, emotion, en (original words), ta (current Tamil).
Return ONLY JSON: {{"lines":[{{"id":0,"tamil":"..."}}]}} with EVERY id exactly once.

INPUT:
{items}"""


TEXT_MODELS = []
for _m in [os.environ.get("GEMINI_TEXT_MODEL"), TEXT_MODEL,
           "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite"]:
    if _m and _m not in TEXT_MODELS:
        TEXT_MODELS.append(_m)


def gen(contents, config=None, what="gemini"):
    """Text/video model-ku call. Oru model fail aana adutha model-ku maarum."""
    last = None
    for m in TEXT_MODELS:
        try:
            r = retry(lambda: client.models.generate_content(
                model=m, contents=contents, config=config), tries=4, what=f"{what}:{m}")
            if m != TEXT_MODELS[0]:
                print(f"  ({m} use aachu)")
            return r
        except Exception as e:  # noqa
            last = e
            print(f"  {m} work aagala, adutha model try pannuren...")
    raise RuntimeError(f"ellaa models-um fail. Kadaisi error: {str(last)[:300]}")


def make_proxy(video):
    proxy = os.path.join(WORK, "proxy.mp4")
    if not os.path.exists(proxy):
        sh(f'ffmpeg -y -i "{video}" -vf "scale=-2:{PROXY_HEIGHT}" -r 12 '
           f'-c:v libx264 -preset veryfast -crf 30 -c:a aac -b:a 64k -ac 1 "{proxy}"')
    return proxy


def upload_wait(path):
    f = retry(lambda: client.files.upload(file=path), what="upload")
    while getattr(f.state, "name", str(f.state)) == "PROCESSING":
        time.sleep(4)
        f = client.files.get(name=f.name)
    if getattr(f.state, "name", str(f.state)) != "ACTIVE":
        raise RuntimeError(f"Upload fail: {f.state}")
    return f


def parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        return json.loads(re.sub(r"^```json|```$", "", text.strip(), flags=re.M))


def run_clip(clip, off, chars):
    """Oru video thundai Gemini-ku anuppi, varigalai (offset serthu) tharum."""
    f = upload_wait(clip)
    roster = ""
    if chars:
        roster = ("Characters already known from earlier clips (reuse these IDs for the same people, "
                  "only create a NEW id for a truly new person):\n" +
                  json.dumps(chars, ensure_ascii=False))
    prompt = SCRIPT_PROMPT.format(roster=roster)
    resp = gen([f, prompt],
               types.GenerateContentConfig(response_mime_type="application/json",
                                           temperature=0.4), what="script")
    try:
        client.files.delete(name=f.name)
    except Exception:
        pass
    data = parse_json(resp.text)
    for c in data.get("characters", []):
        if "id" in c:
            chars.setdefault(c["id"], {k: v for k, v in c.items() if k != "is_new"})
    out = []
    for ln in data.get("lines", []):
        try:
            s_, e_ = ts_to_sec(ln["start"]) + off, ts_to_sec(ln["end"]) + off
        except Exception:
            continue
        if not str(ln.get("tamil", "")).strip() or e_ <= s_:
            continue
        out.append({"start": round(s_, 2), "end": round(e_, 2),
                    "speaker": ln.get("speaker", "C1"),
                    "original": ln.get("original", ""),
                    "tamil": ln["tamil"].strip(),
                    "emotion": ln.get("emotion", "neutral")})
    return out


def cut_clip(src, start, dur, dst):
    sh(f'ffmpeg -y -ss {start:.2f} -t {dur:.2f} -i "{src}" -c:v libx264 -preset ultrafast '
       f'-crf 32 -c:a aac -b:a 64k "{dst}"')


FRAME = 0.05


def speech_mask(vocals_path):
    """vocals.wav-la yengellam kural irukku-nu frame-wise (0.05s) mask."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", vocals_path, "-ac", "1", "-ar", "16000",
                        "-f", "s16le", "-"], capture_output=True)
    x = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    n = int(len(x) // (FRAME * 16000))
    if n == 0:
        return np.zeros(0, dtype=bool)
    fr = x[: n * int(FRAME * 16000)].reshape(n, -1)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    loud = rms[rms > 0.003]
    thr = max(0.01, 0.15 * float(np.percentile(loud, 95))) if len(loud) else 0.01
    act = rms > thr
    dil = np.convolve(act.astype(int), np.ones(9), "same") > 0       # chinna idaiveli moodu
    return np.convolve(dil.astype(int), np.ones(9), "same") >= 9


def find_gaps(mask, lines, min_gap=1.0):
    """Kural kekkuthu aana script-la illatha idangal."""
    n = len(mask)
    covered = np.zeros(n, dtype=bool)
    for ln in lines:
        a = max(0, int((ln["start"] - 0.3) / FRAME))
        b = min(n, int((ln["end"] + 0.3) / FRAME) + 1)
        covered[a:b] = True
    miss = mask & ~covered
    gaps, i = [], 0
    while i < n:
        if miss[i]:
            j = i
            while j < n and (miss[j] or (j + int(1.0 / FRAME) < n and miss[j: j + int(1.0 / FRAME)].any())):
                j += 1
            if (j - i) * FRAME >= min_gap:
                gaps.append((i * FRAME, j * FRAME))
            i = j
        else:
            i += 1
    return gaps


def fill_gaps(proxy, lines, chars, total):
    voc = os.path.join(WORK, "vocals.wav")
    if not os.path.exists(voc):
        return lines
    try:
        gaps = find_gaps(speech_mask(voc), lines)
    except Exception as e:  # noqa
        print("  gap check fail:", str(e)[:150])
        return lines
    if not gaps:
        print("  koraivu illa: ellaa pechum cover aachu")
        return lines
    print(f"  {len(gaps)} idathula pechu miss aachu, marubadi kekkurom...")
    added = 0
    for k, (gs, ge) in enumerate(gaps):
        cs = max(0.0, gs - 0.7)
        ce = min(total, ge + 0.7)
        if ce - cs < 2.5:
            ce = min(total, cs + 2.5)
        clip = os.path.join(WORK, f"gap_{k}.mp4")
        cut_clip(proxy, cs, ce - cs, clip)
        try:
            new = run_clip(clip, cs, chars)
        except Exception as e:  # noqa
            print("  gap fill fail:", str(e)[:150])
            continue
        for nl in new:
            dur = nl["end"] - nl["start"]
            ov = sum(max(0.0, min(nl["end"], o["end"]) - max(nl["start"], o["start"])) for o in lines)
            if dur > 0 and ov / dur < 0.4:
                lines.append(nl)
                added += 1
    lines.sort(key=lambda x: x["start"])
    print(f"  {added} puthu varigal serthom")
    return lines


def polish(lines, chars):
    """Ellaa varigalaiyum 'movie dub' pechu Tamil-a maatri ezhuthum."""
    print("  Tamil-a movie-dub pechu nadaikku maathurom...")
    B = 30
    done = 0
    for i in range(0, len(lines), B):
        batch = lines[i: i + B]
        items = []
        for k, ln in enumerate(batch):
            c = chars.get(ln["speaker"], {})
            items.append({"id": k, "who": f"{c.get('age','adult')} {c.get('gender','male')}",
                          "emotion": ln.get("emotion", "neutral"),
                          "en": ln.get("original", ""), "ta": ln["tamil"]})
        prompt = POLISH_PROMPT.format(items=json.dumps(items, ensure_ascii=False))
        try:
            resp = gen(prompt, types.GenerateContentConfig(
                response_mime_type="application/json", temperature=0.5), what="polish")
            data = parse_json(resp.text)
            for it in data.get("lines", []):
                k = int(it["id"])
                t = str(it.get("tamil", "")).strip()
                if 0 <= k < len(batch) and t and len(t) <= len(batch[k]["tamil"]) * 1.6 + 8:
                    batch[k]["tamil_formal"] = batch[k]["tamil"]
                    batch[k]["tamil"] = t
                    done += 1
        except Exception as e:  # noqa
            print("  polish fail (original Tamil-ae use aagum):", str(e)[:150])
    print(f"  {done}/{len(lines)} varigal maathapattana")


def step_script(video):
    script_path = os.path.join(WORK, "script.json")
    if os.path.exists(script_path):
        print("[2] script.json cache irukku, skip")
        return json.load(open(script_path, encoding="utf-8"))

    print("[2] Gemini-ku video anuppi script eluthurom...")
    proxy = make_proxy(video)
    total = duration(proxy)
    step = CHUNK_MIN * 60
    chars, lines = {}, []
    n = int(np.ceil(total / step))
    for i in range(n):
        off = i * step
        clip = os.path.join(WORK, f"chunk_{i}.mp4")
        cut_clip(proxy, off, step, clip)
        print(f"  chunk {i+1}/{n} ({off/60:.0f} min-la irundhu)")
        lines += run_clip(clip, off, chars)

    lines.sort(key=lambda x: x["start"])
    lines = fill_gaps(proxy, lines, chars, total)
    for sp in {l["speaker"] for l in lines}:
        chars.setdefault(sp, {"id": sp, "gender": "male", "age": "adult",
                              "pitch": "mid", "tone": "neutral"})
    polish(lines, chars)
    out = {"characters": chars, "lines": lines}
    json.dump(out, open(script_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  {len(lines)} varigal, {len(chars)} characters")
    return out


# ----------------------------- STEP 3: voices -------------------------------
# name: (gender, age, pitch)
VOICE_POOL = {
    "Charon": ("male", "adult", "low"), "Orus": ("male", "adult", "low"),
    "Alnilam": ("male", "adult", "low"), "Enceladus": ("male", "adult", "low"),
    "Algenib": ("male", "elder", "low"), "Rasalgethi": ("male", "elder", "mid"),
    "Iapetus": ("male", "adult", "mid"), "Umbriel": ("male", "adult", "mid"),
    "Algieba": ("male", "adult", "mid"), "Schedar": ("male", "adult", "mid"),
    "Zubenelgenubi": ("male", "adult", "mid"), "Sadaltager": ("male", "adult", "mid"),
    "Fenrir": ("male", "young", "mid"), "Achird": ("male", "young", "mid"),
    "Puck": ("male", "young", "high"), "Sadachbia": ("male", "young", "high"),
    "Kore": ("female", "adult", "mid"), "Callirrhoe": ("female", "adult", "mid"),
    "Despina": ("female", "adult", "mid"), "Erinome": ("female", "adult", "mid"),
    "Sulafat": ("female", "adult", "mid"), "Autonoe": ("female", "young", "mid"),
    "Aoede": ("female", "young", "mid"), "Pulcherrima": ("female", "young", "mid"),
    "Zephyr": ("female", "young", "high"), "Leda": ("female", "young", "high"),
    "Laomedeia": ("female", "young", "high"), "Achernar": ("female", "young", "high"),
    "Gacrux": ("female", "elder", "low"), "Vindemiatrix": ("female", "elder", "mid"),
}


def assign_voices(chars):
    used, mapping = set(), {}
    # adhigam pesuravanga mudhalla (nalla voice kidaikkum)
    for cid, c in chars.items():
        g = c.get("gender", "male")
        age = c.get("age", "adult")
        pitch = c.get("pitch", "mid")
        if age == "child":
            age, pitch = "young", "high"
        best, best_score = None, -1
        for name, (vg, va, vp) in VOICE_POOL.items():
            if vg != g:
                continue
            score = (3 if va == age else 0) + (2 if vp == pitch else 0)
            if name not in used:
                score += 4
            if score > best_score:
                best, best_score = name, score
        mapping[cid] = best or "Kore"
        used.add(mapping[cid])
    return mapping


def style_text(c, emotion):
    age = c.get("age", "adult")
    g = c.get("gender", "male")
    who = "a child" if age == "child" else f"a {age} {g}"
    return (f"You are the Tamil dubbing voice actor for {who} in a professional Tamil movie dub. "
            f"Voice character: {c.get('tone','natural')}. Current emotion: {emotion}. "
            f"Speak casual, everyday spoken Tamil exactly as written, with a relaxed conversational rhythm "
            f"and real acting emotion, like dialogue in a Tamil film. Not like a news reader and not formal. "
            f"Read only the user's text, never read these instructions.")


# ----------------------------- STEP 4: TTS ----------------------------------
def pcm_to_np(data):
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def trim_silence(x, thr=0.01, pad=int(0.04 * SR)):
    idx = np.where(np.abs(x) > thr)[0]
    if len(idx) == 0:
        return x
    return x[max(0, idx[0] - pad): min(len(x), idx[-1] + pad)]


def drop_tail_burst(x):
    """Pechu mudinjapuram varra 'kirich' maadhiri chinna burst irundha vettidum."""
    W = int(0.02 * SR)
    n = len(x) // W
    if n < 6:
        return x
    rms = np.sqrt((x[: n * W].reshape(n, W) ** 2).mean(axis=1))
    thr = max(0.008, 0.06 * float(rms.max()))
    act = rms > thr
    segs, i = [], 0
    while i < n:
        if act[i]:
            j = i
            while j < n and act[j]:
                j += 1
            segs.append((i, j))
            i = j
        else:
            i += 1
    if len(segs) >= 2:
        last, prev = segs[-1], segs[-2]
        if (last[1] - last[0]) * 0.02 < 0.2 and (last[0] - prev[1]) * 0.02 >= 0.06:
            x = x[: (prev[1] + 2) * W]
    return x


def declick(x):
    """Clip-oda arambam/mudivu click-ku: tail burst vettu, DC neekki, fade, loudness samam."""
    if len(x) < int(0.2 * SR):
        return x
    x = drop_tail_burst(x)
    x = x - float(np.mean(x[:int(0.01 * SR)]))
    pk = float(np.abs(x).max())
    if pk > 1e-4:
        x = x * (0.8 / pk)
    fi, fo = int(0.015 * SR), int(0.09 * SR)
    x = x.copy()
    x[:fi] *= np.linspace(0.0, 1.0, fi, dtype=np.float32)
    x[-fo:] *= np.linspace(1.0, 0.0, fo, dtype=np.float32)
    return x


def tts_once(text, voice, system, model, use_system=True):
    cfg = dict(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))),
    )
    if use_system:
        cfg["system_instruction"] = system
    resp = client.models.generate_content(
        model=model, contents=text, config=types.GenerateContentConfig(**cfg))
    part = resp.candidates[0].content.parts[0]
    return pcm_to_np(part.inline_data.data)


_state = {"model": TTS_MODEL, "use_system": True}


def tts(text, voice, system):
    def go():
        try:
            return tts_once(text, voice, system, _state["model"], _state["use_system"])
        except Exception as e:  # noqa
            m = str(e)
            if _state["use_system"] and ("system_instruction" in m or "INVALID_ARGUMENT" in m):
                print("  system_instruction support illa pola, adhu illama try pannuren")
                _state["use_system"] = False
                return tts_once(text, voice, system, _state["model"], False)
            if "404" in m or "NOT_FOUND" in m:
                if _state["model"] != TTS_FALLBACK:
                    print(f"  {_state['model']} kidaikkala, {TTS_FALLBACK}-ku maaruren")
                    _state["model"] = TTS_FALLBACK
                    return tts_once(text, voice, system, _state["model"], _state["use_system"])
            raise
    return trim_silence(retry(go, what="tts"))


def shorten(text, ratio):
    prompt = (f"Shorten this colloquial Tamil dubbing line to about {int(ratio*100)}% of its length. "
              f"Keep meaning and tone, keep it spoken Tamil. Return ONLY the Tamil text.\n\n{text}")
    r = gen(prompt, what="shorten")
    return r.text.strip().strip('"')


def speed_np(x, factor):
    """ffmpeg atempo (pitch maaraamal vegam)."""
    if factor <= 1.01:
        return x
    uid = uuid.uuid4().hex[:10]       # thread-ku thani file (munnadi ore file-a share panni mix aachu)
    tmp_in, tmp_out = os.path.join(WORK, f"_s_{uid}_in.wav"), os.path.join(WORK, f"_s_{uid}_out.wav")
    write_wav(tmp_in, x)
    try:
        sh(f'ffmpeg -y -i "{tmp_in}" -filter:a "atempo={factor:.3f}" "{tmp_out}"')
        return read_wav(tmp_out)
    finally:
        for t in (tmp_in, tmp_out):
            if os.path.exists(t):
                os.remove(t)


def write_wav(path, x, sr=SR):
    x16 = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(x16.tobytes())


def read_wav(path):
    with wave.open(path, "rb") as w:
        data = w.readframes(w.getnframes())
    return pcm_to_np(data)


STATS = {"tts_failed": 0}


def step_tts(script):
    chars, lines = script["characters"], script["lines"]
    voices = assign_voices(chars)
    print("[3] Kural thervu:", {k: voices[k] for k in voices})
    print(f"[4] {len(lines)} varigalukku Tamil kural uruvaakkurom...")

    def work(i):
        ln = lines[i]
        path = os.path.join(WORK, "tts", f"{i:05d}.wav")
        if os.path.exists(path):
            return i, read_wav(path)
        c = chars.get(ln["speaker"], {})
        system = style_text(c, ln.get("emotion", "neutral"))
        text = ln["tamil"]
        nxt = lines[i + 1]["start"] if i + 1 < len(lines) else ln["end"] + 1.5
        orig_len = ln["end"] - ln["start"]
        allowed = max(orig_len, min(nxt - ln["start"] - 0.05, orig_len * 1.6 + 0.5))
        try:
            audio = tts(text, voices.get(ln["speaker"], "Kore"), system)
        except Exception as e:  # noqa
            print(f"  line {i} TTS fail: {str(e)[:120]}")
            return i, np.zeros(int(0.1 * SR), dtype=np.float32)
        dur = len(audio) / SR
        if dur > allowed * 1.25 and len(text) > 12:
            try:
                short = shorten(text, max(0.5, allowed / dur))
                audio = tts(short, voices.get(ln["speaker"], "Kore"), system)
                ln["tamil_short"] = short
                dur = len(audio) / SR
            except Exception as e:  # noqa
                print(f"  line {i} shorten fail: {str(e)[:100]}")
        if dur > allowed:
            audio = speed_np(audio, min(dur / allowed, MAX_SPEEDUP))
        audio = declick(audio)
        write_wav(path, audio)
        return i, audio

    results = {}
    with ThreadPoolExecutor(max_workers=TTS_WORKERS) as ex:
        for k, (i, a_) in enumerate(ex.map(work, range(len(lines)))):
            results[i] = a_
            if k % 10 == 0:
                print(f"  {k+1}/{len(lines)}")
    # fail aana varigalai thaniya marubadi (cache-la illaadhavai)
    for rnd in range(2):
        failed = [i for i in range(len(lines))
                  if not os.path.exists(os.path.join(WORK, "tts", f"{i:05d}.wav"))]
        if not failed:
            break
        print(f"  {len(failed)} varigal fail aachu, marubadi try (round {rnd+1})...")
        time.sleep(10)
        for i in failed:
            _, results[i] = work(i)
    failed = [i for i in range(len(lines))
              if not os.path.exists(os.path.join(WORK, "tts", f"{i:05d}.wav"))]
    STATS["tts_failed"] = len(failed)
    return [results[i] for i in range(len(lines))]


# ----------------------------- STEP 5: mix ----------------------------------
def step_mix(video, bgm, lines, audios):
    print("[5] Ellathaiyum serkkirom...")
    total = duration(video)
    track = np.zeros(int((total + 5) * SR), dtype=np.float32)
    for ln, a in zip(lines, audios):
        s = int(ln["start"] * SR)
        e = min(len(track), s + len(a))
        if s < len(track):
            track[s:e] += a[: e - s]
    peak = np.abs(track).max()
    if peak > 1.0:
        track /= peak
    dub_wav = os.path.join(WORK, "dub_track.wav")
    write_wav(dub_wav, track)
    is_mkv = OUTPUT_VIDEO.lower().endswith(".mkv")

    def build(norm):
        n = ":normalize=0" if norm else ""
        fc = (f"[1:a]volume={BGM_VOLUME}[b];[2:a]aresample=44100,volume={VOICE_VOLUME}[d];"
              f"[b][d]amix=inputs=2:duration=first:dropout_transition=0{n}[m];"
              f"[m]alimiter=limit=0.95[a]")
        cmd = (f'ffmpeg -y -i "{video}" -i "{bgm}" -i "{dub_wav}" -filter_complex "{fc}" '
               f'-map 0:v -map "[a]" ')
        if is_mkv:
            # MKV: Tamil dub mudhal track (default) + asal audio rendavadhu track
            cmd += ('-map 0:a:0? -c:v copy -c:a aac -b:a 192k '
                    '-metadata:s:a:0 language=tam -metadata:s:a:0 title="Tamil Dub" '
                    '-metadata:s:a:1 language=und -metadata:s:a:1 title="Original" '
                    '-disposition:a:0 default -disposition:a:1 0 ')
        else:
            cmd += '-c:v copy -c:a aac -b:a 192k -movflags +faststart '
        return cmd + f'-shortest "{OUTPUT_VIDEO}"'

    r = sh(build(True), check=False)
    if r.returncode != 0:  # pazhaya ffmpeg-la normalize illa
        sh(build(False))
    print(f"\nMudinjadhu -> {OUTPUT_VIDEO}")


# ----------------------------- MAIN -----------------------------------------
def dub(video, job_dir, progress=print, fmt="mp4"):
    """Oru video-va Tamil-la dub panni output path-a tharum. (Bot idhai kooppidum)"""
    global WORK, OUTPUT_VIDEO
    WORK = os.path.join(job_dir, "work")
    fmt = "mkv" if str(fmt).lower() == "mkv" else "mp4"
    OUTPUT_VIDEO = os.path.join(job_dir, f"dubbed_tamil.{fmt}")
    os.makedirs(os.path.join(WORK, "tts"), exist_ok=True)
    progress(f"Video: {duration(video)/60:.1f} nimisham. BGM pirikkirom...")
    bgm = step_bgm(video)
    progress("Gemini script eluthuthu (yaar enna pesuranga + pechu Tamil)...")
    script = step_script(video)
    ls = script["lines"]
    span = f" ({ls[0]['start']:.0f}s - {ls[-1]['end']:.0f}s)" if ls else ""
    progress(f"{len(ls)} varigal{span}, {len(script['characters'])} characters. Tamil kural uruvaakkuthu...")
    audios = step_tts(script)
    if STATS["tts_failed"]:
        progress(f"⚠️ {STATS['tts_failed']} varigalukku kural varala (Gemini limit). Marubadi anuppi paaru.")
    progress("Ellathaiyum serkkirom...")
    step_mix(video, bgm, script["lines"], audios)
    return OUTPUT_VIDEO


if __name__ == "__main__":
    video = find_video()
    dub(video, OUT_DIR)
