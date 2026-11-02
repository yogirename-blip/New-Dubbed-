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

import os, sys, re, json, time, glob, shutil, subprocess, wave, uuid, threading
from concurrent.futures import ThreadPoolExecutor

# ----------------------------- CONFIG ---------------------------------------
VIDEO_PATH   = ""        # kaali na /kaggle/input-la irukkura mudhal video edukkum
OUT_DIR      = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
WORK         = os.path.join(OUT_DIR, "work")
OUTPUT_VIDEO = os.path.join(OUT_DIR, "dubbed_tamil.mp4")

TEXT_MODEL   = "gemini-3.8-flash"            # video puriyum + Tamil mozhipeyarpu
TTS_MODEL    = "gemini-3.8-flash-tts"        # nalla quality; fail na lite-ku maarum
TTS_FALLBACK = "gemini-3.8-flash-lite-tts"

CHUNK_MIN    = 3         # video-va ivlo nimisham thundaa pirichu anuppum
PROXY_HEIGHT = 480       # Gemini-ku anuppa chinna video (upload vegam)
MAX_SPEEDUP  = 1.5       # Tamil vari neelamaa irundha max vegappaduthal
TTS_WORKERS  = 3         # ore neraththil TTS call (429 vandha 1-aa kuraikkavum)
BGM_VOLUME   = 0.9
VOICE_VOLUME = 1.0
SR           = 24000     # Gemini TTS output sample rate
# ----------------------------------------------------------------------------


class Cancelled(Exception):
    """User Cancel button-a azhuthinaa idhu varum."""


CANCEL = threading.Event()
_P = lambda msg=None, pct=None: None      # dub() ithai set pannum (progress callback)


def check_cancel():
    if CANCEL.is_set():
        raise Cancelled("cancelled")


def run_cancellable(cmd, on_pct=None):
    """shell command-a odum pothu Cancel check pannum. demucs '%' progress-um padikkum."""
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         start_new_session=True)
    err = bytearray()

    def reader():
        buf = b""
        while True:
            ch = p.stderr.read(1)
            if not ch:
                break
            err.extend(ch)
            if ch in (b"\r", b"\n"):
                m = re.findall(rb"(\d+)%", buf)
                if m and on_pct:
                    try:
                        on_pct(min(100, int(m[-1])))
                    except Exception:  # noqa
                        pass
                buf = b""
            else:
                buf += ch

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    while p.poll() is None:
        if CANCEL.is_set():
            try:
                os.killpg(os.getpgid(p.pid), 9)
            except Exception:  # noqa
                p.kill()
            t.join(2)
            raise Cancelled("cancelled")
        time.sleep(0.5)
    t.join(5)
    return subprocess.CompletedProcess(cmd, p.returncode, "", bytes(err).decode("utf-8", "ignore"))


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
    try:
        import edge_tts  # noqa
    except ImportError:
        sh(f"{sys.executable} -m pip install -q edge-tts", check=False)
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
    check_cancel()
    _P("🎼 BGM பிரிக்கிறேன்...", 10)
    r = run_cancellable(f'{sys.executable} -m demucs --two-stems=vocals -n htdemucs '
                        f'-o "{WORK}/demucs" "{full}"',
                        on_pct=lambda x: _P("🎼 BGM பிரிக்கிறேன்...", 10 + x * 0.2))
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
- Example: "இப்போ நான் இங்கே வந்தால் என்ன செய்வேன்" -> "இப்போ நான் இங்க வந்தா என்ன பண்ணுவேன்".
  Always: -ால் (if) -> -ா, இங்கே/அங்கே/எங்கே -> இங்க/அங்க/எங்க, செய் -> பண்ணு, -கிறேன் -> -றேன்.
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
    pieces = []                                   # perisa gap-a 45s thundaa pirikkirom
    for gs, ge in gaps:
        a = gs
        while ge - a > 45:
            pieces.append((a, a + 45))
            a += 45
        pieces.append((a, ge))
    gaps = pieces
    print(f"  {len(gaps)} idathula pechu miss aachu, marubadi kekkurom...")
    added = 0
    for k, (gs, ge) in enumerate(gaps):
        check_cancel()
        cs = max(0.0, gs - 0.7)
        ce = min(total, ge + 0.7)
        if ce - cs < 2.5:
            ce = min(total, cs + 2.5)
        clip = os.path.join(WORK, f"gap_{uuid.uuid4().hex[:6]}_{k}.mp4")
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


def speech_segments(mask, merge=0.4, min_len=0.25):
    """Mask-la irundhu 'kural irukkura' (start, end) thundugal."""
    segs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            segs.append([i * FRAME, j * FRAME])
            i = j
        else:
            i += 1
    merged = []
    for sg in segs:
        if merged and sg[0] - merged[-1][1] < merge:
            merged[-1][1] = sg[1]
        else:
            merged.append(sg)
    return [m for m in merged if m[1] - m[0] >= min_len]


def align_lines(lines, mask, maxshift=15.0):
    """Oru chunk-la ellaa varigalum orey alavu munnaadi/pinnaadi irundhaa (Gemini neram thappu),
    asal pechoda match aagura shift-a kandupidichu motha chunk-aiyum nagarthum."""
    if len(lines) < 3 or len(mask) == 0:
        return 0.0
    K = int(maxshift / FRAME)
    a0 = max(0, int(min(l["start"] for l in lines) / FRAME))
    a1 = min(len(mask), int(max(l["end"] for l in lines) / FRAME) + 1)
    if a1 - a0 < 20:
        return 0.0
    act = np.zeros(a1 - a0)
    for l in lines:
        s_ = max(0, int(l["start"] / FRAME) - a0)
        e_ = min(len(act), int(l["end"] / FRAME) - a0 + 1)
        if e_ > s_:
            act[s_:e_] = 1.0
    m = np.where(mask, 1.0, -1.0)
    lo, hi = a0 - K, a1 + K
    win = np.full(hi - lo, -1.0)
    sl, sh_ = max(0, lo), min(len(m), hi)
    win[sl - lo: sh_ - lo] = m[sl:sh_]
    sc = np.correlate(win, act, mode="valid")          # 2K+1 shifts
    j = int(np.argmax(sc))
    if j == K or sc[j] - sc[K] < 0.08 * act.sum():
        return 0.0
    shift = (j - K) * FRAME
    for l in lines:
        l["start"] = round(max(0.0, l["start"] + shift), 2)
        l["end"] = round(max(l["start"] + 0.2, l["end"] + shift), 2)
    return shift


def snap_lines(lines, segs, window=5.0):
    """Gemini sollura neram konjam thappa irukkum. Asal kural nijama kekkura idathukku line-a nagarthum
    (adhaan 'kural oru pakkam, video oru pakkam' prachanai-ku theervu)."""
    if not segs:
        return 0

    def speech_ov(a, b):
        return sum(max(0.0, min(b, e) - max(a, s_)) for s_, e in segs if e > a - 1 and s_ < b + 1)

    moved, prev_end = 0, 0.0
    for ln in sorted(lines, key=lambda x: x["start"]):
        a, b = ln["start"], ln["end"]
        d = max(0.3, b - a)
        if speech_ov(a, b) >= 0.5 * d:            # already asal pechu mela than irukku
            prev_end = max(prev_end, ln["end"])
            continue
        best, bd = None, None
        for s_, e in segs:
            if s_ < a - window:
                continue
            if s_ > a + window:
                break
            if s_ < prev_end - 0.2:
                continue
            if any(o is not ln and s_ - 0.1 <= o["start"] <= s_ + d for o in lines):
                continue                          # anga vera line already irukku
            dist = abs(s_ - a)
            if bd is None or dist < bd:
                best, bd = s_, dist
        if best is not None:
            ln["start"], ln["end"] = round(best, 2), round(best + d, 2)
            moved += 1
        prev_end = max(prev_end, ln["end"])
    lines.sort(key=lambda x: x["start"])
    return moved


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


# Ezhuthu-Tamil -> pechu-Tamil. Gemini sila neram formal-aa ezhudhinaalum idhu pidichu maatthum.
_COLLOQ = [
    (r"வருகிறேன்", "வரேன்"), (r"வருகிறாய்", "வர்ற"), (r"வருகிறார்கள்", "வராங்க"), (r"வருகிறார்", "வராரு"),
    (r"வருகிறான்", "வரான்"), (r"வருகிறாள்", "வராள்"), (r"வருகிறது", "வருது"), (r"வருவேன்", "வருவேன்"),
    (r"போகிறேன்", "போறேன்"), (r"போகிறாய்", "போற"), (r"போகிறார்கள்", "போறாங்க"), (r"போகிறான்", "போறான்"),
    (r"போகிறது", "போகுது"), (r"செய்வேன்", "பண்ணுவேன்"), (r"செய்வாய்", "பண்ணுவ"), (r"செய்வோம்", "பண்ணுவோம்"),
    (r"செய்கிறேன்", "பண்றேன்"), (r"செய்கிறாய்", "பண்ற"), (r"செய்கிறார்கள்", "பண்றாங்க"), (r"செய்கிறான்", "பண்றான்"),
    (r"செய்யுங்கள்", "பண்ணுங்க"), (r"செய்யலாம்", "பண்ணலாம்"), (r"செய்யவில்லை", "பண்ணல"), (r"செய்ய", "பண்ண"),
    (r"செய்தேன்", "பண்ணேன்"), (r"செய்தாய்", "பண்ண"), (r"செய்தார்", "பண்ணாரு"), (r"செய்து", "பண்ணி"),
    (r"இருக்கிறது", "இருக்கு"), (r"இருக்கிறேன்", "இருக்கேன்"), (r"இருக்கிறாய்", "இருக்க"), (r"இருக்கிறார்கள்", "இருக்காங்க"),
    (r"இருக்கிறார்", "இருக்காரு"), (r"இருக்கிறான்", "இருக்கான்"), (r"இருக்கின்றன", "இருக்கு"),
    (r"சொல்கிறேன்", "சொல்றேன்"), (r"சொல்கிறாய்", "சொல்ற"), (r"சொல்கிறார்கள்", "சொல்றாங்க"), (r"சொல்கிறான்", "சொல்றான்"),
    (r"பார்க்கிறேன்", "பாக்குறேன்"), (r"பார்க்கிறாய்", "பாக்குற"), (r"பார்க்கிறார்கள்", "பாக்குறாங்க"),
    (r"தெரியவில்லை", "தெரியல"), (r"புரியவில்லை", "புரியல"), (r"முடியவில்லை", "முடியல"), (r"வேண்டாம்", "வேணாம்"),
    (r"வேண்டும்", "வேணும்"), (r"இல்லை", "இல்ல"), (r"ஆனால்", "ஆனா"), (r"ஆகையால்", "அதனால"), (r"எனவே", "அதனால"),
    (r"ஏனெனில்", "ஏன்னா"), (r"மற்றும்", "அப்புறம்"), (r"பிறகு", "அப்புறம்"), (r"இப்போது", "இப்போ"), (r"அப்போது", "அப்போ"),
    (r"எப்போது", "எப்போ"), (r"இங்கே", "இங்க"), (r"அங்கே", "அங்க"), (r"எங்கே", "எங்க"), (r"இன்னும்", "இன்னும்"),
    (r"நீங்கள்", "நீங்க"), (r"உங்கள்", "உங்க"), (r"எங்கள்", "எங்க"), (r"அவர்கள்", "அவங்க"), (r"இவர்கள்", "இவங்க"),
    (r"என்னுடைய", "என்னோட"), (r"உன்னுடைய", "உன்னோட"), (r"அவனுடைய", "அவனோட"), (r"அவளுடைய", "அவளோட"),
    (r"எப்படி இருக்கிறீர்கள்", "எப்படி இருக்கீங்க"),
    # if-clauses: வந்தால் -> வந்தா
    (r"(வந்த|போன|சொன்ன|பார்த்த|செய்த|பண்ணின|இருந்த|கேட்ட|கொடுத்த|சென்ற|தெரிந்த|முடிந்த|ஆன)ால்(?![\u0B80-\u0BFF])", r"\1ா"),
    (r"(இருக்க|போக|வர|செய்ய|பார்க்க|சொல்ல)ிறது", r"\1ுது"),
    (r"க்கிறது", "க்குது"), (r"க்கிற", "க்குற"),
    # generic present tense: ...கிறேன் -> ...றேன், ...கிறாய் -> ...ற, ...கிறார்கள் -> ...றாங்க
    (r"கிறார்கள்", "றாங்க"), (r"கிறேன்", "றேன்"), (r"கிறாய்", "ற"), (r"கிறான்", "றான்"), (r"கிறாள்", "றாள்"),
    (r"கிறார்", "றாரு"), (r"கிறோம்", "றோம்"), (r"கிறீர்கள்", "றீங்க"), (r"கிறது", "குது"),
    (r"வில்லை(?![\u0B80-\u0BFF])", "ல"),
]


def colloquial(text):
    for pat, rp in _COLLOQ:
        text = re.sub(pat, rp, text)
    return text


def step_script(video):
    script_path = os.path.join(WORK, "script.json")
    if os.path.exists(script_path):
        print("[2] script.json cache irukku, skip")
        return json.load(open(script_path, encoding="utf-8"))

    print("[2] Gemini-ku video anuppi script eluthurom...")
    proxy = make_proxy(video)
    total = duration(proxy)
    voc0 = os.path.join(WORK, "vocals.wav")
    mask0 = None
    if os.path.exists(voc0):
        try:
            mask0 = speech_mask(voc0)
        except Exception as e:  # noqa
            print("  speech mask fail:", str(e)[:120])
    step = CHUNK_MIN * 60
    chars, lines = {}, []
    n = int(np.ceil(total / step))
    for i in range(n):
        off = i * step
        check_cancel()
        _P(f"📝 Script எழுதுறேன் (பகுதி {i+1}/{n})", 30 + 20 * i / n)
        clip = os.path.join(WORK, f"chunk_{i}.mp4")
        cut_clip(proxy, off, step, clip)
        print(f"  chunk {i+1}/{n} ({off/60:.0f} min-la irundhu)")
        got = None
        for attempt in range(3):                  # oru chunk fail aanaalum vittuda koodaadhu
            check_cancel()
            try:
                got = run_clip(clip, off, chars)
                break
            except Exception as e:  # noqa
                print(f"  chunk {i+1} fail (try {attempt+1}/3): {str(e)[:150]}")
                time.sleep(5)
        got = got or []
        if mask0 is not None and got:
            sh_ = align_lines(got, mask0)
            if sh_:
                print(f"  chunk {i+1}: neram {sh_:+.1f}s sari pannapattathu")
        lines += got
        _P(f"📝 Script எழுதுறேன் (பகுதி {i+1}/{n})", 30 + 20 * (i + 1) / n)

    lines.sort(key=lambda x: x["start"])
    lines = fill_gaps(proxy, lines, chars, total)
    lines = fill_gaps(proxy, lines, chars, total)         # rendavadhu pass: innum miss aana idam irundhaa
    voc = os.path.join(WORK, "vocals.wav")
    if os.path.exists(voc):
        try:
            n_moved = snap_lines(lines, speech_segments(speech_mask(voc)))
            print(f"  {n_moved} varigal asal kural neram-ku nagarthappattana")
        except Exception as e:  # noqa
            print("  snap fail:", str(e)[:150])
    for sp in {l["speaker"] for l in lines}:
        chars.setdefault(sp, {"id": sp, "gender": "male", "age": "adult",
                              "pitch": "mid", "tone": "neutral"})
    polish(lines, chars)
    for ln in lines:                                       # kadaisiyaa pechu-Tamil-a urudhi seyyum
        ln["tamil"] = colloquial(ln["tamil"])
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


_state_tts = {"fails": 0, "dead_until": 0.0}
EDGE_POOL = {
    "male": ["ta-IN-ValluvarNeural", "ta-LK-KumarNeural", "ta-MY-SuryaNeural", "ta-SG-AnbuNeural"],
    "female": ["ta-IN-PallaviNeural", "ta-LK-SaranyaNeural", "ta-MY-KaniNeural", "ta-SG-VenbaNeural"],
}


def assign_edge_voices(chars):
    """Ovvoru character-kum vera voice + vera pitch (ellarukkum ore kural varaadhu)."""
    cnt = {"male": 0, "female": 0}
    out = {}
    for cid, c in chars.items():
        g = c.get("gender", "male")
        g = g if g in cnt else "male"
        k = cnt[g]
        cnt[g] += 1
        voice = EDGE_POOL[g][k % 4]
        hz = {"child": 45, "young": 8, "adult": 0, "elder": -12}.get(c.get("age", "adult"), 0)
        hz += {"high": 14, "mid": 0, "low": -14}.get(c.get("pitch", "mid"), 0)
        hz += (k // 4) * (-10 if k % 8 < 4 else 10)        # 4-kku mela irundhaa pitch maathi
        out[cid] = (voice, int(max(-50, min(50, hz))))
    return out


EMO_RATE = {"angry": 8, "shout": 10, "happy": 6, "scared": 8, "sad": -10, "whisper": -8, "calm": -4}


def edge_tts_np(text, ev, emotion="neutral"):
    """Gemini TTS limit/fail aana edge-tts (ilavasam, limit illa) Tamil kural."""
    import asyncio as _aio
    import edge_tts
    uid = uuid.uuid4().hex[:10]
    mp3 = os.path.join(WORK, f"_e_{uid}.mp3")
    wav = os.path.join(WORK, f"_e_{uid}.wav")
    voice, hz = ev
    rate = EMO_RATE.get(emotion, 0)

    async def go():
        await edge_tts.Communicate(text, voice, rate=f"{rate:+d}%", pitch=f"{hz:+d}Hz").save(mp3)
    try:
        _aio.run(go())
        sh(f'ffmpeg -y -i "{mp3}" -ac 1 -ar {SR} "{wav}"')
        return trim_silence(read_wav(wav))
    finally:
        for t in (mp3, wav):
            if os.path.exists(t):
                os.remove(t)


def speak(text, voice, system, ev, emotion="neutral", cid=None):
    """OmniVoice -> Gemini -> edge-tts. (ethuvum fail aanaa exception)"""
    if cid is not None and OMNI["model"] is not None:
        try:
            a = omni_speak(text, cid)
            if a is not None:
                OMNI["fails"] = 0
                return a
        except Exception as e:  # noqa
            OMNI["fails"] += 1
            print(f"  OmniVoice fail ({OMNI['fails']}): {str(e)[:100]}")
            if OMNI["fails"] >= 5:
                OMNI["off_until"] = time.time() + 300
                OMNI["fails"] = 0
    if time.time() >= _state_tts["dead_until"]:
        try:
            a = tts(text, voice, system)
            _state_tts["fails"] = 0
            return a
        except Exception as e:  # noqa
            _state_tts["fails"] += 1
            print(f"  Gemini TTS fail ({_state_tts['fails']}): {str(e)[:100]}")
            if _state_tts["fails"] >= 3:           # limit pola: konja neram Gemini-a vittuda
                _state_tts["dead_until"] = time.time() + 120
                _state_tts["fails"] = 0
    last = None
    for _ in range(3):
        try:
            return edge_tts_np(text, ev, emotion)
        except Exception as e:  # noqa
            last = e
            time.sleep(2)
    raise RuntimeError(f"edge-tts fail: {str(last)[:100]}")


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


# ------------------- OmniVoice (GPU-la local, limit illaadha Tamil TTS) -------------------
OMNI_ENABLE = os.environ.get("OMNIVOICE", "1") != "0"        # 0 -> OmniVoice off
OMNI_REF = os.environ.get("OMNI_REF", "orig")                # orig = asal nadigar kural clone, edge = Tamil kural reference
EDGE_REF_TEXT = "வணக்கம், நான் உங்க கூட கொஞ்சம் பேசணும். இன்னைக்கு என்ன நடந்துச்சுன்னு சொல்லுங்க."
OMNI = {"model": None, "refs": {}, "tried": False, "fails": 0, "off_until": 0.0}
_omni_lock = threading.Lock()


def omni_load():
    """OmniVoice model-a (Hugging Face-la irundhu) onnu thadavai ethum. GPU illa / fail na None."""
    if OMNI["tried"]:
        return OMNI["model"]
    OMNI["tried"] = True
    if not OMNI_ENABLE:
        return None
    try:
        import torch
        if not torch.cuda.is_available():
            print("  GPU illa -> OmniVoice skip")
            return None
        try:
            import omnivoice  # noqa
        except ImportError:
            print("  omnivoice install aaguthu...")
            sh(f"{sys.executable} -m pip install -q omnivoice", check=False)
            import importlib
            importlib.invalidate_caches()
        from omnivoice import OmniVoice
        print("  OmniVoice model etthuthu (modhal thadavai download aagum)...")
        OMNI["model"] = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0",
                                                  dtype=torch.float16)
        print("  OmniVoice ready")
    except Exception as e:  # noqa
        print("  OmniVoice load fail:", str(e)[:200])
        OMNI["model"] = None
    return OMNI["model"]


def omni_prepare_refs(chars, lines, edge_voices):
    """Ovvoru character-kum oru reference kural clip thayaar pannum."""
    voc = os.path.join(WORK, "vocals.wav")
    mask = None
    if OMNI_REF != "edge" and os.path.exists(voc):
        try:
            mask = speech_mask(voc)
        except Exception:  # noqa
            mask = None
    for cid in chars:
        path = os.path.join(WORK, f"ref_{cid}.wav")
        got = None
        if mask is not None:
            best, bscore = None, 0.0
            for ln in lines:
                if ln.get("speaker") != cid or not ln.get("original"):
                    continue
                d = ln["end"] - ln["start"]
                if not 3.0 <= d <= 10.0:
                    continue
                a, b = int(ln["start"] / FRAME), int(ln["end"] / FRAME)
                frac = float(mask[a:b].mean()) if b > a else 0.0
                score = frac * min(d, 8.0)
                if frac >= 0.6 and score > bscore:
                    best, bscore = ln, score
            if best is not None:
                r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{best['start']:.2f}",
                                    "-t", f"{best['end'] - best['start']:.2f}", "-i", voc,
                                    "-ac", "1", "-ar", str(SR), path], capture_output=True)
                if r.returncode == 0 and os.path.exists(path):
                    got = (path, best["original"])
        if got is None and OMNI_REF != "orig_only":
            try:                                             # asal clip illa -> edge-tts Tamil clip reference
                ev = edge_voices.get(cid, ("ta-IN-ValluvarNeural", 0))
                write_wav(path, edge_tts_np(EDGE_REF_TEXT, ev), SR)
                got = (path, EDGE_REF_TEXT)
            except Exception as e:  # noqa
                print(f"  ref {cid} fail: {str(e)[:100]}")
        if got:
            OMNI["refs"][cid] = got
    print(f"  OmniVoice reference: {len(OMNI['refs'])}/{len(chars)} characters")


def omni_speak(text, cid):
    m, ref = OMNI["model"], OMNI["refs"].get(cid)
    if m is None or ref is None or time.time() < OMNI["off_until"]:
        return None
    with _omni_lock:                                         # oru GPU model -> orey nerathula onnu
        out = m.generate(text=text, ref_audio=ref[0], ref_text=ref[1])
    a = np.asarray(out[0] if isinstance(out, (list, tuple)) else out, dtype=np.float32).reshape(-1)
    if len(a) < int(0.2 * SR):
        raise RuntimeError("OmniVoice too short")
    return trim_silence(a)


def step_tts(script):
    chars, lines = script["characters"], script["lines"]
    voices = assign_voices(chars)
    edge_voices = assign_edge_voices(chars)
    print("[3] Kural thervu:", {k: voices[k] for k in voices})
    print(f"[4] {len(lines)} varigalukku Tamil kural uruvaakkurom...")
    if omni_load():
        _P("🧠 OmniVoice குரல்களை தயார் பண்றேன்...", 50)
        omni_prepare_refs(chars, lines, edge_voices)

    def work(i):
        check_cancel()
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
        ev = edge_voices.get(ln["speaker"], ("ta-IN-ValluvarNeural", 0))
        emo = ln.get("emotion", "neutral")
        try:
            audio = speak(text, voices.get(ln["speaker"], "Kore"), system, ev, emo, ln["speaker"])
        except Exception as e:  # noqa
            print(f"  line {i} TTS fail: {str(e)[:120]}")
            return i, np.zeros(int(0.1 * SR), dtype=np.float32)
        dur = len(audio) / SR
        if dur > allowed * 1.25 and len(text) > 12:
            try:
                short = colloquial(shorten(text, max(0.5, allowed / dur)))
                audio = speak(short, voices.get(ln["speaker"], "Kore"), system, ev, emo, ln["speaker"])
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
            _P(f"🎙️ தமிழ் குரல் உருவாக்குறேன் ({k+1}/{len(lines)} வரிகள்)", 50 + 40 * (k + 1) / max(1, len(lines)))
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
            check_cancel()
            _, results[i] = work(i)
    failed = [i for i in range(len(lines))
              if not os.path.exists(os.path.join(WORK, "tts", f"{i:05d}.wav"))]
    STATS["tts_failed"] = len(failed)
    return [results[i] for i in range(len(lines))]


# ----------------------------- STEP 5: mix ----------------------------------
def keep_original_where_undubbed(track, lines, audios):
    """Dub kural varaadha idathula 'music mattum' kekkaamal, asal kural-a (vocals.wav) serkkum."""
    voc = os.path.join(WORK, "vocals.wav")
    if not os.path.exists(voc):
        return 0
    try:
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", voc, "-ac", "1", "-ar", str(SR),
                            "-f", "s16le", "-"], capture_output=True)
        x = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        segs = speech_segments(speech_mask(voc))
    except Exception as e:  # noqa
        print("  original voice check fail:", str(e)[:120])
        return 0
    spans = []
    for ln, a in zip(lines, audios):
        dur = len(a) / SR
        if dur > 0.3:                      # fail aana (silent) varigal count aagaadhu
            spans.append((ln["start"], max(ln["end"], ln["start"] + dur) + 0.2))
    kept = 0
    for s_, e_ in segs:
        if any(min(e_, b) - max(s_, a) > 0.15 for a, b in spans):
            continue
        i0, i1 = int(max(0, s_ - 0.1) * SR), min(len(x), len(track), int((e_ + 0.1) * SR))
        if i1 - i0 < int(0.2 * SR):
            continue
        seg = x[i0:i1].copy()
        f = min(len(seg) // 2, int(0.05 * SR))
        if f > 0:
            seg[:f] *= np.linspace(0, 1, f)
            seg[-f:] *= np.linspace(1, 0, f)
        track[i0:i1] += seg * 0.9
        kept += 1
    print(f"  {kept} idathula dub illa -> asal kural serthom")
    return kept


def step_mix(video, bgm, lines, audios):
    check_cancel()
    print("[5] Ellathaiyum serkkirom...")
    total = duration(video)
    track = np.zeros(int((total + 5) * SR), dtype=np.float32)
    for ln, a in zip(lines, audios):
        s = int(ln["start"] * SR)
        e = min(len(track), s + len(a))
        if s < len(track):
            track[s:e] += a[: e - s]
    STATS["orig_kept"] = keep_original_where_undubbed(track, lines, audios)
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
def dub(video, job_dir, progress=print, fmt="mp4", cancel=None):
    """Oru video-va Tamil-la dub panni output path-a tharum. (Bot idhai kooppidum)"""
    global WORK, OUTPUT_VIDEO, CANCEL, _P
    CANCEL = cancel if cancel is not None else threading.Event()

    def _P(msg=None, pct=None):  # noqa: F811  (progress + percentage)
        try:
            progress(msg, pct)
        except TypeError:          # pazhaya progress(msg) mattum edukkum function
            progress(msg)
        except Exception:  # noqa
            pass
    WORK = os.path.join(job_dir, "work")
    fmt = "mkv" if str(fmt).lower() == "mkv" else "mp4"
    OUTPUT_VIDEO = os.path.join(job_dir, f"dubbed_tamil.{fmt}")
    os.makedirs(os.path.join(WORK, "tts"), exist_ok=True)
    _P(f"Video: {duration(video)/60:.1f} நிமிடம். BGM பிரிக்கிறேன்...", 10)
    bgm = step_bgm(video)
    check_cancel()
    _P("📝 Script எழுதுறேன் (யார் என்ன பேசுறாங்க + பேச்சுத் தமிழ்)...", 30)
    script = step_script(video)
    ls = script["lines"]
    span = f" ({ls[0]['start']:.0f}s - {ls[-1]['end']:.0f}s)" if ls else ""
    _P(f"{len(ls)} வரிகள்{span}, {len(script['characters'])} characters. தமிழ் குரல் உருவாக்குறேன்...", 50)
    audios = step_tts(script)
    if STATS["tts_failed"]:
        progress(f"⚠️ {STATS['tts_failed']} வரிகளுக்கு குரல் வரல (Gemini limit). மறுபடி அனுப்பிப் பாருங்க.")
    _P("🎬 எல்லாத்தையும் சேர்க்கிறேன்...", 92)
    step_mix(video, bgm, script["lines"], audios)
    if STATS.get("orig_kept"):
        progress(f"⚠️ {STATS['orig_kept']} இடங்களுக்கு dub குரல் கிடைக்கல, அங்க அசல் குரல் இருக்கும்.")
    return OUTPUT_VIDEO


if __name__ == "__main__":
    video = find_video()
    dub(video, OUT_DIR)
