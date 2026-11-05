"""
Multi-language Auto-Dubbing Pipeline (Tamil / Telugu / Malayalam / Kannada / Hindi)
(Kaggle notebook-ku)
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


# ----------------------------- LANGUAGES ------------------------------------
# code: name, native name, ISO-639-2 tag (video track language), Unicode range of its script,
#       edge-tts voices, reference sentence (OmniVoice), spoken-style note for the translator
LANGS = {
    "ta": dict(name="Tamil", native="தமிழ்", tag="tam", code="ta", rng=(0x0B80, 0x0BFF), script="Tamil",
               edge={"male": ["ta-IN-ValluvarNeural", "ta-LK-KumarNeural", "ta-MY-SuryaNeural", "ta-SG-AnbuNeural"],
                     "female": ["ta-IN-PallaviNeural", "ta-LK-SaranyaNeural", "ta-MY-KaniNeural", "ta-SG-VenbaNeural"]},
               ref="வணக்கம், நான் உங்க கூட கொஞ்சம் பேசணும். இன்னைக்கு என்ன நடந்துச்சுன்னு சொல்லுங்க, நான் கேக்குறேன்.",
               style="Casual Chennai spoken Tamil (இல்ல, வரேன், பண்றேன், நீங்க, அவங்க), never literary Tamil."),
    "te": dict(name="Telugu", native="తెలుగు", tag="tel", code="te", rng=(0x0C00, 0x0C7F), script="Telugu",
               edge={"male": ["te-IN-MohanNeural"], "female": ["te-IN-ShrutiNeural"]},
               ref="నమస్కారం, నేను మీతో కొంచెం మాట్లాడాలి. ఈ రోజు ఏం జరిగిందో చెప్పండి, నేను వింటున్నాను.",
               style="Everyday spoken Telugu (వ్యవహారిక భాష) as in Telugu-dubbed movies, never bookish/grandhika Telugu."),
    "ml": dict(name="Malayalam", native="മലയാളം", tag="mal", code="ml", rng=(0x0D00, 0x0D7F), script="Malayalam",
               edge={"male": ["ml-IN-MidhunNeural"], "female": ["ml-IN-SobhanaNeural"]},
               ref="നമസ്കാരം, എനിക്ക് നിങ്ങളോട് കുറച്ച് സംസാരിക്കണം. ഇന്ന് എന്താണ് സംഭവിച്ചതെന്ന് പറയൂ, ഞാൻ കേൾക്കുന്നുണ്ട്.",
               style="Everyday spoken Malayalam (സംസാരഭാഷ) as in Malayalam-dubbed movies, never bookish Malayalam."),
    "kn": dict(name="Kannada", native="ಕನ್ನಡ", tag="kan", code="kn", rng=(0x0C80, 0x0CFF), script="Kannada",
               edge={"male": ["kn-IN-GaganNeural"], "female": ["kn-IN-SapnaNeural"]},
               ref="ನಮಸ್ಕಾರ, ನಾನು ನಿಮ್ಮ ಜೊತೆ ಸ್ವಲ್ಪ ಮಾತನಾಡಬೇಕು. ಇವತ್ತು ಏನಾಯಿತು ಹೇಳಿ, ನಾನು ಕೇಳುತ್ತಿದ್ದೇನೆ.",
               style="Everyday spoken Kannada (ಆಡುಮಾತು) as in Kannada-dubbed movies, never bookish Kannada."),
    "hi": dict(name="Hindi", native="हिन्दी", tag="hin", code="hi", rng=(0x0900, 0x097F), script="Devanagari (Hindi)",
               edge={"male": ["hi-IN-MadhurNeural"], "female": ["hi-IN-SwaraNeural"]},
               ref="नमस्ते, मुझे आपसे थोड़ी बात करनी है। आज क्या हुआ, बताइए, मैं सुन रहा हूँ।",
               style="Natural everyday spoken Hindi (बोलचाल की हिंदी) as in Hindi-dubbed movies and OTT series; simple words, not shuddh/Sanskritised Hindi."),
}
LANG_ORDER = ["ta", "te", "ml", "kn", "hi"]


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


def gpu_info():
    """('cuda', 'Tesla T4') GPU irundhaa, illaiyaa ('cpu', 'CPU')."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", torch.cuda.get_device_name(0)
    except Exception:  # noqa
        pass
    return "cpu", "CPU"


# ----------------------------- STEP 1: BGM ----------------------------------
def step_bgm(video):
    bgm_path = os.path.join(WORK, "bgm.wav")
    if os.path.exists(bgm_path):
        print("[1] BGM cache irukku, skip")
        return bgm_path
    dev, dev_name = gpu_info()
    tag = "GPU" if dev == "cuda" else "CPU - மெதுவா"
    print(f"[1] Audio eduthu BGM pirikkirom (device: {dev_name})...")
    if dev != "cuda":
        print("  !! GPU illa -> BGM pirikka romba neram aagum. Kaggle Settings-la GPU T4 on pannunga.")
    full = os.path.join(WORK, "full_audio.wav")
    sh(f'ffmpeg -y -i "{video}" -vn -ac 2 -ar 44100 "{full}"')
    check_cancel()
    _P(f"🎼 BGM பிரிக்கிறேன் ({tag})...", 10)
    r = run_cancellable(f'{sys.executable} -m demucs -d {dev} --two-stems=vocals -n htdemucs '
                        f'-o "{WORK}/demucs" "{full}"',
                        on_pct=lambda x: _P(f"🎼 BGM பிரிக்கிறேன் ({tag})...", 10 + x * 0.2))
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
Write the Tamil exactly like a real Tamil speaker talks in a casual voice message to a friend: relaxed slang,
one consistent casual register, never formal. Imagine real friends / family chatting casually and write what THAT character would really say in THAT situation
(a kid talks like a kid, an angry man like an angry man, a boss like a boss). Use everyday English words in Tamil
script when natural (டைம், பிளான், ப்ராப்ளம், ஓகே, சாரி, ப்ளீஸ்).
Never use words like மற்றும், ஏனெனில், எனவே. Use natural interjections (டா, சரி, ஆமா, அய்யோ) only where they fit.
Write in Tamil script. Common loanwords are fine (ஓகே, சாரி, போன்). No Hindi/English sentences.
Keep meaning, emotion and length close to the original so it fits the same time. Short, speakable sentences.

SCENE-AWARE DUBBING (VERY IMPORTANT): do NOT translate word by word. First understand the situation on screen:
where they are, what just happened, who is talking to whom (friend, kid, parent, boss, enemy, stranger), and the mood.
Then write what a real Tamil person would SAY in that exact situation, even if the wording is different from the original.
If the original is a formal or bookish sentence, say the same thing the way people actually speak. Prefer short natural
phrases, fillers and reactions (ஆமா, சரி, என்னடா, அட, ஹ்ம்ம், ச்சே, அய்யோ) over full textbook sentences.
Match respect level to the relationship: friends/younger use நீ/டா/டி, elders/strangers/boss use நீங்க/சார்/அண்ணா/அக்கா.
Read every line aloud in your head; if it sounds like reading from a book or a news bulletin, rewrite it.

Speakers: use the video (who is on screen, who is speaking) and the voices to decide who speaks.
Use stable IDs like C1, C2, C3. The same person must ALWAYS get the same ID.
{roster}

Timestamps are relative to the START OF THIS CLIP in MM:SS.s format (e.g. 00:12.4).
Skip songs/singing/background noise. Include narrator voice-over as its own character.

Return ONLY JSON:
{{
 "characters": [ {{"id":"C1","gender":"male|female","age":"child|young|adult|elder",
                   "pitch":"low|mid|high","tone":"short English description of voice/personality",
                   "look":"short English note: face/hair/clothes/role so the same person is recognised in later clips",
                   "is_new": true}} ],
 "lines": [ {{"start":"00:01.2","end":"00:03.9","speaker":"C1",
              "original":"original words","english":"short natural English meaning","tamil":"colloquial Tamil",
              "to":"who they are talking to and relation, e.g. 'C2 (best friend)' or 'child' or 'boss'",
              "scene":"very short English note: what is happening in this moment",
              "emotion":"neutral|angry|sad|happy|scared|whisper|shout|calm"}} ]
}}
Only list characters that appear in this clip in "characters"."""


POLISH_PROMPT = """You are the top dialogue writer for Tamil-dubbed movies and web series (the kind of dub
people watch on TV/OTT). Rewrite the Tamil of every line below so it sounds like REAL spoken Tamil as people
talk in Chennai, as in a professional movie dub. NOT written / formal / literary Tamil.

STYLE REFERENCE - this is exactly how the audience talks (a real Tamil speaker's voice messages). Match THIS register:
  "நான் உங்ககிட்ட பேசுறேன்ல, நான் பேசுறது உனக்கு புரியுதுல்ல"
  "இப்போ நான் இங்க வந்தா என்ன பண்ணுவேன்"
  "எனக்கு என்ன பிரச்சனைன்னா, இது சரியா வேலை செய்யல"
Typical features: -ல / -ல்ல question tags, ன்னு / -னா endings, இங்க/அங்க/எங்க, பண்ணு, -றேன்/-றோம், -ங்க.
Keep ONE consistent casual register in every sentence. Never mix formal and casual words in the same sentence
(e.g. do not say "நாங்கள் இங்கு வந்தோம், அவர்தான் வந்தாங்க" -> say "நாங்க இங்க வந்தோம், அவருதான் வந்தாரு").

DUBBING STYLE: write it like a good modern Tamil dub of an anime / Hollywood / Korean series (the kind kids and teens watch
on Pogo, Netflix or Amazon): lively Chennai street-casual Tamil, short punchy lines, same energy as the original.
Friends say டா/டி, வாடா, போடா, மச்சி; juniors say அண்ணா/அக்கா/சார்; fights and shouts stay short and loud
("டேய்! என்ன பண்ற நீ?", "நான் உன்ன காப்பாத்துவேன், கண்டிப்பா!", "அது முடியாது, நிறுத்து!").
Never sound like a textbook, a news reader or a speech. If a line has both a formal word and a casual word, rewrite the whole line casual.

MOST IMPORTANT: use the scene info. Each item has "scene" (what is happening), "to" (who is being spoken to) and
prev2/prev/next/next2 (neighbouring lines). Do NOT translate the English literally. Ask: in THIS situation, what would a real
Tamil person blurt out? Rewrite so it flows like a real conversation with the previous and next line (answers, reactions,
"ஆமா", "இல்ல", "ஹே", "ச்சே"), not like separate sentences read from a script. Drop stiff phrases and use the short everyday way.
Bookish -> spoken examples:
  "நீங்கள் என்ன சொல்ல விரும்புகிறீர்கள்?" -> "என்ன சொல்ல வர்றீங்க?"
  "நான் அதைப் பற்றி யோசித்துக் கொண்டிருக்கிறேன்" -> "அதான் யோசிச்சிட்டு இருக்கேன்"
  "எனக்கு உன்னுடைய உதவி தேவைப்படுகிறது" -> "எனக்கு உன் உதவி வேணும்"
  "இது எப்படி சாத்தியம்?" -> "இது எப்படி முடியும்?"
  "அவசியம் இல்லை" -> "அதெல்லாம் வேணாம்"

THINK LIKE THIS: imagine this exact scene happening between real friends / family / colleagues in Chennai.
How would THIS character say it, in THIS situation, to THIS person? Casual, relaxed, a bit lazy, like talking
on the phone with a friend. It should never sound like a speech, a book or a news reader.
Fully re-write the sentence if needed. Do not just swap a few words.
Formal -> casual examples:
  "நாம் இங்கே வந்திருக்கிறோம், அதனால் தான் இங்கே போர் நடக்கிறது" -> "நாம இங்க வந்திருக்கோம், அதனாலதான் இங்க சண்டையே நடக்குது"
  "நீ என்ன செய்து கொண்டிருக்கிறாய்?" -> "நீ என்ன பண்ணிட்டு இருக்க?"
  "நான் உன்னிடம் ஒன்று சொல்ல வேண்டும்" -> "உன்கிட்ட ஒண்ணு சொல்லணும்"
  "தயவுசெய்து என்னை மன்னித்து விடுங்கள்" -> "ப்ளீஸ் என்ன மன்னிச்சிடுங்க"
  "அவர்கள் எங்கே சென்றிருக்கிறார்கள் என்று தெரியவில்லை" -> "அவங்க எங்க போயிருக்காங்கன்னே தெரியல"
Use everyday English words in Tamil script the way people really talk (டைம், பிளான், ப்ராப்ளம், ஃப்ரெண்ட், ஹெல்ப், சீக்கிரம்,
ஓகே, சாரி, ப்ளீஸ்) when it feels natural. Contractions and drops are good: -ிட்டு, -ிருக்கேன், -னு, -ன்னு, -ல்ல.
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

Input is JSON: each item has id, who, emotion, en (original words), ta (current Tamil), prev/next (neighbouring lines, only for\ncontext of the scene - do not rewrite them).
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
        roster = ("Characters already known from earlier clips. The SAME person must ALWAYS keep the SAME id. "
                  "Match by face, clothes/look, voice and role (use each character's 'look' note). The main hero and "
                  "recurring characters must never get a second id. Create a NEW id ONLY for a person who clearly "
                  "has never appeared before; when unsure, reuse the closest existing id:\n" +
                  json.dumps(chars, ensure_ascii=False))
    prompt = SCRIPT_PROMPT.format(roster=roster)
    resp = gen([f, prompt],
               types.GenerateContentConfig(response_mime_type="application/json",
                                           temperature=0.6), what="script")
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
                    "english": ln.get("english", ""),
                    "tamil": ln["tamil"].strip(),
                    "to": ln.get("to", ""),
                    "scene": ln.get("scene", ""),
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


def polish(lines, chars, strict=False):
    """Ellaa varigalaiyum 'movie dub' pechu Tamil-a maatri ezhuthum."""
    print("  Tamil-a movie-dub pechu nadaikku maathurom..." + (" (strict)" if strict else ""))
    pos = {id(l): n for n, l in enumerate(lines)}
    B = 30
    done = 0
    for i in range(0, len(lines), B):
        batch = lines[i: i + B]
        items = []
        for k, ln in enumerate(batch):
            c = chars.get(ln["speaker"], {})
            n_ = pos.get(id(ln), 0)
            prev_ = lines[n_ - 1]["tamil"] if n_ > 0 else ""
            next_ = lines[n_ + 1]["tamil"] if n_ + 1 < len(lines) else ""
            prev2_ = lines[n_ - 2]["tamil"] if n_ > 1 else ""
            next2_ = lines[n_ + 2]["tamil"] if n_ + 2 < len(lines) else ""
            items.append({"id": k, "who": f"{c.get('age','adult')} {c.get('gender','male')} ({c.get('tone','')})",
                          "emotion": ln.get("emotion", "neutral"),
                          "scene": ln.get("scene", ""), "to": ln.get("to", ""),
                          "en": ln.get("english", "") or ln.get("original", ""), "ta": ln["tamil"],
                          "prev2": prev2_, "prev": prev_, "next": next_, "next2": next2_})
        prompt = POLISH_PROMPT.format(items=json.dumps(items, ensure_ascii=False))
        if strict:
            prompt = ("IMPORTANT: these lines are STILL too formal / bookish. Throw the old Tamil away and write each "
                      "line again from scratch, the way a close friend would actually say it in a casual chat.\n\n"
                      + prompt)
        try:
            resp = gen(prompt, types.GenerateContentConfig(
                response_mime_type="application/json", temperature=0.7), what="polish")
            data = parse_json(resp.text)
            for it in data.get("lines", []):
                k = int(it["id"])
                t = str(it.get("tamil", "")).strip()
                if 0 <= k < len(batch) and t and len(t) <= len(batch[k]["tamil"]) * 1.8 + 10:
                    batch[k]["tamil_formal"] = batch[k]["tamil"]
                    batch[k]["tamil"] = t
                    done += 1
        except Exception as e:  # noqa
            print("  polish fail (original Tamil-ae use aagum):", str(e)[:150])
    print(f"  {done}/{len(lines)} varigal maathapattana")


# Ezhuthu-Tamil -> pechu-Tamil. Gemini sila neram formal-aa ezhudhinaalum idhu pidichu maatthum.
_COLLOQ = [
    (r"அனைவரும்", "எல்லாரும்"), (r"அனைத்தும்", "எல்லாமே"), (r"அனைத்து", "எல்லா"), (r"யாவரும்", "எல்லாரும்"),
    (r"அவ்வாறு", "அப்படி"), (r"இவ்வாறு", "இப்படி"), (r"எவ்வாறு", "எப்படி"), (r"தாங்கள்", "நீங்க"),
    (r"உள்ளனர்", "இருக்காங்க"), (r"உள்ளார்", "இருக்காரு"), (r"உள்ளது", "இருக்கு"), (r"உள்ளன", "இருக்கு"),
    (r"குழந்தைகள்", "குழந்தைங்க"), (r"மனிதர்கள்", "மனுஷங்க"), (r"பள்ளிக்கு", "ஸ்கூலுக்கு"), (r"பள்ளியில்", "ஸ்கூல்ல"), (r"பள்ளி", "ஸ்கூல்"), (r"ஆசிரியர்", "டீச்சர்"),
    (r"மட்டுமே", "மட்டும்தான்"), (r"தயாராக", "ரெடியா"), (r"தயார்", "ரெடி"), (r"மன்னிக்கவும்", "சாரி"),
    (r"வாருங்கள்", "வாங்க"), (r"போங்கள்", "போங்க"),
    (r"பேசுவது", "பேசுறது"), (r"செய்வது", "பண்றது"), (r"வருவது", "வர்றது"), (r"போவது", "போறது"), (r"சொல்வது", "சொல்றது"),
    (r"நாங்கள் இங்கு", "நாங்க இங்க"), (r"அவர்தான்", "அவருதான்"), (r"இவர்தான்", "இவருதான்"), (r"அவள்தான்", "அவதான்"),
    (r"அவன்தான்", "அவன்தான்"), (r"அவர் தான்", "அவருதான்"), (r"இங்கு தான்", "இங்கதான்"), (r"இங்க தான்", "இங்கதான்"),
    (r"அங்கு தான்", "அங்கதான்"), (r"நான்தான்", "நான்தான்"), (r"வந்தோம்", "வந்தோம்"), (r"இருக்கிறீர்களா", "இருக்கீங்களா"),
    (r"புரிகிறதா", "புரியுதா"), (r"புரிந்ததா", "புரிஞ்சுதா"), (r"புரிந்தது", "புரிஞ்சுது"), (r"தெரியுமா", "தெரியுமா"),
    (r"கொள்ளுங்கள்", "கோங்க"), (r"கொண்டு", "கிட்டு"), (r"விட்டார்கள்", "விட்டாங்க"), (r"விட்டான்", "விட்டான்"),
    (r"அவர்களுக்கு", "அவங்களுக்கு"), (r"அவர்களை", "அவங்கள"), (r"அவர்களின்", "அவங்களோட"), (r"அவர்களது", "அவங்களோட"),
    (r"நம்முடைய", "நம்மளோட"), (r"உங்களுடைய", "உங்களோட"), (r"அவனுடைய", "அவனோட"), (r"நம்மை", "நம்மள"),
    (r"ன்\s+என்று(?![\u0B80-\u0BFF])", "ன்னு"), (r"ம்\s+என்று(?![\u0B80-\u0BFF])", "ம்னு"),
    (r"\s+என்று(?![\u0B80-\u0BFF])", "ன்னு"), (r"ஒன்று", "ஒண்ணு"),
    (r"சென்றிரு", "போயிரு"), (r"சென்றுவிட", "போயிட"),
    (r"அதனால் தான்", "அதனாலதான்"), (r"அதனால்தான்", "அதனாலதான்"), (r"இதனால்", "இதனால"), (r"அதனால்", "அதனால"),
    (r"தெரிகிறது", "தெரியுது"), (r"புரிகிறது", "புரியுது"), (r"தெரிகிறேன்", "தெரியுறேன்"),
    (r"வந்திருக்கிறோம்", "வந்திருக்கோம்"), (r"இருக்கிறோம்", "இருக்கோம்"), (r"கின்றது", "குது"), (r"கின்ற", "கிற"),
    (r"நாங்கள்", "நாங்க"), (r"\bநாம்(?![\u0B80-\u0BFF])", "நாம"), (r"மிகவும்", "ரொம்ப"), (r"தயவுசெய்து", "ப்ளீஸ்"),
    (r"நிச்சயமாக", "கண்டிப்பா"), (r"ஆமாம்", "ஆமா"), (r"உண்மையில்", "உண்மையா"), (r"அல்லது", "இல்லன்னா"),
    (r"ஏனென்றால்", "ஏன்னா"), (r"இருப்பினும்", "இருந்தாலும்"), (r"எனினும்", "இருந்தாலும்"), (r"ஆகவே", "அதனால"),
    (r"என்னிடம்", "என்கிட்ட"), (r"உன்னிடம்", "உன்கிட்ட"), (r"அவனிடம்", "அவன்கிட்ட"), (r"அவரிடம்", "அவர்கிட்ட"),
    (r"அவளிடம்", "அவகிட்ட"), (r"உங்களிடம்", "உங்ககிட்ட"), (r"நம்மிடம்", "நம்மகிட்ட"), (r"பற்றி", "பத்தி"),
    (r"அங்கு", "அங்க"), (r"இங்கு", "இங்க"), (r"எங்கு", "எங்க"), (r"இன்று", "இன்னைக்கு"), (r"நேற்று", "நேத்து"),
    (r"இவ்வளவு", "இவ்ளோ"), (r"அவ்வளவு", "அவ்ளோ"), (r"எவ்வளவு", "எவ்ளோ"), (r"ஒன்றும்", "ஒண்ணும்"),
    (r"இரண்டு", "ரெண்டு"), (r"மூன்று", "மூணு"), (r"நான்கு", "நாலு"), (r"ஐந்து", "அஞ்சு"),
    (r"சென்றேன்", "போனேன்"), (r"சென்றோம்", "போனோம்"), (r"சென்றான்", "போனான்"), (r"சென்றாள்", "போனாள்"),
    (r"சென்றார்கள்", "போனாங்க"), (r"சென்றார்", "போனாரு"), (r"சென்று", "போய்"),
    (r"கூறினான்", "சொன்னான்"), (r"கூறினாள்", "சொன்னாள்"), (r"கூறினார்", "சொன்னாரு"), (r"கூறு", "சொல்லு"),
    (r"என்றான்", "சொன்னான்"), (r"என்றாள்", "சொன்னாள்"), (r"என்றார்", "சொன்னாரு"),
    (r"தேவையில்லை", "தேவையில்ல"), (r"ாமல்(?![\u0B80-\u0BFF])", "ாம"),
    (r"(?<=\u0BCD)(தால்|றால்|டால்)(?![\u0B80-\u0BFF])", lambda m: m.group(1)[:-2]),
    (r"ுங்கள்", "ுங்க"), (r"ீர்கள்", "ீங்க"), (r"ார்கள்", "ாங்க"),
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
    (r"சென்ற", "போன"),
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


# innum 'pustaga Tamil' maadhiri irukkura varigalai kandupidikka
FORMAL_RE = re.compile(r"(கிறே|கிறா|கிறோ|கிறீ|கிறது|இங்கே|அங்கே|எங்கே|ங்கள்|ார்கள்|ன்றால்|தால்(?![\u0B80-\u0BFF])|"
                       r"வேண்டும்|இல்லை|ஆனால்|மிகவும்|செய்|கொள்கிறே|இயலும்|இயலாது|வேண்டியது|அவர்களை|உள்ளது|உள்ளன|அனைவ|அவ்வாறு|இவ்வாறு|"
                       r"எவ்வாறு|தாங்கள்|யாது|ஆகிய|கொண்டிருக்|ிருக்கின்ற|இருந்தார்கள்|சென்ற|கூறி|அவனுடைய|அவளுடைய|இன்று|நேற்று)")


def _read_script_text(path):
    """.txt/.json mattum illa, .docx / .html-layum irundhu text edukkum."""
    import zipfile, html as _html
    raw_b = open(path, "rb").read()
    if raw_b[:2] == b"PK":                                   # .docx (zip)
        try:
            xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8", "ignore")
            xml = re.sub(r"</w:p>", "\n", xml)
            return _html.unescape(re.sub(r"<[^>]+>", "", xml))
        except Exception:  # noqa
            pass
    txt = raw_b.decode("utf-8-sig", "ignore")
    if re.search(r"<(html|body|script|div)\b", txt[:5000], re.I):   # .html viewer page
        txt = _html.unescape(re.sub(r"<(script|style)[^>]*>|</(script|style)>", "\n", txt, flags=re.I))
    return txt


def _extract_script_parts(raw):
    """Pala JSON paguthi (30 nimisha video-ku 5-5 nimisham) / vettupatta JSON-ilum irundhu characters + lines edukkum."""
    dec = json.JSONDecoder()
    chars, lines = {}, []
    i = 0
    while True:
        i = raw.find("{", i)
        if i < 0:
            break
        try:
            obj, end = dec.raw_decode(raw, i)
        except Exception:
            i += 1                                           # vettupatta/thappa object -> ullae poi thedu
            continue
        if not isinstance(obj, dict):
            i += 1
            continue
        if "lines" in obj or "characters" in obj:
            cin = obj.get("characters") or {}
            if isinstance(cin, dict):
                cin = [dict(v, id=v.get("id", k)) for k, v in cin.items() if isinstance(v, dict)]
            for c in cin:
                if isinstance(c, dict) and c.get("id"):
                    chars.setdefault(str(c["id"]), c)
            try:                                             # "part_start": 600 -> indha paguthi video-vil 600s-la thodangum
                off = ts_to_sec(obj.get("part_start", 0) or 0)
            except Exception:
                off = 0.0
            for ln in obj.get("lines") or []:
                if isinstance(ln, dict):
                    if off:
                        try:
                            ln = dict(ln, start=ts_to_sec(ln["start"]) + off, end=ts_to_sec(ln["end"]) + off)
                        except Exception:
                            continue
                    lines.append(ln)
            i = end
        elif "start" in obj and "end" in obj:
            lines.append(obj)
            i = end
        elif "id" in obj and "gender" in obj:
            chars.setdefault(str(obj["id"]), obj)
            i = end
        else:
            i += 1
    return chars, lines


def load_external_script(path):
    """Gemini app-la nee eluthina script (JSON / pala paguthi / .txt / .docx)-a bot-oda format-ku maathum."""
    raw = _read_script_text(path)
    chars_in, raw_lines = _extract_script_parts(raw)
    data = {"lines": raw_lines}
    lines = []
    for ln in data.get("lines") or []:
        try:
            st, en = ts_to_sec(ln["start"]), ts_to_sec(ln["end"])
        except Exception:
            continue
        tamil = str(ln.get("tamil", "")).strip()
        if en <= st or not tamil:
            continue
        lines.append({"start": round(st, 2), "end": round(en, 2),
                      "speaker": str(ln.get("speaker", "C1")),
                      "original": str(ln.get("original", "")),
                      "english": str(ln.get("english", "")),
                      "tamil": tamil,
                      "to": str(ln.get("to", "")), "scene": str(ln.get("scene", "")),
                      "emotion": str(ln.get("emotion", "neutral")) or "neutral"})
    if not lines:
        raise ValueError("script-la 'lines' onnum illa (start, end, speaker, tamil venum)")
    lines.sort(key=lambda x: x["start"])
    seen, uniq = set(), []
    for l in lines:                                          # rendu paguthi overlap aanaa irukkura dupe varigalai neekkum
        k = (l["start"], l["end"], l["speaker"])
        if k not in seen:
            seen.add(k)
            uniq.append(l)
    lines = uniq
    chars = {}
    for sp in {l["speaker"] for l in lines}:
        c = dict(chars_in.get(sp, {}))
        chars[sp] = {"id": sp, "gender": c.get("gender", "male"), "age": c.get("age", "adult"),
                     "pitch": c.get("pitch", "mid"), "tone": c.get("tone", "neutral"),
                     "look": c.get("look", "")}
    return {"characters": chars, "lines": lines, "tamil_done": False, "external": True}


def step_script(video):
    script_path = os.path.join(WORK, "script.json")
    if os.path.exists(script_path):
        print("[2] script.json cache irukku, skip")
        cached = json.load(open(script_path, encoding="utf-8"))
        voc_ = os.path.join(WORK, "vocals.wav")
        if cached.get("external") and not cached.get("snapped") and os.path.exists(voc_):
            try:                                   # unga script-oda neram-a asal kural neram-ku othukkum
                n_moved = snap_lines(cached["lines"], speech_segments(speech_mask(voc_)))
                print(f"  external script: {n_moved} varigal asal kural neram-ku nagarthappattana")
            except Exception as e:  # noqa
                print("  external snap fail:", str(e)[:150])
            cached["snapped"] = True
            json.dump(cached, open(script_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return cached

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
    out = {"characters": chars, "lines": lines, "tamil_done": False}
    json.dump(out, open(script_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  {len(lines)} varigal, {len(chars)} characters")
    return out


def finish_tamil(script):
    """Tamil mattum thervu aanaa: Tamil-a movie-dub pechu nadaikku maathi script.json-la save pannum."""
    if script.get("tamil_done"):
        return
    lines, chars = script["lines"], script["characters"]
    for ln in lines:
        ln.setdefault("tamil", "")
    polish(lines, chars)
    for ln in lines:                                       # kadaisiyaa pechu-Tamil-a urudhi seyyum
        ln["tamil"] = colloquial(ln["tamil"])
    still = [ln for ln in lines if FORMAL_RE.search(ln["tamil"])]
    if still:                                              # innum formal-aa irukkura varigal -> thirumba strict-aa
        print(f"  {len(still)} varigal innum formal-aa irukku, thirumba maatthurom...")
        polish(still, chars, strict=True)
        for ln in still:
            ln["tamil"] = colloquial(ln["tamil"])
    script["tamil_done"] = True
    json.dump(script, open(os.path.join(WORK, "script.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


TRANSLATE_PROMPT = """You are the top dubbing dialogue writer for {name} movies and web series (the kind of dub people
watch on TV/OTT). Translate every line below into natural SPOKEN {name}, exactly how real people talk in daily
conversation and in a professional {name} dub. {style}
Rules:
- Write ONLY in {script} script. Common everyday English loanwords (ok, sorry, please, time, phone) may be written in
  {script} script. NO Latin letters, NO explanations, NO transliteration notes.
- Keep meaning, emotion and energy of the original. Match the speaker (age, gender, mood). Short, speakable sentences.
- TIME FIT: each item has "sec" = seconds available. The translated line must be speakable in about that time
  (max roughly {cps} characters per second). Shorten wording if needed, never drop the meaning.
- Use prev/next only to understand the scene, do not translate them.
Input items: id, who, emotion, src (original words), en (English meaning), sec, prev, next.
Return ONLY JSON: {{"lines":[{{"id":0,"text":"..."}}]}} with EVERY id exactly once.

INPUT:
{items}"""


def in_script(text, rng):
    letters = [ch for ch in str(text) if ch.isalpha()]
    if not letters:
        return False
    hit = sum(1 for ch in letters if rng[0] <= ord(ch) <= rng[1])
    return hit / len(letters) >= 0.6


def step_translate(script, lang):
    """Ellaa varigalaiyum {lang}-la mozhipeyarkkum. Cache: work/script_<lang>.json. Tamil -> script['tamil']."""
    lines, chars = script["lines"], script["characters"]
    if lang == "ta":
        return [str(ln.get("tamil", "")) for ln in lines]
    L = LANGS[lang]
    path = os.path.join(WORK, f"script_{lang}.json")
    texts = [""] * len(lines)
    if os.path.exists(path):
        try:
            old = json.load(open(path, encoding="utf-8"))
            if isinstance(old, list) and len(old) == len(lines):
                texts = [str(t) for t in old]
        except Exception:  # noqa
            pass
    todo = [i for i in range(len(lines)) if not in_script(texts[i], L["rng"])]
    if not todo:
        return texts
    print(f"  {L['name']} mozhipeyarpu: {len(todo)}/{len(lines)} varigal...")
    B = 25
    for a in range(0, len(todo), B):
        check_cancel()
        ids = todo[a: a + B]
        for attempt in range(3):
            need = [i for i in ids if not in_script(texts[i], L["rng"])]
            if not need:
                break
            items = []
            for k, i in enumerate(need):
                ln = lines[i]
                c = chars.get(ln["speaker"], {})
                prv = lines[i - 1] if i > 0 else {}
                nxt = lines[i + 1] if i + 1 < len(lines) else {}
                items.append({"id": k,
                              "who": f"{c.get('age', 'adult')} {c.get('gender', 'male')} ({c.get('tone', '')})",
                              "emotion": ln.get("emotion", "neutral"),
                              "src": ln.get("original", ""), "en": ln.get("english", "") or ln.get("original", ""),
                              "sec": round(max(0.6, ln["end"] - ln["start"]), 1),
                              "prev": prv.get("english") or prv.get("original", ""),
                              "next": nxt.get("english") or nxt.get("original", "")})
            prompt = TRANSLATE_PROMPT.format(name=L["name"], style=L["style"], script=L["script"],
                                             cps=14, items=json.dumps(items, ensure_ascii=False))
            if attempt:
                prompt = (f"IMPORTANT: answer ONLY in {L['script']} script, every line, no Latin letters.\n\n" + prompt)
            try:
                resp = gen(prompt, types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=0.4), what=f"translate-{lang}")
                data = parse_json(resp.text)
                for it in data.get("lines", []):
                    k = int(it["id"])
                    t = str(it.get("text", "")).strip()
                    if 0 <= k < len(need) and t and in_script(t, L["rng"]):
                        texts[need[k]] = t
            except Exception as e:  # noqa
                print(f"  translate {lang} fail (try {attempt + 1}/3): {str(e)[:150]}")
                time.sleep(3)
        json.dump(texts, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    missing = sum(1 for t in texts if not in_script(t, L["rng"]))
    if missing:
        print(f"  {missing} varigal {L['name']}-la mozhipeyarkka mudiyala")
    return texts


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


def style_text(c, emotion, lang="ta"):
    age = c.get("age", "adult")
    g = c.get("gender", "male")
    who = "a child" if age == "child" else f"a {age} {g}"
    nm = LANGS[lang]["name"]
    return (f"You are the {nm} dubbing voice actor for {who} in a professional {nm} movie dub. "
            f"Voice character: {c.get('tone','natural')}. Current emotion: {emotion}. "
            f"Speak casual, everyday spoken {nm} exactly as written, with a relaxed conversational rhythm "
            f"and real acting emotion, like dialogue in a {nm} film. Not like a news reader and not formal. "
            f"Do NOT read it like a text: talk it naturally, with natural pauses, quick casual pacing, "
            f"stress on the emotional words, and slightly lazy everyday pronunciation, as if chatting with a friend. "
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
    if len(x) < int(0.05 * SR):                 # tail burst vettina piragu romba chinnadhu -> fade vendaam
        return x
    x = x - float(np.mean(x[:int(0.01 * SR)]))
    pk = float(np.abs(x).max())
    if pk > 1e-4:
        x = x * (0.8 / pk)
    fi, fo = int(0.015 * SR), int(0.09 * SR)
    fi, fo = min(fi, len(x) // 2), min(fo, len(x) // 2)   # clip-ai vida fade neelamaa irukka koodaadhu
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
def assign_edge_voices(chars, lang="ta"):
    """Ovvoru character-kum vera voice + vera pitch (ellarukkum ore kural varaadhu)."""
    pool = LANGS[lang]["edge"]
    cnt = {"male": 0, "female": 0}
    out = {}
    for cid, c in chars.items():
        g = c.get("gender", "male")
        g = g if g in cnt else "male"
        k = cnt[g]
        cnt[g] += 1
        n = len(pool[g])
        voice = pool[g][k % n]
        hz = {"child": 45, "young": 8, "adult": 0, "elder": -12}.get(c.get("age", "adult"), 0)
        hz += {"high": 14, "mid": 0, "low": -14}.get(c.get("pitch", "mid"), 0)
        r = k // n                                           # ore voice thirumba vandha pitch maathi
        hz += (0, -10, 10, -20, 20)[min(r, 4)] if n else 0
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


def speak(text, voice, system, ev, emotion="neutral", cid=None, lang="ta"):
    """OmniVoice -> Gemini -> edge-tts. (ethuvum fail aanaa exception)"""
    if cid is not None and OMNI["model"] is not None:
        for _try in range(3):                      # same character voice-ae irukka: 3 thadavai try pannum
            try:
                a = omni_speak(text, cid, ev, emotion, lang)
                if a is not None:
                    OMNI["fails"] = 0
                    return a
                break
            except Exception as e:  # noqa
                OMNI["fails"] += 1
                print(f"  OmniVoice fail ({OMNI['fails']}): {str(e)[:100]}")
                time.sleep(0.5)
        if OMNI["fails"] >= 30:                    # romba fail aanaa mattum konja neram off (voice maaruvathai thavirkka)
            OMNI["off_until"] = time.time() + 60
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


def shorten(text, ratio, lang="ta"):
    nm = LANGS[lang]["name"]
    prompt = (f"Shorten this colloquial {nm} dubbing line to about {int(ratio*100)}% of its length. "
              f"Keep meaning and tone, keep it spoken {nm}, keep the same script. Return ONLY the {nm} text.\n\n{text}")
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
OMNI_REF = os.environ.get("OMNI_REF", "orig")                # edge = Tamil kural reference (Tamil uchcharippu), orig = asal nadigar kural clone
OMNI_PACE = os.environ.get("OMNI_PACE", "none")                # edge = Tamil TTS-oda vegam/nadai-ya eduthukkum, speed = OMNI_SPEED, none = model-oda vegam
OMNI_SPEED = float(os.environ.get("OMNI_SPEED", "1.0"))     # OMNI_PACE=speed na use aagum (1.0 = normal)
OMNI_LANG = os.environ.get("OMNI_LANG", "")                    # "auto" -> ovvoru mozhikkum adhoda code-a sollum | "ta" pola kuduthaa adhai ellaathukkum                    # "ta" kuduthaa language-a sollum (default: sollaadhu - munnaadi mathiri)
EDGE_REF_TEXT = "வணக்கம், நான் உங்க கூட கொஞ்சம் பேசணும். இன்னைக்கு என்ன நடந்துச்சுன்னு சொல்லுங்க, நான் கேக்குறேன்."
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


def omni_prepare_refs(chars, lines, edge_voices, lang="ta"):
    """Ovvoru character-kum oru reference kural clip thayaar pannum."""
    voc = os.path.join(WORK, "vocals.wav")
    mask = None
    OMNI["refs"] = {}
    if OMNI_REF != "edge" and os.path.exists(voc):
        try:
            mask = speech_mask(voc)
        except Exception:  # noqa
            mask = None
    ref_text = LANGS[lang]["ref"]
    first_edge = LANGS[lang]["edge"]["male"][0]
    for cid in chars:
        path = os.path.join(WORK, f"ref_{cid}.wav")
        got = None
        if mask is not None:
            best, bscore = None, 0.0
            for ln in lines:
                if ln.get("speaker") != cid or not ln.get("original"):
                    continue
                d = ln["end"] - ln["start"]
                if not 2.0 <= d <= 14.0:
                    continue
                a, b = int(ln["start"] / FRAME), int(ln["end"] / FRAME)
                frac = float(mask[a:b].mean()) if b > a else 0.0
                score = frac * min(d, 8.0)
                if frac >= 0.5 and score > bscore:
                    best, bscore = ln, score
            if best is not None:
                r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{best['start']:.2f}",
                                    "-t", f"{min(best['end'] - best['start'], 10.0):.2f}", "-i", voc,
                                    "-ac", "1", "-ar", str(SR), path], capture_output=True)
                if r.returncode == 0 and os.path.exists(path):
                    got = (path, best["original"])
        if got is None and OMNI_REF != "orig_only":
            epath = os.path.join(WORK, f"ref_{lang}_{cid}.wav")
            try:                                             # asal clip illa -> edge-tts clip reference (andha mozhiyil)
                ev = edge_voices.get(cid, (first_edge, 0))
                write_wav(epath, edge_tts_np(ref_text, ev), SR)
                got = (epath, ref_text)
            except Exception as e:  # noqa
                print(f"  ref {cid} fail: {str(e)[:100]}")
        if got:
            OMNI["refs"][cid] = got
    n_orig = sum(1 for v in OMNI["refs"].values() if not os.path.basename(v[0]).startswith("ref_" + lang + "_"))
    print(f"  OmniVoice reference: {len(OMNI['refs'])}/{len(chars)} characters ({n_orig} asal kural clone, "
          f"{len(OMNI['refs']) - n_orig} edge-tts reference)")


def omni_speak(text, cid, ev=None, emotion="neutral", lang="ta"):
    m, ref = OMNI["model"], OMNI["refs"].get(cid)
    if m is None or ref is None or time.time() < OMNI["off_until"]:
        return None
    kw = {}
    if OMNI_PACE == "edge" and ev is not None:
        try:                                                 # asal TTS evvalavu neram pesuthu -> adhey vegam
            kw["duration"] = round(len(edge_tts_np(text, ev, emotion)) / SR + 0.15, 2)
        except Exception:  # noqa
            kw = {}
    if not kw and OMNI_PACE != "none" and OMNI_SPEED != 1.0:
        kw["speed"] = OMNI_SPEED
    with _omni_lock:                                         # oru GPU model -> orey nerathula onnu
        if OMNI_LANG:
            kw["language"] = LANGS[lang]["code"] if OMNI_LANG.lower() in ("auto", "1", "on", "yes") else OMNI_LANG
        try:                                                 # oru character-ku oru seed -> ellaa varigalilum orey kural
            import torch
            torch.manual_seed(sum(ord(ch) for ch in str(cid)) * 7919 % 2**31)
        except Exception:  # noqa
            pass
        out = m.generate(text=text, ref_audio=ref[0], ref_text=ref[1], **kw)
    a = np.asarray(out[0] if isinstance(out, (list, tuple)) else out, dtype=np.float32).reshape(-1)
    if len(a) < int(0.2 * SR):
        raise RuntimeError("OmniVoice too short")
    return trim_silence(a)


def step_tts(script, texts, lang="ta", p0=50, p1=90):
    chars, lines = script["characters"], script["lines"]
    L = LANGS[lang]
    voices = assign_voices(chars)
    edge_voices = assign_edge_voices(chars, lang)
    tdir = os.path.join(WORK, "tts", lang)
    os.makedirs(tdir, exist_ok=True)
    print(f"[3] Kural thervu ({L['name']}):", {k: voices[k] for k in voices})
    print(f"[4] {len(lines)} varigalukku {L['name']} kural uruvaakkurom...")
    if omni_load():
        _P("🧠 OmniVoice குரல்களை தயார் பண்றேன்...", p0)
        omni_prepare_refs(chars, lines, edge_voices, lang)
    todo = [i for i in range(len(lines)) if str(texts[i]).strip()]
    first_edge = L["edge"]["male"][0]

    def wav_path(i):
        return os.path.join(tdir, f"{i:05d}.wav")

    def work(i):
        check_cancel()
        ln = lines[i]
        path = wav_path(i)
        if os.path.exists(path):
            return i, read_wav(path)
        c = chars.get(ln["speaker"], {})
        system = style_text(c, ln.get("emotion", "neutral"), lang)
        text = texts[i]
        nxt = lines[i + 1]["start"] if i + 1 < len(lines) else ln["end"] + 1.5
        orig_len = ln["end"] - ln["start"]
        allowed = max(orig_len, min(nxt - ln["start"] - 0.05, orig_len * 1.6 + 0.5))
        ev = edge_voices.get(ln["speaker"], (first_edge, 0))
        emo = ln.get("emotion", "neutral")
        vname = voices.get(ln["speaker"], "Kore")
        try:
            audio = speak(text, vname, system, ev, emo, ln["speaker"], lang)
        except Exception as e:  # noqa
            print(f"  line {i} TTS fail: {str(e)[:120]}")
            return i, np.zeros(int(0.1 * SR), dtype=np.float32)
        dur = len(audio) / SR
        if dur > allowed * 1.25 and len(text) > 12:
            try:
                short = shorten(text, max(0.5, allowed / dur), lang)
                if lang == "ta":
                    short = colloquial(short)
                if in_script(short, L["rng"]):
                    audio = speak(short, vname, system, ev, emo, ln["speaker"], lang)
                    ln.setdefault("short", {})[lang] = short
                    dur = len(audio) / SR
            except Exception as e:  # noqa
                print(f"  line {i} shorten fail: {str(e)[:100]}")
        if dur > allowed:
            audio = speed_np(audio, min(dur / allowed, MAX_SPEEDUP))
        audio = declick(audio)
        write_wav(path, audio)
        return i, audio

    results = {i: np.zeros(0, dtype=np.float32) for i in range(len(lines))}
    with ThreadPoolExecutor(max_workers=TTS_WORKERS) as ex:
        for k, (i, a_) in enumerate(ex.map(work, todo)):
            results[i] = a_
            _P(f"🎙️ {L['native']} குரல் உருவாக்குறேன் ({k+1}/{len(todo)} வரிகள்)",
               p0 + (p1 - p0) * 0.9 * (k + 1) / max(1, len(todo)))
            if k % 10 == 0:
                print(f"  {k+1}/{len(todo)}")
    # fail aana varigalai thaniya marubadi (cache-la illaadhavai)
    for rnd in range(2):
        failed = [i for i in todo if not os.path.exists(wav_path(i))]
        if not failed:
            break
        print(f"  {len(failed)} varigal fail aachu, marubadi try (round {rnd+1})...")
        time.sleep(10)
        for i in failed:
            check_cancel()
            _, results[i] = work(i)
    failed = [i for i in todo if not os.path.exists(wav_path(i))]
    STATS["tts_failed"] = len(failed)
    STATS["untranslated"] = len(lines) - len(todo)
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


def mix_language(total, bgm, lines, audios, lang):
    """Oru mozhiyin kural + BGM-a kalandhu oru audio file (m4a) aakkum. Video length-ke sariyaa irukkum."""
    check_cancel()
    track = np.zeros(int((total + 5) * SR), dtype=np.float32)
    for ln, a in zip(lines, audios):
        s = int(ln["start"] * SR)
        e = min(len(track), s + len(a))
        if s < len(track) and e > s:
            track[s:e] += a[: e - s]
    STATS["orig_kept"] = keep_original_where_undubbed(track, lines, audios)
    peak = np.abs(track).max()
    if peak > 1.0:
        track /= peak
    dub_wav = os.path.join(WORK, f"dub_track_{lang}.wav")
    write_wav(dub_wav, track)
    out = os.path.join(WORK, f"mix_{lang}.m4a")

    def build(norm):
        n = ":normalize=0" if norm else ""
        fc = (f"[0:a]volume={BGM_VOLUME}[b];[1:a]aresample=44100,volume={VOICE_VOLUME}[d];"
              f"[b][d]amix=inputs=2:duration=first:dropout_transition=0{n}[m];"
              f"[m]alimiter=limit=0.95,apad[a]")
        return (f'ffmpeg -y -i "{bgm}" -i "{dub_wav}" -filter_complex "{fc}" -map "[a]" '
                f'-c:a aac -b:a 192k -ar 44100 -ac 2 -t {total:.3f} "{out}"')

    r = sh(build(True), check=False)
    if r.returncode != 0:  # pazhaya ffmpeg-la normalize illa
        sh(build(False))
    return out


def mux_all(video, mixes, fmt):
    """Ellaa mozhi audio track-aiyum ORE video file-la serkkum (thaniththaniyaa varaadhu)."""
    check_cancel()
    is_mkv = fmt == "mkv"
    n = len(mixes)
    inputs = f'-i "{video}" ' + "".join(f'-i "{p}" ' for _, p in mixes)
    maps = "-map 0:v:0 " + "".join(f"-map {k + 1}:a " for k in range(n))
    if is_mkv:
        maps += "-map 0:a:0? "                      # asal audio kadaisi track-aa
    meta = ""
    for k, (lang, _) in enumerate(mixes):
        L = LANGS[lang]
        meta += f'-metadata:s:a:{k} language={L["tag"]} -metadata:s:a:{k} title="{L["name"]} Dub" '
    if is_mkv:
        meta += f'-metadata:s:a:{n} language=und -metadata:s:a:{n} title="Original" '
    disp = "-disposition:a:0 default " + "".join(
        f"-disposition:a:{k} 0 " for k in range(1, n + (1 if is_mkv else 0)))
    tail = "" if is_mkv else "-movflags +faststart "
    out = f'"{OUTPUT_VIDEO}"'
    r = sh(f"ffmpeg -y {inputs}{maps}-c copy {meta}{disp}{tail}{out}", check=False)
    if r.returncode != 0:                           # video codec container-ku set aagala -> re-encode
        print("  copy fail, video-va re-encode pannuren:", r.stderr[-200:])
        sh(f"ffmpeg -y {inputs}{maps}-c:v libx264 -preset veryfast -crf 20 -c:a copy {meta}{disp}{tail}{out}")
    print(f"\nMudinjadhu -> {OUTPUT_VIDEO}")


# ----------------------------- MAIN -----------------------------------------
def dub(video, job_dir, progress=print, fmt="mp4", cancel=None, langs=("ta",)):
    """Oru video-va thervu seytha mozhi(galil) dub panni, ORE video file-aa output path-a tharum. (Bot idhai kooppidum)"""
    global WORK, OUTPUT_VIDEO, CANCEL, _P
    CANCEL = cancel if cancel is not None else threading.Event()

    def _P(msg=None, pct=None):  # noqa: F811  (progress + percentage)
        try:
            progress(msg, pct)
        except TypeError:          # pazhaya progress(msg) mattum edukkum function
            progress(msg)
        except Exception:  # noqa
            pass
    langs = [l for l in LANG_ORDER if l in set(langs)] or ["ta"]
    WORK = os.path.join(job_dir, "work")
    fmt = "mkv" if (str(fmt).lower() == "mkv" or len(langs) > 1) else "mp4"   # pala mozhi -> orey MKV
    OUTPUT_VIDEO = os.path.join(job_dir, f"dubbed_{'_'.join(langs)}.{fmt}")
    os.makedirs(os.path.join(WORK, "tts"), exist_ok=True)
    total = duration(video)
    _P(f"Video: {total/60:.1f} நிமிடம். BGM பிரிக்கிறேன்...", 10)
    bgm = step_bgm(video)
    check_cancel()
    _P("📝 Script எழுதுறேன் (யார் என்ன பேசுறாங்க + நேரம்)...", 30)
    script = step_script(video)
    if "ta" in langs and not script.get("tamil_done"):
        _P("📝 தமிழை பேச்சு நடைக்கு மாத்துறேன்...", 46)
        finish_tamil(script)
    ls = script["lines"]
    voc_ = os.path.join(WORK, "vocals.wav")
    if script.get("external") and ls and os.path.exists(voc_):
        try:                                           # unga script video-vin ethanai sadhaveegatham-a cover pannuthu?
            segs_ = speech_segments(speech_mask(voc_))
            gap_ = sum(1 for a_, b_ in segs_ if not any(min(b_, l["end"]) - max(a_, l["start"]) > 0.15 for l in ls))
            if len(segs_) > 15 and gap_ / len(segs_) > 0.3:
                progress(f"⚠️ Script-ல {len(ls)} வரிகள் மட்டும்தான் இருக்கு, ஆனா வீடியோவுல சுமார் {len(segs_)} பேச்சு "
                         f"இடங்கள் இருக்கு. {gap_} இடங்கள்ல அசல் (English) குரல் அப்படியே இருக்கும். "
                         f"முழு episode script அனுப்பிட்டு மறுபடி முயற்சி பண்ணுங்க.")
        except Exception as e:  # noqa
            print("  script coverage check fail:", str(e)[:100])
    span = f" ({ls[0]['start']:.0f}s - {ls[-1]['end']:.0f}s)" if ls else ""
    names = " + ".join(LANGS[l]["native"] for l in langs)
    _P(f"{len(ls)} வரிகள்{span}, {len(script['characters'])} characters. {names} குரல் உருவாக்குறேன்...", 50)
    mixes, n = [], len(langs)
    for k, lang in enumerate(langs):
        check_cancel()
        L = LANGS[lang]
        p0, p1 = 50 + 40 * k / n, 50 + 40 * (k + 1) / n
        _P(f"🌐 {L['native']} மொழிபெயர்க்கிறேன்...", p0)
        texts = step_translate(script, lang)
        audios = step_tts(script, texts, lang, p0, p1)
        if STATS["tts_failed"]:
            progress(f"⚠️ {L['native']}: {STATS['tts_failed']} வரிகளுக்கு குரல் வரல (Gemini limit). மறுபடி அனுப்பிப் பாருங்க.")
        if STATS.get("untranslated"):
            progress(f"⚠️ {L['native']}: {STATS['untranslated']} வரிகளை மொழிபெயர்க்க முடியல, அங்க அசல் குரல் இருக்கும்.")
        _P(f"🎬 {L['native']} audio கலக்கிறேன்...", p1)
        mixes.append((lang, mix_language(total, bgm, script["lines"], audios, lang)))
        del audios
        if STATS.get("orig_kept"):
            progress(f"⚠️ {L['native']}: script-ல இல்லாத {STATS['orig_kept']} பேச்சு இடங்கள்ல அசல் குரல் இருக்கும் "
                     f"(TTS fail இல்ல - அந்த இடங்களுக்கு dub வசனமே இல்ல).")
    _P("🎬 எல்லாத்தையும் ஒரே வீடியோவில் சேர்க்கிறேன்...", 94)
    mux_all(video, mixes, fmt)
    return OUTPUT_VIDEO


if __name__ == "__main__":
    video = find_video()
    dub(video, OUT_DIR, langs=[x for x in os.environ.get("DUB_LANGS", "ta").split(",") if x])
