"""
Multi-language Dub Telegram Bot (Tamil / Telugu / Malayalam / Kannada / Hindi)
=============================================================================
Telegram-la video anuppunga -> mozhi(galai) button-la thervu pannunga -> bot dub panni ORE video-va anuppum.

Secrets (Kaggle -> Add-ons -> Secrets):
  GEMINI_API_KEY      aistudio.google.com/apikey
  TELEGRAM_BOT_TOKEN  Telegram-la @BotFather -> /newbot
  TELEGRAM_API_ID     my.telegram.org -> API development tools
  TELEGRAM_API_HASH   (adhe pakkam)

Optional: TELEGRAM_ALLOWED = unga Telegram user id (comma-vil pirichu)
          (@userinfobot-la id therium). Kaali na yaar vena use pannalam,
          unga Gemini quota yaar vena kaali pannalam - so ithai set pannunga.

Run:  !python New-Dubbed-/tamil_bot.py
"""
import os, sys, asyncio, shutil, time, traceback, subprocess, json, signal, threading, itertools
from datetime import datetime

try:
    import telethon  # noqa
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "telethon"])

from telethon import TelegramClient, events, Button

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tamil_dub  # noqa  (Gemini key-a ULLE edukkum)

API_ID = int(os.environ["TELEGRAM_API_ID"])
API_HASH = os.environ["TELEGRAM_API_HASH"]
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ALLOWED = {int(x) for x in os.environ.get("TELEGRAM_ALLOWED", "").replace(" ", "").split(",") if x}

BASE = "/kaggle/working/jobs" if os.path.exists("/kaggle/working") else "./jobs"
os.makedirs(BASE, exist_ok=True)

client = TelegramClient(os.path.join(BASE, "bot_session"), API_ID, API_HASH)
queue: asyncio.Queue = asyncio.Queue()

# ---------------- Bot on/off notice + job state ----------------
STATE_FILE = os.path.join(BASE, "state.json")     # heartbeat + yaar-kitta pesanum
USERS_FILE = os.path.join(BASE, "users.json")
JOBS = {}                                         # job_id -> Job
_ids = itertools.count(1)
ACTIVE = {"job": None}
START_TS = time.time()
SESSION_HOURS = float(os.environ.get("SESSION_HOURS", "12"))   # Kaggle session limit


def _load(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa
        return default


def _save(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:  # noqa
        pass


def remember_user(chat_id):
    users = set(_load(USERS_FILE, []))
    if chat_id not in users:
        users.add(chat_id)
        _save(USERS_FILE, sorted(users))


def known_users():
    return set(_load(USERS_FILE, [])) | set(ALLOWED)


async def notify_all(text):
    for uid in known_users():
        try:
            await client.send_message(uid, text)
        except Exception:  # noqa
            pass


def write_state(clean=False):
    j = ACTIVE["job"]
    pending = [x.chat_id for x in JOBS.values() if not x.cancel.is_set()]
    _save(STATE_FILE, {"ts": time.time(), "clean": clean,
                       "busy": bool(j) or bool(pending), "chats": sorted(set(pending))})


def bar(pct, n=10):
    pct = max(0, min(100, int(pct)))
    full = pct * n // 100
    return "▓" * full + "░" * (n - full) + f" {pct}%"


class Job:
    def __init__(self, event, fmt, send, langs):
        self.id = next(_ids)
        self.event, self.fmt, self.send, self.langs = event, fmt, send, list(langs)
        self.chat_id, self.user_id = event.chat_id, event.sender_id
        self.cancel = threading.Event()      # tamil_dub idhai paarkkum
        self.task = None                     # download / upload asyncio task
        self.status = None                   # progress message
        self.msg, self.pct, self.shown = "வரிசையில் காத்திருக்கு...", 0, None
        self.t0 = time.time()

    def buttons(self):
        return [Button.inline("❌ Cancel", data=f"cancel:{self.id}".encode())]


async def render(job, final=None):
    """Status message-a (progress bar + Cancel button) update pannum."""
    if final:
        text, btn = final, None
    else:
        el = int(time.time() - job.t0)
        text = f"{job.msg}\n\n{bar(job.pct)}\n⏱ {el // 60}நி {el % 60}வி"
        btn = job.buttons()
    if text == job.shown:
        return
    job.shown = text
    try:
        await job.status.edit(text, buttons=btn)
    except Exception as e:  # noqa
        if "wait of" in str(e).lower() or "flood" in str(e).lower():
            await asyncio.sleep(5)


SETTINGS = {}  # user_id -> {"fmt": "mp4|mkv", "send": "video|document", "langs": [...]}
PENDING = {}   # panel_id -> {"event", "user_id", "langs", "fmt", "send", "panel", "ts"}
_pids = itertools.count(1)
LANG_BTN = [("ta", "தமிழ்"), ("te", "తెలుగు"), ("ml", "മലയാളം"), ("kn", "ಕನ್ನಡ"), ("hi", "हिन्दी")]
LANG_ORDER = [c for c, _ in LANG_BTN]
LANG_NAME = dict(LANG_BTN)


def prefs(uid):
    return SETTINGS.setdefault(uid, {"fmt": "mp4", "send": "video", "langs": ["ta"]})


def eff_fmt(pd):
    return "mkv" if len(pd["langs"]) > 1 else pd["fmt"]          # pala mozhi -> orey MKV (ella audio track-um ullae)


def eff_send(pd):
    return "document" if eff_fmt(pd) == "mkv" else pd["send"]


def describe(fmt, send, langs):
    names = " + ".join(LANG_NAME[l] for l in LANG_ORDER if l in langs)
    snd = "file (document)" if send == "document" or fmt == "mkv" else "video"
    return f"மொழி: {names}\nFormat: {fmt.upper()} | அனுப்பும் முறை: {snd}"


def panel_text(pd):
    langs = pd["langs"]
    t = "🎬 வீடியோ கிடைத்தது ✅\n\n🌐 எந்த மொழியில் dub வேணும்? (ஒண்ணு அல்லது பல தேர்ந்தெடுங்க)\n\n"
    t += describe(eff_fmt(pd), eff_send(pd), langs)
    if len(langs) > 1:
        t += ("\n\n📦 பல மொழி தேர்ந்ததால், எல்லா மொழி audio-வும் ஒரே MKV வீடியோவுக்குள்ள "
              "தனித்தனி track-ஆ இருக்கும். Player-ல (MX Player / VLC) audio track மாத்தி கேட்கலாம்.")
    t += "\n\nதேர்ந்ததும் 🚀 தொடங்கு அழுத்துங்க."
    return t


def panel_buttons(pid, pd):
    def tick(on):
        return "✅ " if on else "⬜ "

    def b(label, data):
        return Button.inline(label, data=data.encode())
    langs = pd["langs"]
    btn = [LANG_BTN[:3], LANG_BTN[3:]]
    rows = [[b(tick(c in langs) + n, f"sel:{pid}:{c}") for c, n in row] for row in btn]
    rows[1].append(b(tick(len(langs) == len(LANG_BTN)) + "எல்லாம்", f"sel:{pid}:all"))
    f = eff_fmt(pd)
    rows.append([b(tick(f == "mp4") + "MP4", f"fmt:{pid}:mp4"), b(tick(f == "mkv") + "MKV", f"fmt:{pid}:mkv")])
    s = eff_send(pd)
    rows.append([b(tick(s == "video") + "🎞 Video", f"snd:{pid}:video"),
                 b(tick(s == "document") + "📁 File", f"snd:{pid}:document")])
    rows.append([b("🚀 தொடங்கு", f"go:{pid}"), b("❌ ரத்து", f"x:{pid}")])
    return rows


async def refresh_panel(pd, pid):
    try:
        await pd["panel"].edit(panel_text(pd), buttons=panel_buttons(pid, pd))
    except Exception as e:  # noqa
        if "not modified" not in str(e).lower():
            print("panel edit fail:", str(e)[:120])


def is_video(msg):
    if msg.video:
        return True
    d = msg.document
    return bool(d and d.mime_type and d.mime_type.startswith("video/"))


@client.on(events.NewMessage(pattern="/start"))
async def start(event):
    await event.reply("வணக்கம்! 🎬 ஒரு வீடியோ அனுப்புங்க.\n"
                      "அப்புறம் வர்ற பட்டன்களில் மொழி (தமிழ் / తెలుగు / മലയാളം / ಕನ್ನಡ / हिन्दी) தேர்ந்தெடுத்து 🚀 தொடங்கு அழுத்துங்க.\n"
                      "ஒண்ணுக்கு மேல தேர்ந்தா எல்லாம் ஒரே வீடியோவுல (பல audio track) வரும்.\n\n"
                      "ஒரு நேரத்தில் ஒரு வீடியோ மட்டும். நீளமான வீடியோவுக்கு நேரம் ஆகும்.")


@client.on(events.NewMessage(pattern="/id"))
async def my_id(event):
    await event.reply(f"உங்க ID: {event.sender_id}")


@client.on(events.NewMessage(pattern=r"^/settings"))
async def settings_cmd(event):
    p = prefs(event.sender_id)
    await event.reply("⚙️ கடைசியா தேர்ந்தது:\n" + describe(p["fmt"], p["send"], p["langs"]) +
                      "\n\nஒவ்வொரு வீடியோ அனுப்பும்போதும் பட்டன்ல மாத்திக்கலாம்.")


USER_SCRIPTS = {}   # user_id -> converted script.json path (next video-ku use aagum)


async def ingest_script(event, raw):
    """raw = script text file. Convert + save + user-ku confirm."""
    out = os.path.join(BASE, "scripts", f"{event.sender_id}.json")
    try:
        data = await asyncio.to_thread(tamil_dub.load_external_script, raw)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    except Exception as e:  # noqa
        await event.reply(f"❌ Script படிக்க முடியல: {str(e)[:200]}\n"
                          "JSON-ல characters, lines (start, end, speaker, tamil) இருக்கணும். "
                          "Gemini பதிலை முழுசா (கடைசி ]} வரை) copy பண்ணி மறுபடி அனுப்புங்க.")
        return
    USER_SCRIPTS[event.sender_id] = out
    await event.reply(f"✅ Script ரெடி: {len(data['lines'])} வரிகள், {len(data['characters'])} characters.\n"
                      "இப்போ அதே வீடியோவை அனுப்புங்க - script எழுதுற நேரம் மிச்சம்!")


async def handle_script_file(event):
    os.makedirs(os.path.join(BASE, "scripts"), exist_ok=True)
    raw = os.path.join(BASE, "scripts", f"{event.sender_id}_raw.txt")
    try:
        await event.download_media(file=raw)
    except Exception as e:  # noqa
        await event.reply(f"❌ File எடுக்க முடியல: {str(e)[:150]}")
        return
    await ingest_script(event, raw)


SCRIPT_TEXT = {}    # user_id -> [pasted text pieces]  (Gemini-la copy panni nerave paste pannina script)


def _is_script_text(e):
    t = (e.raw_text or "").strip()
    if e.file is not None or not t or e.sender_id is None:
        return False
    if t.startswith("/"):
        return False
    return e.sender_id in SCRIPT_TEXT or t.startswith(("{", "[", "```"))


@client.on(events.NewMessage(func=_is_script_text))
async def on_script_text(event):
    if ALLOWED and event.sender_id not in ALLOWED:
        return
    first = event.sender_id not in SCRIPT_TEXT
    SCRIPT_TEXT.setdefault(event.sender_id, []).append(event.raw_text)
    if first:
        await event.reply("📝 Script வாங்கிக்கிட்டேன். இன்னும் பகுதி இருந்தா தொடர்ந்து paste பண்ணுங்க.\n"
                          "முழுசா அனுப்பினதும் /done அனுப்புங்க.")


@client.on(events.NewMessage(pattern=r"^/done"))
async def on_script_done(event):
    pieces = SCRIPT_TEXT.pop(event.sender_id, None)
    if not pieces:
        await event.reply("Script எதுவும் வரல. Gemini JSON-ஐ copy பண்ணி இங்க paste பண்ணுங்க.")
        return
    os.makedirs(os.path.join(BASE, "scripts"), exist_ok=True)
    raw = os.path.join(BASE, "scripts", f"{event.sender_id}_raw.txt")
    with open(raw, "w", encoding="utf-8") as f:
        f.write("".join(pieces))
    await ingest_script(event, raw)


@client.on(events.NewMessage(func=lambda e: e.file is not None))
async def on_file(event):
    if ALLOWED and event.sender_id not in ALLOWED:
        await event.reply("மன்னிக்கவும், இந்த bot உங்களுக்கு அனுமதி இல்லை.")
        return
    fname = (getattr(event.file, "name", None) or "").lower()
    if not is_video(event.message) and fname.endswith((".json", ".txt")):
        await handle_script_file(event)                  # Gemini app-la eluthina script-a vaanguvom
        return
    if not is_video(event.message):
        await event.reply("வீடியோ file மட்டும் அனுப்புங்க. (Script இருந்தா .json / .txt ஆ அனுப்பலாம்)")
        return
    remember_user(event.chat_id)
    for k in [k for k, v in PENDING.items() if time.time() - v["ts"] > 3600]:
        PENDING.pop(k, None)                       # 1 mani neram thodaadha panel-gal neekkum
    p = prefs(event.sender_id)
    pid = next(_pids)
    pd = {"event": event, "user_id": event.sender_id, "langs": list(p["langs"]) or ["ta"],
          "fmt": p["fmt"], "send": p["send"], "ts": time.time()}
    pd["panel"] = await event.reply(panel_text(pd), buttons=panel_buttons(pid, pd))
    PENDING[pid] = pd


def _owner_ok(event, pd):
    return event.sender_id == pd["user_id"] or (ALLOWED and event.sender_id in ALLOWED)


@client.on(events.CallbackQuery(pattern=rb"(sel|fmt|snd|go|x):(\d+)(?::(\w+))?"))
async def on_panel(event):
    kind = event.data_match.group(1).decode()
    pid = int(event.data_match.group(2))
    arg = (event.data_match.group(3) or b"").decode()
    pd = PENDING.get(pid)
    if not pd:
        await event.answer("இந்த தேர்வு காலாவதி ஆயிடுச்சு. வீடியோவை மறுபடி அனுப்புங்க.", alert=True)
        return
    if not _owner_ok(event, pd):
        await event.answer("இது உங்க வீடியோ இல்ல.", alert=True)
        return
    if kind == "x":
        PENDING.pop(pid, None)
        try:
            await pd["panel"].edit("🛑 ரத்து செய்யப்பட்டது. புது வீடியோ அனுப்பலாம்.", buttons=None)
        except Exception:  # noqa
            pass
        await event.answer("ரத்து")
        return
    if kind == "sel":
        if arg == "all":
            pd["langs"] = list(LANG_ORDER) if len(pd["langs"]) < len(LANG_ORDER) else ["ta"]
        elif arg in LANG_NAME:
            if arg in pd["langs"]:
                if len(pd["langs"]) == 1:
                    await event.answer("குறைந்தது ஒரு மொழி வேணும்.", alert=False)
                    return
                pd["langs"].remove(arg)
            else:
                pd["langs"].append(arg)
            pd["langs"] = [c for c in LANG_ORDER if c in pd["langs"]]
        await event.answer()
        await refresh_panel(pd, pid)
        return
    if kind == "fmt":
        if len(pd["langs"]) > 1 and arg == "mp4":
            await event.answer("பல மொழிக்கு MKV தான் (ஒரே file-ல எல்லா audio track).", alert=True)
            return
        pd["fmt"] = arg if arg in ("mp4", "mkv") else pd["fmt"]
        await event.answer()
        await refresh_panel(pd, pid)
        return
    if kind == "snd":
        if eff_fmt(pd) == "mkv" and arg == "video":
            await event.answer("MKV Telegram-ல file-ஆ தான் போகும்.", alert=True)
            return
        pd["send"] = arg if arg in ("video", "document") else pd["send"]
        await event.answer()
        await refresh_panel(pd, pid)
        return
    # ---- go ----
    PENDING.pop(pid, None)
    fmt, send, langs = eff_fmt(pd), eff_send(pd), list(pd["langs"])
    p = prefs(pd["user_id"])
    p["fmt"], p["send"], p["langs"] = pd["fmt"], pd["send"], langs
    ev = pd["event"]
    job = Job(ev, fmt, send, langs)
    JOBS[job.id] = job
    pos = queue.qsize() + (1 if ACTIVE["job"] else 0) + 1
    job.msg = f"வரிசை எண்: {pos}\n{describe(fmt, send, langs)}"
    job.status = pd["panel"]
    await event.answer("தொடங்குது 🚀")
    try:
        await job.status.edit(job.msg + f"\n\n{bar(0)}", buttons=job.buttons())
    except Exception:  # noqa
        job.status = await ev.reply(job.msg + f"\n\n{bar(0)}", buttons=job.buttons())
    write_state()
    await queue.put(job)


def cancel_job(job):
    job.cancel.set()                              # dub thread nirkum
    if job.task and not job.task.done():
        job.task.cancel()                         # download / upload nirkum


@client.on(events.CallbackQuery(pattern=rb"cancel:(\d+)"))
async def on_cancel_btn(event):
    job = JOBS.get(int(event.data_match.group(1)))
    if not job:
        await event.answer("இந்த job ஏற்கனவே முடிஞ்சிருச்சு.", alert=False)
        return
    if event.sender_id != job.user_id and (not ALLOWED or event.sender_id not in ALLOWED):
        await event.answer("இது உங்க job இல்ல.", alert=True)
        return
    cancel_job(job)
    await event.answer("ரத்து செய்யுறேன்...", alert=False)


@client.on(events.NewMessage(pattern=r"^/cancel"))
async def on_cancel_cmd(event):
    mine = [j for j in JOBS.values() if j.user_id == event.sender_id and not j.cancel.is_set()]
    if not mine:
        await event.reply("ரத்து செய்ய எந்த job-உம் இல்ல.")
        return
    for j in mine:
        cancel_job(j)
    await event.reply(f"🛑 {len(mine)} job ரத்து செய்யப்படுது.")


async def say(event, text):
    try:
        await event.reply(text)
    except Exception:
        pass


async def ticker(job, stop):
    """3 second-ku oru thadavai progress bar-a refresh pannum."""
    while not stop.is_set():
        await render(job)
        try:
            await asyncio.wait_for(stop.wait(), 3)
        except asyncio.TimeoutError:
            pass


async def run_task(job, coro):
    """Download/upload-a cancel panna mudiyura maari task-aa odudum."""
    job.task = asyncio.ensure_future(coro)
    try:
        return await job.task
    except asyncio.CancelledError:
        if job.cancel.is_set():
            raise tamil_dub.Cancelled("cancelled")
        raise
    finally:
        job.task = None


async def worker():
    loop = asyncio.get_running_loop()
    while True:
        job = await queue.get()
        event = job.event
        ACTIVE["job"] = job
        write_state()
        doc = getattr(event.message, "document", None)          # adhey video-va marubadi anuppinaa ippo vittadhulirundhu thodarum
        key = getattr(doc, "id", None) or int(time.time())
        folder = os.path.join(BASE, f"{job.user_id}_{key}")
        os.makedirs(folder, exist_ok=True)
        ext = USER_SCRIPTS.pop(job.user_id, None)           # user script anuppiyirundhaa adhaiye use pannum
        if ext and os.path.exists(ext):
            os.makedirs(os.path.join(folder, "work"), exist_ok=True)
            shutil.copy(ext, os.path.join(folder, "work", "script.json"))
        keep_folder = False
        stop = asyncio.Event()
        tick = asyncio.create_task(ticker(job, stop))
        try:
            if job.cancel.is_set():
                raise tamil_dub.Cancelled("cancelled")
            job.msg = "⬇️ பதிவிறக்குறேன்..."

            def dl_prog(done, total):
                if total:
                    job.pct = max(job.pct, 8 * done / total)

            import glob as _glob
            old = [f for f in _glob.glob(os.path.join(folder, "input.*")) if os.path.getsize(f) > 0]
            if old:
                src = old[0]                                     # munnaadi download aanadhu - thirumba download venaam
            else:
                src = await run_task(job, event.download_media(file=os.path.join(folder, "input"),
                                                              progress_callback=dl_prog))
            job.pct = 8

            def progress(msg=None, pct=None):
                if pct is None:
                    if msg:                       # warning maari - thaniya anuppu
                        asyncio.run_coroutine_threadsafe(say(event, msg), loop)
                    return
                if msg:
                    job.msg = msg
                job.pct = max(job.pct, pct)

            out = await asyncio.to_thread(tamil_dub.dub, src, folder, progress, job.fmt, job.cancel, job.langs)
            if job.cancel.is_set():
                raise tamil_dub.Cancelled("cancelled")
            job.msg, job.pct = "📤 அனுப்புறேன்...", 98

            def up_prog(done, total):
                if total:
                    job.pct = max(job.pct, 98 + 2 * done / total)

            as_doc = job.send == "document" or job.fmt == "mkv"
            if os.path.getsize(out) > 1_990_000_000:
                as_doc = True
                await say(event, "⚠️ Output 2GB-ku mela irukku, Telegram-la anuppa mudiyaama pogalaam.")
            await run_task(job, client.send_file(event.chat_id, out, caption=" + ".join(LANG_NAME[l] for l in LANG_ORDER if l in job.langs) + " dub 🎧",
                                                 force_document=as_doc, supports_streaming=not as_doc,
                                                 reply_to=event.id, progress_callback=up_prog))
            stop.set(); await tick
            await render(job, final=f"✅ முடிந்தது!\n\n{bar(100)}")
        except tamil_dub.Cancelled:
            stop.set(); await tick
            await render(job, final="🛑 ரத்து செய்யப்பட்டது. புது வீடியோ அனுப்பலாம்.")
        except Exception as e:  # noqa
            traceback.print_exc()
            stop.set(); await tick
            keep_folder = True            # fail aanaa folder-a vechuko -> marubadi anuppinaa script/kural cache-la irundhu thodarum
            await render(job, final=f"❌ பிழை: {str(e)[:300]}\n\nஅதே வீடியோவை மறுபடி அனுப்புங்க - முடிஞ்ச வேலை திரும்ப செய்ய வேண்டாம்.")
        finally:
            stop.set()
            if not keep_folder:
                shutil.rmtree(folder, ignore_errors=True)
            JOBS.pop(job.id, None)
            ACTIVE["job"] = None
            write_state()
            queue.task_done()


async def heartbeat():
    warned = False
    while True:
        write_state()
        left = SESSION_HOURS * 3600 - (time.time() - START_TS)
        if not warned and os.path.exists("/kaggle/working") and left < 1800:
            warned = True
            await notify_all("⏰ இன்னும் ~30 நிமிடத்தில் Kaggle session முடியப்போகுது. "
                             "பெரிய வீடியோ இப்போ அனுப்ப வேண்டாம்; முடிஞ்சதும் மறுபடி ON பண்ணுங்க.")
        await asyncio.sleep(20)


async def main():
    prev = _load(STATE_FILE, None)
    try:                                           # 24 mani neram munnaadi irundha pazhaya folders-a azhi
        for d in os.listdir(BASE):
            dp = os.path.join(BASE, d)
            if os.path.isdir(dp) and time.time() - os.path.getmtime(dp) > 86400:
                shutil.rmtree(dp, ignore_errors=True)
    except Exception:  # noqa
        pass
    await client.start(bot_token=BOT_TOKEN)
    me = await client.get_me()
    print(f"Bot ready: @{me.username}  (Telegram-la poi /start anuppunga)")

    # --- mundhaiya run thideernu nindru pochaa? ---
    if prev and not prev.get("clean"):
        when = datetime.fromtimestamp(prev["ts"]).strftime("%d %b %H:%M")
        for cid in prev.get("chats", []):
            try:
                await client.send_message(cid, "⚠️ Bot திடீரென OFF ஆனதால் உங்க வீடியோ பாதியில் நின்னுடுச்சு. "
                                               "இப்போ மறுபடி ON ஆயிடுச்சு — வீடியோவை மறுபடி அனுப்புங்க 🙏")
            except Exception:  # noqa
                pass
        await notify_all(f"🟢 Bot மறுபடி ON ஆயிடுச்சு.\n(கடைசியா {when}-க்கு OFF ஆச்சு)")
    else:
        await notify_all("🟢 Bot ON ஆயிடுச்சு. வீடியோ அனுப்புங்க!")

    asyncio.create_task(worker())
    asyncio.create_task(heartbeat())

    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stopping.set)
        except Exception:  # noqa
            pass

    runner = asyncio.create_task(client.run_until_disconnected())
    waiter = asyncio.create_task(stopping.wait())
    try:
        await asyncio.wait({runner, waiter}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        # normal-aa nirkkum pothu (Stop / Ctrl+C) "OFF" nu sollum
        for j in list(JOBS.values()):
            cancel_job(j)
        try:
            await asyncio.wait_for(notify_all("🔴 Bot OFF ஆயிடுச்சு. மறுபடி ON பண்ணினதும் சொல்றேன்."), 8)
        except Exception:  # noqa
            pass
        write_state(clean=True)
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
