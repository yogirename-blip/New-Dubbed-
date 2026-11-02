"""
Tamil Dub Telegram Bot
======================
Telegram-la video anuppunga -> bot Tamil-la dub panni thirumba anuppum.

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
    def __init__(self, event, fmt, send):
        self.id = next(_ids)
        self.event, self.fmt, self.send = event, fmt, send
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


SETTINGS = {}  # user_id -> {"fmt": "mp4|mkv", "send": "video|document"}


def prefs(uid):
    return SETTINGS.setdefault(uid, {"fmt": "mp4", "send": "video"})


def describe(p):
    send = "document (file)" if p["send"] == "document" or p["fmt"] == "mkv" else "video"
    return f"Format: {p['fmt'].upper()} | அனுப்பும் முறை: {send}"


@client.on(events.NewMessage(pattern=r"^/(mp4|mkv|video|document|settings)"))
async def set_pref(event):
    cmd = event.raw_text.split()[0].lstrip("/").split("@")[0].lower()
    p = prefs(event.sender_id)
    if cmd in ("mp4", "mkv"):
        p["fmt"] = cmd
    elif cmd in ("video", "document"):
        p["send"] = cmd
    extra = ""
    if p["fmt"] == "mkv":
        extra = "\n(MKV: தமிழ் + அசல் ஆடியோ இரண்டு track இருக்கும். Telegram-ல் file-ஆகத்தான் போகும்.)"
    await event.reply("✅ " + describe(p) + extra +
                      "\n\nமாற்ற: /mp4 /mkv /video /document")


def is_video(msg):
    if msg.video:
        return True
    d = msg.document
    return bool(d and d.mime_type and d.mime_type.startswith("video/"))


@client.on(events.NewMessage(pattern="/start"))
async def start(event):
    await event.reply("வணக்கம்! 🎬 ஒரு வீடியோ அனுப்புங்க, நான் அதை பேச்சுத் தமிழில் dub செய்து திருப்பி அனுப்புறேன்.\n"
                      "(நீளமான வீடியோவுக்கு நேரம் ஆகும். ஒரு நேரத்தில் ஒரு வீடியோ மட்டும்.)\n\n"
                      "வடிவம் மாற்ற: /mp4 /mkv   |   அனுப்பும் முறை: /video /document   |   /settings")


@client.on(events.NewMessage(pattern="/id"))
async def my_id(event):
    await event.reply(f"உங்க ID: {event.sender_id}")


@client.on(events.NewMessage(func=lambda e: e.file is not None))
async def on_file(event):
    if ALLOWED and event.sender_id not in ALLOWED:
        await event.reply("மன்னிக்கவும், இந்த bot உங்களுக்கு அனுமதி இல்லை.")
        return
    if not is_video(event.message):
        await event.reply("வீடியோ file மட்டும் அனுப்புங்க.")
        return
    remember_user(event.chat_id)
    p = prefs(event.sender_id)
    cap = (event.raw_text or "").lower()          # caption-la 'mkv' / 'doc' ezhuthina adhu mudhal
    fmt = "mkv" if "mkv" in cap else ("mp4" if "mp4" in cap else p["fmt"])
    send = "document" if ("doc" in cap or "file" in cap) else p["send"]
    job = Job(event, fmt, send)
    event.job_fmt, event.job_send = fmt, send
    JOBS[job.id] = job
    pos = queue.qsize() + (1 if ACTIVE["job"] else 0) + 1
    job.msg = f"வீடியோ கிடைத்தது ✅ வரிசை எண்: {pos}\n{describe({'fmt': fmt, 'send': send})}"
    job.status = await event.reply(job.msg + f"\n\n{bar(0)}", buttons=job.buttons())
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
        folder = os.path.join(BASE, f"{job.user_id}_{int(time.time())}")
        os.makedirs(folder, exist_ok=True)
        stop = asyncio.Event()
        tick = asyncio.create_task(ticker(job, stop))
        try:
            if job.cancel.is_set():
                raise tamil_dub.Cancelled("cancelled")
            job.msg = "⬇️ பதிவிறக்குறேன்..."

            def dl_prog(done, total):
                if total:
                    job.pct = max(job.pct, 8 * done / total)

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

            out = await asyncio.to_thread(tamil_dub.dub, src, folder, progress, job.fmt, job.cancel)
            if job.cancel.is_set():
                raise tamil_dub.Cancelled("cancelled")
            job.msg, job.pct = "📤 அனுப்புறேன்...", 98

            def up_prog(done, total):
                if total:
                    job.pct = max(job.pct, 98 + 2 * done / total)

            as_doc = job.send == "document" or job.fmt == "mkv"
            await run_task(job, client.send_file(event.chat_id, out, caption="தமிழ் dub 🎧",
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
            await render(job, final=f"❌ பிழை: {str(e)[:300]}")
        finally:
            stop.set()
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
