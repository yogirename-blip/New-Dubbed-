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
import os, sys, asyncio, shutil, time, traceback, subprocess

try:
    import telethon  # noqa
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "telethon"])

from telethon import TelegramClient, events

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
    pos = queue.qsize() + 1
    p = prefs(event.sender_id)
    cap = (event.raw_text or "").lower()          # caption-la 'mkv' / 'doc' ezhuthina adhu mudhal
    event.job_fmt = "mkv" if "mkv" in cap else ("mp4" if "mp4" in cap else p["fmt"])
    event.job_send = "document" if ("doc" in cap or "file" in cap) else p["send"]
    await event.reply(f"வீடியோ கிடைத்தது ✅ வரிசை எண்: {pos}\n{describe({'fmt': event.job_fmt, 'send': event.job_send})}")
    await queue.put(event)


async def say(event, text):
    try:
        await event.reply(text)
    except Exception:
        pass


async def worker():
    loop = asyncio.get_running_loop()
    while True:
        event = await queue.get()
        job = os.path.join(BASE, f"{event.sender_id}_{int(time.time())}")
        os.makedirs(job, exist_ok=True)
        try:
            await say(event, "பதிவிறக்குறேன்...")
            src = await event.download_media(file=os.path.join(job, "input"))

            def progress(msg):
                asyncio.run_coroutine_threadsafe(say(event, "⏳ " + msg), loop)

            out = await asyncio.to_thread(tamil_dub.dub, src, job, progress, event.job_fmt)
            await say(event, "முடிந்தது! அனுப்புறேன்...")
            as_doc = event.job_send == "document" or event.job_fmt == "mkv"
            await client.send_file(event.chat_id, out, caption="தமிழ் dub 🎧",
                                   force_document=as_doc,
                                   supports_streaming=not as_doc, reply_to=event.id)
        except Exception as e:  # noqa
            traceback.print_exc()
            await say(event, f"❌ பிழை: {str(e)[:300]}")
        finally:
            shutil.rmtree(job, ignore_errors=True)
            queue.task_done()


async def main():
    await client.start(bot_token=BOT_TOKEN)
    me = await client.get_me()
    print(f"Bot ready: @{me.username}  (Telegram-la poi /start anuppunga)")
    asyncio.create_task(worker())
    await client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
