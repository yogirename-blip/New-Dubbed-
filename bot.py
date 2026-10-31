import asyncio
import os
import time
import tempfile

import httpx
from aiohttp import web
from pyrogram import Client, filters
from pyrogram.types import Message

API_ID = int(os.environ["API_ID"])
API_HASH = os.environ["API_HASH"]
BOT_TOKEN = os.environ["BOT_TOKEN"]
ELEVEN_KEY = os.environ["ELEVEN_API_KEY"]

TARGET_LANG = "ta"          # Tamil
POLL_SECONDS = 20           # how often to check dubbing status
MAX_WAIT_SECONDS = 4 * 3600  # give up after 4 hours
BASE = "https://api.elevenlabs.io/v1"

# One video at a time (ElevenLabs + Render memory friendly)
queue_lock = asyncio.Semaphore(1)

app = Client("tamil_dub_bot", api_id=API_ID, api_hash=API_HASH, bot_token=BOT_TOKEN)


async def safe_edit(msg: Message, text: str):
    """Edit a message, ignore Telegram errors like MESSAGE_NOT_MODIFIED."""
    try:
        await msg.edit_text(text)
    except Exception:
        pass


async def start_dubbing(path: str) -> str:
    """Upload video to ElevenLabs, return dubbing_id."""
    headers = {"xi-api-key": ELEVEN_KEY}
    data = {
        "target_lang": TARGET_LANG,
        "source_lang": "auto",
        "num_speakers": "0",      # 0 = auto-detect speakers
        "watermark": os.environ.get("WATERMARK", "true").lower(),  # Free plan needs true
        "highest_resolution": "true",
    }
    timeout = httpx.Timeout(60.0, read=3600.0, write=3600.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        with open(path, "rb") as f:
            files = {"file": (os.path.basename(path), f, "video/mp4")}
            r = await client.post(f"{BASE}/dubbing", headers=headers, data=data, files=files)
        r.raise_for_status()
        return r.json()["dubbing_id"]


async def wait_for_dub(dubbing_id: str, status_msg: Message) -> None:
    headers = {"xi-api-key": ELEVEN_KEY}
    start = time.time()
    last_edit = 0
    async with httpx.AsyncClient(timeout=60) as client:
        while True:
            r = await client.get(f"{BASE}/dubbing/{dubbing_id}", headers=headers)
            r.raise_for_status()
            info = r.json()
            status = info.get("status")
            if status == "dubbed":
                return
            if status == "failed":
                raise RuntimeError(info.get("error") or "Dubbing failed")
            if time.time() - start > MAX_WAIT_SECONDS:
                raise TimeoutError("Dubbing took too long")
            if time.time() - last_edit > 120:
                mins = int((time.time() - start) // 60)
                await safe_edit(status_msg, f"⏳ தமிழ் டப்பிங் நடக்குது... ({mins} நிமிடம் ஆச்சு)")
                last_edit = time.time()
            await asyncio.sleep(POLL_SECONDS)


async def download_dub(dubbing_id: str, out_path: str) -> None:
    headers = {"xi-api-key": ELEVEN_KEY}
    timeout = httpx.Timeout(60.0, read=3600.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "GET", f"{BASE}/dubbing/{dubbing_id}/audio/{TARGET_LANG}", headers=headers
        ) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                async for chunk in r.aiter_bytes(1024 * 1024):
                    f.write(chunk)


@app.on_message(filters.command("start") & filters.private)
async def start_cmd(_, m: Message):
    await m.reply_text(
        "வணக்கம்! 🎬\nஹிந்தி / ஜப்பானீஸ் வீடியோவை அனுப்பு, "
        "தமிழ் குரலில் (அதே குரல் வகையில்) மாற்றி தருகிறேன்.\n"
        "நேரம் ஆகலாம், பொறுமையா காத்திரு."
    )


@app.on_message(filters.private & (filters.video | filters.document))
async def handle_video(_, m: Message):
    if m.document and not (m.document.mime_type or "").startswith("video/"):
        return await m.reply_text("வீடியோ கோப்பு மட்டும் அனுப்பு.")

    status = await m.reply_text("📥 வீடியோ எடுக்கிறேன்...")
    workdir = tempfile.mkdtemp(prefix="dub_")
    src = os.path.join(workdir, "input.mp4")
    out = os.path.join(workdir, "tamil.mp4")

    try:
        async with queue_lock:
            await m.download(file_name=src)
            await safe_edit(status, "📤 ElevenLabs-க்கு அனுப்புறேன்...")
            dubbing_id = await start_dubbing(src)

            await safe_edit(status, "⏳ தமிழ் டப்பிங் ஆரம்பம்...")
            await wait_for_dub(dubbing_id, status)

            await safe_edit(status, "⬇️ தமிழ் வீடியோ எடுக்கிறேன்...")
            await download_dub(dubbing_id, out)

            await safe_edit(status, "📤 உனக்கு அனுப்புறேன்...")
            await m.reply_video(out, caption="✅ தமிழ் டப்பிங் ரெடி!", supports_streaming=True)
            await status.delete()
    except httpx.HTTPStatusError as e:
        await safe_edit(status, f"❌ ElevenLabs பிழை: {e.response.status_code}\n{e.response.text[:300]}")
    except Exception as e:
        await safe_edit(status, f"❌ பிழை: {e}")
    finally:
        for p in (src, out):
            try:
                os.remove(p)
            except OSError:
                pass
        try:
            os.rmdir(workdir)
        except OSError:
            pass


# Tiny web server so Render "Web Service" stays alive (skip if you use Background Worker)
async def health(_):
    return web.Response(text="ok")


async def main():
    runner = web.AppRunner(web.Application())
    runner.app.router.add_get("/", health)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", int(os.environ.get("PORT", 8080))).start()
    await app.start()
    print("Bot started")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
