# ---- Kaggle: ஒரே cell (நிறுவல் + clone + keys + bot) ----
# பயன்படுத்தும் முறை: Settings-ல Internet=On, GPU=On, Secrets attach பண்ணு.
# அப்புறம் மேல வலது "Save Version" -> "Save & Run All (Commit)" -> Save.
# இப்படி ஓட்டினா போன் ஆஃப் ஆனாலும் / tab மூடினாலும் Kaggle server-ல தொடர்ந்து ஓடும்.
import os, subprocess, time

os.chdir("/kaggle/working")
subprocess.run("rm -rf bot && git clone https://github.com/yogirename-blip/New-Dubbed-.git bot",
               shell=True, check=True)
subprocess.run("pip install -q -r /kaggle/working/bot/requirements.txt && pip install -q 'av<16'",
               shell=True)

# pyannote (விருப்பம்): நிறுவ முடியலன்னா பரவாயில்ல, bot தானா WavLM முறைக்கு மாறும்
subprocess.run("pip install -q pyannote.audio", shell=True)

# IndicF5 (தமிழ் குரல் clone மாடல்)
subprocess.run("pip install -q git+https://github.com/ai4bharat/IndicF5.git", shell=True)

from kaggle_secrets import UserSecretsClient
s = UserSecretsClient()
for k in ["API_ID", "API_HASH", "BOT_TOKEN", "GEMINI_API_KEY", "HF_TOKEN"]:   # HF_TOKEN இப்போ கட்டாயம் (IndicF5)
    os.environ[k] = s.get_secret(k)
for k in ["OWNER_ID"]:                      # விருப்பம் (OWNER_ID = உன் Telegram user id)
    try:
        os.environ[k] = s.get_secret(k)
    except Exception:
        print(f"{k} இல்ல -> தவிர்க்கிறேன்")

# விருப்பம்: இத்தனை நிமிஷம் வேலை இல்லன்னா bot தானா நிற்கும் (GPU நேரம் மிச்சம்). 0 = நிற்காது
os.environ.setdefault("IDLE_EXIT_MIN", "30")
os.environ.setdefault("MAX_RUN_HOURS", "11")
os.environ.setdefault("OMNI_FP32", "1")   # float32: கரகரப்பு/ரோபோ குரல் குறையும் (கொஞ்சம் மெதுவா)

os.chdir("/kaggle/working/bot")

# bot crash ஆனா 20 வினாடில தானா திரும்ப தொடங்கும்.
# bot தானா (idle / session முடிவு) நின்னா exit code 0 -> cell முடியும் -> GPU நிற்கும்.
DEADLINE = time.time() + 11.5 * 3600
crashes = 0
while time.time() < DEADLINE:
    code = subprocess.run(["python", "-u", "bot.py"]).returncode
    if code == 0:
        print("Bot சுத்தமா நின்னுடுச்சு.")
        break
    crashes += 1
    print(f"Bot crash (exit {code}), restart #{crashes} in 20s")
    if crashes >= 20:
        print("அதிக crash, நிறுத்துறேன்.")
        break
    time.sleep(20)
