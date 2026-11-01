# ---- Kaggle: ஒரே cell (நிறுவல் + clone + keys + bot) ----
import os, subprocess
os.chdir("/kaggle/working")
subprocess.run("rm -rf bot && git clone https://github.com/yogirename-blip/New-Dubbed-.git bot", shell=True, check=True)
subprocess.run("pip install -q -r /kaggle/working/bot/requirements.txt && pip install -q 'av<16'", shell=True)

# pyannote (விருப்பம்): நிறுவ முடியலன்னா பரவாயில்ல, bot தானா WavLM முறைக்கு மாறும்
subprocess.run("pip install -q pyannote.audio", shell=True)

from kaggle_secrets import UserSecretsClient
s = UserSecretsClient()
for k in ["API_ID", "API_HASH", "BOT_TOKEN", "GEMINI_API_KEY"]:
    os.environ[k] = s.get_secret(k)
try:
    os.environ["HF_TOKEN"] = s.get_secret("HF_TOKEN")      # விருப்பம்
except Exception:
    print("HF_TOKEN இல்ல -> pyannote இல்லாம ஓடும்")
os.chdir("/kaggle/working/bot")
!python -u bot.py
