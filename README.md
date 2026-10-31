# New-Dubbed (Tamil Auto Dubbing Bot)

வீடியோவை Telegram-ல் அனுப்பினா, அதே கேரக்டர் குரலில், அதே நேரத்தில் தமிழ் டப்பிங் பண்ணி திருப்பி அனுப்பும்.

## Features
- குரல் / BGM பிரிப்பு (demucs) - BGM, effects அப்படியே இருக்கும்
- பேச்சு + நேரம் எடுத்தல் (faster-whisper), மொழி தானா கண்டுபிடிப்பு
- தமிழ் மொழிபெயர்ப்பு (Gemini) - நேரத்துக்கு ஏத்த மாதிரி சுருக்கமா; key இல்லாட்டி Google Translate fallback
- அதே கேரக்டர் குரல் clone (OmniVoice) - தமிழில் பேசும்
- ஒவ்வொரு வசனமும் அசல் நேரத்தில் தொடங்கும், overlap ஆகாது; நீளமா இருந்தா சுருக்கும் (அதிகபட்சம் 1.35x)
- குரல் + BGM கலந்து வீடியோவோடு இணைப்பு (ffmpeg)
- Progress message, ஒரு நேரத்தில் ஒரு வீடியோ (queue)
- ஒரு வசனம் fail ஆனா மத்தது தொடரும்

## Setup
1. ffmpeg install ஆகியிருக்கணும்
2. `pip install torch==2.8.0 torchaudio==2.8.0`
3. `pip install -r requirements.txt`
4. Environment variables:
   - `API_ID`, `API_HASH`, `BOT_TOKEN` (கட்டாயம்)
   - `GEMINI_API_KEY` (aistudio.google.com - இலவசம்)
   - விருப்பம்: `SOURCE_LANG` (ja/hi/en), `BG_GAIN` (0.8), `NUM_STEP` (32; 16 = வேகம்), `WHISPER_MODEL` (large-v3)
5. `python bot.py`

GPU வேணும் (Kaggle / Colab / GPU server). Render CPU-ல் practical இல்ல.
