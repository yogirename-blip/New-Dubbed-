# New-Dubbed (Auto Dubbing Bot: தமிழ் / తెలుగు / മലയാളം / ಕನ್ನಡ / हिन्दी)

வீடியோவை Telegram-ல் அனுப்பினா, பட்டன்களில் மொழி தேர்ந்தெடுக்கலாம் (தமிழ், தெலுங்கு, மலையாளம், கன்னடம், ஹிந்தி - ஒண்ணு அல்லது பல). அதே கேரக்டர் குரலில், அதே நேரத்தில் டப்பிங் பண்ணி ஒரே வீடியோவா திருப்பி அனுப்பும்.

## Features
- வீடியோ அனுப்பினதும் பட்டன் பேனல்: மொழி (பல தேர்வு), MP4/MKV, Video/File, 🚀 தொடங்கு / ❌ ரத்து
- ஒண்ணுக்கு மேல மொழி தேர்ந்தா: எல்லா மொழி audio-வும் ஒரே MKV-க்குள்ள தனித்தனி track (Tamil Dub, Telugu Dub ... + Original). தனித்தனி வீடியோ வராது. Player-ல (MX Player / VLC) audio track மாத்தலாம்
- ஒரே மொழின்னா MP4 (அல்லது MKV)
- மொழிபெயர்ப்பு: அந்த மொழியின் எழுத்துலயே வந்திருக்கான்னு சோதிக்கும்; தப்பா வந்தா திரும்ப கேட்கும்; முடியலன்னா அந்த இடத்துல அசல் குரல் இருக்கும் (அமைதி ஆகாது)
- ஒரு வீடியோவுக்கு BGM பிரிப்பு + script ஒரே தடவை; அப்புறம் ஒவ்வொரு மொழிக்கும் மொழிபெயர்ப்பு + குரல் மட்டும்
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
   - விருப்பம்: `OMNI_LANG=auto` (OmniVoice-க்கு ஒவ்வொரு மொழி code-ஐ சொல்லும்), `SOURCE_LANG` (ja/hi/en), `BG_GAIN` (0.8), `NUM_STEP` (32; 16 = வேகம்), `WHISPER_MODEL` (large-v3)
5. `python tamil_bot.py`

GPU வேணும் (Kaggle / Colab / GPU server). Render CPU-ல் practical இல்ல.
