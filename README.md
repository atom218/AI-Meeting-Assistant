# AI-Meeting-Assistant
AI-powered meeting assistant that records audio in real time, transcribes speech using Whisper, generates concise summaries with Groq AI, and automatically emails the transcript and summary.

# 🎙️ AI Meeting Assistant  

An AI-powered meeting assistant that records audio in real time, transcribes conversations using [Whisper](https://github.com/openai/whisper), summarizes discussions with [Groq AI](https://groq.com/), and delivers structured meeting summaries straight to your inbox.  

---

## 🚀 Features
- 🎧 **Continuous Recording** – Captures meeting audio via microphone or Stereo Mix.  
- ✍️ **Live Transcription** – Uses Whisper for accurate, real-time transcription.  
- 🧠 **AI-Powered Summarization** – Summarizes transcripts into key points, action items, and overall insights.  
- 📂 **Auto-Save Sessions** – Organizes transcripts and summaries into time-stamped folders.  
- 📧 **Email Integration** – Sends transcripts and summaries directly to your email.  

---

## 🛠️ Tech Stack
- **Python**  
- [Whisper](https://github.com/openai/whisper) for speech-to-text  
- [Librosa](https://librosa.org/) & [SoundDevice](https://python-sounddevice.readthedocs.io/) for audio handling  
- [Groq API](https://groq.com/) for AI summarization  
- [SMTP](https://docs.python.org/3/library/smtplib.html) for email automation  

---

## ⚙️ Setup & Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/ai-meeting-assistant.git
   cd ai-meeting-assistant
