import sounddevice as sd
import numpy as np
import whisper
from scipy.io.wavfile import write
import time
import os
from datetime import datetime, timedelta
import threading
import queue
import librosa
import re
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import requests

# ====== CONFIGURATION ======
# Recording Settings
SAMPLERATE = 48000
CHANNELS = 2
DEVICE_INDEX = 12  # Your Stereo Mix device index
CHUNK_DURATION = 20
OVERLAP_DURATION = 5
WHISPER_SAMPLE_RATE = 16000

# API Settings
GROQ_API_KEY = "enter your GROQ API Key here"  # replace with valid key

# Email Settings
EMAIL_ADDRESS = "Enter your personal Email ID"
EMAIL_PASSWORD = "Add your password"  # Gmail App Password
RECIPIENT_EMAIL = "Add recipeint"

# ====== GLOBAL VARIABLES ======
model = None
session_start_time = None
transcript_filename = ""
audio_queue = queue.Queue()
recording_active = True
session_time_offset = 0.0
processed_chunks = []
session_folder = ""

def load_whisper_model():
    """Load Whisper model"""
    global model
    print("🔄 Loading Whisper model...")
    model = whisper.load_model("small") #cann change according to computer specs
    print("✅ Whisper model loaded")

def setup_session():
    """Setup recording session"""
    global session_start_time, transcript_filename, session_folder
    
    session_start_time = datetime.now()
    folder_name = session_start_time.strftime("Meeting_%Y-%m-%d_%H-%M-%S")
    os.makedirs(folder_name, exist_ok=True)
    session_folder = folder_name
    
    transcript_filename = os.path.join(session_folder, "Live_Transcript.txt")
    
    with open(transcript_filename, "w", encoding="utf-8") as f:
        f.write("LIVE MEETING TRANSCRIPT\n")
        f.write(f"Session Start: {session_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
    
    print(f"📝 Session started: {session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📂 Files saved inside: {session_folder}")

def format_timestamp(seconds):
    return str(timedelta(seconds=int(seconds)))

def continuous_recorder():
    global session_time_offset, recording_active
    chunk_id = 1
    previous_audio = None
    print("🎙 Starting continuous recording...")
    
    while recording_active:
        try:
            print(f"🔴 Recording chunk {chunk_id}...")
            
            audio_data = sd.rec(
                int(CHUNK_DURATION * SAMPLERATE),
                samplerate=SAMPLERATE,
                channels=CHANNELS,
                dtype='float32',
                device=DEVICE_INDEX,
                blocking=True
            )
            
            if CHANNELS == 2:
                audio_data = np.mean(audio_data, axis=1)
            
            if previous_audio is not None and chunk_id > 1:
                overlap_samples = int(OVERLAP_DURATION * SAMPLERATE)
                seamless_audio = np.concatenate([
                    previous_audio[-overlap_samples:],
                    audio_data
                ])
                audio_info = {
                    'chunk_id': chunk_id,
                    'audio_data': seamless_audio,
                    'start_time': session_time_offset - OVERLAP_DURATION,
                    'is_overlapped': True
                }
            else:
                audio_info = {
                    'chunk_id': chunk_id,
                    'audio_data': audio_data,
                    'start_time': session_time_offset,
                    'is_overlapped': False
                }
            
            audio_queue.put(audio_info)
            previous_audio = audio_data.copy()
            session_time_offset += CHUNK_DURATION
            chunk_id += 1
            
        except Exception as e:
            print(f"❌ Recording error: {e}")
            break

def transcription_processor():
    global recording_active
    print("📝 Starting transcription processor...")
    
    while recording_active or not audio_queue.empty():
        try:
            try:
                audio_info = audio_queue.get(timeout=2)
            except queue.Empty:
                if not recording_active:
                    break
                continue
            
            print(f"🔍 Transcribing chunk {audio_info['chunk_id']}...")
            
            audio_16k = librosa.resample(
                audio_info['audio_data'], 
                orig_sr=SAMPLERATE, 
                target_sr=WHISPER_SAMPLE_RATE
            )
            
            temp_filename = f"temp_chunk_{audio_info['chunk_id']}.wav"
            write(temp_filename, WHISPER_SAMPLE_RATE, (audio_16k * 32767).astype(np.int16))
            
            result = model.transcribe(temp_filename)
            process_transcription(audio_info, result)
            os.remove(temp_filename)
            
        except Exception as e:
            print(f"❌ Transcription error: {e}")

def process_transcription(audio_info, result):
    chunk_id = audio_info['chunk_id']
    start_time = audio_info['start_time']
    is_overlapped = audio_info['is_overlapped']
    
    segments = result.get('segments', [])
    if not segments:
        return
    
    if is_overlapped:
        filtered_segments = []
        for segment in segments:
            if segment['start'] > OVERLAP_DURATION - 0.5:
                segment['start'] = segment['start'] - OVERLAP_DURATION + start_time
                segment['end'] = segment['end'] - OVERLAP_DURATION + start_time
                filtered_segments.append(segment)
        segments = filtered_segments
    else:
        for segment in segments:
            segment['start'] += start_time
            segment['end'] += start_time
    
    if segments:
        save_transcription(chunk_id, segments, start_time)

def save_transcription(chunk_id, segments, start_time):
    try:
        full_text = " ".join([seg['text'].strip() for seg in segments if seg['text'].strip()])
        if not full_text:
            return
        
        with open(transcript_filename, "a", encoding="utf-8") as f:
            timestamp = format_timestamp(start_time)
            f.write(f"[{timestamp}] {full_text}\n")
        
        timestamp = format_timestamp(start_time)
        print(f"💬 [{timestamp}] {full_text}")
        
        processed_chunks.append({
            'chunk_id': chunk_id,
            'timestamp': start_time,
            'text': full_text
        })
        
    except Exception as e:
        print(f"❌ Error saving transcription: {e}")

def record_meeting():
    global recording_active
    setup_session()
    
    recorder_thread = threading.Thread(target=continuous_recorder, daemon=True)
    transcriber_thread = threading.Thread(target=transcription_processor, daemon=True)
    recorder_thread.start()
    transcriber_thread.start()
    
    print("\n" + "="*60)
    print("🚀 LIVE TRANSCRIPTION ACTIVE")
    print("="*60)
    print(f"📄 Recording to: {transcript_filename}")
    print("⌨️  Press Ctrl+C to stop recording")
    print("="*60 + "\n")
    
    try:
        while recording_active:
            time.sleep(3)
            elapsed = (datetime.now() - session_start_time).total_seconds() / 60
            print(f"📊 Status: {len(processed_chunks)} chunks processed, {elapsed:.1f} min elapsed")
    
    except KeyboardInterrupt:
        print(f"\n🛑 Stopping recording...")
        recording_active = False
        print("⏳ Finishing transcription...")
        recorder_thread.join(timeout=10)
        transcriber_thread.join(timeout=15)
        
        session_end_time = datetime.now()
        duration_minutes = (session_end_time - session_start_time).total_seconds() / 60
        
        with open(transcript_filename, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write("SESSION SUMMARY\n")
            f.write(f"Duration: {duration_minutes:.1f} minutes\n")
            f.write(f"Total Chunks: {len(processed_chunks)}\n")
        
        print(f"✅ Recording completed!")
        print(f"📄 Transcript saved: {transcript_filename}")

def process_transcript_with_ai(transcript_file):
    print(f"🤖 Processing transcript with AI: {transcript_file}")
    with open(transcript_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    spoken_lines = []
    for line in content.split('\n'):
        match = re.match(r'\[(\d{1,2}:\d{2}:\d{2})\]\s*(.*)', line)
        if match and match.group(2).strip():
            spoken_lines.append(match.group(2).strip())
    
    raw_transcript = " ".join(spoken_lines)
    if not raw_transcript.strip():
        print("❌ No spoken content found in transcript")
        return
    
    print(f"📝 Processing {len(raw_transcript)} characters with AI...")
    ai_response = try_ai_processing(raw_transcript)
    
    if ai_response:
        cleaned_transcript, analysis = parse_ai_response(ai_response)
        summary = create_clean_summary(analysis, transcript_file)
    else:
        print("⚠️ AI processing failed, creating basic summary")
        summary = create_basic_summary(raw_transcript)
    
    summary_filename = os.path.join(session_folder, "Meeting_Summary.txt")
    with open(summary_filename, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"💾 Summary saved: {summary_filename}")
    send_email_simple(summary_filename, summary)

def try_ai_processing(raw_transcript):
    prompt = f"""You are a professional meeting summarizer. 
Your task is to extract and organize the most important insights from the transcript.

INSTRUCTIONS:
1. DO NOT include the full transcript.
2. Provide results in this exact format:

MEETING SUMMARY REPORT

📅 Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🔑 MAIN POINTS DISCUSSED:
- Point 1
- Point 2
- Point 3

⚠️ IMPORTANT THINGS TO KEEP IN MIND:
- Item 1
- Item 2

📝 OVERALL SUMMARY:
A concise paragraph (3–5 sentences) summarizing the entire meeting.

Transcript:
{raw_transcript}
"""
    try:
        print("🔄 Processing with Groq AI...")
        if not GROQ_API_KEY:
            print("❌ Please set your GROQ_API_KEY in the configuration")
            return None
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-70b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.3
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()['choices'][0]['message']['content']
            print("✅ Groq AI processing successful")
            return result
        else:
            print(f"❌ Groq failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Groq failed: {e}")
        return None

def parse_ai_response(ai_response):
    try:
        parts = ai_response.split("📝 OVERALL SUMMARY:")
        analysis = {
            "key_points": [],
            "concerns": [],
            "summary": parts[-1].strip()
        }
        
        main_points_match = re.findall(r"- (.*)", ai_response.split("MAIN POINTS DISCUSSED:")[-1].split("⚠️")[0])
        concerns_match = re.findall(r"- (.*)", ai_response.split("THINGS TO KEEP IN MIND:")[-1].split("📝")[0])
        
        analysis["key_points"] = main_points_match
        analysis["concerns"] = concerns_match
        return ai_response, analysis
    except Exception as e:
        print(f"⚠️ Parsing error: {e}")
        return ai_response, {"key_points": [], "concerns": [], "summary": ai_response}

def create_clean_summary(analysis, original_filename):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = f"""MEETING SUMMARY REPORT
📅 Date & Time: {timestamp}
Source: {original_filename}

{'='*50}

🔑 MAIN POINTS DISCUSSED:
"""
    for i, point in enumerate(analysis.get('key_points', []), 1):
        summary += f"{i}. {point}\n"
    
    summary += f"""
{'='*50}

⚠️ IMPORTANT THINGS TO KEEP IN MIND:
"""
    for i, concern in enumerate(analysis.get('concerns', []), 1):
        summary += f"{i}. {concern}\n"
    
    summary += f"""
{'='*50}

📝 OVERALL SUMMARY:
{analysis.get('summary', 'No summary available')}

{'='*50}
"""
    return summary

def create_basic_summary(raw_transcript):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    word_count = len(raw_transcript.split())
    summary = f"""MEETING SUMMARY
Generated: {timestamp}

Total Words: {word_count}
Raw Transcript:
{raw_transcript}
"""
    return summary

def send_email_simple(summary_filename, summary_content):
    print("📧 Sending email...")
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = f"Meeting Summary - {datetime.now().strftime('%Y-%m-%d')}"
        
        body = summary_content
        msg.attach(MIMEText(body, 'plain'))
        
        with open(summary_filename, "rb") as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename= {summary_filename}')
            msg.attach(part)
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        print("✅ Email sent successfully!")
    except Exception as e:
        print(f"❌ Email failed: {e}")
        print("📄 Summary still saved locally")

def main():
    print("🚀 AI Meeting Assistant Starting...")
    load_whisper_model()
    record_meeting()
    print(f"\n🔄 Now processing the transcript with AI...")
    process_transcript_with_ai(transcript_filename)
    print("\n🎉 All done! Check your email for the summary.")

if __name__ == "__main__":
    main()
