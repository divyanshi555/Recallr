import yt_dlp
from pydub import AudioSegment
import os

DOWNLOAD_DIR = 'downloads'


'''
Function: Download audio from a YouTube URL
 - Uses yt-dlp to fetch the best available audio stream
 - Converts it to WAV format using FFmpeg
 - Returns the final WAV file path
'''

def download_youtube_audio(url :str) ->str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
    return filename



'''
Function: Convert any audio/video file to WAV format
 - Uses pydub to read the input file
 - Converts audio to mono channel and resamples to 16kHz
 - Exports the processed audio as a WAV file
'''
def convert_to_wav(input_path: str) -> str:
    filename = os.path.splitext(os.path.basename(input_path))[0] + "_converted.wav"
    output_path = os.path.join(DOWNLOAD_DIR, filename)
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz
    audio.export(output_path, format="wav")
    return output_path



'''
Function: Split a WAV file into smaller chunks
 - Divides audio into segments of specified length (default 10 minutes)
 - Exports each chunk as a separate WAV file
 - Returns a list of chunk file paths
'''

def chunk_audio(wav_path : str , chunk_minutes : int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000 

    chunks = []

    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path , format = "wav")

        chunks.append(chunk_path)
    
    return chunks


'''
 Function: Process input source (YouTube URL or local file)
 - Detects if input is a YouTube link or local file
 - Downloads or converts to WAV accordingly
 - Splits audio into chunks
 - Returns list of chunk file paths
'''

def process_input(source: str) -> list:
    source = source.strip().strip('"').strip("'")
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks