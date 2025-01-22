import torch
import TTS
from TTS.api import TTS
import sounddevice as sd
import queue
import threading
import numpy as np
import re

class StreamingTTS:
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2", sample_rate=24000):
        # Initialize TTS
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = TTS(model_name).to(self.device)
        self.sample_rate = sample_rate
        
        # Initialize audio queue and event flags
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.is_playing = False

    def audio_player_thread(self):
        """Thread to handle audio playback"""
        while not self.stop_event.is_set() or not self.audio_queue.empty():
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                sd.play(audio_chunk, samplerate=self.sample_rate)
                sd.wait()  # Wait until audio is finished playing
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio playback: {e}")
                break

    def preprocess_text(self, text):
        """Split text into sentences using regex"""
        # Split on period followed by space or newline
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Remove empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def generate_and_stream(self, text, speaker_wav=None, language="en"):
        """Generate TTS audio and stream it in real-time"""
        try:
            # Reset flags and clear queue
            self.stop_event.clear()
            while not self.audio_queue.empty():
                self.audio_queue.get()

            # Start audio player thread
            self.is_playing = True
            audio_thread = threading.Thread(target=self.audio_player_thread)
            audio_thread.start()

            # Process each sentence
            sentences = self.preprocess_text(text)
            for sentence in sentences:
                if self.stop_event.is_set():
                    break

                # Generate audio for the sentence
                wav = self.tts.tts(
                    text=sentence,
                    speaker_wav=speaker_wav,
                    language=language
                )

                # Normalize audio
                wav = np.array(wav)
                wav = wav / np.max(np.abs(wav))

                # Add to queue for playback
                self.audio_queue.put(wav)

            # Wait for audio thread to finish
            self.stop_event.set()
            audio_thread.join()
            self.is_playing = False

        except Exception as e:
            print(f"Error in TTS generation: {e}")
            self.stop_event.set()
            self.is_playing = False
            raise

    def stop_stream(self):
        """Stop the current stream"""
        self.stop_event.set()
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        self.is_playing = False

# Example usage
if __name__ == "__main__":
    # Initialize the streaming TTS
    streaming_tts = StreamingTTS()

    # Example text to convert to speech
    long_text = """
    This is a long text that will be converted to speech in real-time.
    You can write multiple paragraphs and they will be processed sentence by sentence.
    The audio will start playing as soon as the first sentence is converted.
    This allows for a more natural listening experience without waiting for the entire text to be processed.
    """

    # Start the streaming TTS
    try:
        streaming_tts.generate_and_stream(
            text=long_text,
            speaker_wav="path/to/your/speaker.wav",  # Optional: for voice cloning
            language="en"
        )
    except KeyboardInterrupt:
        print("\nStopping TTS stream...")
        streaming_tts.stop_stream()