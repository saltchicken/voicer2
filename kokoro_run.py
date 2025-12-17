import sounddevice as sd
import numpy as np
from kokoro import KPipeline
import sys

# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------
LANG_CODE = "a"  # 'a' for American English, 'b' for British
VOICE_NAME = "af_heart"
SAMPLE_RATE = 24000  # Kokoro native rate


def main():
    # 1. Initialize Pipeline
    print("⏳ Loading model... (this takes a second)")
    pipeline = KPipeline(lang_code=LANG_CODE)
    print(f"✅ Model loaded! Voice: {VOICE_NAME}")
    print("--------------------------------------------------")
    print("Type something and press Enter to hear it.")
    print("Type 'exit' or 'quit' to stop.")
    print("--------------------------------------------------")

    while True:
        try:
            # 2. Get User Input
            user_text = input("\n📝 You: ")

            if user_text.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            if not user_text.strip():
                continue

            print("🔊 Speaking...", end="", flush=True)

            # 3. Generate and Stream Audio
            # The pipeline returns a generator. We iterate through it
            # and play chunks as soon as they are ready.
            generator = pipeline(
                user_text, voice=VOICE_NAME, speed=1.0, split_pattern=r"\n+"
            )

            for i, (graphemes, phonemes, audio) in enumerate(generator):
                # 'audio' is a numpy array of float32
                # We play it immediately using sounddevice
                sd.play(audio, SAMPLE_RATE)
                sd.wait()  # Wait for this chunk to finish before playing the next

            print(" Done.")

        except KeyboardInterrupt:
            print("\nStopping...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
