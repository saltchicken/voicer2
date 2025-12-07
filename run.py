import soundfile as sf
from kokoro import KPipeline

# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------
# 'a' = American English, 'b' = British English
LANG_CODE = "a"

# Voice options:
# American: af_heart, af_bella, af_nicole, af_sky, am_adam
# British:  bf_emma, bf_isabella, bm_george, bm_lewis
VOICE_NAME = "af_heart"

TEXT = """
Hello! I am running fully locally on your machine. 
Because I am using Kokoro, I am incredibly fast and lightweight.
"""


def main():
    # 1. Initialize the pipeline
    # This will download the model weights (~300MB) on the first run.
    print(f"Loading Kokoro model for language '{LANG_CODE}'...")
    pipeline = KPipeline(lang_code=LANG_CODE)

    # 2. Generate Audio
    print(f"Generating audio with voice '{VOICE_NAME}'...")

    # The pipeline returns a generator of segments (useful for long text)
    generator = pipeline(
        TEXT,
        voice=VOICE_NAME,
        speed=1.0,
        split_pattern=r"\n+",  # Split by newlines
    )

    # 3. Save to file
    # We loop through the segments and save them.
    # For a simple demo, we just save the first segment or concatenate them.
    for i, (graphemes, phonemes, audio) in enumerate(generator):
        filename = f"output_{i}.wav"
        print(f"Saving {filename}...")
        sf.write(filename, audio, 24000)  # 24kHz is the native sample rate

    print("Done! Check your folder for .wav files.")


if __name__ == "__main__":
    main()
