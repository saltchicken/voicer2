import sounddevice as sd
import soundfile as sf
import torch
import time
import warnings
import os
import sys

# ‼️ Suppress the annoying pydub syntax warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# ‼️ We need huggingface_hub to download the weights file
from huggingface_hub import hf_hub_download

# ‼️ Import CFM (Conditional Flow Matching) wrapper
from f5_tts.model import DiT, CFM
from f5_tts.infer.utils_infer import infer_process, load_vocoder

# ‼️ Need this to load the vocabulary map correctly
from f5_tts.model.utils import get_tokenizer

# --------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------
# ‼️ IMPORTANT: Point this to your actual .wav file
REF_AUDIO = "ref.wav"

# ‼️ IMPORTANT: You MUST write exactly what is said in the reference audio.
REF_TEXT = ""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ‼️ Standard F5-TTS Base Configuration
F5_MODEL_CFG = dict(
    dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
)

# ‼️ Standard Mel Spectrogram Config for F5
MEL_SPEC_KWARGS = dict(
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    mel_spec_type="vocos",
)


def main():
    print(f"🚀 Initializing F5-TTS on {DEVICE}...")

    # ------------------------------------------------------------------
    # 0. CHECK AUDIO & TEXT
    # ------------------------------------------------------------------
    if not os.path.exists(REF_AUDIO):
        print(f"\n❌ Error: Could not find '{REF_AUDIO}'")
        print("   Please rename your audio file to 'ref.wav' or edit the script.")
        return

    global REF_TEXT
    if not REF_TEXT.strip():
        print(f"\n⚠️  WARNING: REF_TEXT is empty.")
        REF_TEXT = input(f"   👉 Please type what is said in '{REF_AUDIO}': ").strip()
        if not REF_TEXT:
            print("❌ Error: You must provide reference text. Exiting.")
            return

    # ------------------------------------------------------------------
    # 1. DOWNLOAD & LOAD MODEL
    # ------------------------------------------------------------------
    print("\n⬇️  Checking for model weights & vocab...")

    ckpt_path = hf_hub_download(
        repo_id="SWivid/F5-TTS", filename="F5TTS_Base/model_1200000.pt"
    )
    # ‼️ FIX: Download the vocab file so we have the correct text mappings
    vocab_path = hf_hub_download(
        repo_id="SWivid/F5-TTS", filename="F5TTS_Base/vocab.txt"
    )
    print(f"✅ Found weights: {ckpt_path}")
    print(f"✅ Found vocab:   {vocab_path}")

    # ‼️ FIX: Load the correct vocabulary map
    # This ensures the text_embedding layer has the correct size (usually 2546)
    vocab_char_map, vocab_size = get_tokenizer(vocab_path, "custom")

    # 1. Initialize the DiT backbone
    transformer = DiT(**F5_MODEL_CFG)

    # 2. Wrap it in CFM
    model = CFM(
        transformer=transformer,
        mel_spec_kwargs=MEL_SPEC_KWARGS,
        vocab_char_map=vocab_char_map,  # ‼️ Pass the actual vocab map here
    ).to(DEVICE)

    # 3. Load State Dict into CFM
    print(f"📦 Loading state dict...")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)

    if "ema_model_state_dict" in checkpoint:
        state_dict = checkpoint["ema_model_state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix (DistributedDataParallel artifacts)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    # Load weights
    # ‼️ strict=False is okay, but now that vocab is correct, text_embed SHOULD load correctly.
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # Load Vocoder
    vocoder = load_vocoder(is_local=False)

    print(f"\n✅ Ready! Ref Audio: '{REF_AUDIO}'")
    print(f'   Ref Text: "{REF_TEXT}"')
    print("--------------------------------------------------")
    print("Type something and press Enter to generate.")
    print("Type 'exit' to quit.")
    print("--------------------------------------------------")

    while True:
        try:
            text_to_gen = input("\n📝 Text to generate: ")

            if text_to_gen.lower() in ["exit", "quit"]:
                break
            if not text_to_gen.strip():
                continue

            print("⏳ Generating... (Diffusion takes a moment)")
            start_time = time.time()

            # 2. Run Inference
            # Now 'model' is the CFM wrapper, so it has .sample()
            audio, sample_rate, _ = infer_process(
                ref_audio=REF_AUDIO,
                ref_text=REF_TEXT,
                gen_text=text_to_gen,
                model_obj=model,
                vocoder=vocoder,
                mel_spec_type="vocos",
                device=DEVICE,
                nfe_step=16,  # ‼️ Speed Hack: Reduce steps from 32 to 16 (faster generation)
            )

            gen_time = time.time() - start_time
            print(f"🔊 Playing... (Generated in {gen_time:.2f}s)")

            # 3. Play Audio
            sd.play(audio, sample_rate)
            sd.wait()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            if "string index out of range" in str(e):
                print("   (Ensure REF_TEXT matches the audio content exactly.)")
            if "CUDA out of memory" in str(e):
                print("⚠️  OOM: Try shorter text or close other GPU apps.")


if __name__ == "__main__":
    main()
