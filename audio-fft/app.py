import io
import math
from dataclasses import dataclass

import numpy as np
try:
    import soundfile as sf
except ImportError:  # the in-browser build has no libsndfile, so it reads WAV only
    sf = None
import streamlit as st
from matplotlib import pyplot as plt
from scipy.signal import stft
from scipy.io.wavfile import read as wav_read, write as wav_write


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Decompose & Reconstruct Audio",
    page_icon="🎧",
    layout="wide",
)

st.title("🎧 Decompose & Reconstruct Audio")
st.caption(
    "Record audio, inspect waveform / FFT / spectrogram, then modify frequency components and hear the difference."
)

if "audio_version" not in st.session_state:
    st.session_state.audio_version = 0

# =========================
# Data container
# =========================
@dataclass
class AudioData:
    samples: np.ndarray   # mono float32 signal in [-1, 1]
    sample_rate: int


# =========================
# Utilities
# =========================
def normalize_audio(x: np.ndarray) -> np.ndarray:
    """Normalize to [-1, 1] safely."""
    x = np.asarray(x, dtype=np.float32)
    peak = np.max(np.abs(x)) if x.size else 0.0
    if peak > 0:
        x = x / peak
    return x.astype(np.float32)


def load_audio_file(uploaded_file) -> AudioData:
    """
    Read uploaded / recorded audio into mono float32.
    Uses soundfile so it can handle wav/ogg/flac supported by libsndfile.
    """
    if sf is not None:
        data, sr = sf.read(uploaded_file, always_2d=False)
    else:
        sr, data = wav_read(io.BytesIO(uploaded_file.getvalue()))
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.float32) / np.iinfo(data.dtype).max

    if data.ndim == 2:
        data = np.mean(data, axis=1)

    data = normalize_audio(data)
    return AudioData(samples=data, sample_rate=int(sr))


def to_wav_bytes(x: np.ndarray, sr: int) -> bytes:
    """Convert float signal in [-1,1] to WAV bytes for playback."""
    x = np.clip(x, -1.0, 1.0)
    x_int16 = (x * 32767).astype(np.int16)

    buffer = io.BytesIO()
    wav_write(buffer, sr, x_int16)
    buffer.seek(0)
    return buffer.read()


def compute_fft(x: np.ndarray, sr: int):
    """
    Return positive-frequency FFT info:
    freqs, complex spectrum, magnitude
    """
    n = len(x)
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    magnitude = np.abs(spectrum)
    return freqs, spectrum, magnitude


def apply_frequency_mask(
    x: np.ndarray,
    sr: int,
    low_cut_hz: float,
    high_cut_hz: float,
    keep_top_k: int | None,
):
    """
    Apply band-limiting and optional top-K strongest-bin retention.
    Returns reconstructed signal and intermediate FFT data.
    """
    freqs, spectrum, magnitude = compute_fft(x, sr)
    masked = spectrum.copy()

    # 1) Band mask
    band_mask = (freqs >= low_cut_hz) & (freqs <= high_cut_hz)
    masked[~band_mask] = 0

    # 2) Top-K strongest positive-frequency bins (excluding DC if possible)
    if keep_top_k is not None and keep_top_k > 0:
        # Work only with bins that remain after band mask
        candidate_idx = np.where(np.abs(masked) > 0)[0]

        if len(candidate_idx) > keep_top_k:
            mags = np.abs(masked[candidate_idx])
            # Keep the largest K bins
            top_local = np.argpartition(mags, -keep_top_k)[-keep_top_k:]
            keep_idx = set(candidate_idx[top_local].tolist())

            new_masked = np.zeros_like(masked)
            for idx in keep_idx:
                new_masked[idx] = masked[idx]

            # Optionally preserve DC if already in band and nonzero
            if 0 in candidate_idx and 0 not in keep_idx:
                pass

            masked = new_masked

    reconstructed = np.fft.irfft(masked, n=len(x))
    reconstructed = normalize_audio(reconstructed)

    return {
        "freqs": freqs,
        "original_spectrum": spectrum,
        "original_magnitude": magnitude,
        "masked_spectrum": masked,
        "masked_magnitude": np.abs(masked),
        "reconstructed": reconstructed,
    }


def top_frequency_peaks(freqs: np.ndarray, magnitude: np.ndarray, top_n: int = 8):
    """
    Return top frequency peaks ignoring the DC bin.
    """
    if len(freqs) <= 1:
        return []

    mags = magnitude.copy()
    mags[0] = 0  # ignore DC

    nonzero_idx = np.where(mags > 0)[0]
    if len(nonzero_idx) == 0:
        return []

    top_n = min(top_n, len(nonzero_idx))
    idx = np.argpartition(mags, -top_n)[-top_n:]
    idx = idx[np.argsort(mags[idx])[::-1]]

    peaks = []
    for i in idx:
        peaks.append((float(freqs[i]), float(magnitude[i])))
    return peaks


def make_waveform_plot(x: np.ndarray, sr: int, title: str):
    t = np.arange(len(x)) / sr
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, x, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def make_fft_plot(
    freqs: np.ndarray,
    orig_mag: np.ndarray,
    mod_mag: np.ndarray,
    max_display_hz: int,
):
    fig, ax = plt.subplots(figsize=(10, 4))
    display_mask = freqs <= max_display_hz

    ax.plot(freqs[display_mask], orig_mag[display_mask], label="Original", linewidth=1.0)
    ax.plot(freqs[display_mask], mod_mag[display_mask], label="Modified", linewidth=1.0)
    ax.set_title("FFT Magnitude Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def make_spectrogram_plot(x: np.ndarray, sr: int, title: str):
    """
    Short-time Fourier transform spectrogram.
    """
    nperseg = min(1024, len(x)) if len(x) >= 64 else len(x)
    noverlap = max(0, int(nperseg * 0.75))

    f, t, zxx = stft(x, fs=sr, nperseg=nperseg, noverlap=noverlap, boundary=None)
    power = np.abs(zxx)
    power_db = 20 * np.log10(power + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 4))
    mesh = ax.pcolormesh(t, f, power_db, shading="gouraud")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(mesh, ax=ax, label="Power (dB)")
    fig.tight_layout()
    return fig


def explain_changes(
    low_cut_hz: float,
    high_cut_hz: float,
    keep_top_k: int | None,
    peaks_before,
    peaks_after,
    sr: int,
) -> str:
    """
    Simple heuristic explanation panel.
    Keeps the app self-contained and submission-safe.
    """
    nyquist = sr / 2
    notes = []

    # Frequency-range explanation
    if low_cut_hz > 0:
        notes.append(
            f"- You removed frequencies below **{low_cut_hz:.0f} Hz**, so the reconstructed audio should lose more of its low-end warmth/body."
        )

    if high_cut_hz < nyquist:
        notes.append(
            f"- You removed frequencies above **{high_cut_hz:.0f} Hz**, so the reconstructed audio should sound more muted or muffled."
        )

    # Top-K explanation
    if keep_top_k is not None:
        notes.append(
            f"- You kept only the **top {keep_top_k} strongest frequency bins**, which forces the signal into a simpler approximation."
        )
        if keep_top_k <= 20:
            notes.append(
                "- With a small K, you are effectively hearing a compressed / simplified version of the signal, so fine details are lost."
            )

    # Peaks explanation
    if peaks_before:
        top_before_str = ", ".join([f"{p[0]:.1f} Hz" for p in peaks_before[:5]])
        notes.append(f"- Dominant frequencies before modification: **{top_before_str}**.")

    if peaks_after:
        top_after_str = ", ".join([f"{p[0]:.1f} Hz" for p in peaks_after[:5]])
        notes.append(f"- Dominant frequencies after modification: **{top_after_str}**.")

    if not notes:
        notes.append("- No strong modification was applied, so the original and reconstructed signals should sound very similar.")

    return "\n".join(notes)


# =========================
# Sidebar controls
# =========================
st.sidebar.header("Controls")

input_mode = st.sidebar.radio(
    "Audio source",
    ["Use a sample sound", "Record from microphone", "Upload audio file"],
)

st.sidebar.markdown("### Modification settings")

default_max_hz = 8000
max_display_hz = st.sidebar.slider(
    "Max FFT display frequency (Hz)",
    min_value=1000,
    max_value=22050,
    value=default_max_hz,
    step=500,
)

top_k_mode = st.sidebar.toggle("Keep only top-K strongest frequency bins", value=False)
keep_top_k = None

# =========================
# Input section
# =========================
audio_obj = None

if st.button("Clear current audio", key="clear_audio_button"):
    st.session_state.audio_version += 1
    for key in ["keep_top_k_value", "keep_top_k_number", "keep_top_k_slider"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

if input_mode == "Use a sample sound":
    st.subheader("1) Sample sound")
    st.write("Two seconds of an A major chord (440, 554 and 659 Hz) with a little noise on top. Try the top-K and high-cut controls on it.")
    sr_s = 22050
    t = np.arange(int(sr_s * 2.0)) / sr_s
    chord = sum(a * np.sin(2 * np.pi * f * t) for f, a in [(440.0, 1.0), (554.37, 0.7), (659.25, 0.5), (880.0, 0.25)])
    chord += 0.08 * np.random.default_rng(1).standard_normal(t.size)
    audio_obj = AudioData(samples=normalize_audio(chord * np.minimum(1, t / 0.02)), sample_rate=sr_s)

elif input_mode == "Record from microphone":
    st.subheader("1) Record audio")
    st.write("Record a short phrase such as “hello”, clap once, or make a humming sound.")

    recorded = st.audio_input(
        "Click to record",
        sample_rate=44100,
        key=f"audio_input_{st.session_state.audio_version}"
    )

    if recorded is not None:
        audio_obj = load_audio_file(recorded)

else:
    st.subheader("1) Upload audio")

    uploaded = st.file_uploader(
        "Upload a WAV / OGG / FLAC file" if sf is not None else "Upload a WAV file",
        type=["wav", "ogg", "flac"] if sf is not None else ["wav"],
        key=f"file_uploader_{st.session_state.audio_version}"
    )

    if uploaded is not None:
        audio_obj = load_audio_file(uploaded)

if audio_obj is None:
    st.info("Record or upload audio to begin.")
    st.stop()

# =========================
# Preprocess
# =========================
x = audio_obj.samples
sr = audio_obj.sample_rate

duration_sec = len(x) / sr
nyquist = sr / 2

st.success(f"Loaded audio: {duration_sec:.2f} seconds at {sr} Hz sample rate.")

max_k_bins = len(np.fft.rfft(x))

# Main filter controls depend on sample rate
# Human-hearing-inspired display range, but still limited by the actual audio
human_low_default = 20
human_high_default = min(20000, int(nyquist))
slider_max_hz = min(22000, int(nyquist))

col_a, col_b = st.columns(2)

with col_a:
    low_cut_hz = st.slider(
        "Low cut (Hz) — remove frequencies below this",
        min_value=0,
        max_value=slider_max_hz,
        value=min(human_low_default, slider_max_hz),
        step=10,
    )

with col_b:
    high_cut_hz = st.slider(
        "High cut (Hz) — remove frequencies above this",
        min_value=0,
        max_value=slider_max_hz,
        value=human_high_default,
        step=10,
    )

st.caption(
    f"Human hearing is roughly 20 Hz to 20 kHz, but this audio can only represent frequencies up to {nyquist:.0f} Hz based on its sample rate."
)

if top_k_mode:
    default_k = max(1, int(0.10 * max_k_bins))

    # Initialize shared K state
    if "keep_top_k_value" not in st.session_state:
        st.session_state.keep_top_k_value = default_k

    # Clamp K if a new audio file has fewer bins
    st.session_state.keep_top_k_value = max(
        1,
        min(int(st.session_state.keep_top_k_value), max_k_bins)
    )

    st.sidebar.markdown("### Top-K frequency selection")

    # Keep widget states aligned with shared value before rendering
    st.session_state["keep_top_k_number"] = int(st.session_state.keep_top_k_value)
    st.session_state["keep_top_k_slider"] = int(st.session_state.keep_top_k_value)

    def sync_k_from_number():
        st.session_state.keep_top_k_value = int(st.session_state.keep_top_k_number)
        st.session_state.keep_top_k_slider = int(st.session_state.keep_top_k_value)

    def sync_k_from_slider():
        st.session_state.keep_top_k_value = int(st.session_state.keep_top_k_slider)
        st.session_state.keep_top_k_number = int(st.session_state.keep_top_k_value)

    st.sidebar.number_input(
        f"Enter exact K value (1 to {max_k_bins})",
        min_value=1,
        max_value=max_k_bins,
        step=1,
        key="keep_top_k_number",
        on_change=sync_k_from_number,
    )

    st.sidebar.slider(
        f"Or adjust K with slider (max = {max_k_bins})",
        min_value=1,
        max_value=max_k_bins,
        step=1,
        key="keep_top_k_slider",
        on_change=sync_k_from_slider,
    )

    keep_top_k = int(st.session_state.keep_top_k_value)

    st.sidebar.caption(
        f"Keeping {keep_top_k} out of {max_k_bins} total frequency bins."
    )

    pct_kept = 100 * keep_top_k / max_k_bins
    st.sidebar.caption(
        f"That is {pct_kept:.2f}% of the available frequency bins."
    )

if low_cut_hz > high_cut_hz:
    st.error("Low cut must be less than or equal to high cut.")
    st.stop()


# =========================
# Original analysis
# =========================
orig_freqs, orig_spectrum, orig_mag = compute_fft(x, sr)
orig_peaks = top_frequency_peaks(orig_freqs, orig_mag, top_n=8)

# Modified analysis
result = apply_frequency_mask(
    x=x,
    sr=sr,
    low_cut_hz=low_cut_hz,
    high_cut_hz=high_cut_hz,
    keep_top_k=keep_top_k,
)

x_mod = result["reconstructed"]
mod_freqs = result["freqs"]
mod_mag = result["masked_magnitude"]
mod_peaks = top_frequency_peaks(mod_freqs, mod_mag, top_n=8)

orig_wav = to_wav_bytes(x, sr)
mod_wav = to_wav_bytes(x_mod, sr)


# =========================
# Audio players
# =========================
st.subheader("2) Listen")
listen_col1, listen_col2 = st.columns(2)

with listen_col1:
    st.markdown("**Original audio**")
    st.audio(orig_wav, format="audio/wav")

with listen_col2:
    st.markdown("**Modified / reconstructed audio**")
    st.audio(mod_wav, format="audio/wav")


# =========================
# Visuals
# =========================
st.subheader("3) Visualize (Waveform, Spectrum, Spectrogram)")

plot_col1, plot_col2 = st.columns(2)

with plot_col1:
    st.pyplot(make_waveform_plot(x, sr, "Original Waveform"))

with plot_col2:
    st.pyplot(make_waveform_plot(x_mod, sr, "Modified / Reconstructed Waveform"))

st.pyplot(make_fft_plot(orig_freqs, orig_mag, mod_mag, max_display_hz=max_display_hz))

spec_col1, spec_col2 = st.columns(2)

with spec_col1:
    st.pyplot(make_spectrogram_plot(x, sr, "Original Spectrogram"))

with spec_col2:
    st.pyplot(make_spectrogram_plot(x_mod, sr, "Modified Spectrogram"))


# =========================
# Frequency peaks table
# =========================
st.subheader("4) Dominant frequencies")

table_col1, table_col2 = st.columns(2)

with table_col1:
    st.markdown("**Top frequencies in original audio**")
    if orig_peaks:
        rows = [{"Frequency (Hz)": round(f, 2), "Magnitude": round(m, 4)} for f, m in orig_peaks]
        st.table(rows)
    else:
        st.write("No strong peaks found.")

with table_col2:
    st.markdown("**Top frequencies after modification**")
    if mod_peaks:
        rows = [{"Frequency (Hz)": round(f, 2), "Magnitude": round(m, 4)} for f, m in mod_peaks]
        st.table(rows)
    else:
        st.write("No strong peaks found.")


# =========================
# Explanation panel
# =========================
st.subheader("5) Explanation / learning notes")

explanation = explain_changes(
    low_cut_hz=low_cut_hz,
    high_cut_hz=high_cut_hz,
    keep_top_k=keep_top_k,
    peaks_before=orig_peaks,
    peaks_after=mod_peaks,
    sr=sr,
)
st.markdown(explanation)

with st.expander("Suggested experiments"):
    st.markdown(
        """
1. **Record your voice** and set a lower **high cut**.  
   Notice how speech becomes more muffled as upper frequencies are removed.

2. **Hum a steady note** and enable **top-K strongest bins** with a small K.  
   Listen to how the reconstructed signal keeps the main tone but loses detail.

3. **Clap once** or make a sharp “tss” sound.  
   Compare its FFT and spectrogram with a vowel sound like “ahhh”.

4. Keep **low cut = 0** and reduce **high cut** gradually.  
   This helps you hear how high-frequency content contributes to clarity.
        """
    )

with st.expander("What this app is teaching"):
    st.markdown(
        """
- A real-world sound can be analyzed in the **time domain** as a waveform.
- The **FFT** reveals which frequencies are present and how strong they are.
- A **spectrogram** shows how those frequencies change over time.
- By removing bins or keeping only top-K bins, you hear how reconstruction quality depends on the retained frequency components.
        """
    )


# =========================
# Download modified audio
# =========================
st.subheader("6) Export")
st.download_button(
    label="Download modified audio as WAV",
    data=mod_wav,
    file_name="modified_audio.wav",
    mime="audio/wav",
)