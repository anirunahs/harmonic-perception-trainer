"""
Instrument sample loader with CDN caching.

"""

import io
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from .config import (
    INSTRUMENTS,
    SALAMANDER_ANCHORS,
    SAMPLE_RATE,
    DEFAULT_CACHE_DIR,
    midi_to_fluidr3_filename,
)

logger = logging.getLogger(__name__)


class SampleLoader:
    """Downloads, caches, and loads individual instrument note samples."""

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.cache_dir = Path(cache_dir)
        self.sr = sample_rate
        self._audio_cache: dict = {}
        self._failed_urls: set = set()


    def get_note(self, instrument_id: str, midi_note: int) -> Optional[np.ndarray]:
        """Load a single note sample as a numpy array (mono, float32)."""
        cache_key = (instrument_id, midi_note)
        if cache_key in self._audio_cache:
            return self._audio_cache[cache_key]

        config = INSTRUMENTS[instrument_id]

        if config.source_type == 'salamander':
            audio = self._load_salamander(midi_note)
        else:
            audio = self._load_fluidr3(config, midi_note)

        if audio is not None:
            self._audio_cache[cache_key] = audio
        return audio

    def clear_memory_cache(self):
        """Free in-memory audio cache."""
        self._audio_cache.clear()

    def preload_instrument(self, instrument_id: str):
        """Download all note samples for an instrument."""
        config = INSTRUMENTS[instrument_id]
        lo, hi = config.midi_range
        downloaded, skipped = 0, 0
        for midi in range(lo, hi + 1):
            note = self.get_note(instrument_id, midi)
            if note is not None:
                downloaded += 1
            else:
                skipped += 1
        logger.info(
            f"  Preloaded {config.name}: {downloaded} notes, {skipped} skipped"
        )
        self.clear_memory_cache()

    # Salamander Piano

    def _load_salamander(self, midi_note: int) -> Optional[np.ndarray]:
        """Load a Salamander piano note, pitch-shifting from nearest anchor."""
        anchor_midi = self._nearest_salamander_anchor(midi_note)
        shift = midi_note - anchor_midi

        anchor_name = SALAMANDER_ANCHORS[anchor_midi]
        url = f"https://tonejs.github.io/audio/salamander/{anchor_name}.mp3"
        local_path = self.cache_dir / 'salamander' / f"{anchor_name}.mp3"

        audio = self._download_and_load(url, local_path)
        if audio is None:
            return None

        if shift != 0:
            audio = self._pitch_shift(audio, shift)
        return audio

    @staticmethod
    def _nearest_salamander_anchor(midi_note: int) -> int:
        """Find the nearest Salamander anchor."""
        anchors = sorted(SALAMANDER_ANCHORS.keys())
        return min(anchors, key=lambda a: abs(a - midi_note))

    def _pitch_shift(self, audio: np.ndarray, semitones: int) -> np.ndarray:
        """Pitch-shift by resampling."""
        factor = 2 ** (semitones / 12)
        new_length = int(len(audio) / factor)
        old_indices = np.arange(len(audio))
        new_indices = np.linspace(0, len(audio) - 1, new_length)
        shifted = np.interp(new_indices, old_indices, audio)
        return shifted.astype(np.float32)

    # FluidR3_GM

    def _load_fluidr3(self, config, midi_note: int) -> Optional[np.ndarray]:
        """Load a FluidR3_GM note sample."""
        filename = midi_to_fluidr3_filename(midi_note)
        url = f"{config.base_url}{filename}"
        local_path = self.cache_dir / config.fluidr3_instrument / filename

        return self._download_and_load(url, local_path)

    # Download & Load

    def _download_and_load(self, url: str, local_path: Path) -> Optional[np.ndarray]:
        """Download mp3 from URL, convert to WAV, load as array."""
        wav_path = local_path.with_suffix('.wav')

        if wav_path.exists():
            return self._load_wav(wav_path)

        if not local_path.exists():
            if url in self._failed_urls:
                return None
            if not self._download_file(url, local_path):
                self._failed_urls.add(url)
                return None

        if not self._mp3_to_wav(local_path, wav_path):
            return None

        return self._load_wav(wav_path)

    def _load_wav(self, wav_path: Path) -> Optional[np.ndarray]:
        """Load a WAV file as mono float32 at target sample rate."""
        try:
            audio, file_sr = sf.read(str(wav_path), dtype='float32', always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if file_sr != self.sr:
                audio = self._resample(audio, file_sr, self.sr)
            return audio
        except Exception as e:
            logger.warning(f"Cannot read {wav_path.name}: {e}")
            return None

    def _mp3_to_wav(self, mp3_path: Path, wav_path: Path) -> bool:
        """Convert mp3 to mono WAV."""
        try:
            segment = AudioSegment.from_mp3(str(mp3_path))
            segment = segment.set_channels(1)
            segment = segment.set_frame_rate(self.sr)
            segment.export(str(wav_path), format='wav')
            return True
        except Exception as e:
            logger.warning(f"mp3→wav failed for {mp3_path.name}: {e}")
            return False

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple linear-interpolation resample."""
        if orig_sr == target_sr:
            return audio
        duration = len(audio) / orig_sr
        target_len = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_len)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    def _download_file(self, url: str, dest: Path, retries: int = 3) -> bool:
        """Download a file from URL to dest with retry logic."""
        dest.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(1, retries + 1):
            try:
                urllib.request.urlretrieve(url, str(dest))
                return True
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    logger.debug(f"Not found: {url}")
                    return False
                if attempt == retries:
                    logger.warning(f"HTTP {e.code} for {url} after {retries} attempts")
            except Exception as e:
                if attempt == retries:
                    logger.warning(f"Download failed {url}: {e}")

        return False
