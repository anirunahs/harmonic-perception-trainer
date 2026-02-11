/**
 * Рендер інтервалу фортепіано через Tone.Offline.
 */
import * as Tone from 'tone';
import { INSTRUMENT_CONFIGS, INTERVALS } from '../hooks/useInstrument';

const PIANO_CONFIG = INSTRUMENT_CONFIGS.piano;

/**
 * @param {string} intervalKey - ключ інтервалу
 * @param {string} baseNote - базова нота
 * @param {number} durationSec - тривалість ноти в секундах
 * @returns {Promise<{ root: AudioBuffer, upper: AudioBuffer, combined: AudioBuffer }>}
 */
export async function renderPianoIntervalToBuffers(intervalKey, baseNote = 'C4', durationSec = 2) {
  const semitones = INTERVALS[intervalKey];
  if (semitones == null) throw new Error(`Unknown interval: ${intervalKey}`);
  const upperNote = Tone.Frequency(baseNote).transpose(semitones).toNote();
  const totalDuration = durationSec + 1.5;

  const createSamplerAndPlay = async (notes, duration) => {
    const sampler = new Tone.Sampler({
      urls: PIANO_CONFIG.samples,
      baseUrl: PIANO_CONFIG.baseUrl,
      ...PIANO_CONFIG.options,
    }).toDestination();
    await Tone.loaded();
    sampler.triggerAttackRelease(notes, duration, 0);
  };

  const root = await Tone.Offline(async () => {
    await createSamplerAndPlay(baseNote, durationSec);
  }, totalDuration);

  const upper = await Tone.Offline(async () => {
    await createSamplerAndPlay(upperNote, durationSec);
  }, totalDuration);

  const combined = await Tone.Offline(async () => {
    await createSamplerAndPlay([baseNote, upperNote], durationSec);
  }, totalDuration);

  const toAudioBuffer = (buf) => {
    if (buf && buf.getChannelData) return buf;
    if (buf && buf._buffer) return buf._buffer;
    return buf;
  };

  return {
    root: toAudioBuffer(root),
    upper: toAudioBuffer(upper),
    combined: toAudioBuffer(combined),
  };
}
