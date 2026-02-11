import React, { useState, useCallback } from "react";
import { Link } from "react-router-dom";
import Header from "../components/Header";
import api from "../api";
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
  ReferenceLine,
} from "recharts";
import * as Tone from "tone";
import { Play, Loader2 } from "lucide-react";
import { useInstrument, INTERVALS } from "../hooks/useInstrument";
import { renderPianoIntervalToBuffers } from "../utils/offlinePiano";
import { audioBufferToWav, arrayBufferToBase64 } from "../utils/audioUtils";

const BASE_NOTE = "C4";

function getUpperNoteName(intervalKey) {
  const semitones = INTERVALS[intervalKey];
  if (semitones == null) return "G4";
  return Tone.Frequency(BASE_NOTE).transpose(semitones).toNote();
}

const INTERVALS_UK = [
  { key: "minor_second", label: "Мала секунда", semitones: 1 },
  { key: "major_second", label: "Велика секунда", semitones: 2 },
  { key: "minor_third", label: "Мала терція", semitones: 3 },
  { key: "major_third", label: "Велика терція", semitones: 4 },
  { key: "perfect_fourth", label: "Чиста кварта", semitones: 5 },
  { key: "tritone", label: "Тритон", semitones: 6 },
  { key: "perfect_fifth", label: "Чиста квінта", semitones: 7 },
  { key: "minor_sixth", label: "Мала секста", semitones: 8 },
  { key: "major_sixth", label: "Велика секста", semitones: 9 },
  { key: "minor_seventh", label: "Мала септима", semitones: 10 },
  { key: "major_seventh", label: "Велика септима", semitones: 11 },
  { key: "perfect_octave", label: "Чиста октава", semitones: 12 },
];

function ExperimentOvertones() {
  const [intervalKey, setIntervalKey] = useState("perfect_fifth");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  /** Який спектр показувати на графіку: all | root | upper | combined */
  const [chartFocus, setChartFocus] = useState("all");
  const [spectrogramImage, setSpectrogramImage] = useState(null);
  const [loadingSpectrogram, setLoadingSpectrogram] = useState(false);

  const { isLoaded: instrumentLoaded, playNote, playHarmonicInterval } = useInstrument("piano");
  const upperNoteName = React.useMemo(() => getUpperNoteName(intervalKey), [intervalKey]);

  const loadExperiment = useCallback(async () => {
    if (!instrumentLoaded) return;
    setLoading(true);
    setError(null);
    try {
      const { root, upper, combined } = await renderPianoIntervalToBuffers(intervalKey, "C4", 2);
      const sampleRate = root.sampleRate || 44100;
      const payload = {
        interval: intervalKey,
        sample_rate: sampleRate,
        audio_root_base64: arrayBufferToBase64(audioBufferToWav(root)),
        audio_upper_base64: arrayBufferToBase64(audioBufferToWav(upper)),
        audio_combined_base64: arrayBufferToBase64(audioBufferToWav(combined)),
      };
      const res = await api.post("/api/experiments/overtones/", payload);
      setData(res.data);
      setChartFocus("all");
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Помилка запиту");
    } finally {
      setLoading(false);
    }
  }, [intervalKey, instrumentLoaded]);

  const chartData = React.useMemo(() => {
    if (!data?.combined_spectrum?.frequencies?.length) return [];
    const freqs = data.combined_spectrum.frequencies;
    const mags = data.combined_spectrum.magnitudes;
    const root = data.root_spectrum?.magnitudes || [];
    const upper = data.upper_spectrum?.magnitudes || [];
    const maxFreq = 2500;
    const raw = freqs.map((f, i) => ({
      freq: Math.round(f),
      combined: mags[i] ?? 0,
      root: root[i] ?? 0,
      upper: upper[i] ?? 0,
    }));
    const filtered = raw.filter((d) => d.freq <= maxFreq);
    const maxVal = Math.max(
      ...filtered.flatMap((d) => [d.combined, d.root, d.upper]),
      1e-10
    );
    return filtered.map((d) => ({
      ...d,
      combined: d.combined / maxVal,
      root: d.root / maxVal,
      upper: d.upper / maxVal,
    }));
  }, [data]);

  const playRoot = useCallback(() => {
    if (!instrumentLoaded || !data) return;
    playNote(BASE_NOTE, "2n");
    setChartFocus("root");
  }, [instrumentLoaded, data, playNote]);

  const playUpper = useCallback(() => {
    if (!instrumentLoaded || !data) return;
    playNote(upperNoteName, "2n");
    setChartFocus("upper");
  }, [instrumentLoaded, data, upperNoteName, playNote]);

  const playInterval = useCallback(() => {
    if (!instrumentLoaded || !data) return;
    playHarmonicInterval(BASE_NOTE, intervalKey, "2n");
    setChartFocus("combined");
  }, [instrumentLoaded, data, intervalKey, playHarmonicInterval]);

  const generateSpectrogramGrid = useCallback(() => {
    setLoadingSpectrogram(true);
    setSpectrogramImage(null);
    api
      .post("/api/experiments/overtones/spectrogram-grid/")
      .then((res) => {
        if (res.data?.image_base64) setSpectrogramImage(res.data.image_base64);
      })
      .catch(() => setError("Не вдалося згенерувати спектрограми"))
      .finally(() => setLoadingSpectrogram(false));
  }, []);

  const downloadSpectrogramImage = useCallback(() => {
    if (!spectrogramImage) return;
    const link = document.createElement("a");
    link.href = `data:image/png;base64,${spectrogramImage}`;
    link.download = "spectrogramy-intervaliv.png";
    link.click();
  }, [spectrogramImage]);

  return (
    <>
      <Header />
      <div className="experiments-page experiment-overtones">
        <div className="experiments-page__container">
          <p className="experiment-overtones__back">
            <Link to="/experiments">← До списку експериментів</Link>
          </p>
          <header className="experiments-page__header">
            <h1 className="experiments-page__title">Обертони інтервалів</h1>
          </header>

          <div className="overtone-controls">
            <label className="overtone-controls__label">
              Інтервал
              <select
                value={intervalKey}
                onChange={(e) => setIntervalKey(e.target.value)}
                className="overtone-controls__select"
              >
                {INTERVALS_UK.map(({ key, label, semitones }) => (
                  <option key={key} value={key}>
                    {label} ({semitones} півт.)
                  </option>
                ))}
              </select>
            </label>
            <button
              type="button"
              className="btn btn--primary"
              onClick={loadExperiment}
              disabled={loading || !instrumentLoaded}
            >
              {loading ? <Loader2 className="spin" /> : <Play />}
              <span>
                {!instrumentLoaded ? "Завантаження семплів…" : loading ? "Генерація…" : "Синтезувати та показати спектр"}
              </span>
            </button>
          </div>

          {error && <div className="alert alert--error">{error}</div>}

          {data && (
            <>
              <div className="overtone-audio">
                <p className="overtone-audio__info">
                  {data.interval_name_uk} — {data.base_note}4 ({data.root_freq_hz} Гц) + верхня нота ({data.upper_freq_hz} Гц)
                </p>
                <div className="overtone-audio__buttons">
                  <button
                    type="button"
                    className={`btn btn--secondary overtone-audio__btn-note ${chartFocus === "root" ? "overtone-audio__btn-note--active" : ""}`}
                    onClick={playRoot}
                    disabled={!instrumentLoaded}
                    title="Відтворити тільки базову ноту та показати її спектр"
                  >
                    <Play /> Базова нота ({BASE_NOTE})
                  </button>
                  <button
                    type="button"
                    className={`btn btn--secondary overtone-audio__btn-note ${chartFocus === "upper" ? "overtone-audio__btn-note--active" : ""}`}
                    onClick={playUpper}
                    disabled={!instrumentLoaded}
                    title="Відтворити тільки верхню ноту та показати її спектр"
                  >
                    <Play /> Верхня нота ({upperNoteName})
                  </button>
                  <button
                    type="button"
                    className={`btn btn--secondary overtone-audio__btn-note ${chartFocus === "combined" ? "overtone-audio__btn-note--active" : ""}`}
                    onClick={playInterval}
                    disabled={!instrumentLoaded}
                    title="Відтворити інтервал (обидві ноти) та показати сумарний спектр"
                  >
                    <Play /> Інтервал (обидві)
                  </button>
                </div>
                {chartFocus !== "all" && (
                  <button
                    type="button"
                    className="overtone-audio__show-all"
                    onClick={() => setChartFocus("all")}
                  >
                    Показати всі три криві на графіку
                  </button>
                )}
              </div>

              <div className="overtone-chart">
                <h3>Спектр інтервалу</h3>
                <ResponsiveContainer width="100%" height={380}>
                  <AreaChart data={chartData} margin={{ top: 32, right: 24, left: 52, bottom: 56 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="freq"
                      type="number"
                      domain={[0, 2500]}
                      ticks={[0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500]}
                      label={{ value: "Частота (Гц)", position: "right", offset: 8 }}
                    />
                    <YAxis
                      domain={[0, 1.05]}
                      tickFormatter={(v) => (v === 1 ? "1" : v.toFixed(1))}
                      label={{ value: "Відносна амплітуда", angle: -90, position: "insideLeft", style: { textAnchor: "middle" } }}
                      width={40}
                    />
                    <Tooltip
                      formatter={(value) => (typeof value === "number" ? value.toFixed(3) : value)}
                      labelFormatter={(freq) => `≈ ${freq} Гц`}
                      contentStyle={{ fontSize: "13px" }}
                    />
                    <Legend wrapperStyle={{ marginTop: 12 }} />
                    {data?.root_freq_hz != null && (chartFocus === "all" || chartFocus === "root" || chartFocus === "combined") && (
                      <ReferenceLine
                        x={data.root_freq_hz}
                        stroke="#059669"
                        strokeDasharray="4 4"
                        label={{ value: `${Math.round(data.root_freq_hz)} Гц (базова)`, position: "left", fontSize: 11 }}
                      />
                    )}
                    {data?.upper_freq_hz != null && (chartFocus === "all" || chartFocus === "upper" || chartFocus === "combined") && (
                      <ReferenceLine
                        x={data.upper_freq_hz}
                        stroke="#d97706"
                        strokeDasharray="4 4"
                        label={{ value: `${Math.round(data.upper_freq_hz)} Гц (верхня)`, position: "right", fontSize: 11 }}
                      />
                    )}
                    {(chartFocus === "all" || chartFocus === "combined") && (
                      <Area type="monotone" dataKey="combined" stroke="#2563eb" fill="#2563eb40" name="Інтервал (обидві ноти)" />
                    )}
                    {(chartFocus === "all" || chartFocus === "root") && (
                      <Area type="monotone" dataKey="root" stroke="#059669" fill="#05966920" name="Базова нота" />
                    )}
                    {(chartFocus === "all" || chartFocus === "upper") && (
                      <Area type="monotone" dataKey="upper" stroke="#d97706" fill="#d9770620" name="Верхня нота" />
                    )}
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </>
          )}

          <div className="overtone-spectrograms">
            <h3>Спектрограми 12 інтервалів (3×4)</h3>
            <div className="overtone-spectrograms__actions">
              <button
                type="button"
                className="btn btn--primary"
                onClick={generateSpectrogramGrid}
                disabled={loadingSpectrogram}
              >
                {loadingSpectrogram ? <Loader2 className="spin" /> : null}
                <span>{loadingSpectrogram ? "Генерація…" : "Згенерувати спектрограми 12 інтервалів"}</span>
              </button>
              {spectrogramImage && (
                <button
                  type="button"
                  className="btn btn--secondary"
                  onClick={downloadSpectrogramImage}
                >
                  Завантажити зображення
                </button>
              )}
            </div>
            {spectrogramImage && (
              <div className="overtone-spectrograms__image-wrap">
                <img
                  src={`data:image/png;base64,${spectrogramImage}`}
                  alt="Спектрограми 12 інтервалів"
                  className="overtone-spectrograms__image"
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

export default ExperimentOvertones;
