import React, { useState } from 'react';

const SAMPLES = [
  { id: 'sample_1_bare_rock', name: 'Bare Rock', desc: 'Rocky ridge & hills' },
  { id: 'sample_2_fallow_fields', name: 'Fallow Fields', desc: 'Dry agricultural soil' },
  { id: 'sample_3_urban_water', name: 'Urban & Water', desc: 'Settlements & lake' },
  { id: 'sample_4_dense_vegetation', name: 'Dense Forest', desc: 'Forest canopy & trees' },
];

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState(null);
  const [activeTitle, setActiveTitle] = useState('');
  const [latency, setLatency] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handlePreset = async (sample) => {
    setLoading(true);
    setError('');
    const t0 = performance.now();

    try {
      const response = await fetch(`/predict_sample/${sample.id}`, {
        method: 'POST',
      });

      const elapsed = Math.round(performance.now() - t0);
      setLatency(elapsed);

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error (${response.status})`);
      }

      const data = await response.json();
      setResults(data);
      setActiveTitle(`Preset: ${sample.name}`);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyze = async (e) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setError('');
    const t0 = performance.now();

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        body: formData,
      });

      const elapsed = Math.round(performance.now() - t0);
      setLatency(elapsed);

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || "Failed to process image. Ensure it is a valid 4-band Sentinel-2 TIFF.");
      }

      const data = await response.json();
      setResults(data);
      setActiveTitle(`Custom File: ${file.name}`);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-transparent font-mono text-text-primary flex flex-col items-center p-6 md:p-10">
      <header className="mb-10 text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/30 text-primary text-xs font-semibold uppercase tracking-wider mb-3">
          <span className="w-2 h-2 rounded-full bg-primary animate-pulse"></span>
          4-Model Ensemble Active
        </div>
        <h1 className="text-3xl md:text-4xl font-bold tracking-tight text-primary uppercase drop-shadow-[0_0_12px_rgba(16,185,129,0.7)]">
          SRM Pipeline
        </h1>
        <p className="mt-3 text-text-secondary max-w-lg mx-auto text-sm">
          Deep Learning Super-Resolution Mapping. Transforms 30m 4-band Sentinel-2 satellite imagery into 10m land-cover maps.
        </p>
      </header>

      <main className="w-full max-w-4xl bg-surface p-6 md:p-8 rounded-2xl border border-border shadow-[0_0_25px_rgba(16,185,129,0.07)]">
        {/* Instant Presets Section */}
        <div className="mb-6 pb-6 border-b border-border">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-text-secondary">
              Instant Sample Presets (Zero-Latency)
            </span>
            <span className="text-xs text-primary font-mono">1-Click Fast Run</span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {SAMPLES.map((sample) => (
              <button
                key={sample.id}
                type="button"
                onClick={() => handlePreset(sample)}
                disabled={loading}
                className="flex flex-col items-start p-3 rounded-xl border border-border bg-background/50 hover:bg-background hover:border-primary/60 transition-all text-left group disabled:opacity-50"
              >
                <span className="text-sm font-semibold text-text-primary group-hover:text-primary transition-colors">
                  {sample.name}
                </span>
                <span className="text-xs text-text-secondary mt-0.5">{sample.desc}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Custom File Upload Form */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-text-secondary">
              Or Upload Custom Sentinel-2 GeoTIFF
            </span>
            <span className="text-xs text-text-secondary">4-band (B2, B3, B4, B8)</span>
          </div>
          <form onSubmit={handleAnalyze} className="flex flex-col md:flex-row items-center gap-4">
            <label className="flex-1 w-full flex items-center justify-center border-2 border-dashed border-border hover:border-primary rounded-xl p-4 cursor-pointer transition-colors bg-background/50">
              <span className="text-text-secondary text-sm truncate">
                {file ? file.name : "Click to select a .tif file"}
              </span>
              <input 
                type="file" 
                accept=".tif,.tiff" 
                className="hidden" 
                onChange={handleFileChange} 
              />
            </label>
            <button 
              type="submit" 
              disabled={!file || loading}
              className="w-full md:w-auto px-6 py-4 bg-transparent border-2 border-primary text-primary rounded-xl font-bold uppercase tracking-wider hover:bg-primary hover:text-primary-foreground transition-all disabled:opacity-40 disabled:cursor-not-allowed drop-shadow-[0_0_8px_rgba(16,185,129,0.5)] text-sm whitespace-nowrap"
            >
              {loading ? 'Processing...' : 'Process File'}
            </button>
          </form>
        </div>

        {error && (
          <div className="mt-6 p-4 bg-error/10 border border-error/40 text-error rounded-xl text-sm leading-relaxed">
            {error}
          </div>
        )}

        {results && (
          <div className="mt-10 space-y-6 animate-in fade-in duration-300">
            <div className="flex items-center justify-between px-1">
              <h2 className="text-base font-bold text-text-primary">{activeTitle}</h2>
              {latency && (
                <div className="text-xs font-mono px-3 py-1 rounded-md bg-background text-primary border border-border">
                  Latency: {latency}ms
                </div>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="flex flex-col items-center bg-background/40 p-4 rounded-xl border border-border">
                <div className="w-full flex items-center justify-between mb-3">
                  <h3 className="text-sm text-primary uppercase tracking-wider font-semibold">Input (30m RGB)</h3>
                  <span className="text-xs text-text-secondary">Percentile Stretched</span>
                </div>
                <img 
                  src={results.input_image} 
                  alt="Input RGB" 
                  className="rounded border border-border shadow-[0_0_12px_rgba(6,182,212,0.1)] w-full object-contain aspect-square"
                  style={{ imageRendering: 'pixelated' }}
                />
              </div>
              <div className="flex flex-col items-center bg-background/40 p-4 rounded-xl border border-border">
                <div className="w-full flex items-center justify-between mb-3">
                  <h3 className="text-sm text-accent uppercase tracking-wider font-semibold">Prediction (10m Map)</h3>
                  <span className="text-xs text-primary font-mono">3x Super-Res</span>
                </div>
                <img 
                  src={results.prediction_image} 
                  alt="Prediction" 
                  className="rounded border border-border shadow-[0_0_12px_rgba(16,185,129,0.1)] w-full object-contain aspect-square"
                  style={{ imageRendering: 'pixelated' }}
                />
              </div>
            </div>

            <div className="p-5 border border-border rounded-xl bg-background/40">
              <h3 className="text-xs text-text-secondary uppercase tracking-widest mb-3">Land-Cover Legend</h3>
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-xs font-medium">
                <div className="flex items-center gap-2 p-2 rounded-lg bg-surface/50 border border-border"><div className="w-3.5 h-3.5 bg-[#0000FF] rounded-sm shrink-0"></div>Water</div>
                <div className="flex items-center gap-2 p-2 rounded-lg bg-surface/50 border border-border"><div className="w-3.5 h-3.5 bg-[#FF0000] rounded-sm shrink-0"></div>Built-up</div>
                <div className="flex items-center gap-2 p-2 rounded-lg bg-surface/50 border border-border"><div className="w-3.5 h-3.5 bg-[#00AA00] rounded-sm shrink-0"></div>Vegetation</div>
                <div className="flex items-center gap-2 p-2 rounded-lg bg-surface/50 border border-border"><div className="w-3.5 h-3.5 bg-[#FFA500] rounded-sm shrink-0"></div>Cropland</div>
                <div className="flex items-center gap-2 p-2 rounded-lg bg-surface/50 border border-border"><div className="w-3.5 h-3.5 bg-[#AAAAAA] rounded-sm shrink-0"></div>Bare Land</div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
