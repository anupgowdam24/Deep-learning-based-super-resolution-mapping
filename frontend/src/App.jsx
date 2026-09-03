import React, { useState } from 'react';

function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleAnalyze = async (e) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    setError('');
    setResults(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to process image. Ensure it is a valid 4-band Sentinel-2 TIFF.");
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-darker font-mono text-white flex flex-col items-center p-8">
      <header className="mb-12 text-center">
        <h1 className="text-4xl font-bold tracking-tight text-neon uppercase drop-shadow-[0_0_10px_rgba(57,255,20,0.8)]">
          SRM Pipeline
        </h1>
        <p className="mt-4 text-gray-400 max-w-lg mx-auto">
          Deep Learning Super-Resolution Mapping. Upload a 30m 4-band TIFF to generate a 10m land-cover map.
        </p>
      </header>

      <main className="w-full max-w-4xl bg-dark p-8 rounded-xl border border-neon/30 shadow-[0_0_15px_rgba(57,255,20,0.1)]">
        <form onSubmit={handleAnalyze} className="flex flex-col md:flex-row items-center gap-6">
          <label className="flex-1 w-full flex items-center justify-center border-2 border-dashed border-neon/50 hover:border-neon rounded-lg p-6 cursor-pointer transition-colors">
            <span className="text-neon/80 text-sm">{file ? file.name : "Click to select a .tif file"}</span>
            <input 
              type="file" 
              accept=".tif,.tiff" 
              className="hidden" 
              onChange={handleFileChange} 
              required
            />
          </label>
          <button 
            type="submit" 
            disabled={!file || loading}
            className="w-full md:w-auto px-8 py-3 bg-transparent border-2 border-neon text-neon rounded font-bold uppercase tracking-widest hover:bg-neon hover:text-black transition-all disabled:opacity-50 disabled:cursor-not-allowed drop-shadow-[0_0_5px_rgba(57,255,20,0.8)]"
          >
            {loading ? 'Processing...' : 'Analyze'}
          </button>
        </form>

        {error && (
          <div className="mt-6 p-4 bg-red-900/20 border border-red-500/50 text-red-400 rounded">
            {error}
          </div>
        )}

        {results && (
          <div className="mt-12 space-y-8 animate-in fade-in duration-500">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="flex flex-col items-center">
                <h3 className="text-lg text-neon mb-4 uppercase tracking-wider">Input (30m RGB)</h3>
                <img 
                  src={results.input_image} 
                  alt="Input RGB" 
                  className="rounded border border-neon/20 shadow-[0_0_10px_rgba(57,255,20,0.1)] w-full object-contain"
                  style={{ imageRendering: 'pixelated' }}
                />
              </div>
              <div className="flex flex-col items-center">
                <h3 className="text-lg text-neon mb-4 uppercase tracking-wider">Prediction (10m Map)</h3>
                <img 
                  src={results.prediction_image} 
                  alt="Prediction" 
                  className="rounded border border-neon/20 shadow-[0_0_10px_rgba(57,255,20,0.1)] w-full object-contain"
                  style={{ imageRendering: 'pixelated' }}
                />
              </div>
            </div>

            <div className="p-6 border border-neon/20 rounded-lg">
              <h3 className="text-sm text-gray-400 uppercase tracking-widest mb-4">Legend</h3>
              <div className="flex flex-wrap gap-6 text-sm">
                <div className="flex items-center gap-2"><div className="w-4 h-4 bg-[#0000FF] rounded-sm"></div>Water</div>
                <div className="flex items-center gap-2"><div className="w-4 h-4 bg-[#FF0000] rounded-sm"></div>Built-up</div>
                <div className="flex items-center gap-2"><div className="w-4 h-4 bg-[#008000] rounded-sm"></div>Vegetation</div>
                <div className="flex items-center gap-2"><div className="w-4 h-4 bg-[#FFFF00] rounded-sm"></div>Cropland</div>
                <div className="flex items-center gap-2"><div className="w-4 h-4 bg-[#A52A2A] rounded-sm"></div>Bare Land</div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
