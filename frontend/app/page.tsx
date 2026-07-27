"use client";

import { useEffect, useState } from "react";

type Video = {
  id: number;
  job_id: number;
  file_path: string;
  quality_score: number;
  approved: boolean;
  published: boolean;
};

type Job = {
  id: string;
  status: string;
  profile_id: number;
  progress: {
    completed: number;
    total: number;
  };
};

type Gpu = {
  id: number;
  name: string;
  vram_total_gb: number;
  vram_used_gb: number;
  vram_free_gb: number;
  backends: string[];
  assigned_tasks: string[];
};

export default function Home() {
  const [videos, setVideos] = useState<Video[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [gpus, setGpus] = useState<Gpu[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [genParams, setGenParams] = useState({ genre: "random", custom_prompt: "", language: "italian", duration_seconds: 30 });
  const [schedulerRunning, setSchedulerRunning] = useState(false);

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const fetchData = async () => {
    try {
      const [videosRes, jobsRes, gpusRes, logsRes] = await Promise.all([
        fetch(`${apiUrl}/videos/`),
        fetch(`${apiUrl}/jobs/`),
        fetch(`${apiUrl}/gpus`),
        fetch(`${apiUrl}/logs`)
      ]);
      setVideos(await videosRes.json());
      setJobs(await jobsRes.json());
      setGpus(await gpusRes.json());
      const logsData = await logsRes.json();
      setLogs(logsData.logs || []);
    } catch (error) {
      console.error("Errore:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    fetchSchedulerStatus();
    const interval = setInterval(() => {
      fetchData();
      fetchSchedulerStatus();
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const res = await fetch(`${apiUrl}/profiles/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: `Gen ${new Date().toLocaleString()}`, ...genParams })
      });
      const profile = await res.json();
      await fetch(`${apiUrl}/jobs/${profile.id}`, { method: "POST" });
      setGenParams({ genre: "random", custom_prompt: "", language: "italian", duration_seconds: 30 });
      fetchData();
    } catch (error) {
      console.error("Errore generazione:", error);
    }
  };

  const handleApprove = async (videoId: number) => {
    await fetch(`${apiUrl}/videos/${videoId}/approve`, { method: "PUT" });
    fetchData();
  };

  const handleReject = async (videoId: number) => {
    await fetch(`${apiUrl}/videos/${videoId}/reject`, { method: "PUT" });
    fetchData();
  };

  const handleInterrupt = async (jobId: string) => {
    await fetch(`${apiUrl}/jobs/${jobId}/interrupt`, { method: "POST" });
    fetchData();
  };

  const handlePublish = async (videoId: number, platform: string) => {
    await fetch(`${apiUrl}/videos/${videoId}/publish/${platform}`, { method: "POST" });
    alert(`Pubblicazione su ${platform} avviata (simulata).`);
  };

  const handleDelete = async (videoId: number) => {
    if (confirm("Sei sicuro di voler eliminare questo video?")) {
      await fetch(`${apiUrl}/videos/${videoId}`, { method: "DELETE" });
      fetchData();
    }
  };

  const fetchSchedulerStatus = async () => {
    try {
      const res = await fetch(`${apiUrl}/scheduler/status`);
      const data = await res.json();
      setSchedulerRunning(data.scheduler_running);
    } catch (error) {
      console.error("Errore stato scheduler:", error);
    }
  };

  const handleStartScheduler = async () => {
    await fetch(`${apiUrl}/scheduler/start?interval=60`, { method: "POST" });
    fetchSchedulerStatus();
  };

  const handleStopScheduler = async () => {
    await fetch(`${apiUrl}/scheduler/stop`, { method: "POST" });
    fetchSchedulerStatus();
  };

  return (
    <main className="min-h-screen bg-gray-900 text-white p-4 sm:p-6 lg:p-8">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
          <h1 className="text-3xl sm:text-4xl font-bold bg-gradient-to-r from-blue-400 to-indigo-500 text-transparent bg-clip-text">
            AI Shorts Factory
          </h1>
          <a href="/jobs" className="text-blue-400 hover:text-blue-300 text-sm sm:text-base underline">
            Dettagli Jobs &rarr;
          </a>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Generazione Video */}
          <div className="bg-gray-800/50 backdrop-blur border border-gray-700 p-6 rounded-xl shadow-lg">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <span className="w-2 h-2 bg-blue-500 rounded-full"></span> Genera Video
            </h2>
            <form onSubmit={handleGenerate} className="space-y-4">
              <div>
                <label className="block text-sm text-gray-400 mb-1">Genere</label>
                <select
                  className="w-full p-2.5 bg-gray-700/50 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                  value={genParams.genre}
                  onChange={(e) => setGenParams({ ...genParams, genre: e.target.value })}
                >
                  <option value="random">Genere Casuale</option>
                  <option value="Horror">Horror</option>
                  <option value="Sci-Fi">Sci-Fi</option>
                  <option value="Fantasy">Fantasy</option>
                  <option value="True Crime">True Crime</option>
                  <option value="Commedia">Commedia</option>
                  <option value="Dramma">Dramma</option>
                  <option value="Thriller">Thriller</option>
                  <option value="Documentario">Documentario</option>
                  <option value="Mistero">Mistero</option>
                  <option value="Azione">Azione</option>
                  <option value="Romantico">Romantico</option>
                  <option value="Storico">Storico</option>
                  <option value="Post-Apocalittico">Post-Apocalittico</option>
                  <option value="Cyberpunk">Cyberpunk</option>
                  <option value="Surrealista">Surrealista</option>
                </select>
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">Prompt (Opzionale)</label>
                <textarea
                  placeholder="Es. Gatti spaziali che esplorano Marte..."
                  className="w-full p-2.5 bg-gray-700/50 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none resize-none"
                  rows={3}
                  value={genParams.custom_prompt}
                  onChange={(e) => setGenParams({ ...genParams, custom_prompt: e.target.value })}
                ></textarea>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Lingua</label>
                  <select
                    className="w-full p-2.5 bg-gray-700/50 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                    value={genParams.language}
                    onChange={(e) => setGenParams({ ...genParams, language: e.target.value })}
                  >
                    <option value="italian">Italiano</option>
                    <option value="english">Inglese</option>
                    <option value="spanish">Spagnolo</option>
                    <option value="french">Francese</option>
                    <option value="german">Tedesco</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Durata (s)</label>
                  <input
                    type="number"
                    className="w-full p-2.5 bg-gray-700/50 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                    value={genParams.duration_seconds}
                    onChange={(e) => setGenParams({ ...genParams, duration_seconds: parseInt(e.target.value) || 30 })}
                  />
                </div>
              </div>
              <button type="submit" className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 p-3 rounded-lg font-bold transition-all shadow-md">
                Avvia Generazione
              </button>
            </form>
          </div>

          {/* Stato Generazioni */}
          <div className="bg-gray-800/50 backdrop-blur border border-gray-700 p-6 rounded-xl shadow-lg">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <span className="w-2 h-2 bg-yellow-500 rounded-full"></span> Stato Generazioni
            </h2>
            {loading ? <p className="text-gray-400">Caricamento...</p> : (
              <ul className="space-y-3 max-h-64 overflow-y-auto pr-2">
                {jobs.length === 0 ? <li className="text-gray-500 text-sm">Nessun job attivo.</li> : jobs.map((j) => {
                  const progressPercent = j.progress.total > 0 ? (j.progress.completed / j.progress.total) * 100 : 0;
                  return (
                    <li key={j.id} className="bg-gray-700/50 p-3 rounded-lg">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm font-medium">Job {String(j.id).substring(0, 8)}</span>
                        <div className="flex items-center gap-2">
                          <span className={`text-xs px-2 py-1 rounded-full ${j.status === 'completed' ? 'bg-green-500/20 text-green-400' : j.status === 'failed' ? 'bg-red-500/20 text-red-400' : j.status === 'interrupted' ? 'bg-orange-500/20 text-orange-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                            {j.status}
                          </span>
                          {(j.status === 'running' || j.status === 'pending') && (
                            <button onClick={() => handleInterrupt(j.id)} className="text-xs bg-red-600/80 hover:bg-red-600 px-2 py-1 rounded-full">
                              Interrompi
                            </button>
                          )}
                        </div>
                      </div>
                      {j.progress.total > 0 ? (
                        <div className="w-full bg-gray-600 rounded-full h-1.5">
                          <div className="bg-blue-500 h-1.5 rounded-full transition-all duration-500" style={{ width: `${progressPercent}%` }}></div>
                        </div>
                      ) : (j.status === 'running' || j.status === 'pending') && (
                        <div className="flex justify-center mt-1">
                          <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                        </div>
                      )}
                    </li>
                  );
                })}
              </ul>
            )}
          </div>

          {/* Stato GPU */}
          <div className="bg-gray-800/50 backdrop-blur border border-gray-700 p-6 rounded-xl shadow-lg">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <span className="w-2 h-2 bg-green-500 rounded-full"></span> Monitor GPU
            </h2>
            {loading ? <p className="text-gray-400">Caricamento...</p> : (
              <ul className="space-y-4">
                {gpus.map((g) => {
                  const vramPercent = g.vram_total_gb > 0 ? (g.vram_used_gb / g.vram_total_gb) * 100 : 0;
                  return (
                    <li key={g.id} className="bg-gray-700/50 p-4 rounded-lg">
                      <div className="flex justify-between items-center mb-2">
                        <strong className="text-sm">{g.name}</strong>
                        <div className="flex gap-1">
                          {g.backends.map(b => (
                            <span key={b} className="text-xs text-gray-300 bg-gray-600 px-2 py-0.5 rounded">{b}</span>
                          ))}
                        </div>
                      </div>
                      <div className="mb-1">
                        <div className="flex justify-between text-xs text-gray-400 mb-1">
                          <span>VRAM</span>
                          <span>{g.vram_used_gb.toFixed(1)} / {g.vram_total_gb.toFixed(1)} GB</span>
                        </div>
                        <div className="w-full bg-gray-600 rounded-full h-2">
                          <div className="bg-gradient-to-r from-green-500 to-yellow-500 h-2 rounded-full transition-all duration-500" style={{ width: `${vramPercent}%` }}></div>
                        </div>
                      </div>
                      <div className="text-xs text-gray-500 mt-2 truncate">Task: {g.assigned_tasks.join(", ")}</div>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>

          {/* Controllo Scheduler */}
          <div className="bg-gray-800/50 backdrop-blur border border-gray-700 p-6 rounded-xl shadow-lg">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <span className="w-2 h-2 bg-indigo-500 rounded-full"></span> Auto Scheduler
            </h2>
            <div className="flex items-center justify-between">
              <span className={`text-sm ${schedulerRunning ? 'text-green-400' : 'text-gray-400'}`}>
                {schedulerRunning ? "In Esecuzione" : "Fermo"}
              </span>
              {schedulerRunning ? (
                <button onClick={handleStopScheduler} className="bg-red-600/80 hover:bg-red-600 px-4 py-2 rounded-lg text-sm font-medium transition-colors">
                  Ferma
                </button>
              ) : (
                <button onClick={handleStartScheduler} className="bg-green-600/80 hover:bg-green-600 px-4 py-2 rounded-lg text-sm font-medium transition-colors">
                  Avvia
                </button>
              )}
            </div>
            <p className="text-xs text-gray-500 mt-2">Genera automaticamente nuovi video ogni 60 minuti per i profili esistenti.</p>
          </div>

          {/* Log Console */}
          <div className="bg-gray-800/50 backdrop-blur border border-gray-700 p-6 rounded-xl shadow-lg md:col-span-2 lg:col-span-3">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <span className="w-2 h-2 bg-purple-500 rounded-full"></span> Log Console
            </h2>
            <div className="bg-black/50 p-4 rounded-lg h-48 overflow-y-auto font-mono text-xs text-gray-300 border border-gray-700">
              {logs.length === 0 ? <p>Nessun log disponibile.</p> : logs.map((log, i) => (
                <div key={i} className="py-0.5 border-b border-gray-800/50">{log}</div>
              ))}
            </div>
          </div>

          {/* Revisione Video */}
          <div className="bg-gray-800/50 backdrop-blur border border-gray-700 p-6 rounded-xl shadow-lg md:col-span-2 lg:col-span-3">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <span className="w-2 h-2 bg-pink-500 rounded-full"></span> Video Generati
            </h2>
            {loading ? <p className="text-gray-400">Caricamento...</p> : videos.length === 0 ? (
              <p className="text-gray-500 text-sm">Nessun video generato.</p>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                {videos.map((v) => {
                  const videoUrl = `${apiUrl}/${v.file_path}`;
                  return (
                    <div key={v.id} className="bg-gray-700/50 border border-gray-600 p-4 rounded-xl">
                      <p className="text-xs text-gray-400 mb-2">ID: {v.id} | Job: {v.job_id}</p>
                      <video src={videoUrl} controls className="w-full rounded-lg mb-3 bg-black aspect-video" />
                      <div className="flex justify-between items-center mb-3">
                        <span className="text-sm font-medium">Score: {v.quality_score.toFixed(1)}/10</span>
                        <span className={`text-xs px-2 py-1 rounded-full ${v.approved ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                          {v.approved ? "Approvato" : "In Attesa"} {v.published && "• Pubblicato"}
                        </span>
                      </div>
                      <div className="flex gap-2 mb-2">
                        {!v.approved ? (
                          <button onClick={() => handleApprove(v.id)} className="flex-1 bg-green-600/80 hover:bg-green-600 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors">
                            Approva
                          </button>
                        ) : (
                          <button onClick={() => handleReject(v.id)} className="flex-1 bg-red-600/80 hover:bg-red-600 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors">
                            Rifiuta
                          </button>
                        )}
                        <button onClick={() => handleDelete(v.id)} className="bg-gray-600/80 hover:bg-gray-600 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors">
                          Elimina
                        </button>
                      </div>
                      {v.approved && (
                        <div className="grid grid-cols-2 gap-2">
                          <button onClick={() => handlePublish(v.id, "tiktok")} className="bg-gray-600/80 hover:bg-gray-600 px-3 py-1.5 rounded-lg text-xs transition-colors">TikTok</button>
                          <button onClick={() => handlePublish(v.id, "youtube")} className="bg-gray-600/80 hover:bg-gray-600 px-3 py-1.5 rounded-lg text-xs transition-colors">YouTube</button>
                          <button onClick={() => handlePublish(v.id, "instagram")} className="bg-gray-600/80 hover:bg-gray-600 px-3 py-1.5 rounded-lg text-xs transition-colors">Instagram</button>
                          <button onClick={() => handlePublish(v.id, "facebook")} className="bg-gray-600/80 hover:bg-gray-600 px-3 py-1.5 rounded-lg text-xs transition-colors">Facebook</button>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
