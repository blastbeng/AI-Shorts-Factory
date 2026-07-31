"use client";

import { useEffect, useState, useRef } from "react";
import Link from "next/link";

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
  gpu_utilization: number;
  backends: string[];
  assigned_tasks: string[];
};

export default function Home() {
  const [videos, setVideos] = useState<Video[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [gpus, setGpus] = useState<Gpu[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [genParams, setGenParams] = useState({ genre: "random", custom_prompt: "", language: "italian", duration_seconds: 16 });
  const [genFrames, setGenFrames] = useState(49);
  const [genFluxSteps, setGenFluxSteps] = useState(4);
  const [genWanSteps, setGenWanSteps] = useState(30);
  const [genLtxSteps, setGenLtxSteps] = useState(50);
  const [videoProvider, setVideoProvider] = useState('wan');
  const [generateSubtitles, setGenerateSubtitles] = useState(false);
  const [inputImage, setInputImage] = useState<string | null>(null);
  const [width, setWidth] = useState(352);
  const [height, setHeight] = useState(640);
  const [resolutionIndex, setResolutionIndex] = useState(0);
  const resolutions = [
    { w: 256, h: 448 },
    { w: 352, h: 640 },
    { w: 480, h: 832 },
    { w: 704, h: 1280 },
    { w: 1088, h: 1920 }
  ];

  const handleResolutionChange = (index: number) => {
    setResolutionIndex(index);
    setWidth(resolutions[index].w);
    setHeight(resolutions[index].h);
  };

  const [schedulerRunning, setSchedulerRunning] = useState(false);
  const [stats, setStats] = useState({total_videos: 0, approved_videos: 0, published_videos: 0, total_jobs: 0});
  const [health, setHealth] = useState<{status: string} | null>(null);
  const logContainerRef = useRef<HTMLDivElement>(null);
  const [progress, setProgress] = useState<{ [key: string]: any }>({});

  const getApiUrl = () => {
    const currentHost = typeof window !== 'undefined' ? window.location.hostname : 'localhost';
    const envUrl = process.env.NEXT_PUBLIC_API_URL;
    
    // If accessing via an IP/hostname that is not localhost, and the env URL is missing or points to localhost/127.0.0.1,
    // dynamically construct the API URL using the current host to avoid NetworkErrors.
    if (currentHost !== 'localhost' && (!envUrl || envUrl.includes('localhost') || envUrl.includes('127.0.0.1'))) {
      return `http://${currentHost}:8000`;
    }
    
    return envUrl || `http://${currentHost}:8000`;
  };
  const apiUrl = getApiUrl();

  const fetchData = async () => {
    try {
      const [videosRes, jobsRes, gpusRes, logsRes, statsRes, healthRes] = await Promise.all([
        fetch(`${apiUrl}/videos/`),
        fetch(`${apiUrl}/jobs/`),
        fetch(`${apiUrl}/gpus`),
        fetch(`${apiUrl}/logs`),
        fetch(`${apiUrl}/stats`),
        fetch(`${apiUrl}/health`)
      ]);
      setVideos(await videosRes.json());
     
      const jobsData = await jobsRes.json();
      setJobs(jobsData);
      
      const runningJobs = jobsData.filter((j: Job) => j.status === 'running');
      if (runningJobs.length > 0) {
        const progressPromises = runningJobs.map((j: Job) =>
          fetch(`${apiUrl}/jobs/${j.id}/progress`).then(res => res.json())
        );
        const progressResults = await Promise.all(progressPromises);
        const newProgress: { [key: string]: any } = {};
        runningJobs.forEach((j: Job, i: number) => {
          newProgress[j.id] = progressResults[i];
        });
        setProgress(newProgress);
      } else {
        setProgress({});
      }
     
      setGpus(await gpusRes.json());
      const logsData = await logsRes.json();
      setLogs(logsData.logs || []);
      setStats(await statsRes.json());
      setHealth(await healthRes.json());
    } catch (error) {
      console.error("Errore:", error);
      setHealth(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetch(`${apiUrl}/settings`)
      .then(res => res.json())
      .then(data => {
        setWidth(data.default_width);
        setHeight(data.default_height);
        setGenerateSubtitles(data.default_generate_subtitles);
      })
      .catch(err => console.error("Failed to fetch settings", err));
  }, []);

  useEffect(() => {
    fetchData();
    fetchSchedulerStatus();
    const interval = setInterval(() => {
      fetchData();
      fetchSchedulerStatus();
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await fetch(`${apiUrl}/jobs/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...genParams,
          gen_width: width,
          gen_height: height,
          width: width,
          height: height,
          gen_frames: genFrames,
          gen_flux_steps: genFluxSteps,
          gen_wan_steps: genWanSteps,
          gen_ltx_steps: genLtxSteps,
          video_provider: videoProvider,
          generate_subtitles: generateSubtitles,
          input_image: inputImage
        })
      });
      setGenParams({ genre: "random", custom_prompt: "", language: "italian", duration_seconds: 16 });
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

  const handleInterruptAll = async () => {
    await fetch(`${apiUrl}/jobs/interrupt_all`, { method: "POST" });
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
          {/* Quick Stats */}
          <div className="bg-gray-800/50 backdrop-blur border border-gray-700 p-6 rounded-xl shadow-lg md:col-span-2 lg:col-span-3">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <span className="w-2 h-2 bg-cyan-500 rounded-full"></span> Statistiche Rapide
            </h2>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="bg-gray-700/50 p-4 rounded-lg text-center">
                <div className="text-3xl font-bold text-blue-400">{stats.total_videos}</div>
                <div className="text-xs text-gray-400 mt-1">Video Totali</div>
              </div>
              <div className="bg-gray-700/50 p-4 rounded-lg text-center">
                <div className="text-3xl font-bold text-green-400">{stats.approved_videos}</div>
                <div className="text-xs text-gray-400 mt-1">Approvati</div>
              </div>
              <div className="bg-gray-700/50 p-4 rounded-lg text-center">
                <div className="text-3xl font-bold text-purple-400">{stats.published_videos}</div>
                <div className="text-xs text-gray-400 mt-1">Pubblicati</div>
              </div>
              <div className="bg-gray-700/50 p-4 rounded-lg text-center">
                <div className="text-3xl font-bold text-yellow-400">{stats.total_jobs}</div>
                <div className="text-xs text-gray-400 mt-1">Jobs Totali</div>
              </div>
            </div>
          </div>

          {/* Generazione Video */}
          <div className="bg-gray-800/50 backdrop-blur border border-gray-700 p-6 rounded-xl shadow-lg h-[500px] flex flex-col">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <span className="w-2 h-2 bg-blue-500 rounded-full"></span> Genera Video
            </h2>
            <form onSubmit={handleGenerate} className="flex-1 flex flex-col min-h-0">
              <button type="submit" className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 p-3 rounded-lg font-bold transition-all shadow-md mb-4">
                Avvia Generazione
              </button>
              <div className="space-y-4 flex-1 overflow-y-scroll pr-2 min-h-0 custom-scrollbar">
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
                      onChange={(e) => setGenParams({ ...genParams, duration_seconds: parseInt(e.target.value) || 16 })}
                    />
                  </div>
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">Video Provider</label>
                  <select
                    className="w-full p-2.5 bg-gray-700/50 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
                    value={videoProvider}
                    onChange={(e) => setVideoProvider(e.target.value)}
                  >
                    <option value="wan">Wan</option>
                    <option value="ltx">LTX</option>
                  </select>
                </div>
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={generateSubtitles}
                    onChange={(e) => setGenerateSubtitles(e.target.checked)}
                    className="form-checkbox h-4 w-4 text-blue-600"
                  />
                  <span className="text-sm text-gray-300">Generate & Embed Subtitles</span>
                </label>
                <div className="flex flex-col">
                  <label className="text-sm text-gray-300 mb-1">Input Image (Optional)</label>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) {
                        const reader = new FileReader();
                        reader.onloadend = () => {
                          setInputImage(reader.result as string);
                        };
                        reader.readAsDataURL(file);
                      } else {
                        setInputImage(null);
                      }
                    }}
                    className="text-sm text-gray-300"
                  />
                </div>
              </div>
            </form>
          </div>

          {/* Parametri di Generazione */}
          <div className="bg-gray-800/50 backdrop-blur border border-gray-700 p-6 rounded-xl shadow-lg h-[500px] flex flex-col">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <span className="w-2 h-2 bg-orange-500 rounded-full"></span> Parametri di Generazione
            </h2>
            <div className="space-y-6 flex-1 overflow-y-auto pr-2">
              {/* Risoluzione */}
              <div>
                <label className="block text-sm text-gray-400 mb-2">Risoluzione ({width}x{height})</label>
                <div className="flex items-center gap-4">
                  <span className="text-xs text-gray-500">480p</span>
                  <input
                    type="range"
                    min="0"
                    max="4"
                    step="1"
                    value={resolutionIndex}
                    onChange={(e) => handleResolutionChange(parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                  />
                  <span className="text-xs text-gray-500">1080p</span>
                </div>
              </div>

              {/* Frames */}
              <div>
                <label className="flex justify-between text-sm text-gray-400 mb-2">
                  <span>Frames per clip</span>
                  <span className="text-white font-medium">{genFrames}</span>
                </label>
                <input
                  type="range"
                  min="9"
                  max="129"
                  step="8"
                  value={genFrames}
                  onChange={(e) => setGenFrames(parseInt(e.target.value))}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
              </div>

              {/* Steps */}
              <div className="space-y-4">
                <div>
                  <label className="flex justify-between text-sm text-gray-400 mb-2">
                    <span>Flux Steps</span>
                    <span className="text-white font-medium">{genFluxSteps}</span>
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="50"
                    step="1"
                    value={genFluxSteps}
                    onChange={(e) => setGenFluxSteps(parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                  />
                </div>
                <div>
                  <label className="flex justify-between text-sm text-gray-400 mb-2">
                    <span>Wan Steps</span>
                    <span className="text-white font-medium">{genWanSteps}</span>
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="100"
                    step="1"
                    value={genWanSteps}
                    onChange={(e) => setGenWanSteps(parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                  />
                </div>
                <div>
                  <label className="flex justify-between text-sm text-gray-400 mb-2">
                    <span>LTX Steps</span>
                    <span className="text-white font-medium">{genLtxSteps}</span>
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="100"
                    step="1"
                    value={genLtxSteps}
                    onChange={(e) => setGenLtxSteps(parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Stato Generazioni */}
          <div className="bg-gray-800/50 backdrop-blur border border-gray-700 p-6 rounded-xl shadow-lg h-[500px] flex flex-col">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <span className="w-2 h-2 bg-yellow-500 rounded-full"></span> Stato Generazioni
              <button onClick={handleInterruptAll} className="ml-auto text-xs bg-red-600/80 hover:bg-red-600 px-2 py-1 rounded-full">
                Ferma Tutti
              </button>
            </h2>
            {loading ? <p className="text-gray-400">Caricamento...</p> : (
              <ul className="space-y-3 flex-1 overflow-y-auto pr-2">
                {jobs.length === 0 ? <li className="text-gray-500 text-sm">Nessun job attivo.</li> : jobs.map((j) => {
                  const progressPercent = j.progress.total > 0 ? (j.progress.completed / j.progress.total) * 100 : 0;
                  return (
                    <li key={j.id} className="bg-gray-700/50 p-3 rounded-lg">
                      <div className="flex justify-between items-center mb-2">
                        <Link href={`/jobs?job_id=${j.id}`} className="text-sm font-medium hover:text-blue-400 hover:underline">
                          Job {String(j.id).substring(0, 8)}
                        </Link>
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
                      {progress[j.id] && progress[j.id].stage && (() => {
                        const p = progress[j.id];
                        let etaInfo = null;
                        if (p.start_time && p.updated_time && p.total_steps > 0 && p.current_step > 0) {
                          const elapsed = p.updated_time - p.start_time;
                          const stepsRemaining = p.total_steps - p.current_step;
                          const timePerStep = elapsed / p.current_step;
                          const etaSeconds = Math.round(timePerStep * stepsRemaining);
                          const endTimeMs = (p.updated_time + etaSeconds) * 1000;
                          
                          const mins = Math.floor(etaSeconds / 60);
                          const secs = etaSeconds % 60;
                          const etaStr = `${mins}m ${secs}s`;
                          const endStr = new Date(endTimeMs).toLocaleTimeString();
                          
                          etaInfo = (
                            <p className="text-xs text-gray-400 mt-1">
                              ETA: {etaStr} (Fine: {endStr})
                            </p>
                          );
                        }
                        return (
                          <div className="mb-2">
                            <p className="text-xs text-blue-400 truncate">
                              {p.message} ({p.current_step}/{p.total_steps})
                            </p>
                            {etaInfo}
                          </div>
                        );
                      })()}
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
          <div className="bg-gray-800/50 backdrop-blur border border-gray-700 p-6 rounded-xl shadow-lg h-[500px] flex flex-col">
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
                      <div className="mb-2">
                        <div className="flex justify-between text-xs text-gray-400 mb-1">
                          <span>Utilizzo GPU</span>
                          <span>{g.gpu_utilization}%</span>
                        </div>
                        <div className="w-full bg-gray-600 rounded-full h-2">
                          <div className="bg-blue-500 h-2 rounded-full transition-all duration-500" style={{ width: `${g.gpu_utilization}%` }}></div>
                        </div>
                      </div>
                      <div>
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

          {/* System Health */}
          <div className="bg-gray-800/50 backdrop-blur border border-gray-700 p-6 rounded-xl shadow-lg">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <span className="w-2 h-2 bg-teal-500 rounded-full"></span> Stato Sistema
            </h2>
            {loading ? <p className="text-gray-400">Caricamento...</p> : (
              <div className="space-y-3">
                <div className="flex justify-between items-center text-sm">
                  <span className="text-gray-300">Backend API</span>
                  <span className={`text-xs px-2 py-1 rounded-full ${health?.status === 'ok' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                    {health?.status === 'ok' ? 'Online' : 'Offline'}
                  </span>
                </div>
                <div className="flex justify-between items-center text-sm">
                  <span className="text-gray-300">API URL</span>
                  <span className="text-xs text-gray-400 truncate ml-2">{apiUrl}</span>
                </div>
                <div className="flex justify-between items-center text-sm">
                  <span className="text-gray-300">Auto Scheduler</span>
                  <span className={`text-xs px-2 py-1 rounded-full ${schedulerRunning ? 'bg-green-500/20 text-green-400' : 'bg-gray-500/20 text-gray-400'}`}>
                    {schedulerRunning ? 'Attivo' : 'Fermo'}
                  </span>
                </div>
              </div>
            )}
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

          {/* Log Console */}
          <div className="bg-gray-800/50 backdrop-blur border border-gray-700 p-6 rounded-xl shadow-lg md:col-span-2 lg:col-span-3">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <span className="w-2 h-2 bg-purple-500 rounded-full"></span> Log Console
            </h2>
            <div ref={logContainerRef} className="bg-black/50 p-4 rounded-lg h-96 overflow-y-auto font-mono text-xs text-gray-300 border border-gray-700">
              {logs.length === 0 ? <p>Nessun log disponibile.</p> : logs.map((log, i) => (
                <div key={i} className="py-0.5 border-b border-gray-800/50">{log}</div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
