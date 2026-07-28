"use client";

import { useEffect, useState } from "react";

type Stage = {
  name: string;
  status: string;
  result: string | null;
  created_at: string;
  updated_at: string;
};

type Video = {
  id: number;
  file_path: string;
  quality_score: number;
  approved: boolean;
  published: boolean;
};

type JobDetails = {
  job_id: string;
  status: string;
  genre: string;
  custom_prompt: string | null;
  language: string;
  style: string;
  duration_seconds: number;
  stages: Stage[];
  video: Video | null;
};

export default function JobsPage() {
  const [jobs, setJobs] = useState<{ id: string; status: string; progress: { completed: number; total: number } }[]>([]);
  const [selectedJob, setSelectedJob] = useState<JobDetails | null>(null);
  const [progress, setProgress] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || `http://${typeof window !== 'undefined' ? window.location.hostname : 'localhost'}:8000`;

  const fetchJobs = async () => {
    try {
      const res = await fetch(`${apiUrl}/jobs/`);
      setJobs(await res.json());
    } catch (error) {
      console.error("Errore:", error);
    } finally {
      setLoading(false);
    }
  };

  const fetchJobDetails = async (jobId: string) => {
    try {
      const [detailsRes, progressRes] = await Promise.all([
        fetch(`${apiUrl}/jobs/${jobId}`),
        fetch(`${apiUrl}/jobs/${jobId}/progress`)
      ]);
      setSelectedJob(await detailsRes.json());
      setProgress(await progressRes.json());
    } catch (error) {
      console.error("Errore:", error);
    }
  };

  useEffect(() => {
    fetchJobs();
    const params = new URLSearchParams(window.location.search);
    const jobId = params.get("job_id");
    if (jobId) {
      fetchJobDetails(jobId);
    }
    const interval = setInterval(() => {
      fetchJobs();
      if (selectedJob && (selectedJob.status === 'running' || selectedJob.status === 'pending')) {
        fetchJobDetails(selectedJob.job_id);
      }
    }, 3000);
    return () => clearInterval(interval);
  }, [selectedJob]);

  return (
    <main className="min-h-screen bg-gray-900 text-white p-4 sm:p-6 lg:p-8">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
          <h1 className="text-3xl sm:text-4xl font-bold bg-gradient-to-r from-blue-400 to-indigo-500 text-transparent bg-clip-text">
            Stato Jobs
          </h1>
          <a href="/" className="text-blue-400 hover:text-blue-300 text-sm sm:text-base underline">
            &larr; Torna alla Dashboard
          </a>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gray-800/50 backdrop-blur border border-gray-700 p-6 rounded-xl shadow-lg">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <span className="w-2 h-2 bg-yellow-500 rounded-full"></span> Lista Jobs
            </h2>
            {loading ? <p className="text-gray-400">Caricamento...</p> : (
              <ul className="space-y-2">
                {jobs.map((j) => (
                  <li 
                    key={j.id} 
                    className={`bg-gray-700/50 p-3 rounded-lg cursor-pointer hover:bg-gray-600 transition-colors ${selectedJob?.job_id === j.id ? 'ring-2 ring-blue-500' : ''}`}
                    onClick={() => fetchJobDetails(j.id)}
                  >
                    <strong>Job {String(j.id).substring(0, 8)}</strong> - 
                    <span className={`ml-2 ${j.status === 'completed' ? 'text-green-400' : j.status === 'failed' ? 'text-red-400' : j.status === 'interrupted' ? 'text-orange-400' : 'text-yellow-400'}`}>
                      {j.status}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="bg-gray-800/50 backdrop-blur border border-gray-700 p-6 rounded-xl shadow-lg">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <span className="w-2 h-2 bg-purple-500 rounded-full"></span> Dettagli Job
            </h2>
            {!selectedJob ? (
              <p className="text-gray-400">Seleziona un job dalla lista per vedere i dettagli.</p>
            ) : (
              <div>
                <p className="mb-4 text-sm">Stato generale: <strong>{selectedJob.status}</strong></p>
                
                {progress && progress.stage && (() => {
                  let etaInfo = null;
                  if (progress.start_time && progress.updated_time && progress.total_steps > 0 && progress.current_step > 0) {
                    const elapsed = progress.updated_time - progress.start_time;
                    const stepsRemaining = progress.total_steps - progress.current_step;
                    const timePerStep = elapsed / progress.current_step;
                    const etaSeconds = Math.round(timePerStep * stepsRemaining);
                    const endTimeMs = (progress.updated_time + etaSeconds) * 1000;
                    
                    const mins = Math.floor(etaSeconds / 60);
                    const secs = etaSeconds % 60;
                    const etaStr = `${mins}m ${secs}s`;
                    const endStr = new Date(endTimeMs).toLocaleTimeString();
                    
                    etaInfo = (
                      <p className="text-xs text-gray-400 mt-1 mb-4">
                        {progress.message} ({progress.current_step}/{progress.total_steps}) - ETA: {etaStr} (Fine: {endStr})
                      </p>
                    );
                  } else if (progress.message) {
                    etaInfo = (
                      <p className="text-xs text-gray-400 mt-1 mb-4">
                        {progress.message} ({progress.current_step}/{progress.total_steps})
                      </p>
                    );
                  }
                  return etaInfo;
                })()}

                <div className="mt-4 mb-4 bg-gray-700/30 p-3 rounded-lg">
                  <h3 className="text-sm font-semibold text-gray-300 mb-2">Parametri Job</h3>
                  <ul className="text-xs text-gray-400 space-y-1">
                    <li><strong>Genere:</strong> {selectedJob.genre}</li>
                    <li><strong>Lingua:</strong> {selectedJob.language}</li>
                    <li><strong>Stile:</strong> {selectedJob.style}</li>
                    <li><strong>Durata:</strong> {selectedJob.duration_seconds}s</li>
                    {selectedJob.custom_prompt && (
                      <li className="mt-1"><strong>Prompt:</strong> {selectedJob.custom_prompt}</li>
                    )}
                  </ul>
                </div>

                {selectedJob.video && (
                  <div className="mt-4 mb-4 bg-gray-700/30 p-3 rounded-lg">
                    <h3 className="text-sm font-semibold text-gray-300 mb-2">Video Generato</h3>
                    <ul className="text-xs text-gray-400 space-y-1 mb-2">
                      <li><strong>ID Video:</strong> {selectedJob.video.id}</li>
                      <li><strong>Score:</strong> {selectedJob.video.quality_score.toFixed(1)}/10</li>
                      <li><strong>Approvato:</strong> {selectedJob.video.approved ? "Sì" : "No"}</li>
                      <li><strong>Pubblicato:</strong> {selectedJob.video.published ? "Sì" : "No"}</li>
                    </ul>
                    {selectedJob.video.file_path && (
                      <video src={`${apiUrl}/${selectedJob.video.file_path}`} controls className="w-full rounded-lg bg-black aspect-video" />
                    )}
                  </div>
                )}

                <ul className="space-y-2">
                  {selectedJob.stages.map((s, index) => (
                    <li key={index} className="bg-gray-700/50 p-3 rounded-lg">
                      <div className="flex justify-between">
                        <span className="text-sm font-medium">{s.name}</span>
                        <span className={`text-xs px-2 py-1 rounded-full ${s.status === 'completed' ? 'bg-green-500/20 text-green-400' : s.status === 'failed' ? 'bg-red-500/20 text-red-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                          {s.status}
                        </span>
                      </div>
                      {s.result && (
                        <div className="text-xs text-gray-400 mt-2 bg-gray-800/50 p-2 rounded max-h-40 overflow-y-auto whitespace-pre-wrap break-words">
                          <span className="font-semibold text-gray-300">Risultato:</span>
                          <br />
                          {s.result}
                        </div>
                      )}
                      {s.updated_at && (
                        <p className="text-xs text-gray-500 mt-1">
                          Updated: {new Date(s.updated_at).toLocaleTimeString()}
                        </p>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
