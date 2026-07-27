"use client";

import { useEffect, useState } from "react";

type Profile = {
  id: number;
  name: string;
  genre: string;
  custom_prompt: string;
  topic: string | null;
  style: string;
  duration_seconds: number;
};

type Video = {
  id: number;
  job_id: number;
  file_path: string;
  quality_score: number;
  approved: boolean;
  published: boolean;
};

type Job = {
  id: number;
  status: string;
  profile_id: number;
};

type Gpu = {
  id: number;
  name: string;
  vram_gb: number;
  backend: string;
  assigned_tasks: string[];
};

export default function Home() {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [videos, setVideos] = useState<Video[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [gpus, setGpus] = useState<Gpu[]>([]);
  const [loading, setLoading] = useState(true);
  const [newProfile, setNewProfile] = useState({ name: "", genre: "random", custom_prompt: "", duration_seconds: 30 });

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const fetchData = async () => {
    try {
      const [profilesRes, videosRes, jobsRes, gpusRes] = await Promise.all([
        fetch(`${apiUrl}/profiles/`),
        fetch(`${apiUrl}/videos/`),
        fetch(`${apiUrl}/jobs/`),
        fetch(`${apiUrl}/gpus`)
      ]);
      setProfiles(await profilesRes.json());
      setVideos(await videosRes.json());
      setJobs(await jobsRes.json());
      setGpus(await gpusRes.json());
    } catch (error) {
      console.error("Errore:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000); // Aggiorna ogni 5 secondi
    return () => clearInterval(interval);
  }, []);

  const handleCreateProfile = async (e: React.FormEvent) => {
    e.preventDefault();
    await fetch(`${apiUrl}/profiles/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(newProfile)
    });
    setNewProfile({ name: "", genre: "random", custom_prompt: "", duration_seconds: 30 });
    fetchData();
  };

  const handleStartJob = async (profileId: number) => {
    await fetch(`${apiUrl}/jobs/${profileId}`, { method: "POST" });
    fetchData();
  };

  const handleApprove = async (videoId: number) => {
    await fetch(`${apiUrl}/videos/${videoId}/approve`, { method: "PUT" });
    fetchData();
  };

  const handleReject = async (videoId: number) => {
    await fetch(`${apiUrl}/videos/${videoId}/reject`, { method: "PUT" });
    fetchData();
  };

  const handlePublish = async (videoId: number, platform: string) => {
    await fetch(`${apiUrl}/videos/${videoId}/publish/${platform}`, { method: "POST" });
    alert(`Pubblicazione su ${platform} avviata (simulata).`);
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-8 bg-gray-900 text-white">
      <h1 className="text-4xl font-bold mb-8">
        AI Shorts Factory{" "}
        <a href="/jobs" className="text-blue-400 hover:text-blue-300 text-2xl underline">
          Vai a Jobs
        </a>
      </h1>
      
      <div className="w-full max-w-6xl grid grid-cols-1 md:grid-cols-3 gap-8">
        {/* Creazione Profilo */}
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl mb-4">Crea Profilo</h2>
          <form onSubmit={handleCreateProfile} className="space-y-4">
            <input
              type="text"
              placeholder="Nome"
              className="w-full p-2 bg-gray-700 rounded"
              value={newProfile.name}
              onChange={(e) => setNewProfile({ ...newProfile, name: e.target.value })}
              required
            />
            <select
              className="w-full p-2 bg-gray-700 rounded"
              value={newProfile.genre}
              onChange={(e) => setNewProfile({ ...newProfile, genre: e.target.value })}
            >
              <option value="random">Random Genre</option>
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
            <textarea
              placeholder="Prompt personalizzato (opzionale)"
              className="w-full p-2 bg-gray-700 rounded"
              value={newProfile.custom_prompt}
              onChange={(e) => setNewProfile({ ...newProfile, custom_prompt: e.target.value })}
            ></textarea>
            <input
              type="number"
              placeholder="Durata (s)"
              className="w-full p-2 bg-gray-700 rounded"
              value={newProfile.duration_seconds}
              onChange={(e) => setNewProfile({ ...newProfile, duration_seconds: parseInt(e.target.value) })}
            />
            <button type="submit" className="w-full bg-blue-600 hover:bg-blue-700 p-2 rounded font-bold">
              Crea
            </button>
          </form>
        </div>

        {/* Lista Profili e Job */}
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl mb-4">Profili & Jobs</h2>
          {loading ? <p>Caricamento...</p> : (
            <ul className="space-y-3 max-h-96 overflow-y-auto">
              {profiles.map((p) => (
                <li key={p.id} className="bg-gray-700 p-3 rounded">
                  <div className="flex justify-between items-center mb-2">
                    <strong>{p.name}</strong>
                    <button onClick={() => handleStartJob(p.id)} className="bg-green-600 hover:bg-green-700 px-3 py-1 rounded text-sm">
                      Avvia
                    </button>
                  </div>
                  <p className="text-xs text-gray-400">Topic: {p.topic}</p>
                  {jobs.filter(j => j.profile_id === p.id).map(j => (
                    <p key={j.id} className={`text-xs mt-1 ${j.status === 'completed' ? 'text-green-400' : j.status === 'failed' ? 'text-red-400' : 'text-yellow-400'}`}>
                      Job #{j.id}: {j.status}
                    </p>
                  ))}
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Stato GPU */}
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl mb-4">Monitor GPU</h2>
          {loading ? <p>Caricamento...</p> : (
            <ul className="space-y-2">
              {gpus.map((g) => (
                <li key={g.id} className="bg-gray-700 p-3 rounded">
                  <strong>{g.name}</strong> ({g.backend})<br/>
                  <span className="text-xs text-gray-400">VRAM: {g.vram_gb}GB | Task: {g.assigned_tasks.join(", ")}</span>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Revisione Video */}
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg col-span-1 md:col-span-3">
          <h2 className="text-2xl mb-4">Video Generati</h2>
          {loading ? <p>Caricamento...</p> : videos.length === 0 ? (
            <p>Nessun video generato.</p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {videos.map((v) => {
                const videoUrl = `${apiUrl}/${v.file_path}`;
                return (
                  <div key={v.id} className="bg-gray-700 p-4 rounded">
                    <p className="text-sm mb-2">Video ID: {v.id} (Job: {v.job_id})</p>
                    <video src={videoUrl} controls className="w-full rounded mb-2 bg-black" />
                    <p className="text-sm mb-2">Score: {v.quality_score.toFixed(1)}/10</p>
                    <p className="text-sm mb-2">Stato: {v.approved ? "Approvato" : "In Attesa"} {v.published && "(Pubblicato)"}</p>
                    <div className="flex gap-2 mt-2">
                      {!v.approved ? (
                        <button onClick={() => handleApprove(v.id)} className="flex-1 bg-green-600 hover:bg-green-700 px-2 py-1 rounded text-xs">
                          Approva
                        </button>
                      ) : (
                        <button onClick={() => handleReject(v.id)} className="flex-1 bg-red-600 hover:bg-red-700 px-2 py-1 rounded text-xs">
                          Rifiuta
                        </button>
                      )}
                    </div>
                    {v.approved && (
                      <div className="grid grid-cols-2 gap-2 mt-2">
                        <button onClick={() => handlePublish(v.id, "tiktok")} className="bg-purple-600 hover:bg-purple-700 px-2 py-1 rounded text-xs">
                          TikTok
                        </button>
                        <button onClick={() => handlePublish(v.id, "youtube")} className="bg-red-600 hover:bg-red-700 px-2 py-1 rounded text-xs">
                          YouTube
                        </button>
                        <button onClick={() => handlePublish(v.id, "instagram")} className="bg-pink-600 hover:bg-pink-700 px-2 py-1 rounded text-xs">
                          Instagram
                        </button>
                        <button onClick={() => handlePublish(v.id, "facebook")} className="bg-blue-600 hover:bg-blue-700 px-2 py-1 rounded text-xs">
                          Facebook
                        </button>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
