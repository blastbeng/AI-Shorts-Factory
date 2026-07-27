"use client";

import { useEffect, useState } from "react";

type Profile = {
  id: number;
  name: string;
  topic: string;
  style: string;
  duration_seconds: number;
};

type Video = {
  id: number;
  job_id: number;
  file_path: string;
  quality_score: number;
  approved: boolean;
};

export default function Home() {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [videos, setVideos] = useState<Video[]>([]);
  const [loading, setLoading] = useState(true);
  const [newProfile, setNewProfile] = useState({ name: "", topic: "", duration_seconds: 30 });

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const fetchData = async () => {
    try {
      const [profilesRes, videosRes] = await Promise.all([
        fetch(`${apiUrl}/profiles/`),
        fetch(`${apiUrl}/videos/`)
      ]);
      setProfiles(await profilesRes.json());
      setVideos(await videosRes.json());
    } catch (error) {
      console.error("Errore:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleCreateProfile = async (e: React.FormEvent) => {
    e.preventDefault();
    await fetch(`${apiUrl}/profiles/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(newProfile)
    });
    setNewProfile({ name: "", topic: "", duration_seconds: 30 });
    fetchData();
  };

  const handleStartJob = async (profileId: number) => {
    await fetch(`${apiUrl}/jobs/${profileId}`, { method: "POST" });
    alert("Job avviato in background!");
  };

  const handleApprove = async (videoId: number) => {
    await fetch(`${apiUrl}/videos/${videoId}/approve`, { method: "PUT" });
    fetchData();
  };

  const handleReject = async (videoId: number) => {
    await fetch(`${apiUrl}/videos/${videoId}/reject`, { method: "PUT" });
    fetchData();
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-8 bg-gray-900 text-white">
      <h1 className="text-4xl font-bold mb-8">AI Shorts Factory</h1>
      
      <div className="w-full max-w-4xl grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Creazione Profilo */}
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl mb-4">Crea Nuovo Profilo</h2>
          <form onSubmit={handleCreateProfile} className="space-y-4">
            <input
              type="text"
              placeholder="Nome Profilo"
              className="w-full p-2 bg-gray-700 rounded"
              value={newProfile.name}
              onChange={(e) => setNewProfile({ ...newProfile, name: e.target.value })}
              required
            />
            <input
              type="text"
              placeholder="Topic"
              className="w-full p-2 bg-gray-700 rounded"
              value={newProfile.topic}
              onChange={(e) => setNewProfile({ ...newProfile, topic: e.target.value })}
              required
            />
            <input
              type="number"
              placeholder="Durata (sec)"
              className="w-full p-2 bg-gray-700 rounded"
              value={newProfile.duration_seconds}
              onChange={(e) => setNewProfile({ ...newProfile, duration_seconds: parseInt(e.target.value) })}
            />
            <button type="submit" className="w-full bg-blue-600 hover:bg-blue-700 p-2 rounded font-bold">
              Crea
            </button>
          </form>
        </div>

        {/* Lista Profili */}
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl mb-4">Profili Esistenti</h2>
          {loading ? (
            <p>Caricamento...</p>
          ) : profiles.length === 0 ? (
            <p>Nessun profilo.</p>
          ) : (
            <ul className="space-y-2">
              {profiles.map((p) => (
                <li key={p.id} className="bg-gray-700 p-3 rounded flex justify-between items-center">
                  <div>
                    <strong>{p.name}</strong> - {p.topic}
                  </div>
                  <button
                    onClick={() => handleStartJob(p.id)}
                    className="bg-green-600 hover:bg-green-700 px-3 py-1 rounded text-sm"
                  >
                    Avvia
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Revisione Video */}
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg col-span-1 md:col-span-2">
          <h2 className="text-2xl mb-4">Video Generati</h2>
          {loading ? (
            <p>Caricamento...</p>
          ) : videos.length === 0 ? (
            <p>Nessun video generato.</p>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {videos.map((v) => (
                <div key={v.id} className="bg-gray-700 p-4 rounded">
                  <p className="text-sm mb-2">Video ID: {v.id} (Job: {v.job_id})</p>
                  <p className="text-sm mb-2">Score: {v.quality_score.toFixed(1)}/10</p>
                  <p className="text-sm mb-2">Stato: {v.approved ? "Approvato" : "In Attesa"}</p>
                  <div className="flex gap-2 mt-2">
                    {!v.approved ? (
                      <button
                        onClick={() => handleApprove(v.id)}
                        className="flex-1 bg-green-600 hover:bg-green-700 px-2 py-1 rounded text-xs"
                      >
                        Approva
                      </button>
                    ) : (
                      <button
                        onClick={() => handleReject(v.id)}
                        className="flex-1 bg-red-600 hover:bg-red-700 px-2 py-1 rounded text-xs"
                      >
                        Rifiuta
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
