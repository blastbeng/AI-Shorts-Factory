"use client";

import { useEffect, useState } from "react";

type Profile = {
  id: number;
  name: string;
  topic: string;
  style: string;
  duration_seconds: number;
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
  const [gpus, setGpus] = useState<Gpu[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Assumendo che il backend giri su localhost:8000
    const fetchData = async () => {
      try {
        const profilesRes = await fetch("http://localhost:8000/profiles/");
        const profilesData = await profilesRes.json();
        setProfiles(profilesData);

        const gpusRes = await fetch("http://localhost:8000/gpus");
        const gpusData = await gpusRes.json();
        setGpus(gpusData);
      } catch (error) {
        console.error("Errore nel recupero dati:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  return (
    <main className="flex min-h-screen flex-col items-center p-24">
      <div className="z-10 max-w-5xl w-full font-mono text-sm lg:flex mb-8">
        <h1 className="text-4xl font-bold">AI Shorts Factory</h1>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 w-full max-w-5xl">
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl mb-4">Profili di Generazione</h2>
          {loading ? (
            <p className="text-gray-400">Caricamento...</p>
          ) : profiles.length === 0 ? (
            <p className="text-gray-400">Nessun profilo trovato.</p>
          ) : (
            <ul className="space-y-2">
              {profiles.map((p) => (
                <li key={p.id} className="bg-gray-700 p-3 rounded">
                  <strong>{p.name}</strong> - {p.topic} ({p.duration_seconds}s)
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl mb-4">Stato GPU</h2>
          {loading ? (
            <p className="text-gray-400">Caricamento...</p>
          ) : (
            <ul className="space-y-2">
              {gpus.map((g) => (
                <li key={g.id} className="bg-gray-700 p-3 rounded">
                  <strong>{g.name}</strong> ({g.backend}) - {g.vram_gb}GB VRAM
                  <br />
                  <span className="text-xs text-gray-400">
                    Task: {g.assigned_tasks.join(", ")}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </main>
  );
}
