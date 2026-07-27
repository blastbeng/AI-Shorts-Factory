"use client";

import { useEffect, useState } from "react";

type Stage = {
  name: string;
  status: string;
  result: string | null;
  created_at: string;
  updated_at: string;
};

type JobDetails = {
  job_id: number;
  status: string;
  profile_id: number;
  stages: Stage[];
};

export default function JobsPage() {
  const [jobs, setJobs] = useState<{ id: number; status: string; profile_id: number }[]>([]);
  const [selectedJob, setSelectedJob] = useState<JobDetails | null>(null);
  const [loading, setLoading] = useState(true);

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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

  const fetchJobDetails = async (jobId: number) => {
    try {
      const res = await fetch(`${apiUrl}/jobs/${jobId}`);
      setSelectedJob(await res.json());
    } catch (error) {
      console.error("Errore:", error);
    }
  };

  useEffect(() => {
    fetchJobs();
    const interval = setInterval(fetchJobs, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <main className="flex min-h-screen flex-col items-center p-8 bg-gray-900 text-white">
      <h1 className="text-4xl font-bold mb-8">Stato Jobs</h1>
      
      <div className="w-full max-w-6xl grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl mb-4">Lista Jobs</h2>
          {loading ? <p>Caricamento...</p> : (
            <ul className="space-y-2">
              {jobs.map((j) => (
                <li 
                  key={j.id} 
                  className={`bg-gray-700 p-3 rounded cursor-pointer hover:bg-gray-600 ${selectedJob?.job_id === j.id ? 'ring-2 ring-blue-500' : ''}`}
                  onClick={() => fetchJobDetails(j.id)}
                >
                  <strong>Job #{j.id}</strong> (Profile: {j.profile_id}) - 
                  <span className={`ml-2 ${j.status === 'completed' ? 'text-green-400' : j.status === 'failed' ? 'text-red-400' : 'text-yellow-400'}`}>
                    {j.status}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
          <h2 className="text-2xl mb-4">Dettagli Job</h2>
          {!selectedJob ? (
            <p className="text-gray-400">Seleziona un job dalla lista per vedere i dettagli.</p>
          ) : (
            <div>
              <p className="mb-4">Stato generale: <strong>{selectedJob.status}</strong></p>
              <ul className="space-y-2">
                {selectedJob.stages.map((s, index) => (
                  <li key={index} className="bg-gray-700 p-3 rounded">
                    <div className="flex justify-between">
                      <span>{s.name}</span>
                      <span className={`text-xs ${s.status === 'completed' ? 'text-green-400' : s.status === 'failed' ? 'text-red-400' : 'text-yellow-400'}`}>
                        {s.status}
                      </span>
                    </div>
                    {s.result && (
                      <p className="text-xs text-gray-400 mt-1 truncate">
                        Result: {s.result.substring(0, 50)}...
                      </p>
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
    </main>
  );
}
