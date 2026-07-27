export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div className="z-10 max-w-5xl w-full items-center justify-between font-mono text-sm lg:flex">
        <h1 className="text-4xl font-bold">AI Shorts Factory</h1>
      </div>
      <div className="bg-gray-800 p-6 rounded-lg shadow-lg w-full max-w-lg">
        <h2 className="text-2xl mb-4">Dashboard</h2>
        <p className="text-gray-400">Benvenuto nella piattaforma di generazione video autonoma.</p>
        <div className="mt-6 flex gap-4">
          <button className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
            Crea Profilo
          </button>
          <button className="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded">
            Avvia Generazione
          </button>
        </div>
      </div>
    </main>
  );
}
