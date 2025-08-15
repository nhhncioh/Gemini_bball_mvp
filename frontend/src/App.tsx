import HomeDashboard from "./HomeDashboard";
/* …existing imports… */

export default function App() {
  /* existing shots state / useEffect … */

  return (
    <div className="min-h-screen bg-slate-900 text-white p-6 space-y-6">
      {/* ⬇️ new top section */}
      <HomeDashboard />

      <h1 className="text-3xl font-bold">ShotSense Dashboard (Mock)</h1>

      {/* existing shot list … */}
    </div>
  );
}
