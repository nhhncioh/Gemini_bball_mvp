import { useEffect, useState } from "react";

interface Stats {
  user_name: string;
  week_attempts: number;
  week_makes: number;
  all_time_makes: number;
}

export default function HomeDashboard() {
  const [stats, setStats] = useState<Stats | null>(null);

  useEffect(() => {
    fetch("/api/stats")
      .then((r) => r.json())
      .then(setStats)
      .catch(console.error);
  }, []);

  if (!stats) {
    return (
      <div className="animate-pulse text-slate-400 text-sm">
        Loading weekly stats…
      </div>
    );
  }

  const { user_name, week_attempts, week_makes, all_time_makes } = stats;
  const weekPct = ((week_makes / week_attempts) * 100).toFixed(1);

  return (
    <section className="mb-8">
      <h2 className="text-2xl font-semibold mb-4">
        Hello&nbsp;{user_name},
      </h2>

      <div className="grid gap-4 sm:grid-cols-3">
        <StatCard label="Shots this week" value={week_attempts} />
        <StatCard label="Makes this week" value={week_makes} />
        <StatCard label="Makes all-time" value={all_time_makes} />
      </div>

      <p className="mt-2 text-sm text-slate-400">
        Weekly FG&nbsp;%: <span className="font-medium">{weekPct}%</span>
      </p>
    </section>
  );
}

/* — Helper sub-component — */
function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-xl bg-slate-800 p-4 flex flex-col">
      <span className="text-xs uppercase tracking-wide text-slate-400">
        {label}
      </span>
      <span className="text-3xl font-bold text-white mt-1">{value}</span>
    </div>
  );
}
