import { http, HttpResponse } from "msw";

export const handlers = [
  // ① SHOTS FEED (already there)
  http.get("/api/shots", () =>
    HttpResponse.json([
      { id: 1, timestamp: "2025-08-05T17:00:00Z", result: "make", shot_type: "3PT", arc_deg: 46 },
      { id: 2, timestamp: "2025-08-05T17:00:02Z", result: "miss", shot_type: "MID", arc_deg: 39 }
    ])
  ),

  // ② NEW: STATS SUMMARY
  http.get("/api/stats", () =>
    HttpResponse.json({
      user_name: "Nic",
      week_attempts: 128,
      week_makes: 83,
      all_time_makes: 7421
    })
  )
];
