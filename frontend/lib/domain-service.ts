// lib/domain-service.ts
// Dipanggil dari /api/chat sebelum nerusin ke model LLM.
// Tujuannya: ambil intelligence dari pipeline Python (case1/2/3),
// gabung jadi satu konteks buat dimasukin ke system prompt LLM.

// --------------------------------------------------
// Tipe bantu
// --------------------------------------------------

type LLMContextBundle = {
  case1?: any;
  case2?: any;
  case3?: any;
};

type DomainBundle = {
  raw: {
    case1?: any;
    case2?: any;
    case3?: any;
  };
  llmContext: LLMContextBundle;
};

// --------------------------------------------------
// Helper generic buat call service python
// --------------------------------------------------

async function postJSON<T>(url: string, body: unknown): Promise<T> {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    // Ini penting: lempar error biar Promise.allSettled() bisa tandain 'rejected'
    throw new Error(`service error: ${url} status=${resp.status}`);
  }

  return resp.json() as Promise<T>;
}

// --------------------------------------------------
// CASE 1
// Ringkasan tren / statistik historis mesin, human_report, dsb.
// Endpoint FastAPI: /infer_case1
// --------------------------------------------------

async function callCase1(query: string) {
  // Contoh response FastAPI case1:
  // {
  //   "raw": {
  //     "human_report": "...",
  //     "intent": {...},
  //     "llm_metadata": {...},
  //     "plot_paths": [...]
  //   },
  //   "llmContext": {
  //     "intent": {...},
  //     "summary_brief": "...",
  //     "metrics": {...}
  //   }
  // }
  const base = process.env.CASE1_SERVICE_URL;
  if (!base) {
    throw new Error("CASE1_SERVICE_URL is not defined");
  }

  return postJSON<{
    raw: any;
    llmContext: any;
  }>(`${base}/infer_case1`, { query });
}

// --------------------------------------------------
// CASE 2
// Anomaly QA → deteksi anomali di window waktu tertentu
// Endpoint FastAPI: /infer_case2
// --------------------------------------------------

async function callCase2(query: string) {
  // Contoh response FastAPI case2:
  // {
  //   "raw": {
  //       "window": {...},
  //       "anomalies": [...],
  //       "machines": [...],
  //       "explanation": "..."
  //   },
  //   "llmContext": {
  //       "summary": {...},
  //       "window": {...},
  //       "explanation": "..."
  //   }
  // }
  const base = process.env.CASE2_SERVICE_URL;
  if (!base) {
    throw new Error("CASE2_SERVICE_URL is not defined");
  }

  return postJSON<{
    raw: any;
    llmContext: any;
  }>(`${base}/infer_case2`, { query });
}

// --------------------------------------------------
// CASE 3
// Forecast ETA threshold ("kapan XP888A turun di bawah 70 C?")
// Endpoint FastAPI: /infer_case3
// --------------------------------------------------

async function callCase3(query: string) {
  // Contoh response FastAPI case3:
  // {
  //   "raw": {
  //      runner_out: { ... hasil interpret intent, threshold, eta ... },
  //      ...
  //   },
  //   "llmContext": {
  //      intent: {...},
  //      forecast_summary: {...},
  //      eta: {...},
  //      operator_briefing: "...",
  //      anomaly_overview: {...}
  //   }
  // }
  const base = process.env.CASE3_SERVICE_URL;
  if (!base) {
    throw new Error("CASE3_SERVICE_URL is not defined");
  }

  return postJSON<{
    raw: any;
    llmContext: any;
  }>(`${base}/infer_case3`, { query });
}

// --------------------------------------------------
// buildLLMContextForUserQuery()
// Ini yang dipanggil /api/chat.
// Dia manggil ketiga service paralel, dan gabung hasilnya.
// --------------------------------------------------

export async function buildLLMContextForUserQuery(
  userText: string
): Promise<DomainBundle> {
  // Panggil semua paralel biar cepat.
  // Promise.allSettled → kalau salah satu down, yang lain tetap jalan.
  const [c1, c2, c3] = await Promise.allSettled([
    callCase1(userText),
    callCase2(userText),
    callCase3(userText),
  ]);

  // Siapkan output final
  const out: DomainBundle = {
    raw: {},
    llmContext: {},
  };

  // Masukin hasil CASE1 kalau sukses
  if (c1.status === "fulfilled") {
    out.raw.case1 = c1.value.raw;
    out.llmContext.case1 = c1.value.llmContext;
  } else {
    // optional logging, biar gak bikin 500
    console.warn("[domain-service] case1 failed:", c1.reason);
  }

  // Masukin hasil CASE2 kalau sukses
  if (c2.status === "fulfilled") {
    out.raw.case2 = c2.value.raw;
    out.llmContext.case2 = c2.value.llmContext;
  } else {
    console.warn("[domain-service] case2 failed:", c2.reason);
  }

  // Masukin hasil CASE3 kalau sukses
  if (c3.status === "fulfilled") {
    out.raw.case3 = c3.value.raw;
    out.llmContext.case3 = c3.value.llmContext;
  } else {
    console.warn("[domain-service] case3 failed:", c3.reason);
  }

  return out;
}
