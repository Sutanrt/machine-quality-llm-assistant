// app/landing/page.tsx
import Link from "next/link";
// import { BackgroundWaves } from "@/components/backGroundWaves";
export default function LandingPreviewPage() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-gray-950 via-gray-900 to-black text-white flex flex-col">
      {/* NAVBAR */}
      {/* <BackgroundWaves/> */}
      <header className="w-full max-w-7xl mx-auto flex items-center justify-between py-6 px-6">
        <div className="text-lg font-semibold tracking-tight">
          <span className="text-emerald-400">MCI</span> AI Assistant
        </div>

        <nav className="hidden md:flex items-center gap-8 text-sm text-gray-300">
          <a href="#features" className="hover:text-white transition">Features</a>
          <a href="#how" className="hover:text-white transition">How it works</a>
          <a href="#contact" className="hover:text-white transition">Contact</a>
        </nav>

        <div className="flex items-center gap-3">
          {/* nanti bisa diarahkan ke login saat auth udah proper */}
          <Link
            href="/"
            className="text-sm text-gray-300 hover:text-white transition"
          >
            Launch App
          </Link>
        </div>
      </header>

      {/* HERO */}
      <section className="w-full max-w-7xl mx-auto flex flex-col-reverse lg:flex-row items-center gap-12 px-6 py-16">
        {/* Text */}
        <div className="flex-1">
          <div className="inline-flex items-center text-xs text-emerald-400 bg-emerald-500/10 border border-emerald-500/30 rounded-full px-3 py-1 mb-4">
            <span className="mr-2 inline-block h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            Internal Preview
          </div>

          <h1 className="text-3xl sm:text-4xl lg:text-5xl font-semibold tracking-tight text-white leading-tight">
            AI Assistant for Production Line
            <span className="text-emerald-400 block">Monitor. Predict. Explain.</span>
          </h1>

          <p className="text-gray-400 text-base sm:text-lg leading-relaxed mt-6 max-w-xl">
            K2-Inhale Assistant bantu operator pabrik ngecek kondisi mesin,
            deteksi anomali, dan jelasin akar masalah dengan bahasa lapangan —
            bukan bahasa data scientist.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 mt-8">
            <Link
              href="/"
              className="rounded-lg bg-emerald-500 text-black font-medium px-5 py-3 text-sm hover:bg-emerald-400 transition text-center"
            >
              Masuk Aplikasi
            </Link>

            <a
              href="#features"
              className="rounded-lg border border-gray-700 text-gray-200 font-medium px-5 py-3 text-sm hover:bg-gray-800/60 transition text-center"
            >
              Lihat Fitur
            </a>
          </div>

          <p className="text-[11px] text-gray-500 mt-4">
            Versi demo internal • Data mesin nyata
          </p>
        </div>

        {/* Visual mock chat */}
        <div className="flex-1 w-full">
          <div className="bg-gray-900/70 border border-gray-700 rounded-xl p-4 shadow-2xl shadow-emerald-500/10 max-w-md mx-auto">
            <div className="text-[10px] uppercase text-gray-400 font-medium tracking-wider mb-2">
              Live Chat Preview
            </div>

            <div className="bg-black/40 border border-gray-800 rounded-lg p-4 text-sm text-gray-200 h-40 overflow-hidden">
              <div className="mb-4">
                <div className="text-[11px] text-gray-500 mb-1">Operator</div>
                <div className="bg-gray-800/60 border border-gray-700 rounded-md p-3">
                  Mesin XP888A aman gak buat shift malam?
                </div>
              </div>

              <div>
                <div className="text-[11px] text-emerald-400 mb-1">Assistant</div>
                <div className="bg-gray-800/20 border border-gray-700/60 rounded-md p-3">
                  Ada kenaikan suhu 14% di luar normal jam 02:10–02:24.
                  Saran: cek coolant flow sebelum start shift berikutnya.
                </div>
              </div>
            </div>

            <div className="text-[10px] text-gray-500 mt-3 text-right">
              AI + Sensor Telemetry
            </div>
          </div>
        </div>
      </section>

      {/* FEATURES */}
      <section
        id="features"
        className="w-full max-w-7xl mx-auto px-6 py-20 grid gap-8 md:grid-cols-3 text-sm"
      >
        <div className="bg-gray-900/40 border border-gray-800 rounded-xl p-6">
          <div className="text-emerald-400 font-medium mb-2">Realtime Alerts</div>
          <div className="text-gray-300 font-semibold mb-2">
            Deteksi anomali langsung.
          </div>
          <div className="text-gray-500 leading-relaxed">
            Sistem bandingin kondisi live vs baseline mesin sehat.
          </div>
        </div>

        <div className="bg-gray-900/40 border border-gray-800 rounded-xl p-6">
          <div className="text-emerald-400 font-medium mb-2">Root Cause Hints</div>
          <div className="text-gray-300 font-semibold mb-2">
            “Kenapa ini kejadian?”
          </div>
          <div className="text-gray-500 leading-relaxed">
            Assistant jelasin kemungkinan penyebab dengan bahasa operator.
          </div>
        </div>

        <div className="bg-gray-900/40 border border-gray-800 rounded-xl p-6">
          <div className="text-emerald-400 font-medium mb-2">Forecast</div>
          <div className="text-gray-300 font-semibold mb-2">
            Prediksi 30–120 menit ke depan.
          </div>
          <div className="text-gray-500 leading-relaxed">
            Biar kamu bisa cegah downtime sebelum kejadian.
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer
        id="contact"
        className="text-[12px] text-gray-500 border-t border-gray-800/70 py-10 px-6 flex flex-col items-center gap-2"
      >
        <div>MCI Internal Preview Build</div>
        <div className="text-gray-600">
          K2-Inhale Project // v0.1-alpha
        </div>
      </footer>
    </main>
  );
}
