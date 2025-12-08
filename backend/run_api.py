# run_api.py
#
# Cara start server:
#   python run_api.py
#
# Server jalan di 0.0.0.0:8001
# Endpoint siap dipakai Next.js kamu.

from industrial_intel_service import bootstrap_all, app
import uvicorn

if __name__ == "__main__":
    # load semua model & data sekali di awal
    bootstrap_all()

    # nyalain FastAPI
    uvicorn.run(
        "industrial_intel_service:app",
        host="0.0.0.0",
        port=8010,
        reload=False
    )

