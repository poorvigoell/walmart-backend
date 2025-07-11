from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from lazyforecast import run_lazy_forecast, get_stockout_alerts

app = FastAPI(title="Lazy AI Forecast API")

# Enable frontend access (customize in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "✅ Lazy AI model is running"}

@app.get("/stockout-alerts")
def stock_alerts():
    return get_stockout_alerts()

@app.get("/forecast")
def forecast():
    df = run_lazy_forecast()
    return df.to_dict(orient="records")
