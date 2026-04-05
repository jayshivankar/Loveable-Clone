from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRouter
from src.routes import router as api_router

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    # Logic to run on startup, e.g., database connection checks
    print("Application startup: Establishing database connection...")

@app.on_event("shutdown")
async def shutdown_event():
    # Logic to run on shutdown, e.g., closing database connections
    print("Application shutdown: Closing database connection...")

app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        print(f"Error occurred while starting the application: {e}")
