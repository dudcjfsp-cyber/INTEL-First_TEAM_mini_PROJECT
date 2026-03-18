import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from model_handler import ModelHandler
from schemas import AnalysisResponse, AnalysisData

app = FastAPI(title="Return To Me API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Model Handler at startup
handler = None

@app.on_event("startup")
async def startup_event():
    global handler
    handler = ModelHandler()
    print("  ✅ FastAPI Backend Models Loaded Successfully")

@app.get("/")
async def root():
    return {"message": "Welcome to Return To Me API"}

@app.post("/api/v1/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    # 1. Validation
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
    
    try:
        # 2. Read Image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 3. Inference
        result_data = handler.analyze(image)
        
        # 4. Response
        return {
            "status": "success",
            "data": result_data
        }
        
    except Exception as e:
        print(f"  ❌ Error during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # backend 디렉토리 내에서 실행하므로 "main:app"으로 변경
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
