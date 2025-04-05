from fastapi.responses import JSONResponse
from fastapi import Request
from api import app, model
from src.evaluate import evaluate_array

@app.post("/process/")
async def process_pixels(request: Request):
    data = await request.json()
    pixels = data.get("grid")

    if not pixels:
        return JSONResponse(status_code=400, content={"error": "there are no pixels"})

    prediction = evaluate_array(model, pixels)

    return JSONResponse(content={"result": prediction})