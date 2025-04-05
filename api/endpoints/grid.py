from fastapi.responses import HTMLResponse
from api import app

@app.get("/", response_class=HTMLResponse)
def get_canvas():
    with open("api/templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())