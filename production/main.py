from fastapi import FastAPI, Request  # type: ignore
from fastapi.responses import HTMLResponse  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from fastapi.templating import Jinja2Templates  # type: ignore
import uvicorn  # type: ignore

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5002, debug=True)
