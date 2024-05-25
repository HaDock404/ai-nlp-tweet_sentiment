from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("templates/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

if __name__ == '__main__':
    uvicorn.run(app, port=8000, host='0.0.0.0')
