from fastapi import FastAPI, WebSocket, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import logging
from contextlib import asynccontextmanager
from app.services.assistant import KnowledgeAssistant
from app.services.transcription import websocket_transcribe

# uvicorn main:app --reload       

async def get_knowledge_assistant(websocket: WebSocket) -> KnowledgeAssistant:
    assistant = websocket.app.state.knowledge_assistant
    if assistant is None:
        raise RuntimeError("Knowledge assistant has not been initialized")
    return assistant

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the assistant on startup
    assistant = KnowledgeAssistant()
    
    # Upload knowledge files to the assistant
    try:
        # upload_result = assistant.upload_knowledge_files()
        # logging.info(f"Knowledge base initialized: {upload_result}")
        # Store the assistant instance on app.state
        app.state.knowledge_assistant = assistant
    except Exception as e:
        logging.error(f"Error initializing knowledge base: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown if needed
    app.state.knowledge_assistant = None

app = FastAPI(
    title="FastAPI Project",
    description="A basic FastAPI project structure",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Provide a route for the index
@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html") as f:
        return f.read()

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket, knowledge_assistant: KnowledgeAssistant = Depends(get_knowledge_assistant)):
    await websocket_transcribe(websocket, knowledge_assistant)

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("main:app", host="localhost", port=8000, reload=True) 