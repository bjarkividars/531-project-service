from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import logging
from typing import Optional
import re
from openai import OpenAI, AssistantEventHandler
from openai.types.beta.threads import TextDelta, Text
from typing_extensions import override
import tempfile
import os

from fastapi.websockets import WebSocketState
from app.config import settings
from app.services.assistant import KnowledgeAssistant

logger = logging.getLogger("app_logger")

class TTSAssistantHandler(AssistantEventHandler):
    def __init__(self, client, audio_queue, synthesis_done):
        super().__init__()
        self.client = client
        self.audio_queue = audio_queue
        self.synthesis_done = synthesis_done
        self.current_sentence = ""
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        
    @override
    def on_text_created(self, text) -> None:
        self.current_sentence = ""
    
    @override
    def on_text_delta(self, delta: TextDelta, snapshot: Text) -> None:
        self.current_sentence += delta.value
        
        # Check if we have a complete sentence
        if any(end in delta.value for end in ['.', '!', '?']) and len(self.current_sentence.strip()) > 0:
            # Process the complete sentence
            asyncio.create_task(self.process_sentence(self.current_sentence.strip()))
            self.current_sentence = ""

    @override
    def on_message_done(self, message) -> None:
        # Process any remaining text
        if len(self.current_sentence.strip()) > 0:
            asyncio.create_task(self.process_sentence(self.current_sentence.strip()))
        self.synthesis_done.set()

    async def process_sentence(self, sentence: str):
        """Convert a sentence to speech using OpenAI TTS and queue the audio."""
        try:
            # Create a temporary file for the speech
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                # Generate speech using OpenAI TTS
                response = await asyncio.to_thread(
                    lambda: self.openai_client.audio.speech.create(
                        model="tts-1",
                        voice="alloy",
                        input=sentence
                    )
                )
                
                # Save the audio to the temporary file
                response.write_to_file(temp_file.name)
                
                # Read the file in chunks and send to queue
                with open(temp_file.name, "rb") as audio_file:
                    while chunk := audio_file.read(32768):  # 32KB chunks
                        await self.audio_queue.put(chunk)
                        
            # Clean up the temporary file
            os.unlink(temp_file.name)
            
        except Exception as e:
            logger.error(f"Error processing sentence: {e}")

async def send_audio_chunks(websocket: WebSocket, audio_queue: asyncio.Queue):
    """Asynchronously send audio chunks from the queue to the WebSocket."""
    try:
        while True:
            chunk = await audio_queue.get()
            if chunk is None:  # End of stream
                break
            await websocket.send_bytes(chunk)
    except Exception as e:
        logger.error(f"Error sending audio chunks: {e}")

async def websocket_synthesize(websocket: WebSocket):
    """Handle WebSocket connection for text-to-speech synthesis with KnowledgeAssistant integration."""
    await websocket.accept()
    logger.info("WebSocket connection accepted for assistant TTS.")

    synthesis_done = asyncio.Event()
    audio_queue = asyncio.Queue()
    
    sender_task = asyncio.create_task(send_audio_chunks(websocket, audio_queue))
    
    # Initialize KnowledgeAssistant
    assistant = KnowledgeAssistant()
    assistant.create_thread()  # Create a new conversation thread

    try:
        while True:
            try:
                data = await websocket.receive_text()
                
                if data == "STOP":
                    logger.info("Received stop command.")
                    break
                
                # Reset the synthesis_done event for new synthesis
                synthesis_done.clear()
                
                # Create custom event handler for this session
                handler = TTSAssistantHandler(
                    assistant.client,
                    audio_queue,
                    synthesis_done
                )
                
                # Add the message to the thread and run the assistant
                await asyncio.to_thread(
                    lambda: assistant.client.beta.threads.messages.create(
                        thread_id=assistant.current_thread.id,
                        role="user",
                        content=data
                    )
                )
                
                # Stream the assistant's response with our custom handler
                with assistant.client.beta.threads.runs.stream(
                    thread_id=assistant.current_thread.id,
                    assistant_id=assistant.assistant.id,
                    event_handler=handler,
                ) as stream:
                    stream.until_done()
                
                # Wait for synthesis to complete
                await synthesis_done.wait()
                await audio_queue.put(None)  # Signal end of audio stream
                
                # Send a marker indicating the end of this synthesis
                await websocket.send_text("CHUNK_COMPLETE")

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected.")
                break
            except Exception as e:
                logger.error(f"Error during synthesis: {e}")
                break

    except Exception as e:
        logger.error(f"Error during WebSocket synthesis: {e}")
    finally:
        logger.info("Cleaning up text-to-speech resources.")
        await audio_queue.put(None)
        await sender_task
        
        try:
            await websocket.send_text("DONE")
            logger.info("Sent 'DONE' to client.")
        except Exception as e:
            logger.error(f"Error sending 'DONE': {e}")

        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close() 