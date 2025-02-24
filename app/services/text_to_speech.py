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

logger = logging.getLogger(__name__)

class TTSAssistantHandler(AssistantEventHandler):
    def __init__(self, client, audio_queue, synthesis_done):
        super().__init__()
        self.client = client
        self.audio_queue = audio_queue
        self.synthesis_done = synthesis_done
        self.current_sentence = ""
        # NEW: Track pending TTS processing tasks
        self.pending_tts_tasks = []
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        print("TTSAssistantHandler initialized")
        
    @override
    def on_text_created(self, text) -> None:
        print("Text creation started")
        self.current_sentence = ""
    
    @override
    def on_text_delta(self, delta: TextDelta, snapshot: Text) -> None:
        logger.debug(f"Received text delta: {delta.value}")
        self.current_sentence += delta.value
        
        # Check if we have a complete sentence
        if any(end in delta.value for end in ['.', '!', '?']) and len(self.current_sentence.strip()) > 0:
            complete_sentence = self.current_sentence.strip()
            print(f"Complete sentence detected: {complete_sentence}")
            # NEW: Create and store the task, so we can await it later
            task = asyncio.create_task(self.process_sentence(complete_sentence))
            self.pending_tts_tasks.append(task)
            self.current_sentence = ""
        else:
            logger.debug(f"Current incomplete sentence: {self.current_sentence}")

    @override
    def on_message_done(self, message) -> None:
        print("Message processing completed")
        # Process any remaining text
        if len(self.current_sentence.strip()) > 0:
            final_sentence = self.current_sentence.strip()
            print(f"Processing final sentence fragment: {final_sentence}")
            task = asyncio.create_task(self.process_sentence(final_sentence))
            self.pending_tts_tasks.append(task)
        else:
            print("No remaining text to process")
        # NEW: Wait for all TTS processing tasks to complete before signaling done
        asyncio.create_task(self._wait_for_all_tts())

    async def _wait_for_all_tts(self):
        if self.pending_tts_tasks:
            await asyncio.gather(*self.pending_tts_tasks)
        print("All TTS tasks completed, setting synthesis_done event")
        self.synthesis_done.set()

    async def process_sentence(self, sentence: str):
        """Convert a sentence to speech using OpenAI TTS and queue the audio."""
        print(f"Starting TTS processing for sentence: {sentence}")
        try:
            # Create a temporary file for the speech
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                print(f"Created temporary file: {temp_file.name}")
                
                # Generate speech using OpenAI TTS
                print("Calling OpenAI TTS API...")
                response = await asyncio.to_thread(
                    lambda: self.openai_client.audio.speech.create(
                        model="tts-1",
                        voice="alloy",
                        input=sentence
                    )
                )
                print("Successfully received TTS response from OpenAI")
                
                # Save the audio to the temporary file
                print(f"Writing audio to temporary file: {temp_file.name}")
                response.write_to_file(temp_file.name)
                
                # Read the file in chunks and send to queue
                print("Starting to read and queue audio chunks")
                chunk_count = 0
                with open(temp_file.name, "rb") as audio_file:
                    while chunk := audio_file.read(32768):  # 32KB chunks
                        await self.audio_queue.put(chunk)
                        chunk_count += 1
                print(f"Queued {chunk_count} audio chunks")
                        
            # Clean up the temporary file
            print(f"Cleaning up temporary file: {temp_file.name}")
            os.unlink(temp_file.name)
            print("TTS processing completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing sentence: {e}", exc_info=True)
            raise  # Re-raise the exception to ensure it's properly handled upstream