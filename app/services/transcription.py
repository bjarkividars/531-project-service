from fastapi import WebSocket, WebSocketDisconnect
import azure.cognitiveservices.speech as speechsdk
import asyncio
import logging
from typing import List, Optional
import google.generativeai as genai

from fastapi.websockets import WebSocketState
from app.config import settings

region = "eastus"
logger = logging.getLogger("app_logger")

# Configure Azure Speech
speech_config = speechsdk.SpeechConfig(
    subscription=settings.azure_speech_key,
    region="eastus"
)

# Configure Gemini
genai.configure(api_key=settings.gemini_api_key)
model = genai.GenerativeModel('gemini-2.0-flash')

class TranscriptionResult:
    def __init__(self):
        self.final_outputs: List[str] = []
        self.complete_text: Optional[str] = None

    def add_final_output(self, text: str):
        if text.startswith("FINAL: "):
            text = text[7:]  # Remove the "FINAL: " prefix
        self.final_outputs.append(text)

    def get_complete_text(self) -> str:
        if not self.complete_text:
            self.complete_text = " ".join(self.final_outputs)
        return self.complete_text

# Function to set up the speech recognizer
def setup_speech_recognition(message_queue, recognition_done):
    stream = speechsdk.audio.PushAudioInputStream()
    audio_config = speechsdk.audio.AudioConfig(stream=stream)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config)

    # Event handlers
    def recognizing_handler(evt):
        message_queue.put_nowait(f"PARTIAL: {evt.result.text}")

    def recognized_handler(evt):
        message_queue.put_nowait(f"FINAL: {evt.result.text}")

    def session_stopped_handler(evt):
        message_queue.put_nowait("SESSION_STOPPED")
        recognition_done.set()

    def canceled_handler(evt):
        logger.error("Recognition canceled")
        recognition_done.set()

    # Attach handlers to recognizer events
    speech_recognizer.recognizing.connect(recognizing_handler)
    speech_recognizer.recognized.connect(recognized_handler)
    speech_recognizer.session_stopped.connect(session_stopped_handler)
    speech_recognizer.canceled.connect(canceled_handler)

    return speech_recognizer, stream

async def send_messages(websocket, message_queue, transcription_result):
    """Asynchronously send messages from the queue to the WebSocket."""
    try:
        while True:
            message = await message_queue.get()
            if message is None:  # Sentinel to stop
                break
            # Save final results to transcription_result for later processing
            if message.startswith("FINAL: "):
                transcription_result.add_final_output(message)
            await websocket.send_text(message)
    except Exception as e:
        logging.error(f"Error sending messages: {e}")

async def analyze_with_gemini(text: str) -> str:
    """Analyze the transcribed text using Gemini API."""
    try:
        prompt = f"""Analyze the following transcribed text and provide insights. 
        Consider the main points, tone, and any key information:

        Transcription: {text}
        """
        response = await asyncio.to_thread(
            lambda: model.generate_content(prompt).text
        )
        return response
    except Exception as e:
        logger.error(f"Error analyzing with Gemini: {e}")
        return f"Error analyzing text: {str(e)}"

async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted.")

    recognition_done = asyncio.Event()
    message_queue = asyncio.Queue()
    transcription_result = TranscriptionResult()

    speech_recognizer, stream = setup_speech_recognition(
        message_queue, recognition_done)

    sender_task = asyncio.create_task(send_messages(
        websocket, message_queue, transcription_result))

    try:
        speech_recognizer.start_continuous_recognition()

        while True:
            try:
                # First, try to receive as bytes (for audio data)
                data = await websocket.receive()
                
                # Check if it's a text message (command) or bytes (audio)
                if "text" in data:
                    command = data["text"]
                    if command == "STOP_DISCARD":
                        logger.info("Received stop command with discard.")
                        break
                    elif command == "STOP_PROCESS":
                        logger.info("Received stop command with process.")
                        # Process the transcription after stopping
                        complete_text = transcription_result.get_complete_text()
                        # Get Gemini analysis
                        analysis = await analyze_with_gemini(complete_text)
                        await websocket.send_text(f"COMPLETE_TRANSCRIPTION: {complete_text}")
                        await websocket.send_text(f"ANALYSIS: {analysis}")
                        break
                    elif command == "CHUNKS_DONE":
                        logger.info("Received 'CHUNKS_DONE' signal from client.")
                        break
                elif "bytes" in data:
                    audio_chunk = data["bytes"]
                    if audio_chunk:
                        stream.write(audio_chunk)
                    else:
                        logger.info("Received empty audio chunk.")
                        break

            except WebSocketDisconnect:
                logger.info("WebSocket disconnected.")
                break
            except Exception as e:
                logger.error(f"Error receiving data: {e}")
                break

        # Wait for transcription to complete
        logger.info("Waiting for transcription to complete.")
        speech_recognizer.stop_continuous_recognition()
        await recognition_done.wait()

    except Exception as e:
        logger.error(f"Error during WebSocket transcription: {e}")
    finally:
        logger.info("Entering finally block.")
        await message_queue.put(None)
        await sender_task
        logger.info("Sender task completed.")
        stream.close()

        try:
            await websocket.send_text("DONE")
            logger.info("Sent 'DONE' to client.")
        except Exception as e:
            logger.error(f"Error sending 'DONE': {e}")

        if websocket.client_state != WebSocketState.DISCONNECTED:
            logger.info("Closing WebSocket connection.")
            await websocket.close()
