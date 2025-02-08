from fastapi import WebSocket, WebSocketDisconnect
import azure.cognitiveservices.speech as speechsdk
import asyncio
import logging

from fastapi.websockets import WebSocketState
from app.config import settings
region = "eastus"
logger = logging.getLogger("app_logger")

speech_config = speechsdk.SpeechConfig(
    subscription=settings.azure_speech_key,
    region="eastus"
)


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


async def send_messages(websocket, message_queue, final_outputs):
    """Asynchronously send messages from the queue to the WebSocket."""
    try:
        while True:
            message = await message_queue.get()
            if message is None:  # Sentinel to stop
                break
            # Save final results to `final_outputs` for later processing
            if message.startswith("FINAL: "):
                final_outputs.append(message)
            await websocket.send_text(message)
    except Exception as e:
        logging.error(f"Error sending messages: {e}")


async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted.")
    logger.info("WebSocket connection accepted.")

    recognition_done = asyncio.Event()  # Use asyncio.Event for async waiting
    message_queue = asyncio.Queue()
    final_outputs = []  # Store final results locally

    speech_recognizer, stream = setup_speech_recognition(
        message_queue, recognition_done)

    sender_task = asyncio.create_task(send_messages(
        websocket, message_queue, final_outputs))

    try:
        speech_recognizer.start_continuous_recognition()

        # Keep receiving audio chunks until "CHUNKS_DONE" is received
        while True:
            try:
                # Receive data as bytes
                audio_chunk = await websocket.receive_bytes()
                if audio_chunk == b"CHUNKS_DONE":  # Check for the signal
                    logger.info("Received 'CHUNKS_DONE' signal from client.")
                    break  # Exit loop to process remaining data
                if audio_chunk:
                    print(
                        'size of audio chunk', len(audio_chunk)
                    )
                    stream.write(audio_chunk)
                else:
                    logger.info("Received empty audio chunk.")
                    break
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected.")
                break
            except Exception as e:
                logger.error(f"Error receiving audio chunk: {
                             e}", level="error")
                break

        # Wait for transcription to complete
        logger.info("Waiting for transcription to complete.")
        speech_recognizer.stop_continuous_recognition()
        await recognition_done.wait()  # Wait for recognition to finish

    except Exception as e:
        logger.error(f"Error during WebSocket transcription: {
                     e}", level="error")
    finally:
        logger.info("Entering finally block.")
        # Push sentinel to stop sender task
        await message_queue.put(None)
        await sender_task
        logger.info("Sender task completed.")
        # Stop recognition and clean up resources
        stream.close()

        # Process final results
        logger.info("Final Transcriptions:")
        for result in final_outputs:
            logger.info(result)

        # Send "DONE" to the client
        try:
            await websocket.send_text("DONE")
            logger.info("Sent 'DONE' to client.")
        except Exception as e:
            logger.error(f"Error sending 'DONE': {e}", level="error")

        # Close WebSocket if not already disconnected
        if websocket.client_state != WebSocketState.DISCONNECTED:
            logger.info("Closing WebSocket connection.")
            await websocket.close()
