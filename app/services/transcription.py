from fastapi import WebSocket, WebSocketDisconnect
import azure.cognitiveservices.speech as speechsdk
import asyncio
import logging
from typing import List, Optional
import google.generativeai as genai

from fastapi.websockets import WebSocketState
from app.config import settings
from app.services.assistant import KnowledgeAssistant
from app.services.text_to_speech import TTSAssistantHandler

region = "eastus"
logger = logging.getLogger("app_logger")

# Configure Azure Speech
speech_config = speechsdk.SpeechConfig(
    subscription=settings.azure_speech_key,
    region="eastus"
)

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
        print('recognizing', evt.result.text)
        message_queue.put_nowait(f"PARTIAL: {evt.result.text}")

    def recognized_handler(evt):
        print('recognized', evt.result.text)
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
    print('send_messages - START')
    try:
        while True:
            message = await message_queue.get()
            print(f"send_messages - Got from queue: {message}")
            if message is None:  # Sentinel to stop
                print("send_messages - Received None sentinel, breaking")
                break
            # Save final results to transcription_result for later processing
            if message.startswith("FINAL: "):
                print(f"send_messages - Adding to transcription result: {message}")
                transcription_result.add_final_output(message)
            await websocket.send_text(message)
            print(f"send_messages - Sent to websocket: {message}")
    except Exception as e:
        logging.error(f"Error sending messages: {e}")
    finally:
        print('send_messages - END')


async def process_with_assistant_and_tts(
    websocket: WebSocket,
    question: str,
    knowledge_assistant: KnowledgeAssistant,
    audio_queue: asyncio.Queue,
    synthesis_done: asyncio.Event
) -> None:
    try:
        # Create conversation thread, etc.
        knowledge_assistant.create_thread()

        # Create TTS handler
        handler = TTSAssistantHandler(
            knowledge_assistant.client,
            audio_queue,
            synthesis_done
        )

        # Send the user question to the language model
        await asyncio.to_thread(
            lambda: knowledge_assistant.client.beta.threads.messages.create(
                thread_id=knowledge_assistant.current_thread.id,
                role="user",
                content=question
            )
        )

        # Start streaming from the language model in a blocking way
        # but TTS generation may happen concurrently in your TTS handler.
        with knowledge_assistant.client.beta.threads.runs.stream(
            thread_id=knowledge_assistant.current_thread.id,
            assistant_id=knowledge_assistant.assistant.id,
            event_handler=handler,
        ) as stream:
            stream.until_done()

        # -----------------------------------------------------------------
        # Now that the LM stream is done, ensure TTS is also truly finished.
        # We await the event we set in on_complete().
        # -----------------------------------------------------------------
        await handler.done_event.wait()

        # Once TTS is done, we can safely enqueue None.
        await audio_queue.put(None)

    except Exception as e:
        logger.error(f"Error processing with assistant and TTS: {e}")
        await websocket.send_text(f"ERROR: {str(e)}")


async def websocket_transcribe(websocket: WebSocket, knowledge_assistant: KnowledgeAssistant):
    await websocket.accept()
    print("WebSocket connection accepted.")

    recognition_done = asyncio.Event()
    message_queue = asyncio.Queue()
    transcription_result = TranscriptionResult()
    audio_queue_tts = asyncio.Queue()  # Queue for TTS audio chunks
    synthesis_done_event = asyncio.Event()  # Event for TTS completion

    speech_recognizer, stream = setup_speech_recognition(
        message_queue, recognition_done
    )

    # Start sending partial/final transcription messages
    message_sender_task = asyncio.create_task(
        send_messages(websocket, message_queue, transcription_result)
    )

    audio_sender_task_tts = None  # Will start later if needed

    try:
        speech_recognizer.start_continuous_recognition()

        while True:
            try:
                data = await websocket.receive()

                # Check if it's a text message (command) or bytes (audio)
                if "text" in data:
                    command = data["text"]
                    if command == "STOP_DISCARD":
                        print("Received stop command with discard.")
                        break
                    elif command == "STOP_PROCESS":
                        print("Received stop command with process.")
                        # Wait for all partial/final messages to be sent
                        speech_recognizer.stop_continuous_recognition()
                        await recognition_done.wait()
                        await message_queue.put(None)
                        await message_sender_task

                        # Now get the complete text
                        complete_text = transcription_result.get_complete_text()
                        print(f"Complete transcription text: {complete_text}")
                        await websocket.send_text(f"COMPLETE_TRANSCRIPTION: {complete_text}")

                        # Process with knowledge assistant and TTS if available
                        if knowledge_assistant and complete_text.strip():
                            print("Processing with assistant and TTS.")
                            print('complete text', complete_text)
                            
                            audio_sender_task_tts = asyncio.create_task(
                                send_audio_chunks(websocket, audio_queue_tts)
                            )
                            
                            # TTS + generative response
                            await process_with_assistant_and_tts(
                                websocket,
                                complete_text,
                                knowledge_assistant,
                                audio_queue_tts,
                                synthesis_done_event
                            )
                        break
                    elif command == "CHUNKS_DONE":
                        print("Received 'CHUNKS_DONE' signal from client.")
                        break
                elif "bytes" in data:
                    audio_chunk = data["bytes"]
                    if audio_chunk:
                        stream.write(audio_chunk)
                    else:
                        print("Received empty audio chunk.")
                        break

            except WebSocketDisconnect:
                print("WebSocket disconnected.")
                break
            except Exception as e:
                logger.error(f"Error receiving data: {e}")
                break

    except Exception as e:
        logger.error(f"Error during WebSocket transcription: {e}")
    finally:
        print("Entering finally block.")
        # Make sure we stop the speech recognizer and finish message sending
        if not recognition_done.is_set():
            speech_recognizer.stop_continuous_recognition()
            await recognition_done.wait()
            await message_queue.put(None)
            await message_sender_task
        print("Message Sender task completed.")

        if audio_sender_task_tts:
            # Wait for the audio sender to drain its queue (which includes `None` from TTS)
            await audio_sender_task_tts
            print("TTS Audio Sender task completed.")

        print("All sender tasks completed.")
        stream.close()

        try:
            await websocket.send_text("DONE")
            print("Sent 'DONE' to client.")
        except Exception as e:
            logger.error(f"Error sending 'DONE': {e}")


async def send_audio_chunks(websocket: WebSocket, audio_queue: asyncio.Queue):
    """Send audio chunks from the queue to the WebSocket."""
    print('send_audio_chunks - START')
    try:
        while True:
            chunk = await audio_queue.get()
            print(f"send_audio_chunks - Got chunk from queue: {chunk is not None}")
            if chunk is None:  # End of stream
                print("send_audio_chunks - Received None chunk, breaking")
                break
            await websocket.send_bytes(chunk)
            print(f"send_audio_chunks - Sent chunk to websocket: {len(chunk) if chunk else 0} bytes")
    except Exception as e:
        logger.error(f"Error sending audio chunks: {e}")
    finally:
        print('send_audio_chunks - END')