import asyncio
import base64
import json
import uuid

import aiohttp
import numpy as np
import websockets
from livekit import rtc
from livekit.rtc._proto.video_frame_pb2 import VideoBufferType
from loguru import logger
from scipy import signal
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    OutputImageRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import AIService


class HeyGenVideoService(AIService):
    """Class to send agent audio to HeyGen using the streaming audio input api"""

    def __init__(
        self,
        *,
        session_id: str,
        session_token: str,
        realtime_endpoint: str,
        session: aiohttp.ClientSession,
        livekit_room_url: str,
        api_base_url: str = "https://api.heygen.com",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._session_id = session_id
        self._session_token = session_token
        self._session = session
        self._api_base_url = api_base_url
        self._websocket = None
        self._buffered_audio_duration_ms = 0
        self._event_id = None
        self._realtime_endpoint = realtime_endpoint
        self._livekit_room_url = livekit_room_url
        self._livekit_room = None
        self._video_task = None
        self._audio_task = None
        self._video_event = asyncio.Event()
        self._video_event.set()

    # AI Service class methods
    async def start(self, frame: StartFrame):
        logger.info(f"HeyGenVideoService starting")
        await super().start(frame)
        await self._ws_connect()
        await self._livekit_connect()

    async def stop(self, frame: EndFrame):
        logger.info(f"HeyGenVideoService stopping")
        await super().stop(frame)
        await self._stop()

    async def cancel(self, frame: CancelFrame):
        logger.info(f"HeyGenVideoService canceling")
        await super().cancel(frame)
        await self._ws_disconnect()
        await self._livekit_disconnect()
        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

    # websocket connection methods
    async def _ws_connect(self):
        """Connect to HeyGen websocket endpoint"""
        try:
            logger.info(f"HeyGenVideoService ws connecting")
            if self._websocket:
                # assume connected
                return
            self._websocket = await websockets.connect(
                uri=self._realtime_endpoint,
            )
            self._receive_task = self.get_event_loop().create_task(
                self._ws_receive_task_handler()
            )
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None

    async def _ws_disconnect(self) -> None:
        """Disconnect from HeyGen websocket endpoint"""
        try:
            if self._websocket:
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} disconnect error: {e}")
        finally:
            self._websocket = None

    async def _ws_receive_task_handler(self) -> None:
        """Handle incoming messages from HeyGen websocket"""
        try:
            while True:
                message = await self._websocket.recv() # type: ignore
                try:
                    parsed_message = json.loads(message)
                    await self._handle_ws_server_event(parsed_message)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse websocket message as JSON: {e}")
                    continue
                if message:
                    logger.info(f"HeyGenVideoService ws received message: {message}")

        except Exception as e:
            logger.error(f"Error receiving message from websocket: {e}")

    async def _handle_ws_server_event(self, event: dict) -> None:
        """Handle an event from HeyGen websocket"""
        event_type = event.get("type")
        
        if event_type == "agent.status":
            logger.info(f"HeyGenVideoService ws received agent status: {event}")
        elif event_type == "agent.state":
            # Handle agent state changes - this is the missing handler!
            logger.info(f"HeyGenVideoService ws received agent state: {event}")
        
        # Check if this is a state change that affects video display
        state = event.get("state")
        if state == "listening":
            # Agent is ready to receive audio
            self._video_event.set()  # Enable video processing
        elif state == "speaking":
            # Agent is generating response
            logger.info("Agent is speaking - avatar should be visible")
        elif state == "idle":
            # Agent is idle
            logger.info("Agent is idle")
            self._video_event.clear()  # Disable video processing
        elif event_type == "session.created":
            logger.info(f"HeyGenVideoService session created: {event}")
        elif event_type == "session.closed":
            logger.info(f"HeyGenVideoService session closed: {event}")
        elif event_type == "avatar.start":
            logger.info(f"HeyGenVideoService avatar started: {event}")
            # Avatar video stream should start - ensure video processing is enabled
            self._video_event.set()
        elif event_type == "avatar.stop":
            logger.info(f"HeyGenVideoService avatar stopped: {event}")
        elif event_type == "error":
            error_msg = event.get("message", "Unknown error")
            logger.error(f"HeyGenVideoService received error: {error_msg}")
            await self.push_error(ErrorFrame(error=f"HeyGen error: {error_msg}", fatal=True))
        else:
            logger.warning(f"HeyGenVideoService ws received unknown event: {event_type}")
            logger.debug(f"Full unknown event: {event}")  # Log full event for debugging

    async def _ws_send(self, message: dict) -> None:
        """Send a message to HeyGen websocket"""
        try:
            logger.info(f"HeyGenVideoService ws sending message: {message.get('type')}")
            if self._websocket:
                await self._websocket.send(json.dumps(message))
            else:
                logger.error(f"{self} websocket not connected")
        except Exception as e:
            logger.error(f"Error sending message to websocket: {e}")
            await self.push_error(
                ErrorFrame(error=f"Error sending client event: {e}", fatal=True)
            )

    async def _interrupt(self) -> None:
        """Interrupt the current session"""
        await self._ws_send({
            "type": "agent.interrupt",
            "event_id": str(uuid.uuid4()),
        })
    
    async def _start_agent_listening(self) -> None:
        await self._ws_send({
            "type": "agent.start_listening",
            "event_id": str(uuid.uuid4()),
        })

    async def _stop_agent_listening(self) -> None:
        """Stop listening animation"""
        await self._ws_send({
            "type": "agent.stop_listening",
            "event_id": str(uuid.uuid4()),
        })

    # heygen api methods
    async def _stop_session(self) -> None:
        """Stop the current session"""
        try:
            await self._ws_disconnect()
        except Exception as e:
            logger.error(f"{self} stop ws error: {e}")
        url = f"{self._api_base_url}/v1/streaming.stop"
        headers = {
            "Content-Type": "application/json",
            "accept": "application/json",
            "Authorization": f"Bearer {self._session_token}",
        }
        body = {"session_id": self._session_id}
        async with self._session.post(url, headers=headers, json=body) as r:
            r.raise_for_status()
    def _resample_audio(self, audio: bytes, original_sample_rate: int, target_sample_rate: int) -> bytes:
        """Resample audio from original sample rate to target sample rate"""
        if original_sample_rate == target_sample_rate:
            return audio
        
        # Convert bytes to numpy array (assuming 16-bit audio)
        audio_array = np.frombuffer(audio, dtype=np.int16)
        
        # Calculate the number of samples in the resampled audio
        num_samples = int(len(audio_array) * target_sample_rate / original_sample_rate)
        
        # Resample the audio
        resampled_audio = signal.resample(audio_array, num_samples)
        
        # Convert back to bytes
        return resampled_audio.astype(np.int16).tobytes() # type: ignore

    # audio buffer methods
    async def _send_audio(
        self, audio: bytes, sample_rate: int, event_id: str, finish: bool = False
    ) -> None:
        try:
            audio = self._resample_audio(audio, sample_rate, 24000)
            self._buffered_audio_duration_ms += self._calculate_audio_duration_ms(
                audio, 24000
            )
            await self._agent_audio_buffer_append(audio, event_id)

            if finish and self._buffered_audio_duration_ms < 80:
                await self._agent_audio_buffer_clear()
                self._buffered_audio_duration_ms = 0

            if finish or self._buffered_audio_duration_ms > 1000:
                logger.info(
                    f"Audio buffer duration from buffer: {self._buffered_audio_duration_ms:.2f}ms"
                )
                await self._agent_audio_buffer_commit(event_id)
                self._buffered_audio_duration_ms = 0
        except Exception as e:
            logger.error(f"Error sending audio: {e}", exc_info=True)
            logger.error(f"Error sending audio: {e}", exc_info=True)

    def _calculate_audio_duration_ms(self, audio: bytes, sample_rate: int) -> float:
        # Each sample is 2 bytes (16-bit audio)
        num_samples = len(audio) / 2
        return (num_samples / sample_rate) * 1000

    async def _agent_audio_buffer_append(self, audio: bytes, event_id: str) -> None:
        audio_base64 = base64.b64encode(audio).decode("utf-8")
        await self._ws_send(
            {
                "type": "agent.audio_buffer_append",
                "audio": audio_base64,
                "event_id": str(uuid.uuid4()),
            }
        )

    async def _agent_audio_buffer_clear(self) -> None:
        await self._ws_send(
            {
                "type": "agent.audio_buffer_clear",
                "event_id": str(uuid.uuid4()),
            }
        )

    async def _agent_audio_buffer_commit(self, event_id: str) -> None:
        audio_base64 = base64.b64encode(b"\x00").decode("utf-8")
        await self._ws_send(
            {
                "type": "agent.audio_buffer_commit",
                "audio": audio_base64,
                "event_id": str(uuid.uuid4()),
            }
        )

    # LiveKit connection methods
    async def _process_audio_frames(self, stream: rtc.AudioStream):
        """Process audio frames from LiveKit stream"""
        frame_count = 0
        try:
            logger.info("Starting audio frame processing...")
            async for frame_event in stream:
                frame_count += 1
                try:
                    audio_frame = frame_event.frame
                    # Convert audio to raw bytes
                    audio_data = bytes(audio_frame.data)

                    # Create TTSAudioRawFrame
                    audio_frame = TTSAudioRawFrame(
                        audio=audio_data,
                        sample_rate=audio_frame.sample_rate,
                        num_channels=1,  # HeyGen uses mono audio
                    )

                    audio_frame._from_heygen = True # type: ignore
                    
                    # Push downstream for Daily output
                    await self.push_frame(audio_frame, FrameDirection.DOWNSTREAM)

                except Exception as frame_error:
                    logger.error(
                        f"Error processing audio frame #{frame_count}: {str(frame_error)}",
                        exc_info=True,
                    )
        except Exception as e:
            logger.error(
                f"Audio frame processing error after {frame_count} frames: {str(e)}",
                exc_info=True,
            )
        finally:
            logger.info(
                f"Audio frame processing ended. Total frames processed: {frame_count}"
            )

    async def _process_video_frames(self, stream: rtc.VideoStream):
        """Process video frames from LiveKit stream"""
        frame_count = 0
        try:
            logger.info("Starting video frame processing...")
            async for frame_event in stream:
                # Wait for video processing to be enabled
                await self._video_event.wait()

                frame_count += 1
                try:
                    video_frame = frame_event.frame

                    # Convert to RGB24 if not already
                    if video_frame.type != VideoBufferType.RGB24:
                        video_frame = video_frame.convert(VideoBufferType.RGB24)

                    # Create frame with original dimensions
                    image_frame = OutputImageRawFrame(
                        image=bytes(video_frame.data),
                        size=(video_frame.width, video_frame.height),
                        format="RGB",
                    )
                    image_frame.pts = (
                        frame_event.timestamp_us // 1000
                    )  # Convert to milliseconds

                    # Log every 30th frame to verify processing
                    if frame_count % 30 == 0:
                        logger.info(f"Processing video frame #{frame_count}, size: {image_frame.size}")

                    # Push downstream for Daily output
                    await self.push_frame(image_frame, FrameDirection.DOWNSTREAM)

                except Exception as frame_error:
                    logger.error(
                        f"Error processing individual frame #{frame_count}: {str(frame_error)}",
                        exc_info=True,
                    )
        except Exception as e:
            logger.error(
                f"Video frame processing error after {frame_count} frames: {str(e)}",
                exc_info=True,
            )
        finally:
            logger.info(
                f"Video frame processing ended. Total frames processed: {frame_count}"
            )

    async def _livekit_connect(self):
        """Connect to LiveKit room"""
        try:
            logger.info(
                f"HeyGenVideoService livekit connecting to room URL: {self._livekit_room_url}"
            )
            self._livekit_room = rtc.Room()

            @self._livekit_room.on("participant_connected")
            def on_participant_connected(participant: rtc.RemoteParticipant):
                logger.info(
                    f"Participant connected - SID: {participant.sid}, Identity: {participant.identity}"
                )
                for track_pub in participant.track_publications.values():
                    logger.info(
                        f"Available track - SID: {track_pub.sid}, Kind: {track_pub.kind}, Name: {track_pub.name}"
                    )

            @self._livekit_room.on("track_subscribed")
            def on_track_subscribed(
                track: rtc.Track,
                publication: rtc.RemoteTrackPublication,
                participant: rtc.RemoteParticipant,
            ):
                logger.info(
                    f"Track subscribed - SID: {publication.sid}, Kind: {track.kind}, Source: {publication.source}"
                )
                if track.kind == rtc.TrackKind.KIND_VIDEO:
                    logger.info(
                        f"Creating video stream processor for track: {publication.sid}"
                    )
                    video_stream = rtc.VideoStream(track)
                    self._video_task = self.create_task(
                        self._process_video_frames(video_stream)
                    )
                elif track.kind == rtc.TrackKind.KIND_AUDIO:
                    logger.info(
                        f"Creating audio stream processor for track: {publication.sid}"
                    )
                    audio_stream = rtc.AudioStream(track)
                    self._audio_task = self.create_task(
                        self._process_audio_frames(audio_stream)
                    )

            @self._livekit_room.on("track_unsubscribed")
            def on_track_unsubscribed(
                track: rtc.Track,
                publication: rtc.RemoteTrackPublication,
                participant: rtc.RemoteParticipant,
            ):
                logger.info(
                    f"Track unsubscribed - SID: {publication.sid}, Kind: {track.kind}"
                )

            @self._livekit_room.on("participant_disconnected")
            def on_participant_disconnected(participant: rtc.RemoteParticipant):
                logger.info(
                    f"Participant disconnected - SID: {participant.sid}, Identity: {participant.identity}"
                )

            logger.info("Attempting to connect to LiveKit room...")
            await self._livekit_room.connect(
                self._livekit_room_url, self._session_token
            )
            logger.info(
                f"Successfully connected to LiveKit room: {self._livekit_room.name}"
            )

            # Log initial room state
            logger.info(f"Room name: {self._livekit_room.name}")
            logger.info(
                f"Local participant SID: {self._livekit_room.local_participant.sid}"
            )
            logger.info(
                f"Number of remote participants: {len(self._livekit_room.remote_participants)}"
            )

            # Log existing participants and their tracks
            for participant in self._livekit_room.remote_participants.values():
                logger.info(
                    f"Existing participant - SID: {participant.sid}, Identity: {participant.identity}"
                )
                for track_pub in participant.track_publications.values():
                    logger.info(
                        f"Existing track - SID: {track_pub.sid}, Kind: {track_pub.kind}, Name: {track_pub.name}"
                    )

        except Exception as e:
            logger.error(f"LiveKit initialization error: {str(e)}", exc_info=True)
            self._livekit_room = None

    async def _livekit_disconnect(self):
        """Disconnect from LiveKit room"""
        try:
            logger.info("Starting LiveKit disconnect...")
            if self._video_task:
                logger.info("Canceling video processing task")
                await self.cancel_task(self._video_task)
                self._video_task = None
                logger.info("Video processing task cancelled successfully")

            if self._audio_task:
                logger.info("Canceling audio processing task")
                await self.cancel_task(self._audio_task)
                self._audio_task = None
                logger.info("Audio processing task cancelled successfully")

            if self._livekit_room:
                logger.info("Disconnecting from LiveKit room")
                await self._livekit_room.disconnect()
                self._livekit_room = None
                logger.info("Successfully disconnected from LiveKit room")
        except Exception as e:
            logger.error(f"LiveKit disconnect error: {str(e)}", exc_info=True)

    async def _stop(self):
        """Stop all processing and disconnect"""
        if self._video_task:
            await self.cancel_task(self._video_task)
            self._video_task = None
        if self._audio_task:
            await self.cancel_task(self._audio_task)
            self._audio_task = None

        # Stop heygen session
        logger.info(f"HeyGenVideoService stopping session")
        try:
            await self._stop_session()
        except Exception as e:
            logger.error(f"Error stopping HeyGen session: {e}", exc_info=True)

        await self._ws_disconnect()
        await self._livekit_disconnect()
        await self._stop_session()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        try:
            if isinstance(frame, UserStartedSpeakingFrame):
                await self._interrupt()
                await self._start_agent_listening()
            elif isinstance(frame, UserStoppedSpeakingFrame):
                await self._stop_agent_listening()
            if isinstance(frame, TTSStartedFrame):
                logger.info(f"HeyGenVideoService TTS started")
                await self.start_processing_metrics()
                await self.start_ttfb_metrics()
                self._event_id = str(uuid.uuid4())
                await self._agent_audio_buffer_clear()
            elif isinstance(frame, TTSAudioRawFrame):
                # Only process TTS audio going TO HeyGen, not FROM HeyGen
                if not hasattr(frame, '_from_heygen'):
                    await self._send_audio(
                        frame.audio, frame.sample_rate, self._event_id, finish=False # type: ignore
                    )
                    await self.stop_ttfb_metrics()
                # If it's from HeyGen, don't reprocess - it's already been pushed downstream
            elif isinstance(frame, TTSStoppedFrame):
                logger.info(f"HeyGenVideoService TTS stopped")
                await self._send_audio(b"\x00\x00", 24000, self._event_id, finish=True) # type: ignore
                await self.stop_processing_metrics()
                self._event_id = None
            elif isinstance(frame, (EndFrame, CancelFrame)):
                logger.info(f"HeyGenVideoService session ended")
                await self._stop()
            else:
                await self.push_frame(frame, direction)
        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
