"""
Pipeline configuration and setup with integrated message service.
"""

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frameworks.rtvi import RTVIObserver
from services.pipeline_manager import PipelineManager
from services.message_service import get_message_service
from actions.rtvi_actions import create_rtvi_actions


class PipelineConfig:
    """Pipeline configuration and setup."""
    
    @staticmethod
    def create_pipeline(
        transport,
        rtvi,
        stt,
        transcript,
        context_aggregator,
        llm,
        tts,
        heygen_video_service
    ) -> Pipeline:
        """Create the main processing pipeline."""
        return Pipeline([
            transport.input(),                    # Input from transport
            rtvi,                                # RTVI processor
            stt,                                 # Speech to text
            transcript.user(),                   # User transcript
            context_aggregator.user(),           # Add user message + vision to context
            llm,                                # LLM processes text + vision
            tts,                                # Text to speech
            transcript.assistant(),              # Assistant transcript
            context_aggregator.assistant(),      # Add assistant response to context
            heygen_video_service,               # Video output
            transport.output(),                 # Output to transport
        ])
    
    @staticmethod
    def create_task(pipeline: Pipeline, rtvi) -> PipelineTask:
        """Create pipeline task with observers."""
        return PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_usage_metrics=True,
            ),
            observers=[RTVIObserver(rtvi)],
        )
    
    @staticmethod
    def setup_rtvi_actions(rtvi_processor):
        """Setup RTVI actions."""
        actions = create_rtvi_actions()
        for action in actions:
            rtvi_processor.register_action(action)
    
    @staticmethod
    def setup_pipeline_manager(rtvi_processor, task, context_aggregator):
        """Setup the global pipeline manager and message service."""
        pipeline_manager = PipelineManager.get_instance()
        pipeline_manager.set_rtvi_processor(rtvi_processor)
        pipeline_manager.set_task(task)
        pipeline_manager.set_context_aggregator(context_aggregator)
        
        # Initialize message service with RTVI processor
        message_service = get_message_service()
        message_service.set_rtvi_processor(rtvi_processor)
