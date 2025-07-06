"""
Pipeline manager for handling global pipeline state.
"""

from typing import Optional
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIProcessor


class PipelineManager:
    """Singleton class to manage pipeline components and global state."""
    
    _instance: Optional['PipelineManager'] = None
    
    def __init__(self):
        self.rtvi_processor: Optional[RTVIProcessor] = None
        self.task: Optional[PipelineTask] = None
        self.context_aggregator = None
        
    @classmethod
    def get_instance(cls) -> 'PipelineManager':
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def set_rtvi_processor(self, processor: RTVIProcessor):
        """Set the RTVI processor."""
        self.rtvi_processor = processor
    
    def get_rtvi_processor(self) -> Optional[RTVIProcessor]:
        """Get the RTVI processor."""
        return self.rtvi_processor
    
    def set_task(self, task: PipelineTask):
        """Set the pipeline task."""
        self.task = task
    
    def get_task(self) -> Optional[PipelineTask]:
        """Get the pipeline task."""
        return self.task
    
    def set_context_aggregator(self, aggregator):
        """Set the context aggregator."""
        self.context_aggregator = aggregator
    
    def get_context_aggregator(self):
        """Get the context aggregator."""
        return self.context_aggregator
