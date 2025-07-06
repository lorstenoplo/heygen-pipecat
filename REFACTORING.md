# Code Refactoring Documentation

## Overview
The large `main.py` file has been refactored into a modular structure following standard software engineering practices. This improves maintainability, testability, and code organization.

## New Project Structure

```
heygen-pipecat/
├── main_refactored.py          # New clean main entry point
├── main.py                     # Original file (kept for reference)
├── services/                   # Service layer
│   ├── __init__.py
│   ├── service_factory.py      # Factory for creating services
│   ├── follow_up_service.py    # Follow-up question generation
│   └── pipeline_manager.py     # Global pipeline state management
├── actions/                    # Action handlers
│   ├── __init__.py
│   └── rtvi_actions.py         # RTVI action handlers
├── handlers/                   # Event handlers
│   ├── __init__.py
│   └── event_handlers.py       # Transport and pipeline event handlers
├── config/                     # Configuration
│   ├── __init__.py
│   └── pipeline_config.py      # Pipeline setup and configuration
└── prompts/                    # System prompts
    ├── __init__.py
    └── system_prompts.py        # AI system prompts
```

## Key Improvements

### 1. **Separation of Concerns**
- **Services**: Business logic and external service integrations
- **Actions**: RTVI action handlers and message processing
- **Handlers**: Event handling for transport and pipeline events
- **Config**: Pipeline configuration and setup
- **Prompts**: AI system prompts and instructions

### 2. **Factory Pattern**
`ServiceFactory` centralizes the creation of all pipeline services:
- Transport configuration
- Speech-to-text (Deepgram)
- Text-to-speech (ElevenLabs)
- LLM services (OpenAI)
- HeyGen video services

### 3. **State Management**
`PipelineManager` singleton handles global state:
- RTVI processor reference
- Pipeline task reference
- Context aggregator reference

### 4. **Event Handler Organization**
`EventHandlers` class consolidates all event handling:
- Client connection/disconnection
- Participant management
- Transcript processing
- Follow-up question generation

### 5. **Configuration Management**
`PipelineConfig` handles pipeline setup:
- Pipeline creation with proper component ordering
- Task configuration with parameters
- RTVI action registration

## Benefits

1. **Maintainability**: Each module has a single responsibility
2. **Testability**: Components can be unit tested in isolation
3. **Reusability**: Services and handlers can be reused across projects
4. **Readability**: Clear separation makes code easier to understand
5. **Scalability**: Easy to add new features without affecting existing code

## Usage

To use the refactored version:

```python
# Run the refactored main file
python main_refactored.py
```

## Migration Notes

- The original `main.py` is preserved for reference
- All functionality has been preserved in the refactored version
- Global variables have been replaced with proper state management
- Event handlers use proper method registration patterns

## Future Improvements

1. **Dependency Injection**: Could implement a DI container for better testability
2. **Configuration Files**: Move hardcoded values to configuration files
3. **Logging Configuration**: Centralize logging setup
4. **Error Handling**: Implement centralized error handling strategies
5. **Testing**: Add unit tests for each module
