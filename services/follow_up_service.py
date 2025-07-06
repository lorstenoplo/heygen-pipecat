"""
Follow-up question generation service using OpenAI API.
"""

import os
import openai
from loguru import logger
from pydantic import BaseModel
from typing import List


class FollowUpQuestions(BaseModel):
    questions: List[str]


class OpenAIFollowUpProcessor:
    """Follow-up question generator using OpenAI API with Pydantic."""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = "gpt-4o"
        self.context = []

    async def generate_follow_ups(self, assistant_response: str) -> List[str]:
        """Generate follow-up questions using OpenAI with structured output."""
        self.context.append(assistant_response)
        if len(self.context) > 5:
            self.context.pop(0)

        try:
            prompt = f"""Based on this AI assistant response, generate 2-3 short, relevant follow-up questions that users might naturally ask next.

            Assistant Response: "{assistant_response}"
            Context: {self.context}

            Generate natural, conversational questions that would logically follow from this response."""

            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=FollowUpQuestions,
                temperature=0.7
            )

            questions_obj = response.choices[0].message.parsed
            if questions_obj and questions_obj.questions and len(questions_obj.questions) >= 2:
                return questions_obj.questions[:3]
            
            # Fallback
            return ["Tell me more about that", "What else should I know?"]

        except Exception as e:
            logger.error(f"Error generating follow-ups: {e}")
            return ["Can you explain further?", "What's next?"]
