import logging
from typing import List, Dict, Literal
from google import genai
import os

# Set tokenizers parallelism to false to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class InstructionManualGenerator:
    """Generates instruction manuals based on task goals and search results."""

    def __init__(
        self,
        gemini_api_key: str,
        task_goal: str,
        results: List[Dict],
        logger: logging.Logger,
        instruction_format: Literal["text_steps", "json_blocks"] = "text_steps",
    ):
        """
        Initialize the instruction manual generator.

        Args:
            gemini_api_key (str): Gemini API key for text generation.
            task_goal (str): The goal of the task that the manual will help accomplish.
            results (List[Dict]): The processed or filtered results that will be included in the manual.
            logger (logging.Logger): Logger instance.
            instruction_format (Literal["text_steps", "json_blocks"]): Format of the output instructions.
        """
        # Initialize Gemini API client
        self.client = genai.Client(api_key=gemini_api_key)
        # Use the correct model name for the current API version
        self.model_name = "gemini-2.5-pro-preview-03-25"


        self.task_goal = task_goal
        self.results = results
        self.logger = logger
        self.instruction_format = instruction_format

    def generate_instruction_manual(self) -> str:
        """
        Generate an instruction manual based on the task goal and search results.

        Returns:
            str: The generated instruction manual content.
        """
        # Format the search results
        results_text = self._format_results()

        # Build the prompt
        prompt = (
            f"Task Goal: {self.task_goal}\n\n"
            "Based on the following search results, generate a clear and concise instruction manual "
            "that will help accomplish this task. If the search results are not relevant to the task, "
            "indicate that and provide general guidance.\n\n"
            "Search Results:\n"
            f"{results_text}\n\n"
            "Please format the instructions as "
            f"{'step-by-step text' if self.instruction_format == 'text_steps' else 'JSON blocks'}. "
            "Focus on practical, actionable steps and include any relevant constraints or requirements "
            "from the search results."
        )

        try:
            # Use the correct API structure for the installed version
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[{"role": "user", "parts": [{"text": prompt}]}]
            )
            return response.text
        except Exception as e:
            self.logger.error(f"Error generating instruction manual: {e}")
            return f"Error: Unable to generate instruction manual. {str(e)}"

    def _format_results(self) -> str:
        """
        Format the search results into a readable string.

        Returns:
            str: Formatted search results.
        """
        formatted_results = []
        for i, result in enumerate(self.results, start=1):
            formatted_results.append(
                f"Result {i}:\n"
                f"  Section: {result.get('section', 'N/A')}\n"
                f"  Content: {result.get('content', 'N/A')}\n"
                f"  Source: {result.get('source', 'N/A')}\n"
            )
        return "\n".join(formatted_results)


# Example Usage
if __name__ == "__main__":
    # Configure logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

    # Retrieve your API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("No GEMINI_API_KEY environment variable set.")
        exit(1)

    task_goal = (
        "查詢資訊工程學系碩士班的課程中，AI代理系統之設計與開發這門課的授課教授是誰?"
    )
    results = [
        {
            "section": "Course Information",
            "content": "The course 'AI Agent System Design and Development' is taught by Professor Zhang.",
            "source": "University Course Announcement"
        },
        {
            "section": "University News",
            "content": "The university is promoting intelligent course development...",
            "source": "University News Website"
        },
        {
            "section": "Student Forum",
            "content": "Does anyone know who teaches the AI agent system course?",
            "source": "Student Forum"
        }
    ]

    manual_generator = InstructionManualGenerator(
        gemini_api_key=api_key,
        task_goal=task_goal,
        results=results,
        logger=logger
    )
    manual = manual_generator.generate_instruction_manual()
    print(manual)

