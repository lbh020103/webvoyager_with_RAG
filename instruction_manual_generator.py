from openai import OpenAI
import json
import os
from typing import Optional, List, Dict
from json.decoder import JSONDecodeError


class InstructionManualGenerator:
    def __init__(
        self,
        openai_api_key: str,
        task_goal: str,
        results: List[Dict],
        openai_org_id: Optional[str] = None,
        output_format: str = "string"
    ):
        """
        Initialize the instruction manual generator for WebVoyager tasks.

        Args:
            openai_api_key (str): OpenAI API key.
            task_goal (str): The task goal string (e.g., identifying the professor of a course).
            results (List[Dict]): A list of dictionaries containing retrieved results.
            openai_org_id (Optional[str]): OpenAI organization ID.
            output_format (str): Determines the output format of the generated instruction manual.
                - "string": Outputs a plain-text, step-by-step instruction manual.
                - "json": Outputs a structured JSON object containing relevant and irrelevant sections.
        """
        self.openai_client = OpenAI(
            api_key=openai_api_key,
            organization=openai_org_id
        )
        self.task_goal = task_goal
        self.results = results
        self.output_format = output_format

    def _generate_prompt(self):
        """
        Generates the prompt for OpenAI's GPT model based on task goal and results.
        :return: The formatted prompt string.
        """
        prompt = f"""
You are a professional technical document assistant for WebVoyager, a web browsing agent. Your task is to filter relevant information from the provided retrieval results based on the given task goal and compile it into a structured instruction manual with actionable, numbered steps to guide the agent in completing the task.

### Task Goal:
{self.task_goal}

### Retrieval Results Example:
Each result contains:
- section: (The title information)
- content: (The information retrieved)
- source: (The source of the information)

### Relevance Criteria:
- The goal is to compile an **instruction manual** that provides actionable steps to achieve the task.
- A result is **relevant** if it:
  - Contains keywords or terminology directly related to any possible approach for completing the task goal
  - Includes step-by-step instructions, procedures, or operations that could contribute to task completion
  - Describes key functions, tools, or settings that might be useful for the task
  - Contains configuration details, system behaviors, or technical information that could aid in achieving the goal
  - Provides partial but useful information, even if it only addresses one aspect of the task
  - Mentions alternative methods or approaches that could accomplish the same goal
- A result is **not relevant** if it:
  - Contains no keywords or terminology related to any approach for completing the task
  - Provides only general theoretical concepts without practical application
  - Is completely unrelated to the task goal or any of its components

### Filtering Process:
1. **Identify Relevant Information**  
   - Consider whether the retrieved content helps in accomplishing the task through ANY possible approach
   - Even if the information describes just one possible method or only a portion of a method, include it
   - If a section contains even one relevant keyword or concept related to task completion, consider it relevant

2. **Structured Output**  
   - Organize the relevant information into a step-by-step instruction manual
   - Each step must be actionable, clearly described, and numbered sequentially
   - Use action-oriented language (e.g., "Click the search button," "Type 'query' into the textbox") to ensure clarity
   - If multiple methods are available, present them as alternative approaches with clear labels (e.g., "Method 1: Step 1")
   - For irrelevant results, provide a clear explanation of why they do not contribute to the task goal

### Output Format:
Return a string containing the structured manual with numbered steps. Each step should be concise and actionable. Format as follows:
```
Task Goal: {self.task_goal}
Steps:
1. [Actionable step description]
2. [Actionable step description]
...

source: [The source of the information]
```

### Example:
For a task like "Search for the latest news on climate change":
```
Task Goal: Search for the latest news on climate change
Steps:
1. Open your web browser and navigate to www.google.com.
2. Type 'climate change latest news' into the search bar and press Enter.
3. Click on a news article from a reputable source like BBC or Reuters.
```

### Retrieval Results
{json.dumps(self.results, ensure_ascii=False, indent=2)}

Please reason step by step and ensure the manual is structured with clear, actionable steps tailored for a web browsing agent.
"""

        if self.output_format == "json":
            prompt = f"""
You are a professional technical document assistant. Your task is to filter the relevant information from the provided retrieval results based on the given task goal and compile it into an instruction manual.

### Task Goal:
{self.task_goal}

### Retrieval Results Example:
Each result contains:
- section: (The title information)
- content: (The information retrieved)
- source: (The source of the information)

### Relevance Criteria:
- The goal is to compile an **instruction manual** that provides actionable steps to achieve the task.
- A result is **relevant** if it:
  - Contains keywords or terminology directly related to any possible approach for completing the task goal
  - Includes step-by-step instructions, procedures, or operations that could contribute to task completion
  - Describes key functions, tools, or settings that might be useful for the task
  - Contains configuration details, system behaviors, or technical information that could aid in achieving the goal
  - Provides partial but useful information, even if it only addresses one aspect of the task
  - Mentions alternative methods or approaches that could accomplish the same goal
- A result is **not relevant** if it:
  - Contains no keywords or terminology related to any approach for completing the task
  - Provides only general theoretical concepts without practical application
  - Is completely unrelated to the task goal or any of its components

### Filtering Process:
1. **Identify Relevant Information**
   - Consider whether the retrieved content helps in accomplishing the task through ANY possible approach
   - Even if the information describes just one possible method or only a portion of a method, include it
   - If a section contains even one relevant keyword or concept related to task completion, consider it relevant

2. **Structured Output**
   - Format relevant results in JSON, including the title, description, and source
   - For irrelevant results, provide a clear explanation of why they do not contribute to the task goal


### Retrieval Results
{json.dumps(self.results, ensure_ascii=False, indent=2)}

### Output Format:
Please output the results in the following JSON format:
```json
{{
    "manual": [
        {{
            "title": "Relevant Title",
            "description": "Operation steps filtered and compiled based on the task goal from the retrieved content",
            "source": "Source of the information"
        }}
    ],
    "irrelevant_explanations": [
        {{
            "section": "Title of the irrelevant section",
            "reason": "Explanation of why this result is not relevant"
        }}
    ]
}}
```
"""
        return prompt

    def _call_openai(self, prompt: str) -> str:
        """
        Call OpenAI's GPT API with the provided prompt and return the response.

        Args:
            prompt (str): The generated prompt string.

        Returns:
            str: The response from OpenAI's API.
        """
        response = self.openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": "You are a professional technical document assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content

    def generate_instruction_manual(self) -> str:
        """
        Generates a structured instruction manual by filtering relevant information from the retrieval results
        based on the defined task goal.

        This method works by:
        1. Generating a prompt using the task goal and retrieved content.
        2. Sending the prompt to the OpenAI API to obtain a filtered and formatted response.
        3. Parsing the response based on the selected output format:
           - If output_format is "json", the response will be parsed and transformed into a readable manual string.
           - Otherwise, the response will be returned as-is (typically as a plain-text manual).

        Returns:
            str: A formatted string containing either:
                - A plain-text instruction manual (if output_format is not "json"), or
                - A structured list of step-by-step instructions (parsed from JSON),
                  each including the title, description, and source of relevant entries.
        """
        prompt = self._generate_prompt()
        response_text = self._call_openai(prompt)

        if self.output_format == "json":
            response_text = response_text.replace("```json", "")
            response_text = response_text.replace("```", "")
            response = json.loads(response_text)
            manual_obj = response["manual"]

            manual_str = ""
            for entry in manual_obj:
                manual_str += f"title: {entry['title']}\ndescription: {entry['description']}\nsource: {entry['source']}\n\n"
            return manual_str
        else:
            return response_text


# Example Usage
if __name__ == "__main__":
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    org_id = os.getenv("OPENAI_ORG_ID")

    task_goal = "查詢資訊工程學系碩士班的課程中，AI代理系統之設計與開發這門課的授課教授是誰?"
    results = [
        {"section": "Course Information",
         "content": "The course 'AI Agent System Design and Development' is taught by Professor Zhang.",
         "source": "University Course Announcement"},
        {"section": "University News", "content": "The university is promoting intelligent course development...",
         "source": "University News Website"},
        {"section": "Student Forum", "content": "Does anyone know who teaches the AI agent system course?",
         "source": "Student Forum"}
    ]

    # Instantiate the class and generate the manual
    manual_generator = InstructionManualGenerator(
        openai_api_key=api_key,
        openai_org_id=org_id,
        task_goal=task_goal,
        results=results
    )
    manual = manual_generator.generate_instruction_manual()

    # Print the resulting manual
    print(manual)
