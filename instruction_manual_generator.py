from openai import OpenAI
import json
import os


class InstructionManualGenerator:
    def __init__(self, openai_api_key, openai_org_id, task_goal, results):
        """
        Initialize the generator with API key, organization ID, task goal, and results.

        Args:
            openai_api_key (str): OpenAI API key.
            openai_org_id (str): OpenAI organization ID.
            task_goal (str): The task goal string (e.g., identifying the professor of a course).
            results (List[Dict]): A list of dictionaries containing retrieved results.
        """
        self.openai_client = OpenAI(
            api_key=openai_api_key,
            organization=openai_org_id
        )
        self.task_goal = task_goal
        self.results = results

    def _generate_prompt(self):
        """
        Generates the prompt for OpenAI's GPT model based on task goal and results.
        :return: The formatted prompt string.
        """
        # prompt = f"""
        # You are a professional technical document assistant. Your task is to filter the relevant information from the provided retrieval results based on the given task goal and compile it into an instruction manual.
        #
        # ### Task Goal:
        # {self.task_goal}
        #
        # ### Retrieval Results Example:
        # Each result contains:
        # - section: (The title information)
        # - content: (The information retrieved)
        # - source: (The source of the information)
        #
        # Please filter the results based on relevance to the task goal and output the relevant details in a structured format, including the title, description, and source:
        # {json.dumps(self.results, ensure_ascii=False, indent=2)}
        #
        # ### Output Format:
        # Please output the results in JSON format as shown below:
        # ```json
        # {{
        #     "manual": [
        #         {{
        #             "title": "Relevant Title",
        #             "description": "Detailed operation steps or findings",
        #             "source": "Source of the information"
        #         }},
        #         ...
        #     ]
        # }}
        # ```
        # """

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
                        "description": "Detailed operation steps or findings",
                        "source": "Source of the information"
                    }}
                    ...
                ],
                "irrelevant_explanations": [
                    {{
                        "section": "Title of the irrelevant section",
                        "reason": "Explanation of why this result is not relevant"
                    }}
                    ...
                ]
            }}
            ```
            """
        return prompt

    def _call_openai(self, prompt):
        """
        Call OpenAI's GPT API with the provided prompt and return the response.

        Args:
            prompt (str): The generated prompt string.

        Returns:
            Dict: The response from OpenAI's API.
        """
        response = self.openai_client.chat.completions.create(
            model="gpt-4-turbo",
            # model="gpt-4o",
            messages=[{"role": "system", "content": "You are a professional technical document assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.3
        )

        return response.choices[0].message.content

    def generate_instruction_manual(self):
        """
        Generate an instruction manual by filtering relevant results based on the task goal.

        This method generates an instruction manual by extracting JSON data from the OpenAI API response,
        and concatenates each entry's title, description, and source in order to form a string.

        Returns:
            str: A string containing the manual, with each entry's title, description, and source.
        """
        prompt = self._generate_prompt()
        response_text = self._call_openai(prompt)
        response_text = response_text.replace("```json", "")
        response_text = response_text.replace("```", "")
        response = json.loads(response_text)
        manual_obj = response["manual"]
        # print(json.dumps(manual_obj, indent=2, ensure_ascii=False))

        manual_str = ""
        for entry in manual_obj:
            manual_str += f"title: {entry['title']}\ndescription: {entry['description']}\nsource: {entry['source']}\n\n"

        return manual_str


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
