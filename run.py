import platform
import argparse
import time
import json
import re
import os
import shutil
import logging
import base64

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

from prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_TEXT_ONLY  # Keep your existing prompts
from google import genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from utils import get_web_element_rect, encode_image, extract_information, print_message, \
    get_webarena_accessibility_tree, get_pdf_retrieval_ans_from_assistant, clip_message_and_obs, \
    clip_message_and_obs_text_only
from pdf_rag import PDFEnhancementPipeline
from instruction_manual_generator import InstructionManualGenerator
from typing import List, Dict, Optional, Any, Literal
from datetime import datetime


def setup_logger(folder_path):
    log_file_path = os.path.join(folder_path, 'agent.log')

    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(log_file_path, encoding='utf-8')
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger


def driver_config(args):
    options = webdriver.ChromeOptions()

    if args.save_accessibility_tree:
        args.force_device_scale = True

    if args.force_device_scale:
        options.add_argument("--force-device-scale-factor=1")
    if args.headless:
        options.add_argument("--headless")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
    options.add_experimental_option(
        "prefs", {
            "download.default_directory": args.download_dir,
            "plugins.always_open_pdf_externally": True
        }
    )

    options.add_argument("disable-blink-features=AutomationControlled")
    return options


def format_msg_for_gemini(it, init_msg, pdf_obs, warn_obs, web_img_b64, web_text):
    """Format messages for Gemini API"""
    if it == 1:
        init_msg += f"I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"
        
        # Add reminder about following instructions
        init_msg += "\n\nREMINDER: Follow the step-by-step instructions in the manual above. Indicate which step you are following with each action."
        
        gemini_msg = {
            'contents': [
                {'role': 'user', 
                 'parts': [
                    {'text': init_msg},
                    {'inline_data': {'mime_type': 'image/jpeg', 'data': web_img_b64}}
                 ]
                }
            ]
        }
        return gemini_msg
    else:
        # Extract the manual from init_msg to include in subsequent messages
        manual_start = init_msg.find("[Manuals and QA pairs]\n")
        manual_end = init_msg.find("\n\nIMPORTANT:")
        
        if manual_start != -1 and manual_end != -1:
            manual_text = init_msg[manual_start:manual_end]
        else:
            manual_text = "[Manuals and QA pairs] (Not found in initial message)"
        
        if not pdf_obs:
            msg_text = f"Observation:{warn_obs} please analyze the attached screenshot and give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"
            
            # Add the manual and reminder
            msg_text += f"\n\n{manual_text}\n\nREMINDER: Continue following the step-by-step instructions in the manual above. Indicate which step you are following with each action."
            
            # Add dropdown menu reminder if this is iteration 3 or more (likely struggling with dropdowns)
            if it >= 3:
                msg_text += "\n\nDROPDOWN MENU TIPS: If you're having trouble with dropdown menus, try these approaches:\n1. Click the dropdown again to fully expand it\n2. Look for the option as a separate element with its own number\n3. If the option isn't visible as a separate element, try clicking where the text of the option would be within the dropdown\n4. Try a different approach to achieve the same goal"
            
            # Add date input reminder if the agent is likely working with dates
            if any(date_term in web_text.lower() for date_term in ["date", "year", "month", "day", "from", "to", "2024"]):
                msg_text += "\n\nDATE INPUT TIPS: For date inputs, try these approaches:\n1. If there are separate fields for year, month, day, fill them one by one\n2. If there's a single field, try the format YYYY-MM-DD (e.g., 2024-01-01)\n3. After entering a date, check if the page automatically submits the form - if so, you'll need to go back and complete all fields before submitting\n4. If the date picker is complex, look for text input alternatives"
        else:
            msg_text = f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The screenshot of the current page is also attached, give the Thought and Action. I've provided the tag name of each element and the text it contains (if text exists). Note that <textarea> or <input> may be textbox, but not exactly. Please focus more on the screenshot and then refer to the textual information.\n{web_text}"
            
            # Add the manual and reminder
            msg_text += f"\n\n{manual_text}\n\nREMINDER: Continue following the step-by-step instructions in the manual above. Indicate which step you are following with each action."
        
        gemini_msg = {
            'contents': [
                {'role': 'user',
                 'parts': [
                    {'text': msg_text},
                    {'inline_data': {'mime_type': 'image/jpeg', 'data': web_img_b64}}
                 ]
                }
            ]
        }
        return gemini_msg


def format_msg_text_only_for_gemini(it, init_msg, pdf_obs, warn_obs, ac_tree):
    """Format text-only messages for Gemini API"""
    if it == 1:
        # Add reminder about following instructions
        init_msg += "\n\nREMINDER: Follow the step-by-step instructions in the manual above. Indicate which step you are following with each action."
        
        gemini_msg = {
            'contents': [
                {'role': 'user',
                 'parts': [
                    {'text': init_msg + '\n' + ac_tree}
                 ]
                }
            ]
        }
        return gemini_msg
    else:
        # Extract the manual from init_msg to include in subsequent messages
        manual_start = init_msg.find("[Manuals and QA pairs]\n")
        manual_end = init_msg.find("\n\nIMPORTANT:")
        
        if manual_start != -1 and manual_end != -1:
            manual_text = init_msg[manual_start:manual_end]
        else:
            manual_text = "[Manuals and QA pairs] (Not found in initial message)"
        
        if not pdf_obs:
            msg_text = f"Observation:{warn_obs} please analyze the accessibility tree and give the Thought and Action.\n{ac_tree}"
            
            # Add the manual and reminder
            msg_text += f"\n\n{manual_text}\n\nREMINDER: Continue following the step-by-step instructions in the manual above. Indicate which step you are following with each action."
            
            # Add dropdown menu reminder if this is iteration 3 or more (likely struggling with dropdowns)
            if it >= 3:
                msg_text += "\n\nDROPDOWN MENU TIPS: If you're having trouble with dropdown menus, try these approaches:\n1. Click the dropdown again to fully expand it\n2. Look for the option as a separate element with its own number\n3. If the option isn't visible as a separate element, try clicking where the text of the option would be within the dropdown\n4. Try a different approach to achieve the same goal"
            
            # Add date input reminder if the agent is likely working with dates
            if any(date_term in ac_tree.lower() for date_term in ["date", "year", "month", "day", "from", "to", "2024"]):
                msg_text += "\n\nDATE INPUT TIPS: For date inputs, try these approaches:\n1. If there are separate fields for year, month, day, fill them one by one\n2. If there's a single field, try the format YYYY-MM-DD (e.g., 2024-01-01)\n3. After entering a date, check if the page automatically submits the form - if so, you'll need to go back and complete all fields before submitting\n4. If the date picker is complex, look for text input alternatives"
        else:
            msg_text = f"Observation: {pdf_obs} Please analyze the response given by Assistant, then consider whether to continue iterating or not. The accessibility tree of the current page is also given, give the Thought and Action.\n{ac_tree}"
            
            # Add the manual and reminder
            msg_text += f"\n\n{manual_text}\n\nREMINDER: Continue following the step-by-step instructions in the manual above. Indicate which step you are following with each action."
        
        gemini_msg = {
            'contents': [
                {'role': 'user',
                 'parts': [
                    {'text': msg_text}
                 ]
                }
            ]
        }
        return gemini_msg


def call_gemini_api(args, gemini_client, messages, conversation_history=None):
    """Call the Gemini API with proper error handling"""
    retry_times = 0
    while True:
        try:
            logging.info('Calling Gemini API...')
            
            # Extract parts from the message format
            user_message = messages['contents'][0]
            
            # Add system prompt if needed
            system_instructions = SYSTEM_PROMPT if not args.text_only else SYSTEM_PROMPT_TEXT_ONLY
            
            # Add emphasis to the manual following part of the system instructions
            system_instructions = system_instructions.replace(
                "* Manual Following Guidelines *",
                "*** CRITICAL: MANUAL FOLLOWING GUIDELINES ***"
            )
            
            # Initialize or continue conversation
            if conversation_history is None:
                # Create a new conversation with system instructions as part of the first user message
                first_message = {
                    "role": "user",
                    "parts": []
                }
                
                # Add system instructions and user text
                first_message["parts"].append({
                    "text": f"{system_instructions}\n\n{user_message['parts'][0]['text']}"
                })
                
                # Add image if present
                if len(user_message['parts']) > 1 and 'inline_data' in user_message['parts'][1]:
                    image_data = user_message['parts'][1]['inline_data']
                    first_message["parts"].append({
                        "inline_data": {
                            "mime_type": image_data['mime_type'],
                            "data": image_data['data']
                        }
                    })
                
                # Create a chat session
                response = gemini_client.models.generate_content(
                    model="gemini-2.5-pro-preview-03-25",
                    contents=[first_message]
                )
                conversation = {"history": [first_message, {"role": "model", "parts": [{"text": response.text}]}]}
            else:
                conversation = conversation_history
            
            # Format the current message for sending
            current_message = {
                "role": "user",
                "parts": []
            }
            
            for part in user_message['parts']:
                if 'text' in part:
                    # a simple text part
                    current_message["parts"].append({"text": part['text']})
                elif 'inline_data' in part:
                    # a valid PartDict
                    current_message["parts"].append({
                        "inline_data": {
                            "mime_type": part['inline_data']['mime_type'],
                            "data": part['inline_data']['data']
                        }
                    })
            
            # Add the current message to the conversation history
            conversation["history"].append(current_message)
            
            # Send the message to the model
            response = gemini_client.models.generate_content(
                model="gemini-2.5-pro-preview-03-25",
                contents=conversation["history"]
            )
            
            # Add the model's response to the conversation history
            conversation["history"].append({"role": "model", "parts": [{"text": response.text}]})
            
            # Estimate token usage (since Gemini doesn't provide this directly)
            # This is a rough estimate based on word count - adjust as needed
            message_text = ' '.join([part.get('text', '') for part in user_message['parts'] if 'text' in part])
            prompt_tokens = len(message_text.split()) * 1.3  # Rough approximation
            completion_tokens = len(response.text.split()) * 1.3  # Rough approximation
            
            logging.info(f'Estimated Prompt Tokens: {int(prompt_tokens)}; Estimated Completion Tokens: {int(completion_tokens)}')
            
            gpt_call_error = False
            return int(prompt_tokens), int(completion_tokens), gpt_call_error, response, conversation

        except Exception as e:
            logging.info(f'Error occurred, retrying. Error type: {type(e).__name__}')
            logging.info(f'Error details: {str(e)}')
            
            # Handle specific error types
            if "rate limit" in str(e).lower():
                time.sleep(10)
            elif "server error" in str(e).lower():
                time.sleep(15)
            elif "invalid request" in str(e).lower():
                gpt_call_error = True
                return None, None, gpt_call_error, None, None
            else:
                time.sleep(5)
                
            retry_times += 1
            if retry_times == 10:
                logging.info('Retrying too many times')
                return None, None, True, None, None


def exec_action_click(info, web_ele, driver_task):
    driver_task.execute_script("arguments[0].setAttribute('target', '_self')", web_ele)
    web_ele.click()
    time.sleep(3)


def exec_action_type(info, web_ele, driver_task):
    warn_obs = ""
    type_content = info['content']

    ele_tag_name = web_ele.tag_name.lower()
    ele_type = web_ele.get_attribute("type")
    # outer_html = web_ele.get_attribute("outerHTML")
    if (ele_tag_name != 'input' and ele_tag_name != 'textarea') or (
            ele_tag_name == 'input' and ele_type not in ['text', 'search', 'password', 'email', 'tel']):
        warn_obs = f"note: The web element you're trying to type may not be a textbox, and its tag name is <{web_ele.tag_name}>, type is {ele_type}."
    try:
        # Not always work to delete
        web_ele.clear()
        # Another way to delete
        if platform.system() == 'Darwin':
            web_ele.send_keys(Keys.COMMAND + "a")
        else:
            web_ele.send_keys(Keys.CONTROL + "a")
        web_ele.send_keys(" ")
        web_ele.send_keys(Keys.BACKSPACE)
    except:
        pass

    actions = ActionChains(driver_task)
    actions.click(web_ele).perform()
    actions.pause(1)

    try:
        driver_task.execute_script(
            """window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea' && e.target.type != 'search') {e.preventDefault();}};""")
    except:
        pass

    actions.send_keys(type_content)
    actions.pause(2)

    actions.send_keys(Keys.ENTER)
    actions.perform()
    time.sleep(10)
    return warn_obs


def exec_action_scroll(info, web_eles, driver_task, args, obs_info):
    scroll_ele_number = info['number']
    scroll_content = info['content']
    if scroll_ele_number == "WINDOW":
        if scroll_content == 'down':
            driver_task.execute_script(f"window.scrollBy(0, {args.window_height * 2 // 3});")
        else:
            driver_task.execute_script(f"window.scrollBy(0, {-args.window_height * 2 // 3});")
    else:
        if not args.text_only:
            scroll_ele_number = int(scroll_ele_number)
            web_ele = web_eles[scroll_ele_number]
        else:
            element_box = obs_info[scroll_ele_number]['union_bound']
            element_box_center = (element_box[0] + element_box[2] // 2, element_box[1] + element_box[3] // 2)
            web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);",
                                                 element_box_center[0], element_box_center[1])
        actions = ActionChains(driver_task)
        driver_task.execute_script("arguments[0].focus();", web_ele)
        if scroll_content == 'down':
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_DOWN).key_up(Keys.ALT).perform()
        else:
            actions.key_down(Keys.ALT).send_keys(Keys.ARROW_UP).key_up(Keys.ALT).perform()
    time.sleep(3)


def get_pdf_retrieval_ans_from_gemini(gemini_client, pdf_path, query):
    """Use Gemini to extract information from PDF instead of OpenAI"""
    try:
        with open(pdf_path, "rb") as file:
            pdf_data = file.read()
            pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
        
        # Create prompt for PDF analysis
        prompt = f"""Please analyze this PDF and answer the following query:
        
        {query}
        
        Provide a detailed and accurate answer based solely on the PDF content."""
        
        # Configure and call Gemini with PDF content
        response = gemini_client.models.generate_content(
            model='gemini-2.5-pro-preview-03-25',
            contents=[
                {"role": "user", "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "application/pdf", "data": pdf_base64}}
                ]}
            ]
        )
        
        return response.text
    except Exception as e:
        logging.error(f"Error in PDF analysis: {str(e)}")
        return f"Error analyzing PDF: {str(e)}"


def index_pdf(
        pdf_path: str,
        output_dir: str,
        api_key: str,
        logger: logging.Logger,
        persist_directory: str = "./chroma_db",
        org_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Indexes a PDF and converts it to Markdown.
    """
    # Initialize the pipeline
    pipeline = PDFEnhancementPipeline(
        gemini_api_key=api_key,
        logger=logger,
        persist_directory=persist_directory
    )

    logger.info(f"Starting to process {pdf_path}...")
    result = pipeline.process_pdf(
        pdf_path=pdf_path,
        output_dir=output_dir,
        add_image_descriptions=True,
        index_for_rag=True,
        overwrite_enhanced_md=False
    )

    logger.info("Processing completed:")
    logger.info(f"- Original PDF: {result['original_pdf']}")
    logger.info(f"- Markdown file: {result['markdown_path']}")
    logger.info(f"- Number of processed images: {result['image_count']}")
    if 'enhanced_markdown_path' in result:
        logger.info(f"- Enhanced Markdown: {result['enhanced_markdown_path']}")

    return result


def search_rag(
        query: str,
        api_key: str,
        logger: logging.Logger,
        persist_directory: str = "./chroma_db",
        k: int = 20,
        org_id: Optional[str] = None
) -> List[Dict]:
    """
    Performs a search on the indexed data.
    """
    # Initialize the pipeline
    pipeline = PDFEnhancementPipeline(
        gemini_api_key=api_key,
        logger=logger,
        persist_directory=persist_directory
    )

    logger.info(f"Searching for: {query}")
    results = pipeline.search(query=query, k=k)
    
    # Process the results to ensure they have consistent keys
    filtered_results = []
    for d in results:
        # Create a standardized result entry with proper fallbacks
        entry = {
            "content": d.get("page_content", d.get("content", "No content available")),
            "source": d.get("source", "Unknown"),
            "section": d.get("section", "N/A"),
            "page": d.get("page", "N/A")
        }
        filtered_results.append(entry)
    
    # Format results for logging
    results_str = ""
    for entry in filtered_results:
        results_str += f"section: {entry.get('section', 'N/A')}\ncontent: {entry.get('content', 'No content available')}\nsource: {entry.get('source', 'Unknown')}\n\n"
    
    logger.info(f"Searching results:\n {results_str}")
    return filtered_results


def generate_instruction_manual_with_gemini(
        gemini_client,
        task_goal: str,
        filtered_results: List[Dict],
        logger: logging.Logger,
) -> str:
    """
    Generates an instruction manual using Gemini API based on filtered results.
    """
    # Create a prompt for the instruction manual generation
    context_text = "\n\n".join([
        f"Section: {result.get('section', 'N/A')}\nContent: {result.get('content', '')}"
        for result in filtered_results
    ])
    
    prompt = f"""Task Goal: {task_goal}

Context Information:
{context_text}

Please create a comprehensive, step-by-step instruction manual for achieving the task goal.
The manual should:
1. Be well-structured with clear, numbered steps (Step 1, Step 2, etc.)
2. Incorporate all relevant information from the context provided
3. Be concise yet thorough
4. Include any necessary cautions or best practices
5. Focus on EXACTLY how to accomplish the task with the website mentioned in the task goal

IMPORTANT FORMATTING REQUIREMENTS:
- Format each step as "Step X: [Clear action instruction]"
- Make each step a single, clear action that can be followed
- Include visual cues like "Look for a button labeled X" or "Find the dropdown menu"
- For UI interactions, be EXTREMELY specific about what to click, what to type, and in what order

DROPDOWN MENU INTERACTION INSTRUCTIONS:
- For dropdown menus, be extremely specific about how to interact with them
- First specify clicking on the dropdown to open it
- Then in a SEPARATE step, specify clicking on the specific option you want to select
- If the dropdown options are not visible as separate elements, provide instructions to try clicking on the visible text of the option within the dropdown element
- If the dropdown is still not working, suggest alternative approaches like using Tab key to navigate to the option

Your instruction manual should be so clear that anyone could follow it without additional guidance.
"""

    try:
        # Generate the instruction manual using Gemini
        response = gemini_client.models.generate_content(
            model='gemini-2.5-pro-preview-03-25',
            contents=[{"role": "user", "parts": [{"text": prompt}]}]
        )
        
        # Add a header to make the instructions more prominent
        manual = "## STEP-BY-STEP INSTRUCTIONS\n\n" + response.text
        
        # Add a footer with a reminder about dropdown menus
        manual += "\n\n## IMPORTANT REMINDER\nFollow the steps above IN ORDER. Do not skip steps or create your own approach. For each UI interaction:\n1. First identify the exact element to interact with\n2. Then perform the action (click, type, etc.)\n3. Confirm the result before moving to the next step\n\nFor dropdown menus:\n- First click on the dropdown to open it\n- Then look for the option you need to select as a separate element\n- If the option is not visible as a separate element, try clicking on the text of the option within the dropdown element"
        
        return manual
    except Exception as e:
        logger.error(f"Error generating instruction manual: {str(e)}")
        return f"Error generating manual: {str(e)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='data/test.json')
    parser.add_argument('--max_iter', type=int, default=25)
    parser.add_argument("--api_key", default="key", type=str, help="YOUR_GOOGLE_API_KEY")
    parser.add_argument("--openai_api_key", default=None, type=str, help="YOUR_OPENAI_API_KEY (for PDF indexing)")
    parser.add_argument("--api_organization_id", default=None, type=str, help="YOUR_OPENAI_ORGANIZATION_ID")
    parser.add_argument("--api_model", default="gemini-1.5-pro-latest", type=str, help="Gemini model name")
    parser.add_argument("--output_dir", type=str, default='results')
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_attached_imgs", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--download_dir", type=str, default="downloads")
    parser.add_argument("--text_only", action='store_true')
    # for web browser
    parser.add_argument("--headless", action='store_true', help='The window of selenium')
    parser.add_argument("--save_accessibility_tree", action='store_true')
    parser.add_argument("--force_device_scale", action='store_true')
    parser.add_argument("--window_width", type=int, default=1024)
    parser.add_argument("--window_height", type=int, default=768)  # for headless mode, there is no address bar
    parser.add_argument("--fix_box_color", action='store_true')

    parser.add_argument("--pdf_path", type=str, default='data/arXiv.pdf')
    args = parser.parse_args()

    # Configure Google Generative AI
    genai_client = genai.Client(api_key=args.api_key)

    options = driver_config(args)

    # Save Result file
    current_time = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    result_dir = os.path.join(args.output_dir, current_time)
    os.makedirs(result_dir, exist_ok=True)

    # Load tasks
    tasks = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append(json.loads(line))

    init_dir = os.path.join(result_dir, 'init')
    os.makedirs(init_dir, exist_ok=True)
    init_logger = setup_logger(init_dir)
    markdown_output_dir = "output"
    
    # For PDF indexing, we might still need OpenAI temporarily
    # (or adapt PDFEnhancementPipeline to use Gemini)
    if args.openai_api_key:
        openai_key = args.openai_api_key
    else:
        # Fallback to using the same key for everything
        openai_key = args.api_key
    
    index_pdf(pdf_path=args.pdf_path, output_dir=markdown_output_dir, api_key=openai_key,
              logger=init_logger, org_id=args.api_organization_id)

    for task_id in range(len(tasks)):
        task = tasks[task_id]
        task_dir = os.path.join(result_dir, 'task{}'.format(task["id"]))
        os.makedirs(task_dir, exist_ok=True)
        task_logger = setup_logger(task_dir)
        logging.info(f'########## TASK{task["id"]} ##########')

        driver_task = webdriver.Chrome(options=options)

        # About window size, 765 tokens
        # You can resize to height = 512 by yourself (255 tokens, Maybe bad performance)
        driver_task.set_window_size(args.window_width,
                                    args.window_height)  # larger height may contain more web information
        driver_task.get(task['web'])
        try:
            driver_task.find_element(By.TAG_NAME, 'body').click()
        except:
            pass
        # sometimes enter SPACE, the page will sroll down
        driver_task.execute_script(
            """window.onkeydown = function(e) {if(e.keyCode == 32 && e.target.type != 'text' && e.target.type != 'textarea') {e.preventDefault();}};""")
        time.sleep(5)

        # We only deal with PDF file
        for filename in os.listdir(args.download_dir):
            file_path = os.path.join(args.download_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        download_files = []  # sorted(os.listdir(args.download_dir))

        fail_obs = ""  # When error execute the action
        pdf_obs = ""  # When download PDF file
        warn_obs = ""  # Type warning
        pattern = r'Thought:|Action:|Observation:'

        # No need to maintain messages list in the OpenAI format
        # Instead, we'll use Gemini's conversation
        conversation = None
        
        obs_prompt = "Observation: please analyze the attached screenshot and give the Thought and Action. "
        if args.text_only:
            obs_prompt = "Observation: please analyze the accessibility tree and give the Thought and Action."

        # Search RAG and generate manual using either OpenAI (temporary) or Gemini
        rag_results = search_rag(query=task['ques'], api_key=openai_key,
                               logger=task_logger, org_id=args.api_organization_id)
        
        # Use Gemini for manual generation
        manual = generate_instruction_manual_with_gemini(
            gemini_client=genai_client,
            task_goal=task['ques'], 
            filtered_results=rag_results, 
            logger=task_logger
        )
        
        logging.info(f"manual:\n {manual}")

        today_date = datetime.today().strftime('%Y-%m-%d')
        init_msg = f"""Today is {today_date}. Now given a task: {task['ques']}  Please interact with https://www.example.com and get the answer. \n"""
        init_msg = init_msg.replace('https://www.example.com', task['web'])
        init_msg += """Before taking action, carefully analyze the contents in [Manuals and QA pairs] below.
        Determine whether [Manuals and QA pairs] contain relevant procedures, constraints, or guidelines that should be followed for this task.
        If so, follow their guidance accordingly. If not, proceed with a logical and complete approach.\n"""

        init_msg += f"""[Key Guidelines You MUST follow]
        Before taking any action, analyze the provided [Manuals and QA pairs] as a whole to determine if they contain useful procedures, constraints, 
        or guidelines relevant to this task.
        - If [Manuals and QA pairs] provide comprehensive guidance, strictly follow their instructions in an ordered and structured manner.
        - If [Manuals and QA pairs] contain partial but useful information, integrate it into your approach while filling in the gaps logically.
        - If [Manuals and QA pairs] are entirely irrelevant or insufficient, proceed with the best available method while ensuring completeness.\n
        [Manuals and QA pairs]
        {manual}\n"""

        # Add stronger emphasis on following the instructions step by step
        init_msg += """
        IMPORTANT: You MUST follow the step-by-step instructions provided in the manual above. 
        Do not skip steps or create your own approach unless absolutely necessary.
        For each step you take, explicitly mention which step from the manual you are following.
        If the manual contains numbered steps, follow them in order from Step 1 to completion.

        SPECIFIC UI INTERACTION GUIDELINES:
        1. For dropdown menus: 
        - First click on the dropdown to open it
        - In the next step, look for the option as a separate element with its own numerical label
        - If the option doesn't appear as a separate element, try clicking on the dropdown again
        - If that doesn't work, try to locate the text of the option within the dropdown element and click there
        - As a last resort, try using keyboard navigation (Tab key) to navigate to the option

        2. For date inputs: Be precise about where to click and what format to use.
        3. For search fields: Type the exact search terms specified in the manual.
        4. For navigation: Follow the exact path described in the manual.

        TROUBLESHOOTING TIPS:
        - If you're stuck on a dropdown menu, try clicking it multiple times or look for alternative ways to select the option
        - If a UI element doesn't respond as expected, try a different approach or look for an alternative path
        - Always check if the expected result occurred after each action before proceeding
        """
        
        init_msg = init_msg + obs_prompt

        it = 0
        accumulate_prompt_token = 0
        accumulate_completion_token = 0

        while it < args.max_iter:
            logging.info(f'Iter: {it}')
            it += 1
            if not fail_obs:
                try:
                    if not args.text_only:
                        rects, web_eles, web_eles_text = get_web_element_rect(driver_task, fix_color=args.fix_box_color)
                    else:
                        accessibility_tree_path = os.path.join(task_dir, 'accessibility_tree{}'.format(it))
                        ac_tree, obs_info = get_webarena_accessibility_tree(driver_task, accessibility_tree_path)

                except Exception as e:
                    if not args.text_only:
                        logging.error('Driver error when adding set-of-mark.')
                    else:
                        logging.error('Driver error when obtaining accessibility tree.')
                    logging.error(e)
                    break

                img_path = os.path.join(task_dir, 'screenshot{}.png'.format(it))
                driver_task.save_screenshot(img_path)

                # accessibility tree
                if (not args.text_only) and args.save_accessibility_tree:
                    accessibility_tree_path = os.path.join(task_dir, 'accessibility_tree{}'.format(it))
                    get_webarena_accessibility_tree(driver_task, accessibility_tree_path)

                # encode image
                b64_img = encode_image(img_path)

                # format msg for Gemini
                if not args.text_only:
                    curr_msg = format_msg_for_gemini(it, init_msg, pdf_obs, warn_obs, b64_img, web_eles_text)
                else:
                    curr_msg = format_msg_text_only_for_gemini(it, init_msg, pdf_obs, warn_obs, ac_tree)
            else:
                curr_msg = {
                    'contents': [
                        {'role': 'user',
                         'parts': [
                             {'text': fail_obs}
                         ]
                        }
                    ]
                }

            # Call Gemini API
            prompt_tokens, completion_tokens, api_call_error, response, conversation = call_gemini_api(
                args, genai_client, curr_msg, conversation
            )

            if api_call_error:
                break
            else:
                accumulate_prompt_token += prompt_tokens
                accumulate_completion_token += completion_tokens
                logging.info(
                    f'Accumulate Prompt Tokens: {accumulate_prompt_token}; Accumulate Completion Tokens: {accumulate_completion_token}')
                logging.info('API call complete...')
            
            gemini_response = response.text
            logging.info(f"Gemini response: {gemini_response}")
            
            # remove the rects on the website
            if (not args.text_only) and rects:
                logging.info(f"Num of interactive elements: {len(rects)}")
                for rect_ele in rects:
                    driver_task.execute_script("arguments[0].remove()", rect_ele)
                rects = []

            # extract action info
            try:
                assert 'Thought:' in gemini_response and 'Action:' in gemini_response
            except AssertionError as e:
                logging.error(e)
                fail_obs = "Format ERROR: Both 'Thought' and 'Action' should be included in your reply."
                continue

            chosen_action = re.split(pattern, gemini_response)[2].strip()
            # print(chosen_action)
            action_key, info = extract_information(chosen_action)

            fail_obs = ""
            pdf_obs = ""
            warn_obs = ""
            # execute action
            try:
                window_handle_task = driver_task.current_window_handle
                driver_task.switch_to.window(window_handle_task)

                if action_key == 'click':
                    if not args.text_only:
                        click_ele_number = int(info[0])
                        web_ele = web_eles[click_ele_number]
                    else:
                        click_ele_number = info[0]
                        element_box = obs_info[click_ele_number]['union_bound']
                        element_box_center = (element_box[0] + element_box[2] // 2,
                                            element_box[1] + element_box[3] // 2)
                        web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])

                    ele_tag_name = web_ele.tag_name.lower()
                    ele_type = web_ele.get_attribute("type")

                    exec_action_click(info, web_ele, driver_task)

                    # deal with PDF file
                    current_files = sorted(os.listdir(args.download_dir))
                    if current_files != download_files:
                        # wait for download finish
                        time.sleep(10)
                        current_files = sorted(os.listdir(args.download_dir))

                        current_download_file = [pdf_file for pdf_file in current_files if pdf_file not in download_files and pdf_file.endswith('.pdf')]
                        if current_download_file:
                            pdf_file = current_download_file[0]
                            pdf_obs = get_pdf_retrieval_ans_from_gemini(genai_client, os.path.join(args.download_dir, pdf_file), task['ques'])
                            shutil.copy(os.path.join(args.download_dir, pdf_file), task_dir)
                            pdf_obs = "You downloaded a PDF file, I ask the Assistant API to answer the task based on the PDF file and get the following response: " + pdf_obs
                        download_files = current_files

                    if ele_tag_name == 'button' and ele_type == 'submit':
                        time.sleep(10)

                elif action_key == 'wait':
                    time.sleep(5)

                elif action_key == 'type':
                    if not args.text_only:
                        type_ele_number = int(info['number'])
                        web_ele = web_eles[type_ele_number]
                    else:
                        type_ele_number = info['number']
                        element_box = obs_info[type_ele_number]['union_bound']
                        element_box_center = (element_box[0] + element_box[2] // 2,
                                            element_box[1] + element_box[3] // 2)
                        web_ele = driver_task.execute_script("return document.elementFromPoint(arguments[0], arguments[1]);", element_box_center[0], element_box_center[1])

                    warn_obs = exec_action_type(info, web_ele, driver_task)
                    if 'wolfram' in task['web']:
                        time.sleep(5)

                elif action_key == 'scroll':
                    if not args.text_only:
                        exec_action_scroll(info, web_eles, driver_task, args, None)
                    else:
                        exec_action_scroll(info, None, driver_task, args, obs_info)

                elif action_key == 'goback':
                    driver_task.back()
                    time.sleep(2)

                elif action_key == 'google':
                    driver_task.get('https://www.google.com/')
                    time.sleep(2)

                # This represents the end
                elif action_key == 'answer':
                    logging.info(info['content'])
                    logging.info('finish!!')
                    break

                else:
                    raise NotImplementedError
                fail_obs = ""
            except Exception as e:
                logging.error('driver error info:')
                logging.error(e)
                if 'element click intercepted' not in str(e):
                    fail_obs = "The action you have chosen cannot be executed. Please double-check if you have selected the wrong Numerical Label or Action or Action format. Then provide the revised Thought and Action."
                else:
                    fail_obs = ""
                time.sleep(2)

        driver_task.quit()
        # Since Gemini might not provide token usage in the same format as OpenAI
        logging.info(f'Task {task["id"]} completed')


if __name__ == '__main__':
    main()
    print('End of process')







