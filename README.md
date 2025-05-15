<div align="center">
<h1> WebVoyager  
<img src="./assets/icon.png" width="45px">  
<br> Building an End-to-End Web Agent with Large Multimodal Models </h1>
</div>

<div align="center">
![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)  
![Python 3.10+](https://img.shields.io/badge/python-3.10.13-green.svg)  
![Selenium](https://img.shields.io/badge/Selenium-4.15.2-red.svg)
</div>

<div align="center">
<img src="./assets/overall_process_crop.png" width="90%">
</div>

An online web-browsing agent built on top of Selenium, driven by Google’s Gemini API for decision-making and vision understanding.  
- **End-to-End**. From task parsing → retrieval → interaction → synthesis.  
- **Multimodal**. Supports screenshots + accessibility trees.  
- **Diverse Tasks**. Prebuilt scenarios on Booking, Google Flights, Wikipedia, etc., and an easy prompt-expansion method.  
- **Automated Evaluation**. Human-in-the-loop review plus GPT-4V auto-scoring.

## Setup Environment

1. Install Chrome (or Chromium).  
2. (Linux only) on CentOS/RHEL:  
   ```bash
   yum install chromium-browser
   ```  
3. Create and activate a conda env:  
   ```bash
   conda create -n webvoyager python=3.10
   conda activate webvoyager
   ```  
4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   pip install langchain
   pip install sentence-transformers torch
   pip install pdfplumber
   pip install langchain-community
   pip install chromadb
   pip install httpx==0.27.2
   pip install pymupdf
   pip install pypdf
   pip install pymupdf4llm
   pip install tiktoken
   pip install google-cloud-aiplatform
   ```

## Data

### Overview

- **Task file**: `data/WebVoyager_data.jsonl`  
- **Reference answers**: `data/reference_answer.json`  
- **GAIA subset**: `data/GAIA_web.jsonl`  

For Booking/Flights tasks, update the dates manually before running.

### Expand Tasks

Use our prompt in `prompts.py` to generate new, diverse web-interaction tasks. See the comments there for best practices.

## Running

### 1. Prepare your test tasks

Copy your target examples into `data/tasks_test.jsonl`. For any outdated date fields, update them manually.

### 2. Configure your Gemini credentials

Export your Gemini API key as an environment variable:
```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

### 3. Launch WebVoyager

Edit `run.sh` to set your model choice and key placeholder if desired, then run:
```bash
bash run.sh
```

Here’s a sample `run.sh`:
```bash
#!/bin/bash
nohup python -u run.py     --test_file ./data/tasks_test.jsonl     --gemini_api_key "$GEMINI_API_KEY"     --headless     --max_iter 15     --max_attached_imgs 1     --temperature 1     --text_only     --api_model gemini-1.5-pro     --seed 42 > test_tasks_text_only.log &
```

### Parameters

- `--test_file`  
  Path to your JSONL task file.  
- `--gemini_api_key`  
  Your Gemini API key (from `GEMINI_API_KEY`).  
- `--output_dir`  
  Directory to save full interaction logs.  
- `--download_dir`  
  Where downloaded PDFs or artifacts are stored.  

**Model & Sampling**  
- `--api_model`  
  A Gemini model identifier (e.g. `gemini-1.0`, `gemini-1.5-pro`).  
- `--temperature`  
  Sampling diversity (0 → deterministic, up to 1.0).  
- `--seed`  
  For reproducibility.  

**Web Navigation**  
- `--headless`  
  Run Chrome headlessly.  
- `--text_only`  
  Don’t capture screenshots, only accessibility tree.  
- `--max_attached_imgs`  
  Keep only the last K screenshots for context.  
- `--window_width` / `--window_height`  
  Browser viewport size (default: 1024×768).  
- `--save_accessibility_tree`  
  Dump the page’s accessibility tree JSON.  
- `--fix_box_color`  
  Use black bounding-boxes instead of random colors.

## Evaluation

### Human Review

Inspect `results/examples/...` for screenshots + action logs.

### Automated (GPT-4V)

Edit `evaluation/run_eval.sh` to point at your key and results folder:

```bash
#!/bin/bash
nohup python -u auto_eval.py     --gemini_api_key "$GEMINI_API_KEY"     --process_dir ../results/examples     --max_attached_imgs 15 > evaluation.log &
```
Then:
```bash
cd evaluation
bash run_eval.sh
```

## Citation

If you find WebVoyager useful, please cite:
```bibtex
@article{he2024webvoyager,
  title={WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models},
  author={He, Hongliang and Yao, Wenlin and Ma, Kaixin and Yu, Weixin and Dai, Yong and Zhang, Hongming and Lan, Zhenzhong and Yu, Dong},
  journal={arXiv preprint arXiv:2401.13919},
  year={2024}
}
```

## Disclaimer

This is an independent project, not an official Google or Tencent product. Any opinions or results are the authors’ own.
