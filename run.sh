#!/bin/bash
nohup python -u run.py \
    --test_file ./data/arxiv_tasks_test5.jsonl \
    --api_key YOUR_OPENAI_API_KEY \
    --api_organization_id YOUR_OPENAI_ORGANIZATION_ID \
    --max_iter 15 \
    --max_attached_imgs 3 \
    --temperature 1 \
    --fix_box_color \
    --seed 42 \
    --window_width 1920 \
    --window_height 1080 \
    --pdf_path data/arXiv.pdf
