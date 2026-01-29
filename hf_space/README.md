---
title: MinerU PDF Parser
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.1
app_file: app.py
pinned: false
license: agpl-3.0
short_description: PDF to Markdown converter with GPU acceleration
tags:
  - pdf
  - markdown
  - ocr
  - document-parsing
---

# MinerU PDF Parser

Convert PDF documents to Markdown format using MinerU with GPU acceleration on HuggingFace ZeroGPU (NVIDIA H200).

## Features

- **PDF to Markdown conversion** with high accuracy
- **Table recognition** for structured data extraction
- **Formula recognition** for mathematical expressions
- **Multi-language support**: Chinese, English, Japanese, Korean, Latin
- **GPU accelerated** using NVIDIA H200 70GB

## Usage

1. Upload a PDF or image file
2. Select parsing backend (Pipeline recommended for stability)
3. Choose document language
4. Set maximum pages to process
5. Click "Start Parsing"

## Backends

- **Pipeline Mode**: Most stable, recommended for ZeroGPU
- **Hybrid Mode**: Balance between accuracy and speed
- **VLM Mode**: Highest accuracy for complex documents

## Powered by

- [MinerU](https://github.com/opendatalab/MinerU) - High-quality PDF parsing
- HuggingFace ZeroGPU - Free GPU compute
