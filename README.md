# moe-caption-specialization

A project for exploring caption generation with mixture-of-experts (MoE) specialization.

## Overview

This project studies how mixture-of-experts architectures can be used to improve image caption generation by encouraging expert specialization.

## Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate moe-caption
python -m spacy download en_core_web_sm
```

## Current Progress

The project currently includes:
- T5 encoder smoke tests
- spaCy-based coarse semantic labeling
- an initial token alignment pipeline for building a word-level token table
- a toy clustering analysis over aligned token representations

## Run

From the project root, run:

```bash
python -m scripts.build_token_table
python -m scripts.analyze_token_table
```