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

## Current Progress

The project currently includes:
- a T5 smoke test
- a spaCy-based coarse labeling smoke test
- an initial token alignment pipeline that builds a word-level token table from captions

## Run

Run the main token-table pipeline from the project root:

```bash
python -m scripts.build_token_table