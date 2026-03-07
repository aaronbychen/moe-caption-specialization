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