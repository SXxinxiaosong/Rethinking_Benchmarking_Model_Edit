# Rethinking_Benchmarking_Model_Edit

# Project Overview

This repository contains tools and scripts for model editing and evaluation. The structure is as follows:

## Folders and Their Purpose

### `EasyEdit`
The `EasyEdit` folder contains scripts and methods for **model editing**. It allows you to apply various editing techniques to a pretrained model, based on a series of predefined methods. This folder includes tools to facilitate model updates and knowledge adaptation.

### `Edit_RAG`
The `Edit_RAG` folder is designed to execute the **SCR (Selective Context Reasoning)** method. It leverages a **Retrieval-Augmented Generation (RAG)** approach, which allows you to update a model's knowledge base dynamically while maintaining efficient inference performance.

### `LUFFY`
The `LUFFY` folder provides tools for **evaluating the inference abilities** of models after they have been edited. It includes functionality for assessing how well the model can perform reasoning tasks and generate accurate outputs based on newly integrated knowledge.

## Usage

1. **Model Editing**: Navigate to the `EasyEdit` folder and follow the instructions to apply editing methods to your model.
2. **SCR Execution**: Use the scripts in `Edit_RAG` to apply the SCR method, which integrates new knowledge into the model through a RAG-based approach.
3. **Evaluation**: After editing the model, use the tools in `LUFFY` to evaluate the model's inference ability and performance on relevant tasks.

## Acknowledge

Our code is based on [EasyEdit](https://github.com/zjunlp/EasyEdit.git) and [LUFFY](https://github.com/ElliottYan/LUFFY.git).

