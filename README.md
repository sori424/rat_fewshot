# Few-shot Learning of Remote Associates Test With Language Models

This is a revised codebase to perform few-shot "in-context" learning using language models similar to the [GPT-3 paper](https://arxiv.org/abs/2005.14165). In particular, a few training examples are placed into a natural language "prompt" and predictions are made by generating from the language model. See the [GPT-3 paper](https://arxiv.org/abs/2005.14165) and [Calibrate Before Use](http://arxiv.org/abs/2102.09690) for more information.

You can run this codebase with GPT-3 (if you have a key from OpenAI), GPT-2, and any other language model available in [HuggingFace Transformers](https://huggingface.co/models). If you have a GPT-3 key, you should place your API key into a file named `openai_key.txt`. The underlying model you use is abstracted away using a common API.

Running this codebase will report results with and without [contextual calibration](http://arxiv.org/abs/2102.09690).


## Installation

The easiest way to install the code is to create a fresh anaconda environment:
```
conda create -n fewshot python=3.6
source activate fewshot
pip install -r requirements.txt
```
Now you should be ready to go!

## Replicating Our Results

Here is how to replicate the results for GPT-2. To replicate the results for classification tasks:


```
CUDA_VISIBLE_DEVICES=0 python testst.py \
--model="gpt2-xl" \
--dataset="rat_reason, rat" \
--num_seeds=5 \
--all_shots="0, 1, 4, 8" \
--subsample_test_set=300
```
