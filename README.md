# Attention

### by Dyan Ahmadi

## Overview

This project explores how BERT, a pre-trained transformer-based language model, uses **self-attention** to understand text. The program takes a sentence containing a masked word (`[MASK]`), asks BERT to predict what that word most likely is, and then generates a set of diagrams visualizing how much attention each attention head pays to each word in the sentence. Comparing these diagrams across layers and heads can reveal patterns in what the model has learned about language, such as tracking word relationships, tracking sentence structure, or attending to specific token types.

## Explanation
 
The program is built around Hugging Face's `transformers` library and a pre-trained `bert-base-uncased` model.
 
1. **Input and tokenization** — The user types a sentence that must contain exactly one `[MASK]` token (BERT's placeholder for "the word I don't know"). The tokenizer converts this sentence into the numeric token IDs BERT expects.
2. **Finding the mask** — `get_mask_token_index` scans the tokenized input and returns the position of the `[MASK]` token, so the program knows which output to look at.
3. **Prediction** — The tokenized input is run through BERT with `output_attentions=True`, which returns both the predicted logits for every token in the vocabulary and the internal self-attention weights from every layer and head. The top 3 highest-scoring tokens for the masked position are printed as BERT's best guesses. The model is loaded with `use_safetensors=False` so it downloads `bert-base-uncased`'s native TensorFlow weights (`tf_model.h5`) instead of converting the PyTorch `safetensors` checkpoint — a conversion path that has a known bug in recent `transformers` versions.
4. **Attention visualization** — BERT has 12 layers, each with 12 attention heads (144 attention patterns total for the base model). `visualize_attentions` loops through every layer and every head, and for each one calls `generate_diagram`.
5. **Diagram generation** — `generate_diagram` draws a grid where each row and column represents a token in the sentence. Each cell is shaded using `get_color_for_attention_score`, which converts an attention score (between 0 and 1) into a grayscale color — lighter cells mean the model is paying more attention from that row's token to that column's token. Each diagram is saved as its own PNG, named after its layer and head number.
The result is a full set of attention diagrams that can be visually inspected to look for interpretable patterns, which are then written up in `analysis.md`.

## Project structure

```
attention/
├── mask.py
├── analysis.md
├── assets/
│   └── fonts/
│       └── OpenSans-Regular.ttf
├── README.md
└── requirements.txt
```

### Files

- **`mask.py`** — The main program. Loads BERT, takes a masked sentence as input, prints the top 3 predicted words, and generates one attention diagram (PNG) per layer/head combination.
- **`analysis.md`** — A short write-up identifying one specific layer and attention head that shows a clear, interpretable pattern, along with two example sentences that demonstrate it.
- **`assets/fonts/OpenSans-Regular.ttf`** — The font file used to label tokens on the generated attention diagrams.
- **`README.md`** — This file.
- **`requirements.txt`** — The Python package dependencies needed to run the project.

## Usage

1. Make sure you have Python 3.9–3.11 installed, then create and activate a virtual environment in the project folder:
   ```
   python3 -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the program:
   ```
   python mask.py
   ```
4. When prompted, enter a sentence containing exactly one `[MASK]` token, for example:
   ```
   Text: The chef seasoned the [MASK] before serving it.
   ```
5. The program prints BERT's top 3 predictions for the masked word, and saves an `Attention_Layer<N>_Head<N>.png` diagram for every layer/head combination in the project folder.

Note: the first run downloads the `bert-base-uncased` model from Hugging Face, so an internet connection is required at least once.

## course 
This project was a part of CS50's Introduction to Artificial Intelligence with Python

Project: Attention
