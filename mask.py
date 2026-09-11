import sys
import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200


def main():
    text = input("Text: ")

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(f"Input must include mask token {tokenizer.mask_token}.")

    # Use model to process input
    model = TFBertForMaskedLM.from_pretrained(MODEL, use_safetensors=False)
    result = model(**inputs, output_attentions=True)

    # Generate predictions
    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Visualize attentions
    visualize_attentions(inputs.tokens(), result.attentions)


def get_mask_token_index(mask_token_id, inputs):
    """
    Return the index of the token with the specified `mask_token_id`, or
    `None` if not present in the `inputs`.
    """
    # BERT doesn't know where your [MASK] is unless you tell it. 
    # When the tokenizer turns your sentence into a sequence of numeric token IDs, 
    # the [MASK] word becomes one specific ID. 
    # The job of this function is simple: 
    # search through the tokenized input and find which position holds that ID, 
    # so later code knows exactly which output slot to read BERT's prediction from. 
    # If the mask token never shows up, 
    # return None so the program can tell the user their input was invalid.
    
    # inputs["input_ids"] is a TensorFlow tensor. [0] grabs our single
    # sentence's row of token IDs out of it then Calling .numpy() converts
    # the whole tensor to a plain NumPy array in one bulk operation,
    # instead of touching TensorFlow's slow per-element machinery later.
    ids = inputs["input_ids"][0].numpy()
    # (ids == mask_token_id) compares every element of the array to
    # mask_token_id all at once, producing a boolean array of the same
    # length (True wherever the mask token appears, False elsewhere)
    # .nonzero()[0] then returns the indices where that array is True.
    matches = (ids == mask_token_id).nonzero()[0]
    return int(matches[0]) if len(matches) else None


def get_color_for_attention_score(attention_score):
    """
    Return a tuple of three integers representing a shade of gray for the
    given `attention_score`. Each value should be in the range [0, 255].
    """
    # Attention scores from BERT are floating-point numbers between 0 and 1, 
    # they come out of a softmax, so they behave like probabilities. 
    # To draw them as a grayscale image, 
    # each score needs to become an RGB color,
    # where a higher score is a lighter pixel and a lower score is darker.
    
    gray = round(float(attention_score) * 255)
    return (gray, gray, gray)


def visualize_attentions(tokens, attentions):
    """
    Produce a graphical representation of self-attention scores.

    For each attention layer, one diagram should be generated for each
    attention head in the layer. Each diagram should include the list of
    `tokens` in the sentence. The filename for each diagram should
    include both the layer number (starting count from 1) and head number
    (starting count from 1).
    """
    # BERT-base has 12 transformer layers, 
    # and each layer has 12 independent attention heads 
    # 144 attention patterns in total for a single sentence.
    # The plan is to loop through every layer, 
    # and within each layer loop through every head, 
    # generating one diagram per (layer, head) pair
    
    for layer_index, layer in enumerate(attentions):
        for head_index in range(layer.shape[1]):
            generate_diagram(
                layer_index + 1,
                head_index + 1,
                tokens,
                attentions[layer_index][0][head_index]
            )


def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Generate a diagram representing the self-attention scores for a single
    attention head. The diagram shows one row and column for each of the
    `tokens`, and cells are shaded based on `attention_weights`, with lighter
    cells corresponding to higher attention scores.

    The diagram is saved with a filename that includes both the `layer_number`
    and `head_number`.
    """
    # Create new image
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw each token onto the image
    for i, token in enumerate(tokens):
        # Draw token columns
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)

        # Draw token rows
        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )

    # Draw each word
    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Save image
    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")


if __name__ == "__main__":
    main()

# example sentence : The chef seasoned the [MASK] before serving it.
# example sentence : She left her umbrella at home, so she got [MASK] in the rain.
# example sentence : The dog chased the cat because it was [MASK].
# example sentence : The [MASK] on the table belongs to my sister.