# Analysis

## Layer 3, Head 1

This head shows a strong diagonal pattern, where nearly every token
attends almost entirely to itself rather than to surrounding words.
This suggests the head preserves a token's own identity rather than
mixing in context from the rest of the sentence.

Example Sentences:
- The chef seasoned the [MASK] before serving it.
- She left her umbrella at home, so she got [MASK] in the rain.

## Layer 12, Head 5

This head shows an "attention sink" pattern, where most tokens direct
a noticeable portion of their attention to the [SEP] token at the end
of the sentence, regardless of the sentence's content. This is a known
behavior in later BERT layers, where heads route unused attention to a
fixed special token rather than any specific word.

Example Sentences:
- The chef seasoned the [MASK] before serving it.
- She left her umbrella at home, so she got [MASK] in the rain.
