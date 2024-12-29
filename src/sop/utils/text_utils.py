from transformers import BertTokenizerFast
from collections import namedtuple
import numpy as np

MappedSpansOutput = namedtuple("MappedSpansOutput",
                               ["original_span_indices",
                                "token_text_spans",
                                "subtoken_span_lists",
                                "original_text_tokens",
                                "bert_tokens"])

def find_start_end_tokens(binary_mask):
    # print('binary_mask:', binary_mask.shape)
    mask = np.array(binary_mask)
    changes = np.diff(mask)
    start_tokens = np.where(changes == 1)[0] + 1
    end_tokens = np.where(changes == -1)[0]
    
    if mask[0] == 1:
        start_tokens = np.insert(start_tokens, 0, 0)
    if mask[-1] == 1:
        end_tokens = np.append(end_tokens, len(mask) - 1)
    
    spans = list(zip(start_tokens, end_tokens))
    return spans

def map_token_spans_to_original_text(text, spans, tokenizer):
    """
    Maps token spans to the original text, with token_text_spans covering full tokens
    and subtoken_spans reflecting the exact subtokens selected by the binary mask.
    """
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    token_offsets = encoded['offset_mapping']
    bert_tokens = encoded.tokens()  # Includes special tokens [CLS] and [SEP]
    original_text_tokens = text.split()

    words = []
    token_text_spans = []
    subtoken_spans = []

    for start_token, end_token in spans:
        # Adjust spans for full tokens for token_text_spans
        full_start_token, full_end_token = start_token, end_token
        if full_start_token > len(bert_tokens) - 1 or full_end_token > len(bert_tokens) - 1:
            continue
        while full_start_token > 0 and bert_tokens[full_start_token].startswith('##'):
            full_start_token -= 1
        while full_end_token < len(bert_tokens) - 1 and bert_tokens[full_end_token + 1].startswith('##'):
            full_end_token += 1

        # Handle offsets for full tokens
        full_start_char = token_offsets[full_start_token][0]
        full_end_char = token_offsets[full_end_token][1] - 1

        # Determine word spans for token_text_spans
        start_word_index, end_word_index = None, None
        current_char_index = 0

        for index, word in enumerate(original_text_tokens):
            word_end_char_index = current_char_index + len(word) - 1
            if full_start_char >= current_char_index and full_start_char <= word_end_char_index:
                start_word_index = index
            if full_end_char >= current_char_index and full_end_char <= word_end_char_index:
                end_word_index = index
                break
            current_char_index += len(word) + 1

        if start_word_index is not None and end_word_index is not None:
            words.append((start_word_index, end_word_index))
            span_text = ' '.join(original_text_tokens[start_word_index:end_word_index + 1])
            token_text_spans.append(span_text)

        # Capture exact subtokens for subtoken_spans
        exact_subtokens = bert_tokens[start_token:end_token + 1]
        subtoken_spans.append(exact_subtokens)

    return MappedSpansOutput(words, token_text_spans, subtoken_spans, original_text_tokens, bert_tokens)


def find_evidence_indices(passage, evidence):
    # Optional: Convert both strings to lower case to ensure case-insensitive matching
    
    passage_lower = passage.lower()
    evidence_lower = evidence.lower()
    
    # Find the start index of the evidence in the passage
    start_index = passage_lower.find(evidence_lower)
    if start_index == -1:
        return None  # or raise an error if the evidence is not found in the passage
    
    # Extract the part of the passage before the evidence
    before_evidence = passage[:start_index]
    
    # Tokenize the passage and the part before the evidence
    passage_tokens = passage.split()
    before_tokens = before_evidence.split()
    
    # The start token index is the number of tokens before the evidence
    start_token_index = len(before_tokens)
    
    # The end token index is the start index plus the number of tokens in the evidence - 1
    evidence_tokens = evidence.split()
    end_token_index = start_token_index + len(evidence_tokens) - 1
    
    return start_token_index, end_token_index

def find_evidence_indices_list(passage, evidences):
    evidence_indices_list = []
    for evidence in evidences:
        evidence_indices_list.append(find_evidence_indices(passage, evidence))
    return evidence_indices_list


def main():
    text = "This is an example sentence, showing off the tokenizer."
    binary_mask = [0, 0, 1, 1, 0, 1, 1, 1, 0]
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    spans = find_start_end_tokens(binary_mask)
    output = map_token_spans_to_original_text(text, spans, tokenizer)
    
    print('Text:', text)
    print("Original text spans:", output.original_span_indices)
    print("Spanned texts:", output.token_text_spans)
    print("Subtokens in spans:", output.subtoken_span_lists)
    print("Original text tokenized:", output.original_text_tokens)
    print("BERT tokenized text:", output.bert_tokens)

if __name__ == "__main__":
    main()
