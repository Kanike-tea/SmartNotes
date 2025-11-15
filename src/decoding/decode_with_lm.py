import numpy as np
from pyctcdecode.decoder import build_ctcdecoder

def load_lm_decoder(tokenizer, kenlm_model_path=None, vocab_file=None):
    # Build vocabulary from tokenizer
    vocab = list(tokenizer.chars)
    vocab.append("-")  # CTC blank token

    # Build decoder with optional KenLM model
    decoder = build_ctcdecoder(
        labels=vocab,
        kenlm_model_path=kenlm_model_path,
        unigrams=None if vocab_file is None else open(vocab_file).read().splitlines(),
    )
    return decoder


def ctc_beam_search_decode(logits, decoder):
    """
    logits: (T, C) numpy array after softmax (logits.exp())
    decoder: loaded pyctcdecode decoder
    """
    log_probs = np.log(logits + 1e-8)  # ensure numerical stability
    return decoder.decode(log_probs)
