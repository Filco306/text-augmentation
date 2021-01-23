import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Union, List

# Code from https://towardsdatascience.com/paraphrase-any-question-with-t5-text-to-text-transfer-transformer-pretrained-model-and-cbb9e35f1555 # noqa: E501


model = T5ForConditionalGeneration.from_pretrained("ramsrigouthamg/t5_paraphraser")
tokenizer = T5Tokenizer.from_pretrained("ramsrigouthamg/t5_paraphraser")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device ", device)
model = model.to(device)
max_len = 256


def generate(
    sentence: str,
    verbose: int = 0,
    skip_special_tokens: bool = True,
    clean_up_tokenization_spaces: bool = True,
) -> List[str]:
    text = "paraphrase: " + sentence + " </s>"

    encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = (
        encoding["input_ids"].to(device),
        encoding["attention_mask"].to(device),
    )

    # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
    beam_outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_masks,
        do_sample=True,
        max_length=max_len,
        top_k=120,
        top_p=0.98,
        early_stopping=True,
        num_return_sequences=10,
    )

    if verbose > 0:
        print("\nOriginal Question ::")
        print(sentence)
        print("\n")
        print("Paraphrased Questions :: ")
    final_outputs = []
    for beam_output in beam_outputs:
        sent = tokenizer.decode(
            beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        if sent.lower() != sentence.lower() and sent not in final_outputs:
            final_outputs.append(sent)
    return final_outputs


def paraphrase(texts: Union[List, str], n_samples: int = 10):
    if isinstance(texts, str) is True:
        texts = [texts]

    paraphrases = {}
    for text in texts:
        print(text)
        paraphrases[text] = generate(text)
    return paraphrases


if __name__ == "__main__":
    print(paraphrase("Which course should I take to get started in data science?"))
