import json
import os

from transformers import EncoderDecoderModel, BertTokenizerFast

if __name__== "__main__":
    configfile = "config.json"
    round_point = 3
    with open(configfile, "r") as f:
        config = json.load(f)
    model_path = "D:/pycharmProject/encdecmodel-hf/data/try_2_significant_3/"
    encoder_decoder = EncoderDecoderModel.from_pretrained(model_path)
    en_tok_path = os.getcwd().replace("\\", "/") + "/" + config["encoder_params"][
        "tokenizer_path"] + f"_{round_point}/"
    en_tokenizer = BertTokenizerFast.from_pretrained(en_tok_path)
    de_tok_path = os.getcwd().replace("\\", "/") + "/" + config["decoder_params"][
        "tokenizer_path"] + f"_{round_point}_{10}/"
    de_tokenizer = BertTokenizerFast.from_pretrained(de_tok_path)
    path = f"./data/sequences_F_{round_point}_preprocessed.json"
    with open(path, "r") as data_file:
        data = json.load(data_file)
    test_entry = data["data"][0]
    test_entry_encodings = en_tokenizer(test_entry, return_tensors="pt",add_special_tokens=True,truncation=True)
    generation = encoder_decoder.generate(input_ids=test_entry_encodings["input_ids"], decoder_start_token_id=en_tokenizer.cls_token_id)
    print(generation)
    decoded_gen = de_tokenizer.decode(generation[0])
    print(decoded_gen)