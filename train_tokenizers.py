import hashlib
import sys
import os
import json
from pathlib import Path

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from tqdm.auto import tqdm


def train_tokenizer( params, round_point, data):
    training_tokenizer = ByteLevelBPETokenizer(
        lowercase=True,
    )
    training_tokenizer.train_from_iterator(data, vocab_size=params["vocab_size"], min_frequency=10, special_tokens=[
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
    ])
    save_location = params["tokenizer_path"]
    if not os.path.exists(f"{save_location}_{round_point}_{10}/"):
        Path(f"{save_location}_{round_point}_{10}/").mkdir(parents=True, exist_ok=True)
    training_tokenizer.save(f"{save_location}_{round_point}_{10}/tokenizer.json")
def round_data(data,rounding_point):
    rounded_data = []
    print("Rounding data ...")
    for index in tqdm(range(len(data)), position=0, leave=True):
        entry = data[index]
        tokens = entry.split(",")
        rounded_tokens = []
        for token in tokens:
            rounded_tokens.append(round(float(token), rounding_point))
        rounded_data.append(",".join(str(x) for x in rounded_tokens))
    return rounded_data
def apply_hash(data):
    hashed_data = []
    print("Hashing data ....")
    for index in tqdm(range(len(data)), position=0, leave=True):
        entry = data[index]
        hash_object = hashlib.sha512(entry.encode())
        hex_dig = hash_object.hexdigest()
        hashed_data.append(hex_dig)
    return hashed_data
if __name__ == "__main__":
    configfile = "config.json"
    round_point = 3
    # Read the params
    with open(configfile, "r") as f:
        config = json.load(f)
    path = "./data/sequences_F.json"
    with open(path, "r") as data_file:
        data = json.load(data_file)

    data["data"] = round_data(data["data"], round_point)
    data["data"] = apply_hash(data["data"])
    with open(f"./data/sequences_F_{round_point}_preprocessed.json", "w") as f:
        json.dump(data, f, indent=4)
    # Train the tokenizers
    train_tokenizer(config["encoder_params"], round_point,data["data"])
    train_tokenizer(config["decoder_params"], round_point,data["labels"])
