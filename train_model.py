import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import tqdm

from data import TranslationDataset
from transformers import BertTokenizerFast, BertLMHeadModel
from transformers import BertModel, BertForMaskedLM, BertConfig, EncoderDecoderModel
from sklearn.model_selection import train_test_split

def get_model():
    vocabsize = encparams["vocab_size"]
    max_length = encparams["max_length"]

    encoder_config = BertConfig(vocab_size=vocabsize,
                                max_position_embeddings=max_length + 64,  # this shuold be some large value
                                num_attention_heads=encparams["num_attn_heads"],
                                num_hidden_layers=encparams["num_hidden_layers"],
                                hidden_size=encparams["hidden_size"],
                                type_vocab_size=1)

    encoder = BertModel(config=encoder_config)

    vocabsize = decparams["vocab_size"]
    max_length = decparams["max_length"]
    decoder_config = BertConfig(vocab_size=vocabsize,
                                max_position_embeddings=max_length + 64,  # this shuold be some large value
                                num_attention_heads=decparams["num_attn_heads"],
                                num_hidden_layers=decparams["num_hidden_layers"],
                                hidden_size=decparams["hidden_size"],
                                type_vocab_size=1,
                                is_decoder=True,
                                add_cross_attention=True)  # Very Important

    decoder = BertLMHeadModel(config=decoder_config)

    # Define encoder decoder model
    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
    model.to(device)
    return model




def compute_loss(predictions, targets):
    """Compute our custom loss"""
    predictions = predictions[:, :-1, :].contiguous()
    targets = targets[:, 1:]

    rearranged_output = predictions.view(predictions.shape[0]*predictions.shape[1], -1)
    rearranged_target = targets.contiguous().view(-1)

    loss = criterion(rearranged_output, rearranged_target)

    return loss

def train_model():
    model.train()
    epoch_loss = 0

    for i, (en_input, en_masks, de_output, de_masks) in enumerate(train_dataloader):

        optimizer.zero_grad()

        en_input = en_input.to(device)
        de_output = de_output.to(device)
        en_masks = en_masks.to(device)
        de_masks = de_masks.to(device)

        lm_labels = de_output.clone()
        out = model(input_ids=en_input, attention_mask=en_masks,
                                        decoder_input_ids=de_output, decoder_attention_mask=de_masks,labels=lm_labels)
        prediction_scores = out[1]
        predictions = F.log_softmax(prediction_scores, dim=2)
        loss = compute_loss(predictions, de_output)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()

    print("Mean epoch loss:", (epoch_loss / num_train_batches))

def eval_model():
    model.eval()
    epoch_loss = 0

    for i, (en_input, en_masks, de_output, de_masks) in enumerate(train_dataloader):

        optimizer.zero_grad()

        en_input = en_input.to(device)
        de_output = de_output.to(device)
        en_masks = en_masks.to(device)
        de_masks = de_masks.to(device)

        lm_labels = de_output.clone()

        out = model(input_ids=en_input, attention_mask=en_masks,
                                        decoder_input_ids=de_output, decoder_attention_mask=de_masks,labels=lm_labels)

        prediction_scores = out[1]
        predictions = F.log_softmax(prediction_scores, dim=2)
        loss = compute_loss(predictions, de_output)
        epoch_loss += loss.item()

    print("Mean validation loss:", (epoch_loss / num_valid_batches))

if __name__=="__main__":
    # MAIN TRAINING LOOP
    configfile = "config.json"
    round_point = 3
    with open(configfile, "r") as f:
        config = json.load(f)

    globalparams = config["global_params"]
    encparams = config["encoder_params"]
    decparams = config["decoder_params"]
    modelparams = config["model_params"]

    en_tok_path = os.getcwd().replace("\\", "/") + "/" + config["encoder_params"][
            "tokenizer_path"] + f"_{round_point}_{10}/"
    en_tokenizer = BertTokenizerFast.from_pretrained(en_tok_path)
    de_tok_path = os.getcwd().replace("\\", "/") + "/" + config["decoder_params"][
            "tokenizer_path"] + f"_{round_point}_{10}/"
    de_tokenizer = BertTokenizerFast.from_pretrained(de_tok_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    print("Loading models ..")

    enc_maxlength = encparams["max_length"]
    dec_maxlength = decparams["max_length"]

    batch_size = modelparams["batch_size"]


    path = f"./data/sequences_F_{round_point}_preprocessed.json"
    with open(path, "r") as data_file:
        data = json.load(data_file)
    train_input, test_input, train_label, test_label = train_test_split(data["data"], data["labels"], test_size=0.2,
                                                                        random_state=42)

    train_dataset = TranslationDataset(train_input, train_label, en_tokenizer, de_tokenizer, enc_maxlength,dec_maxlength)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False,
                                                   drop_last=True, num_workers=1,
                                                   collate_fn=train_dataset.collate_function)

    valid_dataset = TranslationDataset(test_input, test_label, en_tokenizer, de_tokenizer, enc_maxlength,
                                       dec_maxlength)
    valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False,
                                                   drop_last=True, num_workers=1,
                                                   collate_fn=valid_dataset.collate_function)



    model = get_model()

    optimizer = optim.Adam(model.parameters(), lr=modelparams['lr'])
    criterion = nn.NLLLoss(ignore_index=de_tokenizer.pad_token_id)

    num_train_batches = len(train_dataloader)
    num_valid_batches = len(valid_dataloader)


    for epoch in tqdm(range(modelparams['num_epochs']), position=0, leave=True):
        print("Starting epoch", epoch+1)
        train_model()
        eval_model()

    print("Saving model ..")
    save_location = modelparams['model_path']
    model_name = modelparams['model_name']
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    model.save_pretrained(save_location)
