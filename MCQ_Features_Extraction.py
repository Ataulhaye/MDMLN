# from __future__ import unicode_literals

import argparse
import collections
import copy
import json
import os
import pickle
from pathlib import Path

import data_loader.data_loader as module_data
import en_core_web_sm
import model.model_dist_MCQ as module_arch
import nltk

# nltk.download()
import pandas as pd
import spacy
import textacy
import torch
import transformers
from base.base_dataset import build_question, sample_phrase
from parse_config_dist_multi import ConfigParser
from spacy.matcher import Matcher
from spacy.util import filter_spans
from textblob import TextBlob
from utils.util import replace_nested_dict_item, state_dict_data_parallel_fix

# To use this script download the MCQ from the following site
# https://github.com/TencentARC/MCQ
# A couple of things are required to set in order to run the script
# ......


def init_dataloaders(config, module_data):
    """
    We need a way to change split from 'train' to 'val'.
    """
    if "type" in config["data_loader"] and "args" in config["data_loader"]:
        # then its a single dataloader
        data_loader = [config.initialize("data_loader", module_data)]
        config["data_loader"]["args"] = replace_nested_dict_item(
            config["data_loader"]["args"], "split", "val"
        )
        valid_data_loader = [config.initialize("data_loader", module_data)]
    elif isinstance(config["data_loader"], list):
        data_loader = [
            config.initialize("data_loader", module_data, index=idx)
            for idx in range(len(config["data_loader"]))
        ]
        new_cfg_li = []
        for dl_cfg in config["data_loader"]:
            dl_cfg["args"] = replace_nested_dict_item(dl_cfg["args"], "split", "val")
            new_cfg_li.append(dl_cfg)
        config._config["data_loader"] = new_cfg_li
        valid_data_loader = [
            config.initialize("data_loader", module_data, index=idx)
            for idx in range(len(config["data_loader"]))
        ]
    else:
        raise ValueError("Check data_loader config, not correct format.")

    return data_loader, valid_data_loader


def process_text(text_data, tokenizer, device):
    text_data = tokenizer(text_data, return_tensors="pt", padding=True, truncation=True)
    text_data = {key: val.to(device) for key, val in text_data.items()}
    return text_data


def extract_verbs(caption):

    verbs = None
    nlp = spacy.load("en_core_web_sm")

    pattern = [
        {"POS": "VERB", "OP": "?"},
        {"POS": "ADV", "OP": "*"},
        {"POS": "AUX", "OP": "*"},
        {"POS": "VERB", "OP": "+"},
    ]

    # instantiate a Matcher instance
    matcher = Matcher(nlp.vocab)
    matcher.add("Verb phrase", [pattern])
    # matcher.add("Verb phrase", None, pattern)

    doc = nlp(caption)
    # call the matcher to find matches
    matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in matches]

    verbs = filter_spans(spans)

    if verbs is None or len(verbs) == 0:

        word_tokens = nltk.word_tokenize(caption)

        is_verb = lambda pos: pos[:2] == "VB"
        verbs = [word for (word, pos) in nltk.pos_tag(word_tokens) if is_verb(pos)]

    if verbs is None or len(verbs) == 0:
        word_tokens = nltk.word_tokenize(caption)

        is_verb = lambda pos: pos[:3] == "NNS"
        verbs = [word for (word, pos) in nltk.pos_tag(word_tokens) if is_verb(pos)]

    return str(verbs)


def extract_nouns(caption):
    nouns = None
    try:
        blob = TextBlob(caption)
        nouns = blob.noun_phrases
    except LookupError as ex:
        print(ex)
    if nouns is None or len(nouns) == 0:
        word_tokens = nltk.word_tokenize(caption)
        is_noun = lambda pos: pos[:2] == "NN"
        nouns = [word for (word, pos) in nltk.pos_tag(word_tokens) if is_noun(pos)]

    # if nouns is None or len(nouns) == 0:
    # word_tokens = nltk.word_tokenize(caption)
    # nouns = []
    # tags = nltk.pos_tag(word_tokens)
    # for tag in tags:
    # if tag[1] == "NNS":
    # nouns.append(tag[0])

    return str(nouns)


# def gather_tensor(config, embed):
# embed_all = [torch.zeros_like(embed) for _ in range(config["n_gpu"])]
# torch.distributed.all_gather(embed_all, embed)
# embed_all = torch.cat(embed_all, dim=0)
# return embed_all


def generate_encodings(query):

    ###########################################################
    # Test section

    """
    config = model_config()
    caption = "the boy has a stormy disposition"
    caption = "the symptoms can only be found sporadically"
    caption = "the staircase to the top is a spiral staircase"
    # build tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config["arch"]["args"]["text_params"]["model"], TOKENIZERS_PARALLELISM=False
    )
    if query == "verb":
        verbs = extract_verbs(caption)
        phrase = sample_phrase(verbs, None, caption, True)
    elif query == "noun":
        nouns = extract_nouns(caption)
        phrase = sample_phrase(None, nouns, caption, False)

    question, answer = build_question(caption, phrase, tokenizer)
    """
    ###################################################################

    # trained_model = torch.load("C:/Users/ataul/source/Uni/MCQ/MCQ.pth")
    trained_model = torch.load(Path("MCQ.pth").resolve())

    config = model_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model architecture, then print to console
    model = config.initialize("arch", module_arch)

    # Load the trained model
    model = model.to(device)

    new_state_dict = state_dict_data_parallel_fix(
        trained_model["state_dict"], model.state_dict()
    )

    # txtWordEmbed = copy.deepcopy(model.text_model.embeddings.word_embeddings.weight)
    # vidNorm1w = copy.deepcopy(model.video_model.blocks[0].norm1.weight)

    model.load_state_dict(new_state_dict)

    # txtWordEmbed_new = model.text_model.embeddings.word_embeddings.weight
    # vidNorm1w_new = model.video_model.blocks[0].norm1.weight
    model.eval()

    import gc

    del trained_model
    del new_state_dict
    torch.cuda.empty_cache()
    gc.collect()

    # setup data_loader instances
    data_loader, valid_data_loader = init_dataloaders(config, module_data)

    print("Val dataset: ", [x.n_samples for x in valid_data_loader], " samples")

    # build tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config["arch"]["args"]["text_params"]["model"], TOKENIZERS_PARALLELISM=False
    )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for dl_idx, dl in enumerate(valid_data_loader):
        for i, data in enumerate(dl):
            caption = data["meta"]["raw_captions"][0]
            phrase = None
            if query == "verb":
                verbs = extract_verbs(caption)
                phrase = sample_phrase(verbs, None, caption, True)
            elif query == "noun":
                nouns = extract_nouns(caption)
                phrase = sample_phrase(None, nouns, caption, False)

            data["question"], data["answer"] = build_question(
                caption, phrase, tokenizer
            )
            data["text"] = process_text(data["text"], tokenizer, device)
            data["question"] = process_text(data["question"], tokenizer, device)
            data["answer"] = process_text(data["answer"], tokenizer, device)
            # data["question"] = data["text"]
            # data["answer"] = data["text"]
            data["video"] = data["video"].to(device)

    print(data["video"].shape)

    with torch.inference_mode():
        text_embed, answer_embed, bridge_embed, vid_embed = model(data)

    # text_embed_all = gather_tensor(config, text_embed)
    # answer_embed_all = gather_tensor(config, answer_embed)
    # bridge_embed_all = gather_tensor(config, bridge_embed)
    # vid_embed_all = gather_tensor(config, vid_embed)

    # print(text_embed_all.shape)
    # print(answer_embed_all.shape)
    # print(bridge_embed_all.shape)
    # print(vid_embed_all.shape)

    print(text_embed.shape)
    print(answer_embed.shape)
    print(bridge_embed.shape)
    print(vid_embed.shape)

    return text_embed, answer_embed, bridge_embed, vid_embed


def model_config():
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default="./configs/zero_msrvtt_4f_i21k.json",
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o", "--observe", action="store_true", help="Whether to observe (neptune)"
    )
    args.add_argument(
        "-l",
        "--launcher",
        choices=["none", "pytorch"],
        default="none",
        help="job launcher",
    )
    args.add_argument("-k", "--local_rank", type=int, default=0)

    # master_address = os.environ["MASTER_ADDR"]
    # master_port = int(os.environ["MASTER_PORT"])
    # world_size = int(os.environ["WORLD_SIZE"])
    # rank = int(os.environ["RANK"])

    # args.add_argument("-ma", "--master_address", default=master_address)
    # args.add_argument("-mp", "--master_port", type=int, default=master_port)
    # args.add_argument("-ws", "--world_size", type=int, default=world_size)
    # args.add_argument("-rk", "--rank", type=int, default=rank)

    # config = ConfigParser(args)
    # args = args.parse_args()

    # torch.distributed.init_process_group(backend="nccl",init_method="tcp://{}:{}".format(args.master_address, args.master_port),rank=args.rank,world_size=args.world_size, )

    config = ConfigParser(args)
    return config


def extract_features(query, set_path, set_name):

    transcripts = []

    with open(set_path) as f:
        transcripts = f.readlines()
    #####################################
    # Test Section
    """
    for transcript in transcripts:
    transcript = transcript.strip("\n")
    script = transcript.split(";")
    test_all_text_query(query, script[2].replace('"', ""))
    """

    #############################################
    pickle_path = (
        Path(
            "data/MSRVTT/high-quality/structured-symlinks/jsfusion_val_caption_idx.pkl"
        )
        .absolute()
        .resolve()
    )
    jsonfile = Path("data/MSRVTT/annotation/MSR_VTT.json").absolute().resolve()
    txtfile = (
        Path("data/MSRVTT/high-quality/structured-symlinks/val_list_jsfusion.txt")
        .absolute()
        .resolve()
    )

    all_embedings = []

    print(f"Total {len(transcripts)} transcripts")

    for transcript in transcripts:
        transcript = transcript.strip("\n")
        script = transcript.split(";")
        key = script[0].split(".")[0].replace('"', "")
        msr_vtt_dict = {
            "annotations": [
                {"caption": script[2].replace('"', ""), "id": "0", "image_id": key}
            ]
        }
        val_list_jsfusion_dict = {key: 0}
        with open(pickle_path, "wb") as output:
            pickle.dump(val_list_jsfusion_dict, output)

        fjson = open(jsonfile, "w")
        fjson.write(json.dumps(msr_vtt_dict))
        fjson.close()

        ftxt = open(txtfile, "w")
        ftxt.write(key)
        ftxt.close()
        en_de_txt = script[1] + "; " + script[2]
        text_embed, answer_embed, bridge_embed, vid_embed = generate_encodings(query)
        all_embedings.append(
            (
                key,
                text_embed,
                answer_embed,
                bridge_embed,
                vid_embed,
                en_de_txt,
            )
        )

    path = r"C:\Users\ataul\source\Uni\BachelorThesis\poc\PickleFiles"
    f_path = os.path.join(path, f"MCQ_Embeddings_Set{set_name}_{query}.pickle")
    with open(f_path, "wb") as output:
        pickle.dump(all_embedings, output)

    print(f"Set{set_name} features are extracted")


def test_all_text_query(query, caption):
    config = model_config()
    # build tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config["arch"]["args"]["text_params"]["model"], TOKENIZERS_PARALLELISM=False
    )
    if query == "verb":
        verbs = extract_verbs(caption)
        phrase = sample_phrase(verbs, None, caption, True)
    elif query == "noun":
        nouns = extract_nouns(caption)
        phrase = sample_phrase(None, nouns, caption, False)

    question, answer = build_question(caption, phrase, tokenizer)

    print("Question", question)
    print("Answer", answer)


if __name__ == "__main__":
    # "verb",
    # "noun",
    extract_features(
        "verb",
        r"C:\Users\ataul\source\Uni\BachelorThesis\poc\CSV_Data_Files\AudioTranscriptsSet1_DE_EN.csv",
        "1",
    )
    extract_features(
        "noun",
        r"C:\Users\ataul\source\Uni\BachelorThesis\poc\CSV_Data_Files\AudioTranscriptsSet1_DE_EN.csv",
        "1",
    )
