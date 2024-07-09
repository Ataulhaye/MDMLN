import argparse
import collections
import copy
import json
import os
import pickle

import pandas as pd
import torch
import transformers

import data_loader.data_loader as module_data
import model.model_dist_MCQ as module_arch
from parse_config_dist_multi import ConfigParser
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


def generate_encodings():
    trained_model = torch.load("C:/Users/ataul/source/Uni/MCQ/MCQ.pth")

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
            data["text"] = process_text(data["text"], tokenizer, device)
            # data['question'] = process_text(data['question'])
            # data['answer'] = process_text(data['answer'])
            data["question"] = data["text"]
            data["answer"] = data["text"]
            data["video"] = data["video"].to(device)

    print(data["video"].shape)

    with torch.inference_mode():
        text_embed, answer_embed, bridge_embed, vid_embed = model(data)

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

    config = ConfigParser(args)
    return config


def extract_features(set_path, set_name):

    transcripts = []

    with open(set_path) as f:
        transcripts = f.readlines()

    pickle_path = r"C:\Users\ataul\source\Uni\MCQ\data\MSRVTT\high-quality\structured-symlinks\jsfusion_val_caption_idx.pkl"
    jsonfile = r"C:\Users\ataul\source\Uni\MCQ\data\MSRVTT\annotation\MSR_VTT.json"
    txtfile = r"C:\Users\ataul\source\Uni\MCQ\data\MSRVTT\high-quality\structured-symlinks\val_list_jsfusion.txt"

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
        text_embed, answer_embed, bridge_embed, vid_embed = generate_encodings()
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
    f_path = os.path.join(path, f"MCQ_Embeddings_Set{set_name}.pickle")
    with open(f_path, "wb") as output:
        pickle.dump(all_embedings, output)

    print(f"Set{set_name} features are extracted")


if __name__ == "__main__":
    extract_features(
        r"C:\Users\ataul\source\Uni\BachelorThesis\poc\AudioTranscriptsSet1_DE_EN.csv",
        "1",
    )
