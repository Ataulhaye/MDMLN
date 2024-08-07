import glob
import os
from pathlib import Path

import moviepy.editor as mp
import pandas as pd
import torch

from Brain import Brain
from BrainDataConfig import BrainDataConfig
from Enums import Lobe
from HyperParameterSearch import load_bestmodel_and_test


class Utility:

    @staticmethod
    def ensure_dir(root_dir, directory):
        directory_path = Path(f"{root_dir}/{directory}").absolute().resolve()
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return directory_path

    @staticmethod
    def test_load_model():
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        path = r"C:\Users\ataul\ray_results\Best-IFG\tune_with_parameters_2024-03-16_17-40-53\tune_with_parameters_f4bd1_00011_11_embedding_dim=2,hidden_dim1=1024,hidden_dim2=8,lr=0.0016_2024-03-16_18-09-11\checkpoint_000024"
        # path = r"C:\Users\ataul\source\Uni\BachelorThesis\poc\STG_BestModel_With_TSNE\STG_mean_18-03-2024_09-57-37_209122_model.pt"
        # path = r"C:\Users\ataul\source\Uni\BachelorThesis\poc\STG_BestModel_With_TSNE\STG_mean_18-03-2024_14-02-01_780484_model.pt"
        model_path = path.replace(os.sep, "/")
        # lobe = "IFG"
        lobe = "STG"
        load_bestmodel_and_test(lobe, model_path, device, gpus_per_trial=1)

    @staticmethod
    def analyse_nans():
        config = BrainDataConfig()
        stg = Brain(lobe=Lobe.STG, data_path=config.STG_path)
        nans_column_wise = stg.calculate_nans_voxel_wise(stg.voxels)
        print("stg nans_column_wise", len(nans_column_wise))
        nans_voxel_wise = stg.calculate_nans_trail_wise(stg.voxels)
        print("stg nans_voxel_wise", len(nans_voxel_wise))
        print("------------")

        ifg = Brain(lobe=Lobe.IFG, data_path=config.IFG_path)
        nans_column_wise_ifg = ifg.calculate_nans_voxel_wise(ifg.voxels)
        print("IFG nans_column_wise", len(nans_column_wise_ifg))
        nans_voxel_wise_ifg = ifg.calculate_nans_trail_wise(ifg.voxels)
        print("IFG nans_voxel_wise", len(nans_voxel_wise_ifg))
        print("------------")

    @staticmethod
    def get_set_files(set_path, set_name):
        set_keys = []
        with open(set_path) as f:
            for line in f:
                k = line.split(" ")[0]
                assert k.split(".")[1].replace('"', "") in "mpg"
                set_keys.append(k)

        all_scripts = r"C:\Users\ataul\source\Uni\BachelorThesis\poc\Result_Files\AudioTranscripts_corrected.csv"

        set_transcripts = []
        scripts_d = []
        with open(all_scripts) as f:
            scripts_d = f.readlines()

        for key in set_keys:
            for script in scripts_d:
                if key in script:
                    set_transcripts.append(script)
                    break

        assert len(set_transcripts) == len(set_keys)
        dest_file = f"AudioTranscriptsSet{set_name}.csv"

        f = open(dest_file, "w")
        for trans in set_transcripts:
            f.write(trans)
        f.close()

    @staticmethod
    def copy_en_txt(source_en_file, source_de_file, set_name):
        de_txt = None
        with open(source_de_file) as f:
            de_txt = f.readlines()

        en_txt = None
        with open(source_en_file) as f:
            en_txt = f.readlines()

        assert len(en_txt) == len(de_txt)

        set_transcripts = []

        for txt in de_txt:
            de_en = Utility.get_translation(txt, en_txt)
            if de_en is not None:
                set_transcripts.append(de_en)

        assert len(set_transcripts) == len(de_txt)

        dest_file = f"AudioTranscriptsSet{set_name}_DE_EN.csv"

        f = open(dest_file, "w")
        for trans in set_transcripts:
            f.write(trans)
        f.close()

    @staticmethod
    def get_translation(txt, en_txt):
        de_en = None
        for script in en_txt:
            s_script = script.strip("\n")
            txt_seg = txt.split(";")
            if txt_seg[0] in script:
                de_en = txt.strip("\n") + ";" + s_script.split(";")[1].strip(" ") + "\n"
                break
        if de_en is None:
            print("This must not happened")
        return de_en

    @staticmethod
    def load_embeddings():
        path = r"C:\Users\ataul\source\Uni\BachelorThesis\poc\PickleFiles\MCQ_Embeddings_Set1.pickle"
        embeddings = pd.read_pickle(path)
        for embed in embeddings:
            key, text_embed, answer_embed, bridge_embed, vid_embed, en_de_txt = embed
            print(key)
        print("End")

    @staticmethod
    def mpg_to_mp4():
        path = r"C:\Users\ataul\source\Uni\BachelorThesis\Stimuli"
        directory_path = Path("Stimuli_mp4").absolute().resolve()
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        for video_file_path in glob.glob(os.path.join(path, "*.mpg")):
            file_name = video_file_path.split("\\")[-1].split(".")[0]
            final_file_name = f"{file_name}.mp4"
            save_path = os.path.join(directory_path, final_file_name)
            clip = mp.VideoFileClip(video_file_path, fps_source="fps")
            clip.write_videofile(save_path)
