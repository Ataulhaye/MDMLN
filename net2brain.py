from net2brain.evaluations.rsa import RSA
from net2brain.feature_extraction import FeatureExtractor
from net2brain.utils.download_datasets import load_dataset

# Load the ROIs
stimuli_path, roi_path = load_dataset("bonner_pnas2017")

# Initialize the FeatureExtractor with a pretrained model
fx = FeatureExtractor(model="AlexNet", netset="standard", device="cpu")

# Extract features from a dataset
fx.extract(dataset_path=stimuli_path, save_format="npz", save_path="AlexNet_Feat")
