from model import Combiner
import torch
import torch.nn.functional as F
import json
import clip
from pathlib import Path
import pickle
from tqdm import tqdm
import numpy as np

# Constants
server_base_path = Path(__file__).absolute().parent.absolute()
data_path = server_base_path / 'data'

fashioniq_path = Path('/kaggle/input/fashion-iq-dataset')

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    data_type = torch.float16
else:
    device = torch.device("cpu")
    data_type = torch.float32

def load_fashionIQ_assets():
    print("Loading FashionIQ assets...")

    global fashionIQ_val_triplets
    fashionIQ_val_triplets = []
    for dress_type in ['dress', 'toptee', 'shirt']:
        with open(fashioniq_path / 'fashionIQ_dataset' / 'captions' / f'cap.{dress_type}.val.json') as f:
            dress_type_captions = json.load(f)
            captions = [dict(caption, dress_type=f'{dress_type}') for caption in dress_type_captions]
            fashionIQ_val_triplets.extend(captions)
    print(f"Loaded {len(fashionIQ_val_triplets)} validation triplets")

    global fashionIQ_val_dress_index_features
    fashionIQ_val_dress_index_features = torch.load(
        data_path / 'fashionIQ_val_dress_index_features.pt', map_location=device).type(data_type).cpu()
    print(f"Loaded FashionIQ dress index features of shape {fashionIQ_val_dress_index_features.shape}")

    global fashionIQ_val_dress_index_names
    with open(data_path / 'fashionIQ_val_dress_index_names.pkl', 'rb') as f:
        fashionIQ_val_dress_index_names = pickle.load(f)
    print(f"Loaded {len(fashionIQ_val_dress_index_names)} dress image names")

    global fashionIQ_val_shirt_index_features
    fashionIQ_val_shirt_index_features = torch.load(
        data_path / 'fashionIQ_val_shirt_index_features.pt', map_location=device).type(data_type).cpu()
    print(f"Loaded FashionIQ shirt index features of shape {fashionIQ_val_shirt_index_features.shape}")

    global fashionIQ_val_shirt_index_names
    with open(data_path / 'fashionIQ_val_shirt_index_names.pkl', 'rb') as f:
        fashionIQ_val_shirt_index_names = pickle.load(f)
    print(f"Loaded {len(fashionIQ_val_shirt_index_names)} shirt image names")

    global fashionIQ_val_toptee_index_features
    fashionIQ_val_toptee_index_features = torch.load(
        data_path / 'fashionIQ_val_toptee_index_features.pt', map_location=device).type(data_type).cpu()
    print(f"Loaded FashionIQ toptee index features of shape {fashionIQ_val_toptee_index_features.shape}")

    global fashionIQ_val_toptee_index_names
    with open(data_path / 'fashionIQ_val_toptee_index_names.pkl', 'rb') as f:
        fashionIQ_val_toptee_index_names = pickle.load(f)
    print(f"Loaded {len(fashionIQ_val_toptee_index_names)} toptee image names")

def evaluate_recall_fashionIQ(combiner: Combiner, ks=[10, 50]):
    """
    Evaluate Recall@K for each category in FashionIQ, and report mean.
    """
    print("===== FashionIQ Recall Evaluation =====")

    categories = ['dress', 'toptee', 'shirt']
    recall_per_category = {cat: {k: 0 for k in ks} for cat in categories}
    total_per_category = {cat: 0 for cat in categories}

    for triplet in tqdm(fashionIQ_val_triplets, desc="Evaluating"):
        caption = f"{triplet['captions'][0].strip('?,. ').capitalize()} and {triplet['captions'][1].strip('?,. ')}"
        reference_name = triplet['candidate']
        target_name = triplet['target']
        dress_type = triplet['dress_type']

        if dress_type == "dress":
            index_features = fashionIQ_val_dress_index_features
            index_names = fashionIQ_val_dress_index_names
        elif dress_type == "toptee":
            index_features = fashionIQ_val_toptee_index_features
            index_names = fashionIQ_val_toptee_index_names
        elif dress_type == "shirt":
            index_features = fashionIQ_val_shirt_index_features
            index_names = fashionIQ_val_shirt_index_names

        index_features = index_features.to(device)

        # Get visual features, extract textual features and compute combined features
        reference_index = index_names.index(reference_name)
        reference_features = index_features[reference_index].unsqueeze(0)

        text_inputs = clip.tokenize(caption, truncate=True).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
            predicted_features = combiner.combine_features(reference_features, text_features).squeeze(0)

        # Sort the results
        index_features = F.normalize(index_features)
        cos_similarity = index_features @ predicted_features.T
        sorted_indices = torch.topk(cos_similarity, max(ks), largest=True).indices.cpu()
        sorted_index_names = np.array(index_names)[sorted_indices].flatten()

        for k in ks:
            if target_name in sorted_index_names[:k]:
                recall_per_category[dress_type][k] += 1

        total_per_category[dress_type] += 1

    print("\n--- FashionIQ Retrieval Performance ---")
    avg_recall = {}
    for k in ks:
        category_recalls = []
        for cat in categories:
            recall = recall_per_category[cat][k] / total_per_category[cat]
            category_recalls.append(recall)
            print(f"{cat.capitalize()} Recall@{k}: {recall * 100:.2f}%")
        avg = sum(category_recalls) / len(category_recalls)
        avg_recall[k] = avg
        print(f"Mean Recall@{k}: {avg * 100:.2f}%\n")

    return avg_recall

def main():
    # Load CLIP model and Combiner networks
    global clip_model
    global clip_preprocess
    print("Loading CLIP model...")
    clip_model, clip_preprocess = clip.load("RN50x4")
    clip_model = clip_model.eval().to(device)

    print("Loading Combiner model...")
    combiner = torch.hub.load(server_base_path, source='local', model='combiner', dataset='fashionIQ')
    combiner = torch.jit.script(combiner).type(data_type).to(device).eval()

    # Load dataset assets
    load_fashionIQ_assets()

    # Run evaluation
    evaluate_recall_fashionIQ(combiner, ks=[10, 50])

# Run if main
if __name__ == "__main__":
    main()
