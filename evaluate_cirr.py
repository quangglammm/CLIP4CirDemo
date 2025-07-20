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
server_base_path = Path(__file__).absolute().parent
data_path = server_base_path / 'data'

cirr_path = Path('/kaggle/input/cirr-cir')

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    data_type = torch.float16
else:
    device = torch.device("cpu")
    data_type = torch.float32


def load_cirr_assets():
    print("Loading CIRR assets...")

    global cirr_val_triplets, cirr_val_index_features, cirr_val_index_names

    with open(cirr_path / 'CIRR' / 'cirr' / 'captions' / 'cap.rc2.val.json') as f:
        cirr_val_triplets = json.load(f)
    print(f"Loaded {len(cirr_val_triplets)} validation triplets")

    cirr_val_index_features = torch.load(data_path / 'cirr_val_index_features.pt', map_location=device).type(data_type)
    print(f"Loaded CIRR index features of shape {cirr_val_index_features.shape}")

    with open(data_path / 'cirr_val_index_names.pkl', 'rb') as f:
        cirr_val_index_names = pickle.load(f)
    print(f"Loaded {len(cirr_val_index_names)} image names")


def evaluate_recall_cirr(combiner: Combiner, ks=[1, 5, 10, 50], subset_ks=[1, 2, 3]):
    """
    Evaluate Recall@K and Recall_subset@K on CIRR dataset.
    """
    print("===== CIRR Recall Evaluation =====")

    index_features = cirr_val_index_features.to(device)
    index_names = list(cirr_val_index_names)
    total = len(cirr_val_triplets)

    # Init counters
    recall_counts = {k: 0 for k in ks}
    recall_subset_counts = {k: 0 for k in subset_ks}

    for triplet in tqdm(cirr_val_triplets, desc="Evaluating CIRR"):
        target_name = triplet['target_hard']
        group_members = triplet['img_set']['members']
        reference_name = triplet['reference']
        caption = triplet['caption']

        # Get visual features, extract textual features and compute combined features
        text_inputs = clip.tokenize(caption, truncate=True).to(device)
        try:
            reference_index = index_names.index(reference_name)
            reference_features = index_features[reference_index].unsqueeze(0)
        except ValueError:
            print(f"[Warning] Reference image '{reference_name}' not found in index.")
            continue

        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
            predicted_features = combiner.combine_features(reference_features, text_features).squeeze(0)

        # Compute similarity with all index features (Recall@K)
        index_features = F.normalize(index_features)
        cos_similarity = index_features @ predicted_features.T
        sorted_indices = torch.topk(cos_similarity, k=max(ks), largest=True).indices.cpu()
        sorted_index_names = np.array(index_names)[sorted_indices].flatten()
        sorted_index_names = np.delete(sorted_index_names, np.where(sorted_index_names == reference_name))

        for k in ks:
            if target_name in sorted_index_names[:k]:
                recall_counts[k] += 1

        # Compute subset recall@K (within group)
        group_indices = [index_names.index(name) for name in group_members]
        group_features = index_features[group_indices]
        cos_similarity = group_features @ predicted_features.T
        group_sorted_indices = torch.argsort(cos_similarity, descending=True).cpu()
        sorted_group_names = np.array(group_members)[group_sorted_indices]
        sorted_group_names = np.delete(sorted_group_names, np.where(sorted_group_names == reference_name)).tolist()

        for k in subset_ks:
            if target_name in sorted_group_names[:k]:
                recall_subset_counts[k] += 1

    # Print results
    print("\n--- CIRR Retrieval Performance ---")
    for k in ks:
        r = recall_counts[k] / total
        print(f"Recall@{k}: {r * 100:.2f}%")
    for k in subset_ks:
        r = recall_subset_counts[k] / total
        print(f"Recall_subset@{k}: {r * 100:.2f}%")

    # Return for logging or analysis
    return {
        f"recall@{k}": recall_counts[k] / total for k in ks
    } | {
        f"recall_subset@{k}": recall_subset_counts[k] / total for k in subset_ks
    }

def main():
    global clip_model, clip_preprocess
    print("Loading CLIP model...")
    clip_model, clip_preprocess = clip.load("RN50x4")
    clip_model = clip_model.eval().to(device)

    print("Loading Combiner model...")
    combiner = torch.hub.load(server_base_path, source='local', model='combiner', dataset='cirr')
    combiner = torch.jit.script(combiner).type(data_type).to(device).eval()

    load_cirr_assets()

    evaluate_recall_cirr(combiner, ks=[1, 5, 10, 50], subset_ks=[1, 2, 3])

if __name__ == "__main__":
    main()
