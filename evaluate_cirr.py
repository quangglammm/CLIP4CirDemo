import torch
import torch.nn.functional as F
import json
import clip
from pathlib import Path
import pickle
from tqdm import tqdm

# Constants
server_base_path = Path(__file__).absolute().parent
data_path = server_base_path / 'data'

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

    with open(server_base_path / 'cirr_dataset' / 'cirr' / 'captions' / 'cap.rc2.val.json') as f:
        cirr_val_triplets = json.load(f)
    print(f"Loaded {len(cirr_val_triplets)} validation triplets")

    cirr_val_index_features = torch.load(data_path / 'cirr_val_index_features.pt', map_location=device).type(data_type)
    print(f"Loaded CIRR index features of shape {cirr_val_index_features.shape}")

    with open(data_path / 'cirr_val_index_names.pkl', 'rb') as f:
        cirr_val_index_names = pickle.load(f)
    print(f"Loaded {len(cirr_val_index_names)} image names")


def evaluate_recall_cirr(combiner, ks=[1, 5, 10, 50], subset_ks=[1, 2, 3]):
    """
    Evaluate Recall@K and Recall_subset@K on CIRR dataset.
    """
    print("===== CIRR Recall Evaluation =====")

    cirr_index_features = cirr_val_index_features.to(device)
    cirr_index_names = list(cirr_val_index_names)
    total = len(cirr_val_triplets)

    # Init counters
    recall_counts = {k: 0 for k in ks}
    recall_subset_counts = {k: 0 for k in subset_ks}

    for triplet in tqdm(cirr_val_triplets, desc="Evaluating CIRR"):
        caption = triplet['caption']
        reference_name = triplet['reference']
        target_name = triplet['target_hard']

        # 1. Encode reference image
        try:
            ref_index = cirr_index_names.index(reference_name)
        except ValueError:
            print(f"[Warning] Reference image '{reference_name}' not found in index.")
            continue
        reference_feature = cirr_index_features[ref_index].unsqueeze(0)

        # 2. Encode text and combine
        with torch.no_grad():
            text_input = clip.tokenize(caption, truncate=True).to(device)
            text_feature = clip_model.encode_text(text_input)
            combined_feature = combiner.combine_features(reference_feature, text_feature).squeeze(0)

        # 3. Compute similarity with all index features (Recall@K)
        sim = cirr_index_features @ combined_feature.unsqueeze(1)
        sim = sim.squeeze(1)  # Shape: [2297]
        sim[ref_index] = sim.min() - 1  # Exclude the reference image safely
        top_indices = torch.topk(sim, k=max(ks), largest=True).indices.cpu().numpy()
        top_names = [cirr_index_names[i] for i in top_indices]

        for k in ks:
            if target_name in top_names[:k]:
                recall_counts[k] += 1

        # 4. Compute subset recall@K (within group)
        group_members = triplet['img_set']['members']
        try:
            group_indices = [cirr_index_names.index(name) for name in group_members]
        except ValueError as e:
            print(f"[Warning] Image missing in group set: {e}")
            continue

        group_features = cirr_index_features[group_indices]
        group_sim = group_features @ combined_feature.T
        sorted_group_indices = torch.argsort(group_sim, descending=True).cpu().numpy()
        sorted_group_names = [group_members[i] for i in sorted_group_indices]

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
    clip_model, clip_preprocess = clip.load("RN50x4", device=device)
    clip_model.eval()

    print("Loading Combiner model...")
    combiner = torch.hub.load(server_base_path, source='local', model='combiner', dataset='cirr')
    combiner = torch.jit.script(combiner).type(data_type).to(device).eval()

    load_cirr_assets()

    evaluate_recall_cirr(combiner, ks=[1, 5, 10, 50], subset_ks=[1, 2, 3])


if __name__ == "__main__":
    main()
