import torch
import torch.nn.functional as F
import json
import clip
from pathlib import Path
import pickle
from tqdm import tqdm

# Constants
server_base_path = Path(__file__).absolute().parent.absolute()
data_path = Path(__file__).absolute().parent.absolute() / 'data'

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    data_type = torch.float16
else:
    device = torch.device("cpu")
    data_type = torch.float32

def load_fashionIQ_assets():
    """
    Load fashionIQ assets
    """
    global fashionIQ_val_triplets
    fashionIQ_val_triplets = []
    for dress_type in ['dress', 'toptee', 'shirt']:
        with open(server_base_path / 'fashionIQ_dataset' / 'captions' / f'cap.{dress_type}.val.json') as f:
            dress_type_captions = json.load(f)
            captions = [dict(caption, dress_type=f'{dress_type}') for caption in dress_type_captions]
            fashionIQ_val_triplets.extend(captions)

    global fashionIQ_val_dress_index_features
    fashionIQ_val_dress_index_features = torch.load(
        data_path / 'fashionIQ_val_dress_index_features.pt', map_location=device).type(data_type).cpu()

    global fashionIQ_val_dress_index_names
    with open(data_path / 'fashionIQ_val_dress_index_names.pkl', 'rb') as f:
        fashionIQ_val_dress_index_names = pickle.load(f)

    global fashionIQ_val_shirt_index_features
    fashionIQ_val_shirt_index_features = torch.load(
        data_path / 'fashionIQ_val_shirt_index_features.pt', map_location=device).type(data_type).cpu()

    global fashionIQ_val_shirt_index_names
    with open(data_path / 'fashionIQ_val_shirt_index_names.pkl', 'rb') as f:
        fashionIQ_val_shirt_index_names = pickle.load(f)

    global fashionIQ_val_toptee_index_features
    fashionIQ_val_toptee_index_features = torch.load(
        data_path / 'fashionIQ_val_toptee_index_features.pt', map_location=device).type(data_type).cpu()

    global fashionIQ_val_toptee_index_names
    with open(data_path / 'fashionIQ_val_toptee_index_names.pkl', 'rb') as f:
        fashionIQ_val_toptee_index_names = pickle.load(f)

def evaluate_recall_fashionIQ(combiner, ks=[10, 50]):
    """
    Evaluate Recall@K for each category in FashionIQ, and report mean.
    """
    categories = ['dress', 'toptee', 'shirt']
    recall_per_category = {cat: {k: 0 for k in ks} for cat in categories}
    total_per_category = {cat: 0 for cat in categories}

    for triplet in tqdm(fashionIQ_val_triplets, desc="Evaluating"):
        caption = f"{triplet['captions'][0].strip('?,. ').capitalize()} and {triplet['captions'][1].strip('?,. ')}"
        reference_name = triplet['candidate']
        target_name = triplet['target']
        dress_type = triplet['dress_type']

        if dress_type not in categories:
            continue

        if dress_type == "dress":
            index_features = fashionIQ_val_dress_index_features.to(device)
            index_names = fashionIQ_val_dress_index_names
        elif dress_type == "toptee":
            index_features = fashionIQ_val_toptee_index_features.to(device)
            index_names = fashionIQ_val_toptee_index_names
        elif dress_type == "shirt":
            index_features = fashionIQ_val_shirt_index_features.to(device)
            index_names = fashionIQ_val_shirt_index_names

        if reference_name not in index_names:
            continue

        ref_idx = index_names.index(reference_name)
        reference_feat = index_features[ref_idx].unsqueeze(0)

        text_tokens = clip.tokenize(caption, truncate=True).to(device)
        with torch.no_grad():
            text_feat = clip_model.encode_text(text_tokens)

        with torch.no_grad():
            combined_feat = combiner.combine_features(reference_feat, text_feat).squeeze(0)

        index_features = F.normalize(index_features, dim=1)
        combined_feat = F.normalize(combined_feat, dim=0)
        sims = index_features @ combined_feat.T
        top_indices = torch.topk(sims, max(ks), largest=True).indices.cpu().numpy()
        top_names = [index_names[i] for i in top_indices]

        for k in ks:
            if target_name in top_names[:k]:
                recall_per_category[dress_type][k] += 1

        total_per_category[dress_type] += 1

    print("\nFashionIQ Recall Evaluation")
    avg_recall = {}
    for k in ks:
        category_recalls = []
        for cat in categories:
            recall = recall_per_category[cat][k] / total_per_category[cat]
            category_recalls.append(recall)
            print(f"{cat.capitalize()} Recall@{k}: {recall * 100:.2f}%")
        avg = sum(category_recalls) / len(category_recalls)
        avg_recall[k] = avg
        print(f"Mean Recall@{k}: {avg * 100:.2f}%")

    return avg_recall

def main():
    # Load CLIP model and Combiner networks
    global clip_model
    global clip_preprocess
    clip_model, clip_preprocess = clip.load("RN50x4")
    clip_model = clip_model.eval().to(device)

    combiner = torch.hub.load(server_base_path, source='local', model='combiner', dataset='fashionIQ')
    combiner = torch.jit.script(combiner).type(data_type).to(device).eval()

    # Load dataset assets
    load_fashionIQ_assets()

    # Run evaluation
    evaluate_recall_fashionIQ(combiner, ks=[10, 50])


# Run if main
if __name__ == "__main__":
    main()
