from typing import List, Dict, Tuple

import clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from data_utils import CIRCODataset
from utils import extract_image_features, device, data_type, collate_fn
from pathlib import Path

torch.multiprocessing.set_sharing_strategy('file_system')

server_base_path = Path(__file__).absolute().parent.absolute()

@torch.no_grad()
def circo_generate_val_predictions(clip_model: CLIP, relative_val_dataset: Dataset, index_features: torch.Tensor,
                                    index_names: List[str]) -> Tuple[torch.Tensor, List[str], list]:
    """
    Generates features predictions for the validation set of CIRCO
    """

    # Create the data loader
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=10,
                                     pin_memory=False, collate_fn=collate_fn, shuffle=False)

    predicted_features_list = []
    target_names_list = []
    gts_img_ids_list = []

    index_features = index_features.to(device)

    # Compute the features
    for batch in tqdm(relative_val_loader):
        reference_names = batch['reference_name']
        target_names = batch['target_name']
        relative_captions = batch['relative_caption']
        gt_img_ids = batch['gt_img_ids']

        gt_img_ids = np.array(gt_img_ids).T.tolist()
        input_captions = [f"a photo of $ that {caption}" for caption in relative_captions]
        tokenized_input_captions = clip.tokenize(input_captions, truncate=True).to(device)
        text_features = clip_model.encode_text(tokenized_input_captions)

        print("\nLoading Combiner model pretrained on FashionIQ...")
        combiner = torch.hub.load(server_base_path, source='local', model='combiner', dataset='fashionIQ')
        combiner = torch.jit.script(combiner).type(data_type).to(device).eval()

        index_name_to_feature = {name: feature for name, feature in zip(index_names, index_features)}
        reference_features = torch.stack([index_name_to_feature[name] for name in reference_names])

        predicted_features = combiner.combine_features(reference_features, text_features).squeeze(0)

        predicted_features_list.append(predicted_features)
        target_names_list.extend(target_names)
        gts_img_ids_list.extend(gt_img_ids)

    predicted_features = torch.vstack(predicted_features_list)

    return predicted_features, target_names_list, gts_img_ids_list


@torch.no_grad()
def circo_compute_val_metrics(relative_val_dataset: Dataset, clip_model: CLIP, index_features: torch.Tensor,
                              index_names: List[str]) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the CIRCO validation set given the dataset, pseudo tokens and the reference names
    """

    # Generate the predicted features
    predicted_features, target_names, gts_img_ids = circo_generate_val_predictions(clip_model, relative_val_dataset, index_features, index_names)
    ap_at5 = []
    ap_at10 = []
    ap_at25 = []
    ap_at50 = []

    recall_at5 = []
    recall_at10 = []
    recall_at25 = []
    recall_at50 = []

    # Move the features to the device
    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)

    # Normalize the features
    index_features = F.normalize(index_features)

    for predicted_feature, target_name, gt_img_ids in tqdm(zip(predicted_features, target_names, gts_img_ids)):
        gt_img_ids = np.array(gt_img_ids)[
            np.array(gt_img_ids) != '']  # remove trailing empty strings added for collate_fn
        similarity = predicted_feature @ index_features.T
        sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
        sorted_index_names = np.array(index_names)[sorted_indices]
        map_labels = torch.tensor(np.isin(sorted_index_names, gt_img_ids), dtype=torch.uint8)
        precisions = torch.cumsum(map_labels, dim=0) * map_labels  # Consider only positions corresponding to GTs
        precisions = precisions / torch.arange(1, map_labels.shape[0] + 1)  # Compute precision for each position

        ap_at5.append(float(torch.sum(precisions[:5]) / min(len(gt_img_ids), 5)))
        ap_at10.append(float(torch.sum(precisions[:10]) / min(len(gt_img_ids), 10)))
        ap_at25.append(float(torch.sum(precisions[:25]) / min(len(gt_img_ids), 25)))
        ap_at50.append(float(torch.sum(precisions[:50]) / min(len(gt_img_ids), 50)))

        assert target_name == gt_img_ids[0], f"Target name not in GTs {target_name} {gt_img_ids}"
        single_gt_labels = torch.tensor(sorted_index_names == target_name)
        recall_at5.append(float(torch.sum(single_gt_labels[:5])))
        recall_at10.append(float(torch.sum(single_gt_labels[:10])))
        recall_at25.append(float(torch.sum(single_gt_labels[:25])))
        recall_at50.append(float(torch.sum(single_gt_labels[:50])))

    map_at5 = np.mean(ap_at5) * 100
    map_at10 = np.mean(ap_at10) * 100
    map_at25 = np.mean(ap_at25) * 100
    map_at50 = np.mean(ap_at50) * 100
    recall_at5 = np.mean(recall_at5) * 100
    recall_at10 = np.mean(recall_at10) * 100
    recall_at25 = np.mean(recall_at25) * 100
    recall_at50 = np.mean(recall_at50) * 100

    return {
        'circo_map_at5': map_at5,
        'circo_map_at10': map_at10,
        'circo_map_at25': map_at25,
        'circo_map_at50': map_at50,
        'circo_recall_at5': recall_at5,
        'circo_recall_at10': recall_at10,
        'circo_recall_at25': recall_at25,
        'circo_recall_at50': recall_at50,
    }


@torch.no_grad()
def circo_val_retrieval(dataset_path: str, clip_model_name: str) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the CIRCO validation set given the pseudo tokens and the reference names
    """
    # Load the model
    clip_model, preprocess = clip.load(clip_model_name, device=device, jit=False)
    clip_model = clip_model.eval().requires_grad_(False)

    # Extract the index features
    classic_val_dataset = CIRCODataset(dataset_path, 'val', 'classic', preprocess)
    index_features, index_names = extract_image_features(classic_val_dataset, clip_model)

    # Define the relative validation dataset
    relative_val_dataset = CIRCODataset(dataset_path, 'val', 'relative', preprocess)

    return circo_compute_val_metrics(relative_val_dataset, clip_model, index_features, index_names)


def main():
    dataset_path = Path('/kaggle/input/circo-circo')
    clip_model_name = 'RN50x4'
    
    circo_metrics = circo_val_retrieval(dataset_path, clip_model_name)

    for k, v in circo_metrics.items():
            print(f"{k} = {v:.2f}")


if __name__ == '__main__':
    main()
