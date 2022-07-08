import numpy as np
from PIL import Image
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from LUPerson_inference import LUPersonInferenceModel

if __name__ == '__main__':
    config_file = './configs/LaST/mgn_R50_moco_cache_test.yml'
    img_path_file = './tests/data/sampleinputimagearray.npy'
    gt_query_feat_file = './tests/data/gt_inference_query_feat.npy'

    image_array = np.load(img_path_file, allow_pickle=True)
    gt_query_feat = np.load(gt_query_feat_file, allow_pickle=True)

    inference_model = LUPersonInferenceModel()
    inference_model.initialize(config_file=config_file)
    query_feat = inference_model.get_embeddings(image_array)

    assert gt_query_feat.shape == query_feat.shape
    assert np.sum((gt_query_feat - query_feat)**2) == 0

    print("Test successful!!")


