import numpy as np
import sys
import os
import json
from google.protobuf import text_format as pbtf
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from luperson_inference.LUPerson_inference import TritonPythonModel

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_array_path", type=str, default='./tests/data/sampleinputimagearray.npy')
    parser.add_argument("--gt_query_feat_path", type=str, default='./tests/data/gt_inference_query_feat.npy')
    args = parser.parse_args()

    img_path_file = args.img_array_path
    gt_query_feat_file = args.gt_query_feat_path

    image_array = np.load(img_path_file, allow_pickle=True).astype(np.float32)
    gt_query_feat = np.load(gt_query_feat_file, allow_pickle=True)

    dummy_model_config = {}
    dummy_model_config['output'] = []
    dummy_model_config['output'].append({'name': "OUTPUT0", 'data_type': 'TYPE_FP32', 'dims': [-1, 2048]})

    inference_model = TritonPythonModel()
    inference_model.initialize(args={'model_config': json.dumps(dummy_model_config, indent=2)})

    query_feat = inference_model.get_embeddings(image_array)
    query_feat2 = inference_model.get_embeddings_batch(image_array[None, :])
    print(query_feat.shape)

    assert gt_query_feat.shape == query_feat.shape == query_feat2.shape
    assert np.sum((gt_query_feat - query_feat)**2) == 0
    assert np.sum((gt_query_feat - query_feat2)**2) == 0

    print("Test successful!!")


