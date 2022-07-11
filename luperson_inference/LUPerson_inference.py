import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from minifastreid.transforms.transforms import ToTensor

from minifastreid.utils.checkpoint import Checkpointer
from minifastreid.modeling.meta_arch import build_model
from minifastreid.config import get_cfg
import numpy as np
import os

import triton_python_backend_utils as pb_utils
import json

class LUPersonInferenceModel:
    def initialize(self, args):
        """
        ----------
        args : 
          * Path to config file containing all params
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT0")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),\
             'configs', 'LaST', 'mgn_R50_moco_cache_test.yml')

        assert config_file is not None, "Please provide a config file for the inference model"

        self.config_file = config_file
        self.cfg = get_cfg()
        self.cfg.merge_from_file(self.config_file)
        self.cfg.defrost()
        self.cfg.MODEL.BACKBONE.PRETRAIN = False
        self.model = build_model(self.cfg)
        self.transform = self.build_transforms()
        self.model.eval()

        Checkpointer(self.model).load(self.cfg.MODEL.WEIGHTS)
        print("Loaded pretrained weights from: ", self.cfg.MODEL.WEIGHTS)

    def build_transforms(self):
        res = []
        size_test = self.cfg.INPUT.SIZE_TEST
        res.append(T.ToPILImage())
        res.append(T.Resize(size_test, interpolation=3))
        res.append(ToTensor())
        return T.Compose(res)

    def execute(self, requests):
        """
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []
        for request in requests:
            # [H, W, C] in float32
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0").as_numpy()

            # Add extra 0th dimension if only a single image
            if len(in_0.shape) == 3:
                in_0 = in_0[None, :]

            out_0 = self.get_embeddings_batch(in_0)

            out_tensor_0 = pb_utils.Tensor("OUTPUT0",
                                           out_0.astype(self.output0_dtype))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            responses.append(inference_response)

        return responses


    def get_embeddings_batch(self, image):
        """
        Args:
            image (numpy.ndarray): a numpy array of shape (B, H, W, C) in [0.0, 255.0]
        Returns:
            predictions (torch.tensor): the output features of the model
        """
        # Convert to uint8
        image = image.astype(np.uint8)

        # Apply transforms to get tensor
        transformed_images = []
        for image_item in image:
            transformed_images.append(self.transform(image_item).unsqueeze(0))

        transformed_images = torch.cat(transformed_images)
        
        # Expects image_content (torch.tensor): an image tensor of shape (B, C, H, W).
        inputs = {"images": transformed_images}

        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            features = self.model(inputs)
            # Normalize feature to compute cosine distance
            # features = F.normalize(features)
            return features.cpu().data.numpy()


    def get_embeddings(self, image):
        """
        Args:
            image (numpy.ndarray): a numpy array of shape (H, W, C) in [0.0, 255.0]
        Returns:
            predictions (torch.tensor): the output features of the model
        """
        # Convert to uint8
        image = image.astype(np.uint8)

        # Apply transforms to get tensor
        image_content = self.transform(image)
        image_content = image_content[None, :]
        
        # Expects image_content (torch.tensor): an image tensor of shape (B, C, H, W).
        inputs = {"images": image_content}

        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            features = self.model(inputs)
            # Normalize feature to compute cosine distance
            # features = F.normalize(features)
            return features.cpu().data.numpy()

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
        del self.model