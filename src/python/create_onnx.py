import torch
import onnx
import onnxruntime

import util
import model_architectures
import training_config

import util

import json

if __name__ == "__main__":
    with open('secrets.json', 'r') as file:
        secrets = json.load(file)
    file.close()
    
    multi_classifier_config = training_config.bracketClassifierConfig()
    util.export_to_onnx(
        onnx_base_path = secrets['onnx_models']['base_path'],
        onnx_name="bracketClassifier",
        model= model_architectures.multiOutputNet(
            size_in=32, 
            hidden_sizes_shared=multi_classifier_config.HIDDEN_SIZES_SHARED, 
            hidden_sizes_a=multi_classifier_config.HIDDEN_SIZES_A, 
            hidden_sizes_b=multi_classifier_config.HIDDEN_SIZES_B, 
            size_out1=4, 
            size_out2=5, 
            dropout_p=multi_classifier_config.DROPOUT_RATE
        ),
        checkpoint = util.load_checkpoint(
            util.find_latest_file(multi_classifier_config.BASE_PATH)
        ),
        input_size=32
    )
    
    modality_classifier_config = training_config.modalityClassifierConfig()
    util.export_to_onnx(
        onnx_base_path = secrets['onnx_models']['base_path'],
        onnx_name="modalityClassifier",
        model= model_architectures.ModalityTypeClassifier(
            size_in=41,
            num_hidden=modality_classifier_config.NUM_HIDDEN,
            hidden_size=modality_classifier_config.HIDDEN_SIZE,
            size_out=3,
            dropout_p=modality_classifier_config.DROPOUT_RATE
        ),
        checkpoint = util.load_checkpoint(
            util.find_latest_file(modality_classifier_config.BASE_PATH)
        ),
        input_size=41
    )
    