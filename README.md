# PeopleCounter
Detects and counts people in a video using a pretrained object detection model. The model is converted with Intel's OpenVino for inference.
The detection model is ssdlite_mobilenet_v2_coco and can be obtained from:

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#mobile-models

The model is converted with OpenVino to use on Intel devices with
python ..intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config ..intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

![Person_detect1](https://user-images.githubusercontent.com/5798711/104991295-45442f00-59d3-11eb-85f0-6487a7c36699.PNG)




