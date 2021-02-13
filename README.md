# PeopleCounter
Detects and counts people in a video using a pretrained object detection model. The model is converted with Intel's OpenVino for inference.
The detection model is ssdlite_mobilenet_v2_coco and can be obtained from:

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#mobile-models

The model is converted with OpenVino to use on Intel devices with

python ..intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config ..intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

The model is about 73% accurate. To avoid double counting a person, simple features using the mean and standard deviation of pixels in the detected box are used to keep track of the person. 


![detector](https://user-images.githubusercontent.com/5798711/105663512-4cfa4c80-5e87-11eb-8377-cd83b101793e.gif)

![Person_detect1](https://user-images.githubusercontent.com/5798711/104991295-45442f00-59d3-11eb-85f0-6487a7c36699.PNG)


Sample application in detecting congestion in train systems.

![detectorTrain](https://user-images.githubusercontent.com/5798711/107834934-a8b04b00-6d4c-11eb-928c-14df6eadb920.gif)

Frames/second processing time on various devices tested on Intel development cloud
1. CPU (i7)
2. GPU (i7 with integrated GPU)
3. VPU (Myriad visual processing unit)
4. FPGA



![fps_trains](https://user-images.githubusercontent.com/5798711/107835157-62a7b700-6d4d-11eb-8750-da3429dae9c3.png)




