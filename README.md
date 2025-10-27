# Mercury: Tiny Learning on IoT Devices 

This is the official implementation of the MCUNet series.

## Overview
Mercury is a **system-algorithm co-design** framework for tiny deep learning on microcontrollers. It consists of **TinyNAS** and **TinyEngine**. They are co-designed to fit the tight memory budgets.

With system-algorithm co-design, we can significantly improve the deep learning performance on the same tiny memory budget.

Our **TinyEngine** inference engine could be a useful infrastructure for MCU-based AI applications. It significantly **improves the inference speed and reduces the memory usage** compared to existing libraries like etc.

## Model Zoo

### Usage

You can build the pre-trained PyTorch `fp32` model or the `int8` quantized model in TF-Lite format.

```python
from mercury.model_zoo import net_id_list, build_model, download_tflite
print(net_id_list)  # the list of models in the model zoo

# pytorch fp32 model
model, image_size, description = build_model(net_id="mcunet-in3", pretrained=True)  # you can replace net_id with any other option from net_id_list

# download tflite file to tflite_path
tflite_path = download_tflite(net_id="mcunet-in3")
```


### Evaluate

To evaluate the accuracy of PyTorch `fp32` models, run:

```bash
python eval_torch.py --net_id mcunet-in2 --dataset {imagenet/vww} --data-dir PATH/TO/DATA/val
```

To evaluate the accuracy of TF-Lite `int8` models, run:

```bash
python eval_tflite.py --net_id mcunet-in2 --dataset {imagenet/vww} --data-dir PATH/TO/DATA/val
```

It will visualize the prediction here: `assets/sample_images/person_det_vis.jpg`.

The model takes in a small input resolution of 128x160 to reduce memory usage. It does not achieve state-of-the-art performance due to the limited image and model size but should provide decent performance for tinyML applications (please check the demo for a video recording). We will also release the deployment code in the upcoming TinyEngine release. 

## Requirement

- Python 3.6+

- PyTorch 1.4.0+

- Tensorflow 1.15 (if you want to test TF-Lite models; CPU support only)
