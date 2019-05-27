<p align="center">
    <a href="#readme">
        <img alt="logo" width="70%" src="computer-vision.png">
    </a>
</p>

## How-to, original competition, https://www.aiforsea.com/computer-vision

1. Download and unzip dataset using [dataset.ipynb](dataset.ipynb).

2. Preprocessing using [preprocessing.ipynb](preprocessing.ipynb).

3. Augmentation using [augmentation.ipynb](augmentation.ipynb).

    1. Random flip
    2. Random rotate
    3. Random shift
    4. Random zoom
    5. Random shear
    6. Random channel shift
    7. Random gray
    8. Random contrast
    9. Random brightness
    10. Random saturation
    11. Change light to night
    12. Random shadow
    13. Random snow
    14. Random rain
    15. Random fog

4. Download pretrained Inception-Resnet-V2 [download-checkpoint.ipynb](download-checkpoint.ipynb)

5. Transfer learning Inception-Resnet-V2 [transfer-learning.ipynb](transfer-learning.ipynb)

## To get better accuracy and less overfit on private dataset

1. Put dropout layer after first logits on [transfer-learning.ipynb](transfer-learning.ipynb).

2. Apply L2 / L1 / L1 + L2 penalty on cost function on [transfer-learning.ipynb](transfer-learning.ipynb).

3. Train another pretrained model like Inception V3 and average the results.

4. Stack the results using any trees algorithm.
