# Ideas to test

* [x] Gradient accumulation => helps to reduce validation metric variance and produces better metrics

## Other loss functions

* [x] CrossEntropy + Jaccard loss => seems to give similar result as single CrossEntropy
    * [x] DeeplabV3-Resnet101
        * [x] compare with CrossEntropy only
            - CrossEntropy: train/test mIoU: 0.857969033992809	0.689686381429643
            - CrossEntropy + Jaccard loss: 0.856403252601405	0.6932224292220975

## Models

* [ ] SE-ResNeXt50-FPN
    - CrossEntropy + Jaccard loss: train/test mIoU: 


## Faster training

* [ ] minibatch persistence/data echoing to accelerate training  

## More complex trainings

* [ ] Unsupervised Data Augmentation
* [ ] XEntropy + "Predict on Prediction" consistency:
    - loss = xentropy(y_pred, y_true) + a * l1_loss(model(y_pred), y_pred)
