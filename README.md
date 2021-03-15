# incubitChallenge

# Code and results

[link 1]

[![open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/avasalya/7645c4a1aa919eeff0046b3561434bf1/satellitev5.ipynb)

[link 2]

[![open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://gist.github.com/avasalya/7645c4a1aa919eeff0046b3561434bf1#file-satellitev5-ipynb
)



# Problem Analysis

I believe the given problem lies within the broad domain of `object detection` and `instance segmentation`. Since we need to detect and classify the type of structure(s) present and also be able to generate a closed-polygon around the object of interest.

Based on the annotation data available:
- I will address the problem using `Mask_RCNN`, a state-of-the-art object detection and segmentation method. Reference [https://github.com/matterport/Mask_RCNN]
- I also worked on a new segmentation model from scratch using the pre-trained models, however due to time and GPU constraint, the results obtained are limited to only `semantically segment` the object of interest(s). Please find the kaggle link [https://www.kaggle.com/avasalya/segmentation], for semantic segmentation approach.



# Approach

- I noticed, annotation data doesn't contain `bounding box` of the object of interest, therefore I will first extract closed-bounding box from the given segmentation polygon coordinates using `OpenCV` library.

- The input images are huge in size and due to GPU limitations, input images needs to be resized/scaled down to handle batch_size of atleast 2.

- I have created a `SatelliteDataset` Class to load, process, scale the given dataset and also to generate mask images.

- The corresponding mask image has same (H x W x C) as resize-colored image and the number of channels `C` corresponds to the total no of object polygons present within each frame.

- There are 4 classes in total, namely ('Bg', 'Houses', 'Buildings', 'Sheds/Garages')

- Since our images are huge and many of the object of interest are very small, therefore I have selected RPN (region proposal network) anchor scale range as (8, 16, 32, 64, 128). This range should be able to identify both small and huge objects.

- The datasets were randomly divided using `Sklearn` library.

- Losses used within Mask RCNN, all losses had default starting value of 1.
  - rpn_class_loss : Region Proposal Network, which separates background with objects
  - rpn_bbox_loss : Measures RPN's ability to localize objects
  - mrcnn_bbox_loss : Measures Mask RCNN's ability to localize objects
  - mrcnn_class_loss : How well the Mask RCNN recognize each class of object
  - mrcnn_mask_loss : How well the Mask RCNN segment objects
  - loss : A combination of all the above losses.


- I have trained and tested the Mask_RCNN base-model with the following variations of hyper-parameters using `TensorFlow 1.5`:

  - V1: pre-trained coco, lr =0.001 data-split (80, 20), STEPS_PER_EPOCH=100, VALIDATION_STEPS=5, heads->all_layers epochs 10, 40
  - V2: pre-trained coco, lr =0.001 data-split (80, 20), STEPS_PER_EPOCH=200, VALIDATION_STEPS=10, heads->all_layers epochs 40, 60
  - V3: pre-trained imagenet, lr =0.01  data-split (80, 20), STEPS_PER_EPOCH=200, VALIDATION_STEPS=15, heads->all_layers epochs 50, 100
  - V4: pre-trained coco, lr =0.01  data-split (60, 40), STEPS_PER_EPOCH=150, VALIDATION_STEPS=20, heads->all_layers epochs 30, 60
  - V5: pre-trained imagenet, lr =0.001 data-split (70, 30), STEPS_PER_EPOCH=1000, VALIDATION_STEPS=50, heads->res4+->all_layers epochs 50,100,131



# Results & Conclusion

- Since our dataset is small and the number of object of interest are huge (in some samples total no of instances are more than 600). We have possible issue of `overtfitting`, which can be seen in the `validation loss curves`. (see section EVALUATION [block 9 onwards])
  - The overtfitting mainly occurs in the `Class` and `Mask` losses. Doing a sanity check, this is however understandable since our dataset is small and there is a huge class imbalance.

- Within all 5 versions as mentioned above, I have tried to address the issue by tuning hyper-parameters. V5 (*this notebook*) has the most promising result.

- The mean APs on all validation dataset samples is ***0.44*** which is acceptable, however addressing overtfitting issue can considerably improve the performance.

- The Precision-Recall with AP@50 is ***0.3***.

- Please go through section ***EVALUATION*** for further results/figures.

- The trained model is tested on 9 (non annotated) samples which were not part of training and validation datasets.

- The output of the model is `json file` with the following format.
   
```
{'filename':file_name,

'labels': [{'name': label_name, 'annotations': [{'id':some_unique_integer_id, 'segmentation':[x,y,x,y,x,y....]}
                                             ....] }
        ....]
}
```





# Hurdles
- I believe, the problem of `overtfitting` needs to be addressed thoroughly. However it is difficult to address it with the small size of given dataset and GPU constraint.

- Moreoever, in the given dataset, there is a huge `class imbalance` problem, especially with the class `sheds/garages`. In most samples, the no of instances are in the ratio (approximate) of 5:10:1 (houses:buildings:garages)

- Since the input images are huge in size, both the `Kaggle` and `Google colab` can't handle batch size of more than 2, even after resizing the images to (512, 512). lowering further will reduce the resolution and hence more difficult to extract relevant features.

- I believe, having access to a powerful GPU is essential.



# Ideas

In future iterations, given enough time and resources, I would try the following to improve the performance.

- To minimize `Class imbalance` issue, replace simple dataset split by `k-fold stratified cross validation`.

- Add batch normalization to the already existing model.

- Add several data augmentation, since the dataset is small.

- One way to increase the number of samples is by spliting each full sized image into 3x3 grids and add zero padding wherever required. This will elevate two important issues.
  - increase the number of samples by x9 times.
  - assist in reducing `class` and `mask` losses, without changing `anchor scales`.
  - Possibly third, improve class imbalance problem as well.

- Another method, replace default `L2` regularization of Mask RCNN with `elastic` regularization (`L1 + L2`) or even the dropout regularization.

- Also, increaing the iterations per epoch and total number of epochs during the training and the validation steps.
