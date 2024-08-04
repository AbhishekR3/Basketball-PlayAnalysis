********************************************************************************************************************
To see the dataset and model please refer to the below link
https://universe.roboflow.com/basketballplayanalysis/customobjectdetection_data

License: CC BY 4.0
********************************************************************************************************************

The dataset includes 63 images.
Classes annotated in YOLOv10 format
- Basketball
- Team_A
- Team_B

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)
* Grayscale

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Randomly crop between 0 and 15 percent of the image
* Random brigthness adjustment of between -20 and +20 percent
* Random exposure adjustment of between -13 and +13 percent
* Random Gaussian blur of between 0 and 4 pixels
* Salt and pepper noise was applied to 1.53 percent of pixels

********************************************************************************************************************

Overall, the model performs significantly better than the generic pre-trained YOLOv10m model.
The custom model records the following metrics:
- mAP50     - 97.7%
- mAP50-95  - 73.1%
- Precision - 97.3%
- Recall    - 89.1%

Training/Validation results and relevant metrics can be found in Training Results document in the same folder 

********************************************************************************************************************