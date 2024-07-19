********************************************************************************************************************

I had a great experience using Roboflow to annotate and create training/validation/test data for my project.
I would recommend it to others who are building personal project.  

********************************************************************************************************************
To see the dataset and model please refer to the below link
https://universe.roboflow.com/basketballplayanalysis/customobjectdetection_data

License: CC BY 4.0

BasketballPlayAnalysis - v1 2024-07-17 9:32pm

********************************************************************************************************************

This dataset was exported via roboflow.com on July 18, 2024 at 3:42 AM GMT

The dataset includes 63 images.
Classes annotated in YOLOv9 format
- Basketball
- Team_A
- Team_B

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

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

Overall, the model performs significantly better than the generic pre-trained YOLOv9c model.
The custom model records a mAP50 value of 98.4% and recall value of 98.2%

Training/Validation results and relevant metrics can be found in Training Results document in the same folder 

********************************************************************************************************************