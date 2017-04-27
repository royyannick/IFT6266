# IFT6266 - Convolutional Auto-Encoder
This CAE is for the class project for IFT6266 : Conditional Image Generation

### Model :
[Insert Image Here]

### Data :
You need to modify the paths in the code to point to you inpainting folder that should contain:
1. train2014/    - Images for training. (64x64 RGB - not cropped)
2. val2014/      - Images for validation.
3. *.pkl         - Pickle file with labels.

The Grayscale Images will be excluded by the script.
The Images will be cropped by the script.

### Results : 
Some early results, trying to see if longer training is better.
Trained on 50 000 images from the train2014 folder of IFT6266's COCO image package.
![Alt text](img/Results_Img_0.png?raw=true "Image #1")
![Alt text](img/Results_Img_1.png?raw=true "Image #1")
![Alt text](img/Results_Img_2.png?raw=true "Image #1")
![Alt text](img/Results_Img_3.png?raw=true "Image #1")
![Alt text](img/Results_Img_4.png?raw=true "Image #1")
![Alt text](img/Results_Img_5.png?raw=true "Image #1")
![Alt text](img/Results_Img_6.png?raw=true "Image #1")
![Alt text](img/Results_Img_7.png?raw=true "Image #1")
![Alt text](img/Results_Img_8.png?raw=true "Image #1")

It seems that more training could be good if the features learned where actually good, because when it's wrong, less training gives a better image by just roughly inpainting the context colors... Wrong guesses trying to paint something that doesn't fit makes it worse. Next step would be to try a deeper network, to learn more features.


### Credits : 
Most of this work is build on top of [Massimiliano Comin's work](https://ift6266mcomin.wordpress.com/)
Thanks for putting simple code that actually works!
