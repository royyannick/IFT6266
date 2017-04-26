# IFT6266 - Convolutional Auto-Encoder
This CAE is for the class project for IFT6266 : Conditional Image Generation

### Model :
[Insert Image Here]

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
