# ImageStyler
**A python script (can be used as module) used to apply Neural Style Transfer on input images, based on the pretrained VGG19 network**  

### 1) The script

usage: image_styler.py [-h] -c CONTENT_IMAGE_PATH -s STYLE_IMAGE_PATH  
                       [-n NUM_ITERATIONS]  

**Arguments:**  
  + -h, --help            : show this help message and exit  
  + -c CONTENT_IMAGE_PATH, --content CONTENT_IMAGE_PATH : The path of the content image  
  + -s STYLE_IMAGE_PATH, --style STYLE_IMAGE_PATH : The path of the style image  
  + -n NUM_ITERATIONS, --nbrIter NUM_ITERATIONS (optional) : The number of iterations to be used for the generation(default=200)  
  
**Example :**  
./image_styler.py -c 'content_image.jpg' -s 'style_image.jpg' -n 100  
  
### 2) The module
You can use the ImageStyler as a module and integrate it into you application.  
To generate the image you need to :  
+ **Instantiate the NstVgg19 class** specifying the path of the pretrained model of the VGG19 network, check the note.
+ Call the **generate()** method specifying the content image path, the style image path and the optional number of iterations for the generation (default=200)



**The Generated Image (generated_image.jpg) will be Saved under ./output.**  

**Note :**  
To be able to run the tool you have first to download the pretrained model (imagenet-vgg-verydeep-19.mat) from https://www.kaggle.com/teksab/imagenetvggverydeep19mat/version/1 and copy it to the forlder ./pretrained_model .
