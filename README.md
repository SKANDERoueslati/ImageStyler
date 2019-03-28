# ImageStylor
A python script and a simple API used to apply Neural Style Transfer on input images  


usage: image_styler.py [-h] -c CONTENT_IMAGE_PATH -s STYLE_IMAGE_PATH  
                       [-n NUM_ITERATIONS]  

Generate Styled image  

optional arguments:  
  -h, --help            show this help message and exit  
  -c CONTENT_IMAGE_PATH, --content CONTENT_IMAGE_PATH  
                        The path of the content image  
  -s STYLE_IMAGE_PATH, --style STYLE_IMAGE_PATH  
                        The path of the style image  
  -n NUM_ITERATIONS, --nbrIter NUM_ITERATIONS  
                        The number of iteration to be used for the generation  
  
Example :  
./image_styler.py -c 'content_image.jpg' -s 'style_image.jpg' -n 5  
  
  
The Generated Image (generated_image.jpg) will be Saved under ./output.  
