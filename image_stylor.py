#!/usr/bin/python

import argparse
import neural_style_transfer as nst

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Generate Styled image')
  parser.add_argument('-c', '--content', type=str,action='store', dest='content_image_path', required=True, help='The path of the content image')
  parser.add_argument('-s', '--style', type=str, action='store', dest='style_image_path', required=True, help='The path of the style image')
  parser.add_argument('-n', '--nbrIter', type=int, action='store', dest='num_iterations', help='The number of iteration to be used for the generation')
  
  args = parser.parse_args()
  
  #Construct the Network
  net = nst.NstVgg19("pretrained_model/imagenet-vgg-verydeep-19.mat")
   
  if(args.num_iterations is None):
    num_iterations = 200
  else:
    num_iterations = args.num_iterations
    
  #Generate the image
  net.generate(args.content_image_path, args.style_image_path, num_iterations)
  
    
    
  
  
  
  
  
  
  
  
  