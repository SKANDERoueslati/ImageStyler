import scipy.misc
from tools import *
import tensorflow as tf

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]

class NstVgg19:
  
  def __init__(self, pretrained_vgg_model_path = "pretrained_model/imagenet-vgg-verydeep-19.mat", alpha = 10, beta = 40):
      
      print('###################### Start Initialization ######################')
      print('Initializing ...')
      # Reset the graph
      tf.reset_default_graph()
      self.model = load_vgg_model(pretrained_vgg_model_path)
      self.alpha = alpha
      self.beta  = beta
      
      self.J_total   = None
      self.J_content = None
      self.J_style   = None
      print('####################### End Initialization #######################\n')
      
  def compute_content_cost(self, a_C, a_G):
      """
      Computes the content cost
      
      Arguments:
      a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
      a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
      
      Returns: 
      J_content"""
      
      m, n_H, n_W, n_C =  a_G.get_shape().as_list()
      
      a_C_unrolled = tf.reshape(a_C, [m, n_C, n_H*n_W])
      a_G_unrolled = tf.reshape(a_G, [m, n_C, n_H*n_W])

      J_content = (tf.reduce_sum(tf.square(a_C_unrolled - a_G_unrolled)))/(4*n_H*n_W*n_C)
      
      return J_content
    
  def gram_matrix(self, A):
      """
      Argument:
      A -- matrix of shape (n_C, n_H*n_W)
      
      Returns:
      GA -- Gram matrix of A, of shape (n_C, n_C)
      """
      
      GA = tf.matmul(A, A, transpose_b = True)
      
      return GA
    
  def compute_layer_style_cost(self, a_S, a_G):
      """
      Arguments:
      a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
      a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
      
      Returns: 
      J_style_layer -- tensor representing a scalar value
      """
      m, n_H, n_W, n_C = a_G.get_shape().as_list()

      a_S = tf.reshape(tf.transpose(a_S), [n_C, n_H*n_W])
      a_G = tf.reshape(tf.transpose(a_G), [n_C, n_H*n_W])

      GS = self.gram_matrix(a_S)
      GG = self.gram_matrix(a_G)

      J_style_layer = tf.reduce_sum( tf.square(GS - GG) ) / ((2*n_H*n_W*n_C)**2)
      
      return J_style_layer
    
  def compute_style_cost(self, model, session, STYLE_LAYERS):
      """
      Computes the overall style cost from several chosen layers
      
      Arguments:
      model -- our tensorflow model
      STYLE_LAYERS -- A python list containing:
                          - the names of the layers we would like to extract style from
                          - a coefficient for each of them
      
      Returns: 
      J_style -- tensor representing a scalar value
      """

      J_style = 0
  
      for layer_name, coeff in STYLE_LAYERS:
  
          # Select the output tensor of the currently selected layer
          out = model[layer_name]
  
          # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
          a_S = session.run(out)
  
          # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
          # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
          # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
          a_G = out
          
          # Compute style_cost for the current layer
          J_style_layer = self.compute_layer_style_cost(a_S, a_G)
  
          # Add coeff * J_style_layer of this layer to overall style cost
          J_style += coeff * J_style_layer
  
      return J_style
  
  def total_cost(self, J_content, J_style, alpha = 10, beta = 40):
      """
      Computes the total cost function
      
      Arguments:
      J_content -- content cost coded above
      J_style -- style cost coded above
      alpha -- hyperparameter weighting the importance of the content cost
      beta -- hyperparameter weighting the importance of the style cost
      
      Returns:
      J -- total cost as defined by the formula above.
      """
      
      J = alpha*J_content + beta*J_style
      
      return J
  
  
  def optimize(self, sess, model, input_image, optimizer, num_iterations = 200):
      
      # Initialize global variables (you need to run the session on the initializer)
      sess.run(tf.global_variables_initializer())
      
      # Run the noisy input image (initial generated image) through the model. Use assign().
      sess.run(model['input'].assign(input_image))
      
      print('Processing: ======> ' + str(0) + '%')
      
      for i in range(num_iterations):
          # Run the session on the train_step to minimize the total cost
          train_step = optimizer.minimize(self.J_total)
          sess.run(train_step)
          
          # Compute the generated image by running the session on the current model['input']
          generated_image = sess.run(model['input'])
          
          #Print the percentage
          print('Processing: ======> ' + str(( (i+1)*100) / num_iterations) + '%')
  
          # Save Intermidiate images every 20 iteration.
          #if i%20 == 0:
              #Jt, Jc, Js = sess.run([self.J_total, self.J_content, self.J_style])
              #print("Iteration " + str(i) + " :")
              #print("total cost = " + str(Jt))
              #print("content cost = " + str(Jc))
              #print("style cost = " + str(Js))
              
              # save current generated image in the "/output" directory
              #save_image("output/step_" + str(i) + ".png", generated_image)
      
      # save last generated image
      print('\nGenerated Image (generated_image.jpg) is Saved under ./output')
      save_image('output/generated_image.jpg', generated_image)
      
      return generated_image  
  
  
  def generate(self, content_image_path, style_image_path, num_iterations=200):

      # Start interactive session
      sess = tf.InteractiveSession()
      
      content_image = scipy.misc.imread(content_image_path)
      content_image = reshape_and_normalize_image(content_image)
      
      style_image = scipy.misc.imread(style_image_path)
      style_image = reshape_and_normalize_image(style_image)
      
      generated_image = generate_noise_image(content_image)
      
      
      # Assign the content image to be the input of the VGG model.  
      sess.run(self.model['input'].assign(content_image))
      
      # Select the output tensor of layer conv4_2
      out = self.model['conv4_2']
      
      # Set a_C to be the hidden layer activation from the layer we have selected
      a_C = sess.run(out)
      
      # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
      # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
      # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
      a_G = out
      
      # Compute the content cost
      self.J_content = self.compute_content_cost(a_C, a_G)
      
      # Assign the input of the model to be the "style" image 
      sess.run(self.model['input'].assign(style_image))
      
      # Compute the style cost
      self.J_style = self.compute_style_cost(self.model, sess, STYLE_LAYERS)
      
      
      
      self.J_total = self.total_cost(self.J_content, self.J_style, self.alpha, self.beta)
      
      
      
      
      # define optimizer
      optimizer = tf.train.AdamOptimizer(2.0)
      
      # define train_step
      train_step = optimizer.minimize(self.J_total)
        
      print('######################## Start Processing ########################')
      generated_image = self.optimize(sess, self.model, generated_image, optimizer, 5)
      print('######################### End Processing #########################')
      
      return generated_image
 
  


