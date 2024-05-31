import numpy as np
from PIL import Image
import click
import os
import sys
# sys.path.insert(0,r'C:/Users/Manish/Downloads/deblur-gan-master/deblur-gan-master/deblurgan')

from model import generator_model
from utils import load_image, deprocess_image, preprocess_image


def deblur(weight_path, input_dir, output_dir):
 	g = generator_model()
 	g.load_weights(weight_path)
 	for image_name in os.listdir(input_dir):
 	    image = np.array([preprocess_image(load_image(os.path.join(input_dir, image_name)))])
 	    x_test = image
 	    generated_images = g.predict(x=x_test)
 	    generated = np.array([deprocess_image(img) for img in generated_images])
 	    x_test = deprocess_image(x_test)
 	    for i in range(generated_images.shape[0]):
 	        x = x_test[i, :, :, :]
 	        img = generated[i, :, :, :]
 	        output = np.concatenate((x, img), axis=1)
 	        # im = Image.fromarray(output.astype(np.uint8))
 	        # im.save(os.path.join(output_dir, image_name))
 			#hit and d trail 
 			# output = np.concatenate((x, img), axis=1)
 	        im = Image.fromarray(img.astype(np.uint8))
 	        im.save(os.path.join(output_dir, image_name))


# def deblur(weight_path, input_dir, output_dir):
#     g = generator_model()
#     g.load_weights(weight_path)
#     for image_name in os.listdir(input_dir):
#         input_image_path = os.path.join(input_dir, image_name)
#         input_image = load_image(input_image_path)
#         input_size = input_image.size
        
#         image = np.array([preprocess_image(input_image)])
#         x_test = image
#         generated_images = g.predict(x=x_test)
#         generated = np.array([deprocess_image(img) for img in generated_images])
#         x_test = deprocess_image(x_test)
#         for i in range(generated_images.shape[0]):
#             x = x_test[i, :, :, :]
#             img = generated[i, :, :, :]
#             # Resize the output image to match the size of the input image
#             img_resized = np.array(Image.fromarray(img.astype(np.uint8)).resize(input_size, Image.BICUBIC))
#             output = np.concatenate((x, img_resized), axis=1)
#             im = Image.fromarray(output.astype(np.uint8))
#             im.save(os.path.join(output_dir, image_name))


@click.command()
@click.option('--weight_path', help='Model weight')
@click.option('--input_dir', help='Image to deblur')
@click.option('--output_dir', help='Deblurred image')
def deblur_command(weight_path, input_dir, output_dir):
    return deblur(weight_path, input_dir, output_dir)


if __name__ == "__main__":
    deblur_command()
