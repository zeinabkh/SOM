import os
from matplotlib import image,pyplot
# from PIL import Image
def load_data(dir_path):
    # get all image path in directory by dir_path.
    """

    :param dir_path: path of directory that contain data set

    :return:
     images_data_pix: return each image as vector of pixel.
     images_label: return label of each pixel
    """
    images_path = os.listdir(dir_path)
    # read each image.
    images_data_pix = []
    images_label = []
    for img_path in images_path:
        images_label.append( img_path[img_path.index(".")+1:].replace(".gif",""))
        image_data = pyplot.imread(dir_path+"\\"+img_path)
        images_data_pix.append( image_data.reshape(-1,image_data.shape[0]*image_data.shape[1])[0]/255)
    # print(images_data_pix[0].shape[0])
    return images_data_pix, images_label
# load_data("G:\master_matus\99_2\\neural network\yalefaces")
