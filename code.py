import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, floor, ceil
from PIL import Image
import math
from time import time

import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import sys
from time import perf_counter 


#GPU parallelization requirements
from math import ceil
from pycuda import gpuarray
from pycuda.compiler import SourceModule


def read_image(path):
    '''Read image and return the image propertis.
    Parameters:
    path (string): Image path
    Returns:
    numpy.ndarray: Image exists in "path"
    list: Image size
    tuple: Image dimension (number of rows and columns)
    '''
    img = cv2.imread(path)  # cv2.IMREAD_GRAYSCALE)
    size = img.shape
    dimension = (size[0], size[1])

    return img, size, dimension


def image_change_scale(img, dimension, scale=100, interpolation=cv2.INTER_LINEAR):
    '''Resize image to a specificall scale of original image.
    Parameters:
    img (numpy.ndarray): Original image
    dimension (tuple): Original image dimension
    scale (int): Multiply the size of the original image
    Returns:
    numpy.ndarray: Resized image
    '''
    scale /= 100
    new_dimension = (int(dimension[1]*scale), int(dimension[0]*scale))
    resized_img = cv2.resize(img, new_dimension, interpolation=interpolation)

    return resized_img


    """
    Parallel GPU solution with CUDA implemented by:
    Nicolas Ortiz, Juan Rengifo, Fancisco Suarez, Juan Gomez
    https://github.com/Nicortiz72/Interpolation_GPU/
    """
def parallel_nearest_interpolation(image, dimension):
    '''Nearest neighbor interpolation method to convert small image to original image
    Parameters:
    img (numpy.ndarray): Small image
    dimension (tuple): resizing image dimension
    Returns:
    numpy.ndarray: Resized image
    '''
    enlarge_time = int(
        sqrt((dimension[0] * dimension[1]) / (image.shape[0]*image.shape[1])))
    
    dim=image.shape[2]
    image=image.astype(np.int32)
    image_gpu = gpuarray.to_gpu(image)
    new_image_gpu = gpuarray.empty((dimension[0],dimension[1],dim), np.int32)
    
    ker = SourceModule("""
                   __global__ void nearest_kernel(
                   int *image, int *new_image){
                    int high = %(high)s;
                    int width = %(width)s;
                    int dim = %(dim)s;
                    int enlarge_time = %(enlarge_time)s;
                    int width_original = (int)(width/enlarge_time);
                    int tx = blockIdx.x*blockDim.x+threadIdx.x;
                    int ty = blockIdx.y*blockDim.y+threadIdx.y;
                    int tz = blockIdx.z*blockDim.z+threadIdx.z;
                    int row,column;
                    if (tx < high && ty < width){
                      row = (int)(tx/enlarge_time);
                      column = (int)(ty/enlarge_time);
                      new_image[(tx * width * dim) + (ty * dim) + tz] = image[(row * width_original * dim) + (column * dim) + tz];
                    }
                   }"""%{'high': dimension[0], 'width': dimension[1], 'enlarge_time' : enlarge_time,'dim' : dim})
    
    nearest_gpu = ker.get_function("nearest_kernel")
    start = drv.Event()
    end=drv.Event()
    #Start Time
    start.record()

    nearest_gpu(image_gpu, new_image_gpu, block=(18,18,3), grid=(ceil(dimension[0]/18),ceil(dimension[1]/18),1))
    new_image = new_image_gpu.get()

    #End Time
    end.record()
    end.synchronize()
    #Measure time difference, give time in milliseconds, which is converted to seconds.
    secs = start.time_till(end)*1e-3
    print("Parallel time %fs" % (secs))
    
    image_gpu.gpudata.free()
    new_image_gpu.gpudata.free()
    return new_image,secs


def nearest_interpolation(image, dimension):
    '''Nearest neighbor interpolation method to convert small image to original image
    Parameters:
    img (numpy.ndarray): Small image
    dimension (tuple): resizing image dimension
    Returns:
    numpy.ndarray: Resized image
    '''
    new_image = np.zeros((dimension[0], dimension[1], image.shape[2]))

    enlarge_time = int(
        sqrt((dimension[0] * dimension[1]) / (image.shape[0]*image.shape[1])))
    start = time()
    for i in range(dimension[0]):
        for j in range(dimension[1]):
            row = floor(i / enlarge_time)
            column = floor(j / enlarge_time)

            new_image[i, j] = image[row, column]
    T=time()-start
    print("Serial Time:",T)
    return new_image,T


    """
    Parallel GPU solution with CUDA implemented by:
    Nicolas Ortiz, Juan Rengifo, Fancisco Suarez, Juan Gomez
    https://github.com/Nicortiz72/Interpolation_GPU/
    """
def parallel_bilinear_interpolation(image, dimension):
    '''Bilinear interpolation method to convert small image to original image
    Parameters:
    img (numpy.ndarray): Small image
    dimension (tuple): resizing image dimension
    Returns:
    numpy.ndarray: Resized image
    '''
    height = image.shape[0]
    width = image.shape[1]

    scale_x = (width)/(dimension[1])
    scale_y = (height)/(dimension[0])

    dim=image.shape[2]
    image=image.astype(np.int32)
    image_gpu = gpuarray.to_gpu(image)
    new_image_gpu = gpuarray.empty((dimension[0],dimension[1],dim), np.float32)
    ker = SourceModule("""
                   __global__ void bilinear_kernel(
                   int *image, float *new_image){
                    int high = %(high)s;
                    int width = %(width)s;
                    int dim = %(dim)s;
                    int tx = blockIdx.x*blockDim.x+threadIdx.x;
                    int ty = blockIdx.y*blockDim.y+threadIdx.y;
                    int tz = blockIdx.z*blockDim.z+threadIdx.z;
                    float x,y,x_diff,y_diff,pixel,a,b,c,d;
                    int x_int,y_int;
                    if (tx < high && ty < width){
                      x = (((float)ty)+0.5) * (%(scale_x)s) - 0.5;
                      y = (((float)tx)+0.5) * (%(scale_y)s) - 0.5;

                      x_int = (int)(x);
                      y_int = (int)(y);
                      if(x_int>%(width_original)s -2){x_int=%(width_original)s -2;}
                      if(y_int>%(height_original)s -2){y_int=%(height_original)s -2;}

                      x_diff = x - ((float)x_int);
                      y_diff = y - ((float)y_int);

                      a = (float)image[(y_int * %(width_original)s * dim) + (x_int * dim) + tz];
                      b = (float)image[(y_int * %(width_original)s * dim) + ((x_int+1) * dim) + tz];
                      c = (float)image[((y_int+1) * %(width_original)s * dim) + (x_int * dim) + tz];
                      d = (float)image[((y_int+1) * %(width_original)s * dim) + ((x_int+1) * dim) + tz];

                      pixel = (a*(1-x_diff)*(1-y_diff)) + (b*(x_diff) * (1-y_diff)) + (c*(1-x_diff) * (y_diff)) + (d*x_diff*y_diff);

                      new_image[(tx * width * dim) + (ty * dim) + tz] = pixel;
                    }
                   }"""%{'high': dimension[0], 'width': dimension[1],'dim' : dim, "scale_x" : scale_x,
                         "scale_y" : scale_y,'height_original': image.shape[0], 'width_original': image.shape[1]})
    
    bilinear_gpu = ker.get_function("bilinear_kernel")
    start = drv.Event()
    end=drv.Event()
    #Start Time
    start.record()

    bilinear_gpu(image_gpu, new_image_gpu, block=(18,18,3), grid=(ceil(dimension[0]/18),ceil(dimension[1]/18),1))
    new_image = new_image_gpu.get()
    
    #End Time
    end.record()
    end.synchronize()
    #Measure time difference, give time in milliseconds, which is converted to seconds.
    secs = start.time_till(end)*1e-3
    print("Parallel time %fs" % (secs))
    
    image_gpu.gpudata.free()
    new_image_gpu.gpudata.free()
    return new_image.astype(np.uint8),secs


def bilinear_interpolation(image, dimension):
    '''Bilinear interpolation method to convert small image to original image
    Parameters:
    img (numpy.ndarray): Small image
    dimension (tuple): resizing image dimension
    Returns:
    numpy.ndarray: Resized image
    '''
    height = image.shape[0]
    width = image.shape[1]

    scale_x = (width)/(dimension[1])
    scale_y = (height)/(dimension[0])

    new_image = np.zeros((dimension[0], dimension[1], image.shape[2]))
    start=time()
    for k in range(3):
        for i in range(dimension[0]):
            for j in range(dimension[1]):
                x = (j+0.5) * (scale_x) - 0.5
                y = (i+0.5) * (scale_y) - 0.5

                x_int = int(x)
                y_int = int(y)

                # Prevent crossing
                x_int = min(x_int, width-2)
                y_int = min(y_int, height-2)

                x_diff = x - x_int
                y_diff = y - y_int

                a = image[y_int, x_int, k]
                b = image[y_int, x_int+1, k]
                c = image[y_int+1, x_int, k]
                d = image[y_int+1, x_int+1, k]

                pixel = a*(1-x_diff)*(1-y_diff) + b*(x_diff) * \
                    (1-y_diff) + c*(1-x_diff) * (y_diff) + d*x_diff*y_diff
                
                new_image[i, j, k] = pixel.astype(np.uint8)
    T=time()-start
    print("Serial time: ",T)

    return new_image,T


def main():
    images_list = {}

    # Read Image
    img, size, dimension = read_image("img1000.png")
    print(f"Image size is: {size}")
    images_list['Original Image'] = img

    # Change Image Size
    scale_percent = 25  # percent of original image size
    resized_img = image_change_scale(img, dimension, scale_percent)
    print(f"Smalled Image size is: {resized_img.shape}")
    images_list['Smalled Image'] = resized_img

    #fig, axs = plt.subplots(2, 2) 
    #fig.suptitle('My Implementation', fontsize=16)

    # Change image to original size using nearest neighbor interpolation
    #s_nn_img = image_change_scale(
    #    resized_img, dimension, interpolation=cv2.INTER_NEAREST)
    #images_list['Nearest Neighbor Interpolation'] = s_nn_img

    # Change image to original size using bilinear interpolation
    #bil_img = image_change_scale(
    #    resized_img, dimension, interpolation=cv2.INTER_LINEAR)
    #images_list['Bilinear Interpolation'] = bil_img

    #######    nearest_interpolation   #########
    print("#"*6,"Nearest Interpolation","#"*6)
    nn_img_algo_S,TSN = nearest_interpolation(resized_img, dimension)
    #nn_img_algo_SN = Image.fromarray(nn_img_algo_S.astype('uint8')).convert('RGB')

    nn_img_algo_P,TPN = parallel_nearest_interpolation(resized_img, dimension)
    nn_img_algo_PN = Image.fromarray(nn_img_algo_P.astype('uint8')).convert('RGB')

    print("Serial and Parallel Nearest interpolation have the same result:",(nn_img_algo_S==nn_img_algo_P).all())
    plt.title("Nearest interpolation")
    plt.imshow(cv2.cvtColor(np.array(nn_img_algo_PN), cv2.COLOR_BGR2RGB))
    plt.show()

    #######   bilinear interpolation ###########
    print("#"*6,"Bilinear Interpolation","#"*6)
    bil_img_algo_S,TSB = bilinear_interpolation(resized_img, dimension)
    #bil_img_algo_SN = Image.fromarray(bil_img_algo_S.astype('uint8')).convert('RGB')

    bil_img_algo_P,TPB=parallel_bilinear_interpolation(resized_img, dimension)
    bil_img_algo_NP = Image.fromarray(bil_img_algo_P.astype('uint8')).convert('RGB')

    print("Serial and Parallel Bilinear interpolation have the same result:",(bil_img_algo_S==bil_img_algo_P).all())
    plt.title("Bilinear interpolation")
    plt.imshow(cv2.cvtColor(np.array(bil_img_algo_NP), cv2.COLOR_BGR2RGB))
    plt.show()


def testCase(resized_img,dimension):
    #######    nearest_interpolation   #########
    nn_img_algo_S,TSN = nearest_interpolation(resized_img, dimension)
    #nn_img_algo_SN = Image.fromarray(nn_img_algo_S.astype('uint8')).convert('RGB')

    nn_img_algo_P,TPN = parallel_nearest_interpolation(resized_img, dimension)
    #nn_img_algo_PN = Image.fromarray(nn_img_algo.astype('uint8')).convert('RGB')

    #print((nn_img_algo_S==nn_img_algo_P).all())
    #plt.imshow(cv2.cvtColor(np.array(nn_img_algo_PN), cv2.COLOR_BGR2RGB))

    #######   bilinear interpolation ###########
    bil_img_algo_S,TSB = bilinear_interpolation(resized_img, dimension)
    #bil_img_algo_SN = Image.fromarray(bil_img_algo_S.astype('uint8')).convert('RGB')

    bil_img_algo_P,TPB=parallel_bilinear_interpolation(resized_img, dimension)
    #bil_img_algo_NP = Image.fromarray(bil_img_algo_P.astype('uint8')).convert('RGB')

    #print((bil_img_algo_S==bil_img_algo_P).all())
    #plt.imshow(cv2.cvtColor(np.array(bil_img_algo_NP), cv2.COLOR_BGR2RGB))

    return TSN,TPN,TSB,TPB


def test():
    #nearest_interpolation
    StimesMean_N=list()
    PtimesMean_N=list()
    StimesSTD_N=list()
    PtimesSTD_N=list()
    #bilinear interpolation
    StimesMean_B=list()
    PtimesMean_B=list()
    StimesSTD_B=list()
    PtimesSTD_B=list()
    for i in range(1,15):
      #generate image
      size=300
      dimension=(i*size,i*size,3)
      resized_img = np.random.randn(int((i*size)/4), int((i*size)/4),3).astype(np.int32)
      print(f"Image size is: {i*size} x {i*size}")
      print(f"Smalled Image size is: {resized_img.shape}")

      eachS_B=list()
      eachP_B=list()
      eachS_N=list()
      eachP_N=list()
      for _ in range(5):
        TSN,TPN,TSB,TPB = testCase(resized_img,dimension)
        eachS_B.append(TSB)
        eachP_B.append(TPB)
        eachS_N.append(TSN)
        eachP_N.append(TPN)

      StimesMean_N.append(np.mean(eachS_N))
      PtimesMean_N.append(np.mean(eachP_N))
      StimesSTD_N.append(np.std(eachS_N))
      PtimesSTD_N.append(np.std(eachP_N))
      StimesMean_B.append(np.mean(eachS_B))
      PtimesMean_B.append(np.mean(eachP_B))
      StimesSTD_B.append(np.std(eachS_B))
      PtimesSTD_B.append(np.std(eachP_B))
    print("#"*10, "nearest_interpolation")
    print("Means S:")
    print("\n".join([str(x) for x in StimesMean_N]))
    print("Means P:")
    print("\n".join([str(x) for x in PtimesMean_N]))
    print("STD S:")
    print("\n".join([str(x) for x in StimesSTD_N]))
    print("STD P:")
    print("\n".join([str(x) for x in PtimesSTD_N]))
    print("#"*10, "bilinear interpolation")
    print("Means S:")
    print("\n".join([str(x) for x in StimesMean_B]))
    print("Means P:")
    print("\n".join([str(x) for x in PtimesMean_B]))
    print("STD S:")
    print("\n".join([str(x) for x in StimesSTD_B]))
    print("STD P:")
    print("\n".join([str(x) for x in PtimesSTD_B]))


if __name__ == "__main__":
    main()