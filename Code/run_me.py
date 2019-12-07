import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tqdm
import copy
import sys
from skimage import io
from sklearn.cluster import KMeans

def visualize(im1, im2, k):
	# displays two images
    im1 = im1.astype('uint8')
    im2 = im2.astype('uint8')
    f = plt.figure(1, figsize=(12,4))
    f.add_subplot(1,2, 1)
    plt.imshow(im1)
    plt.axis('off')
    plt.title('Original')
    f.add_subplot(1,2, 2)
    plt.imshow(im2)
    plt.axis('off')
    plt.title('Cluster: '+str(k))
    plt.savefig('k_means_'+str(k)+'.jpg')
    plt.show()
    return None

def MSE(Im1, Im2):
	# computes error
	Diff_Im = Im2-Im1
	Diff_Im = np.power(Diff_Im, 2)
	Diff_Im = np.sum(Diff_Im, axis=2)
	Diff_Im = np.sqrt(Diff_Im)
	sum_diff = np.sum(np.sum(Diff_Im))
	avg_error = sum_diff / float(Im1.shape[0]*Im2.shape[1])
	return avg_error

# Open the image
original_image = np.array(Image.open('../../Data/singapore.jpg'))
# Resize the image
image_resize = original_image.reshape(original_image.shape[0]*original_image.shape[1],3)
# Initalize and fit the KMeans
kmeans = KMeans(n_clusters=5, random_state=10).fit(image_resize)
# Extract the cluster centers
clusters = np.asarray(kmeans.cluster_centers_,dtype=np.uint8) 
# Labels for each pixel 
labels = np.asarray(kmeans.labels_,dtype=np.uint8 )  
labels_2 = labels.reshape(original_image.shape[0],original_image.shape[1]); 
# Saving the images
np.save('../../Data_comp/Compressed_data_5.npy',clusters)    
io.imsave('../../Data_comp/compressed_SNP_5.jpg',labels_2);
io.imsave('../../Data_comp/compressed_SNP_5_png.png',labels_2);
#centers = np.load('codebook_tiger.npy')
centers = clusters
# Load the compressed pixels
c_image = io.imread('../../Data_comp/compressed_SNP_5_png.png')

# initialize the empty array for recontructed image
image = np.zeros((c_image.shape[0],c_image.shape[1],3),dtype=np.uint8 )

# Assigning centroid values to entire clusters
for i in range(c_image.shape[0]):
    for j in range(c_image.shape[1]):
            image[i,j,:] = centers[c_image[i,j],:]
# Saving the reconstructed image
io.imsave('../../Data_comp/reconstructed_SNP_5.png',image);
io.imshow(image)
io.show()
# calculating the error sum of squared errors
error= MSE(original_image,image)