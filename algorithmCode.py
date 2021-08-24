"""
Algorithm code implementing the Laplacian of Gaussian method for detecting smudges on images. 
"""


import numpy as np
import matplotlib.pyplot as plt
import os 
import skimage.io as skio
from scipy import stats

def extractImages(directory):
	"""
	Reads tiff images in a specified directory as numpy arrays. 

	Inputs: 
		directory: string 
			Directory in which image files are stored. 

	Outputs: 
		images: list
			List of images read. 
	"""
	images = []

	#change into given directory
	os.chdir(directory)
	for image in os.listdir('./'):
		print('IMAGE: ', image)
		image = skio.imread(image, plugin = "tifffile")

		# if tiff file has more than one image (i.e., input is three dimensional)
		if len(image.shape) > 2: 

			# loop through images and append
			for index in range(image.shape[0]):
				images.append(image[index])
		else: 
			images.append(image)

	return images

def gaussian(sigma):
	# """
	# Returns gaussian kernel of size 3*sigma. 

	# Inputs: 
	# 	sigma: int
	# 		Sigma of gaussian kernel. 

	# Outputs: 
	# 	kernel: numpy array 
	# 		Gaussian kernel. 
	# """

	# Decide half length in x and y direction
	dim_x,dim_y = ((3*sigma-1)/2, (3*sigma-1)/2)

	# Create meshgrid
	yy,xx = np.ogrid[-dim_x:dim_x+1,-dim_y:dim_y+1]

	# Calculate kernel 
	kernel = np.exp(-(xx**2 + yy**2)/(2*sigma**2))

	# Divide kernel by sum to ensure that kernel adds up to 1.
	# This ensures that image isn't brighter/darker after convolution
	kernel /= kernel.sum()

	return kernel

def gaussianSmooth(image, sigma):
	"""
	Returns image smoothened by a gaussian kernel of specified sigma

	Inputs: 
		image: numpy array
			Original image that is to be smoothened. 

		sigma: int
			Sigma of gaussian kernel that smoothens the image. 

	Outputs: 
		smoothened_image: numpy array 
			Image smoothened by the gaussian kernel
	"""
	
	# Get gaussian kernel
	gauss = gaussian(sigma)

	# Perform convolution of the gaussian kernel and image 
	smoothened_image = convolve2D(image, gauss) 

	# Return smoothened image
	return smoothened_image

def laplacian_of_gaussian(sigma):
	"""
	Returns Laplacian of Gaussian kernel of size 3*sigma. 

	Inputs: 
		sigma: int
			Sigma of Laplacian of Gaussian kernel. 

	Outputs: 
		kernel: numpy array 
			Laplacian of Gaussian kernel. 
	"""

	# Decide half length in x and y direction
	dim_x,dim_y = ((3*sigma-1)/2, (3*sigma-1)/2)

	# Create meshgrid
	yy,xx = np.ogrid[-dim_x:dim_x+1,-dim_y:dim_y+1]

	# Calculate kernel 
	kernel = (1 - (xx**2+yy**2)/(2*sigma**2))*np.exp(-(xx**2+yy**2)/(2*sigma**2))

	# Subtract mean from entire kernel so that kernel adds up to zero
	kernel = kernel - np.mean(kernel)

	# Return kernel
	return kernel

def padImage(image, kernel_length):
	"""
	Returns padded image. 

	Inputs: 
		image: numpy array 
			Image that is to be padded. 

		kernel_length: int 
			Max length of kernels that will be convolved with the image. 
	Outputs:
		padded_image: numpy array
			Padded image.
	"""

	# Estimate background pixel as mode of image. If more than one mode exists, pick the first one.
	background_pixel = stats.mode(image)[0][0]
	if len(background_pixel)>1: 
		background_pixel = background_pixel[0]

	# Copy image into padded_image
	padded_image = image.copy()

	# Calculate length of padding based on kernel size. 
	padding_length = int(np.ceil(kernel_length/2))

	# Estimate image dimensions and calculate padding in vertical direction
	(rows, cols) = image.shape
	vertical_padding = np.array([[background_pixel]*cols]*padding_length)

	# Add zeros to top of image
	padded_image = np.append(vertical_padding, padded_image, axis = 0)

	# Add zeros to bottom of image
	padded_image = np.append(padded_image, vertical_padding, axis = 0)

	# Re-estimate image dimensions and calculate padding in horizontal direction
	(rows, cols) = padded_image.shape
	horizontal_padding = np.array([[background_pixel]*padding_length]*rows)

	# Add zeros to left of image
	padded_image = np.append(horizontal_padding, padded_image, axis = 1)

	# Add zeros to right of image
	padded_image = np.append(padded_image, horizontal_padding, axis = 1)

	# Return padded image
	return padded_image

def mirrorPadding(image, kernel_length):
	"""
	Returns mirror padded image. 

	Inputs: 
		image: numpy array 
			Image that is to be padded. 

		kernel_length: int 
			Max length of kernels that will be convolved with the image. 
	Outputs: 
		padded_image: numpy array 
			Padded image.
	"""  

	# Estimate background pixel as mode of image. If more than one mode exists, pick the first one.
	background_pixel = stats.mode(image)[0][0]
	if len(background_pixel)>1: 
		background_pixel = background_pixel[0]

	# Copy image into padded_image
	padded_image = image.copy()

	# Calculate length of padding based on kernel size. 
	padding_length = int(np.ceil(kernel_length/2))

	# Estimate image dimensions and calculate padding in vertical direction
	(rows, cols) = image.shape
	top_padding = np.flip(image[:padding_length,:], axis=0)
	bot_padding = np.flip(image[rows-padding_length:,:], axis=0)

	# Mirror top of image
	padded_image = np.append(top_padding, padded_image, axis = 0)

	# Mirror bottom of image
	padded_image = np.append(padded_image, bot_padding, axis = 0)

	# Re-estimate image dimensions and calculate padding in horizontal direction
	(rows, cols) = padded_image.shape
	left_padding = np.flip(padded_image[:, :padding_length], axis=1)
	right_padding = np.flip(padded_image[:, cols-padding_length:], axis=1)

	# Mirror left of image
	padded_image = np.append(left_padding, padded_image, axis = 1)

	# Mirror right of image
	padded_image = np.append(padded_image, right_padding, axis = 1)

	# Return padded image
	return padded_image



def convolve2D(image, kernel):
	"""
	returns convolution of image with kernel

	Inputs
		image: numpy array
			An array of image intensities.

		kernel: numpy array 
			Kernel with which image is to be convolved.

	Ouputs
		convolved_image: numpy array 
			Output of convolution of image and kernel.

	"""

	# copy image
	convolved_image = image.copy()

	# extract rows and columns in image
	(rows, cols) = image.shape

	# extract kernel size 
	kernel_size = len(kernel)
	
	# iterate over image to extract submatrices of size kernel_size
	for j in range(rows-kernel_size): 
		for i in range(cols-kernel_size): 
			# extract submatrix 
			submatrix = image[j:j+kernel_size,i:i+kernel_size]

			# multiply submatrix with kernel and find sum 
			convolved_pixel = sum(sum(submatrix*kernel))

			# replace pixel in image with convolved pixel value
			midpoint = int((kernel_size-1)/2)
			convolved_image[j+midpoint,i+midpoint] = convolved_pixel

	# return list of submatrices as a numpy array 
	return convolved_image

def detectSmudges(image, sigma = 3/np.sqrt(2)):
	"""  
	Returns numpy array of center coordinates of detected smudges. 
	
	Inputs: 
		image: numpy array
			Image with smudges to be detected. 

		sigma: int
			Sigma of Laplacian of Gaussian that is used for blob detection.
			For a blob of radius r, sigma = r/root(2)
			As blob radius is known to be 3 pixels, default value is sigma = 3/root(2)

	Outputs: 
		blob_center_coordinates: np array
			List of center coordinates of detected smudges.

	"""
	# Max kernel length determined by gaussian smoothing kernel whose sigma is 5
	max_kernel_length = 3*5

	# Step 1: Normalise image.
	normalised_image = (image - np.min(image)) / (np.max(image) - np.min(image))

	# Step 2: Mirror padding normalised image so that edges are included in laplacian of gaussian convolution step.
	padded_image = mirrorPadding(normalised_image, max_kernel_length)

	# Step 3: Remove background noise by subtracting smoothed image from original image
	print('Removing background noise...')
	smoothed = gaussianSmooth(padded_image, sigma = 5)
	noise_removed_image = padded_image - smoothed

	# # Step 3: Pad image so that edges are included in laplacian of gaussian convolution step.
	# print('Padding image...')
	# padded_image = padImage(noise_removed_image, max_kernel_length)
	# padded_image = noise_removed_image

	# Step 4: Get laplacian of gaussian for given sigma and convolve image with kernel. 
	print('Convolving image with Laplacian of Gaussian kernel...')
	laplacian_gaussian = laplacian_of_gaussian(sigma)
	convolved_image = convolve2D(noise_removed_image, laplacian_gaussian)

	# Step 5: Threshhold convolved image to get smudges. Pixels below the threshold are set to zero.
	# Threshold is calculated as 999th percentile of image distribution. 
	print('Thresholding image...')
	threshold = np.quantile(convolved_image, 0.992)
	thresholded_image = thresholdImage(convolved_image, threshold)

	# Step 6: Calculate positions of blob centers. A pixel is a local maximum if its intensity is the local maximum of a 5x5 square and it is the center of the 5x5 square.
	print('Finding local maxima...')
	blob_center_coordinates = np.array(localMaximas(thresholded_image))

	# Step 7: Adjust coordinates to account for padding
	padding_adjustment = max_kernel_length/2
	blob_center_coordinates -= [int(np.ceil(padding_adjustment)), int(np.ceil(padding_adjustment))]

	number_of_smudges = len(blob_center_coordinates)
	print(number_of_smudges, "number of smudges found!! ")

	# return coordinates of the blob centers
	return blob_center_coordinates

def thresholdImage(image, threshhold): 
	"""
	Returns images with pixels below a threshold set to zero. 

	Inputs: 
		image: numpy array 
			Image (post-convolution with Laplacian of Gaussian) to which threshold is to be applied. 

		threshold: int
			Threshold below which pixels are set to zero. 
			A good threshold will set the background to zero, leaving smudges unchanged. 

	Outputs: 
		thresholded_image: numpy array
			Image to which threshold has been applied. 
	"""

	# # Step 1: Copy original image
	# thresholded_image = image.copy()

	# # Step 2: Loop through entire image and set intensity to zero if pixel intensity < threshold
	# for i, row in enumerate(image): 
	# 	for j, pixel in enumerate(row):
	# 		if pixel < threshhold:
	# 			thresholded_image[i,j] = 0

	booleans = image<threshhold
	indices = np.where(booleans == True)
	image[indices] = 0

	# Return thesholded image
	return image

def localMaximas(image):
	"""
	Returns a list of local maximas in 6x6 square regions within the image. 

	Input: 
		image: numpy array 
			Image whose local maximas are to be identified.
			Image MUST BE thresholded prior to local maximum identification for good results. 

	Output: 
		maxima_locations: list 
			List of coordinates of local maximas. 
	""" 

	# initialise list of locations
	maxima_locations = []

	# extract number of rows and columns in the image
	(rows, cols) = image.shape

	# iterate over image to extract submatrices of size 5
	for j in range(rows-6): 
		for i in range(cols-6): 

			# extract submatrix of size 6x6
			submatrix = image[j:j+6,i:i+6]

			# if the submatrix isn't all 0s (which is the background in an appropriately thresholded image)
			if set(submatrix.flatten()) != {0}:

				# if center of submatrix is the local maximum, append maxima indices to maxima_location list
				if submatrix[2][2] == np.amax(submatrix):
					maxima_locations.append([i+2,j+2])

	return maxima_locations

def drawSmudges(image, locations, edgecolor = 'red'):
	"""
	Draws rectangles of height and width 6 based on the blob center coordinates 

	Input: 
		image: numpy array 
			Image on which blob is to be identified

		locations: list 
			List of center coordinates around which rectangles are to be drawn. 
			Center coordinates are specified as (x,y). 
	"""

	fig,ax = plt.subplots(1)
	ax.set_aspect('equal')

	ax.imshow(image)

	for location in locations:
		rect = plt.Rectangle((location[0]-3, location[1]-3), width=6, height=6,edgecolor=edgecolor, fill=False)
		ax.add_patch(rect)

	plt.title('Smudges located on image.')


