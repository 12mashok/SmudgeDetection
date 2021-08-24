# SmudgeDetection
This python code was written to solve an image processing coding challenge. The challenge is to identify small (around 6x6 pixels) light areas in images obtained from digital microscopy without the use of image processing libraries such as scikit-image.

The Laplacian of Gaussian method has been implemented in "algorithmCode.py" to locate smudges in the images.

Functions have been written to extract images and draw detected smudges. Briefly, 
    extractImages(directory) reads tiff images in a specified directory as numpy arrays. 
    detectSmudges(image) returns center coordinates of 6x6 rectangles bounding smudges detected.
    drawSmudges(image, locations) draws rectangles around detected smudges on image. 

"AlgorithmDescription.pdf" contains more information on the algorithm implementation and describes a use case.
