# Assignment 4: Histogram Equalization

Assignment No 4 for the multi-core programming course. Implement histogram equalization for a gray scale image in CPU and GPU. The result of applying the algorithm to an image with low contrast can be seen in Figure 1:

![Figure 1](Images/histogram_equalization.png)
<br/>Figure 1: Expected Result.

The programs have to do the following:

1. Using Opencv, load and image and convert it to grayscale.
2. Calculate de histogram of the image.
3. Calculate the normalized sum of the histogram.
4. Create an output image based on the normalized histogram.
5. Display both the input and output images.

Test your code with the different images that are included in the *Images* folder. Include the average calculation time for both the CPU and GPU versions, as well as the speedup obtained, in the Readme.

Rubric:

1. Image is loaded correctly.
2. The histogram is calculated correctly using atomic operations.
3. The normalized histogram is correctly calculated.
4. The output image is correctly calculated.
5. For the GPU version, used shared memory where necessary.
6. Both images are displayed at the end.
7. Calculation times and speedup obtained are incuded in the Readme.

Results:

| Image         | GPU           | CPU           | Speedup       |
| ------------- |:-------------:|:-------------:| -------------:|
| dog1          | 0.027981 ms   | 220.384827 ms | 7876.23126407 |
| dog2          | 0.018761 ms   | 220.370773 ms | 11746.2167795 |
| dog3          | 0.018036 ms   | 232.162842 ms | 12872.1912841 |
| scenery       | 0.017705 ms   |   7.069092 ms | 399.270940412 |
| woman         | 0.018327 ms   | 178.494812 ms | 9739.44519016 |
| woman2        | 0.020057 ms   | 172.892899 ms | 8620.07772847 |
| woman3        | 0.018146 ms   | 177.573990 ms | 9785.84756971 |
