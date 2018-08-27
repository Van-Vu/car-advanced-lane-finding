**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup_images/chessboard_distort.jpg "ChessboardUndistorted"
[image2]: ./writeup_images/testimage_distort.jpg "TestImageUndistorted"
[image3]: ./writeup_images/threshold_binary.jpg "Threshold Binary"
[image4]: ./writeup_images/warped_image.jpg "Warp Example"
[image5]: ./writeup_images/masked_image.jpg "Masked Example"
[image6]: ./writeup_images/histogram.jpg "histogram"
[image7]: ./writeup_images/sliding_window.jpg "sliding window"
[image8]: ./writeup_images/draw_line.jpg "draw line"
[image9]: ./writeup_images/real_draw.jpg "real draw"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. README that includes all the rubric points and how I addressed each one
You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./advanced_lane_finding.ipynb"

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Distortion-corrected image.

Below is an example of a undistorted test image:
![alt text][image2]

#### 2. Threshold binary: color transforms, gradients or other methods to create a thresholded binary image.

In code block 7, I layout all possible color transform & grandient method so that I can work on each of them one by one to find a good pair of threshold min / max to generate binary image
Below are different threshold values that I tried

![alt text][image3]

#### 3. Perspective transform:

The code for my perspective transform includes a function called `warp_image(image, source, dest)`, which appears in code block 10 of the IPython notebook.
The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
bottom_left = [150,720]
bottom_right = [1130, 720]
top_left = [540, 480]
top_right = [762, 480]
source = np.float32([bottom_left,bottom_right,top_right,top_left])

bottom_left = [320,720]
bottom_right = [920, 720]
top_left = [320, 1]
top_right = [920, 1]
dst = np.float32([bottom_left,bottom_right,top_right,top_left])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 150, 720      | 320, 720        | 
| 540, 480      | 320, 1      |
| 762, 480     | 920, 1      |
| 1130, 720      | 920, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Region of Interest and mask white & yellow color
I use 2 masking techniques:
- Region of Interest to focus only on the center of each image which likely has the lane line
```python
def get_vertices(image):
    bottom_left = [180,720]
    bottom_right = [1230, 720]
    top_left = [500, 450]
    top_right = [762, 450]
    return np.array([[bottom_left,top_left,top_right, bottom_right]], dtype=np.int32) 
```
- White and Yellow color
```python
def select_rgb_white_yellow(image): 
    # white color mask
    lower = np.uint8([220, 220, 220])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    #lower = np.uint8([170, 170, 0])
    upper = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    
    return masked
```

Below is the result:
![alt text][image5]

#### 5. Identified lane-line pixels and fit their positions with a polynomial

For finding lane-line pixels, I use the histogram sliding window technique, the code is in function find_lane_pixels with the following hyperparam:
- nwindows = 10
- margin = 100
- minpix = 40

Histogram of the test image

![alt text][image6]

Sliding window polyfit

![alt text][image7]

Once found the line, I search around that area to draw a line in the next frame

![alt text][image8]

#### 6.Calculate the radius of curvature of the lane and the position of the vehicle with respect to center.

I do this in function `calculate_curv_rad_and_center_dist`. Based on the histogram, I guesstimate 100 pixel according to y and 400 pixel according to x
Then based on binary image, polynomial fit, and L and R lane pixel indices ... I apply the formula `f(y)=Ayy+By+C` to find the curvature as well as car position on the test image


#### 7. Plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in function `draw_real_world` which takes in original_img, binary_img, left_fit, right_fit, M_inverse


![alt text][image9]

---

### Pipeline (video)

Here's a [link to my video result](https://youtu.be/hKtBd94Xa10)

---

### Discussion

- Finding out suitable threshold numbers take quite a bit of time
- I tried the second convolution technique from the lecture but it doesn't seem to work
- When I remove the Region of Interest feature, my pipeline fails to recognize lane line. This is a sign that some gradient & color transfomations doesn't work as expected
- The same pipeline applies to challenge video doesn't work.  


