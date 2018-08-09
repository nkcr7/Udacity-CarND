import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import pickle
from pathlib import Path

if Path('./calibration.p').exists():
    with open('./calibration.p', 'rb') as handle:
        calibration = pickle.load(handle)
        ret = calibration['ret']
        mtx = calibration['mtx']
        dist = calibration['dist']
        rvecs = calibration['rvecs']
        tvecs = calibration['tvecs']


def abs_sobel_thresh(img_channel, orient='x', ksize=3, thresh=(0, 255)):
    if orient is 'x':
        direction = [1, 0]
    elif orient is 'y':
        direction = [0, 1]
    sobel = cv2.Sobel(img_channel, cv2.CV_64F, direction[0], direction[1], ksize=ksize)
    abs_sobel = np.abs(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output


def mag_thresh(img_channel, ksize=3, mag_thresh=(0, 255)):
    sobelx = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary_output


def dir_threshold(img_channel, ksize=3, thresh=(0, np.pi / 2)):
    sobelx = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    direction = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(direction)
    binary_output[(direction > thresh[0]) & (direction <= thresh[1])] = 1
    return binary_output


def color_threshold(img_channel, thresh=(0, 255)):
    binary_output = np.zeros_like(img_channel)
    binary_output[(img_channel > thresh[0]) & (img_channel <= thresh[1])] = 1
    return binary_output


# pipeline for video
if Path('./calibration.p').exists():
    with open('./calibration.p', 'rb') as handle:
        calibration = pickle.load(handle)
        ret = calibration['ret']
        mtx = calibration['mtx']
        dist = calibration['dist']
        rvecs = calibration['rvecs']
        tvecs = calibration['tvecs']
else:
    print('calibrate camera first')

if Path('./perspective.p').exists():
    with open('./perspective.p', 'rb') as handle:
        perspective = pickle.load(handle)
        M = perspective['M']
        inv_M = perspective['inv_M']
else:
    print('calculate perspective matrix first')


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None


# create line instances
left_line = Line()
right_line = Line()
fail_count = 0


def process_image(image):
    global left_line
    global right_line
    global fail_count

    # image in RGB mode
    # distorted image
    dst = cv2.undistort(image, mtx, dist, None, mtx)

    # r channel
    r_channel = dst[:, :, 0]

    # x gradient threshold R channel
    sobel_mask_x = abs_sobel_thresh(r_channel, 'x', ksize=7, thresh=(25, 200))

    # y gradient threshold
    sobel_mask_y = abs_sobel_thresh(r_channel, 'y', ksize=7, thresh=(50, 150))

    # magnitude gradient threshold
    mag_mask = mag_thresh(r_channel, ksize=7, mag_thresh=(50, 200))

    # direction gradient threshold
    dir_mask = dir_threshold(r_channel, ksize=7, thresh=(-0.5, 0.5))

    # s-channel color threshold
    hls = cv2.cvtColor(dst, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_color_mask = color_threshold(s_channel, thresh=(180, 255))

    # R-channel color threshold
    r_color_mask = color_threshold(r_channel, thresh=(200, 255))

    # combined gradient threshold
    combined = np.zeros_like(dir_mask)
    combined[(((sobel_mask_x == 1) & (sobel_mask_y == 1)) | ((mag_mask == 1) & (dir_mask == 1))) | (
    (s_color_mask == 1) & (r_color_mask == 1))] = 1

    # perspective transform
    dsize = combined.shape[0:2][::-1]
    after_perspective = cv2.warpPerspective(combined, M, dsize, flags=cv2.INTER_LINEAR)

    # find the lines using HOG
    # Take a histogram of the bottom half of the image
    histogram = np.sum(after_perspective[after_perspective.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((after_perspective, after_perspective, after_perspective)) * 255
    # plot windows
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(after_perspective.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = after_perspective.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = after_perspective.shape[0] - (window + 1) * window_height
        win_y_high = after_perspective.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        out_img = cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                                (0, 255, 0), 10)
        out_img = cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                                (0, 255, 0), 10)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, after_perspective.shape[0] - 1, after_perspective.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 3 / 80  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 662  # meters per pixel in x dimension
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # calculate offset, define right offset as positve
    #     offset = ((left_fitx[-1] + right_fitx[-1]) / 2 - after_perspective.shape[1] / 2) * xm_per_pix
    # offset to left line
    #     left_offset = (after_perspective.shape[1] / 2 - left_fitx[-1]) * xm_per_pix
    # offset to right line
    #     right_offset= (right_fitx[-1] - after_perspective.shape[1] / 2) * xm_per_pix

    avg_n = 5  # average horizon
    fail_count_limit = 10
    gamma = 0.8  # discount weight in average
    # sanity check
    if ((abs(left_curverad - right_curverad) < 2000) &
            (abs(abs(left_fitx[-1] - right_fitx[-1]) * xm_per_pix - 3.7) / 3.7 < 0.4) &
            (np.max(abs(left_fitx - right_fitx) - np.mean(abs(left_fitx - right_fitx))) / np.mean(
                abs(left_fitx - right_fitx)) < 0.4)):
        # left line
        # was the line detected in the last iteration?
        left_line.detected = True
        # x values of the last n fits of the line
        if len(left_line.recent_xfitted) < avg_n:
            left_line.recent_xfitted.append(left_fitx)
            left_line.bestx = left_fitx
        else:
            left_line.recent_xfitted.pop(0)
            left_line.recent_xfitted.append(left_fitx)
            # average x values of the fitted line over the last n iterations
            # left_line.bestx = np.average(np.array(left_line.recent_xfitted), axis=1,
            #                              weights=np.array([gamma ** i for i in range(avg_n - 1, 0, -1)]))
            left_line.bestx = np.mean(np.array(left_line.recent_xfitted), axis=0)
            # polynomial coefficients averaged over the last n iterations
        left_line.best_fit_cr = np.polyfit(ploty * ym_per_pix, left_line.bestx * xm_per_pix, 2)
        # polynomial coefficients for the most recent fit
        #         left_line.current_fit = left_fit
        # radius of curvature of the line in some units
        left_line.radius_of_curvature = ((1 + (
        2 * left_line.best_fit_cr[0] * y_eval * ym_per_pix + left_line.best_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_line.best_fit_cr[0])
        # distance in meters of vehicle center from the line
        left_line.line_base_pos = (left_line.bestx[-1] - after_perspective.shape[1] / 2) * xm_per_pix
        # difference in fit coefficients between last and new fits
        #         left_line.diffs = left_line.current_fit - left_line.best_fit
        # x values for detected line pixels
        #         left_line.allx = leftx
        # y values for detected line pixels
        #         left_line.ally = lefty

        # right line
        # was the line detected in the last iteration?
        right_line.detected = True
        # x values of the last n fits of the line
        if len(right_line.recent_xfitted) < avg_n:
            right_line.recent_xfitted.append(right_fitx)
            right_line.bestx = right_fitx
        else:
            right_line.recent_xfitted.pop(0)
            right_line.recent_xfitted.append(right_fitx)
            # average x values of the fitted line over the last n iterations
            # right_line.bestx = np.average(np.array(right_line.recent_xfitted), axis=1,
                                          # weights=np.array([gamma ** i for i in range(avg_n - 1, 0, -1)]))
            right_line.bestx = np.mean(np.array(right_line.recent_xfitted), axis=0)
            # polynomial coefficients averaged over the last n iterations
        right_line.best_fit_cr = np.polyfit(ploty * ym_per_pix, right_line.bestx * xm_per_pix, 2)
        # polynomial coefficients for the most recent fit
        #         right_line.current_fit = right_fit
        # radius of curvature of the line in some units
        right_line.radius_of_curvature = ((1 + (
        2 * right_line.best_fit_cr[0] * y_eval * ym_per_pix + right_line.best_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_line.best_fit_cr[0])
        # distance in meters of vehicle center from the line
        right_line.line_base_pos = (right_line.bestx[-1] - after_perspective.shape[1] / 2) * xm_per_pix
        # difference in fit coefficients between last and new fits
    #         right_line.diffs = right_line.current_fit - right_line.best_fit
    # x values for detected line pixels
    #         right_line.allx = rightx
    # y values for detected line pixels
    #         right_line.ally = righty

    else:

        fail_count += 1
        left_line.detected = False
        right_line.detected = False



        # plotting detected line pixels
    #     out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    #     out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # Warp the detected lane boundaries back onto the original image
    lane_mask = np.zeros_like(out_img, dtype=np.uint8)
    for y_idx in range(after_perspective.shape[0]):
        lane_mask[y_idx, int(left_line.bestx[y_idx]):int(right_line.bestx[y_idx])] = [0, 255, 0]
        # inverse perspective transform
        dsize = lane_mask.shape[0:2][::-1]
        lane_mask_inv = cv2.warpPerspective(lane_mask, inv_M, dsize, flags=cv2.INTER_LINEAR)
        lane_detected_img = cv2.addWeighted(dst, 0.8, lane_mask_inv, 0.6, 0.)
        # add text
        cv2.putText(lane_detected_img,
                    'left_curverad: %.2f m'
                    % (left_line.radius_of_curvature), (250, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 2)
        cv2.putText(lane_detected_img,
                    'right_curverad: %.2f m'
                    % (right_line.radius_of_curvature), (250, 100), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 2)
        offset = (right_line.line_base_pos - left_line.line_base_pos) / 2
        cv2.putText(lane_detected_img,
                    '%s offset: %.2f m'
                    % ('right' if offset > 0 else 'left', offset), (250, 150), cv2.FONT_HERSHEY_COMPLEX, 1.5,
                    (255, 255, 255), 2)

    return lane_detected_img

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


white_output = './project_video/project_video_output.mp4'
clip1 = VideoFileClip("./project_video.mp4") # RGB mode
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

