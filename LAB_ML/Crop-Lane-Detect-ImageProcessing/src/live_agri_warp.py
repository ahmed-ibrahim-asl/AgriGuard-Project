import cv2
import numpy as np
import psutil
import pynvml
from DBSCAN import DBSCAN  # Ensure DBSCAN.py is in the same directory
import time
from filterpy.kalman import KalmanFilter  # For Kalman filter
import warnings

# Initialize GPU monitoring
try:
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except:
    gpu_handle = None

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Constants
ORIGINAL_SIZE = (1280, 720)
WARPED_SIZE = (500, 600)

# Global variables for temporal smoothing
left_buffer = []
right_buffer = []
buffer_size = 5

# Initialize Kalman Filters for lane lines
kf_left = KalmanFilter(dim_x=2, dim_z=2)
kf_right = KalmanFilter(dim_x=2, dim_z=2)

for kf in [kf_left, kf_right]:
    kf.F = np.eye(2)  # State transition matrix
    kf.H = np.eye(2)  # Measurement function
    kf.P *= 1000  # Initial uncertainty
    kf.R *= 10  # Measurement noise
    kf.Q *= 1e-3  # Process noise
    kf.x = np.zeros((2, 1))  # Initial state estimate

# Create a psutil Process object for the current process
process = psutil.Process()

def getROI():
    roi_points = np.array([
        [0, 360],
        [1280, 360],
        [1280, 665],
        [0, 665]
    ])
    roi = np.zeros((720, 1280), np.uint8)
    cv2.fillPoly(roi, [roi_points], 1)
    return roi

def fillAvgs(lines):
    l = []
    for i in range(len(lines)):
        line = lines[i]
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]
        if x2 - x1 == 0:
            continue  # Avoid division by zero
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - (slope * x1)
        l.append([slope, intercept])
    return l

def getLines(img):
    roi = getROI()

    # Gaussian Blur and Morphological Operations
    img = cv2.GaussianBlur(img, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # Convert to HLS and extract the L channel
    img_HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    _l_channel = img_HLS[:, :, 1]
    _l_channel = cv2.equalizeHist(_l_channel)

    # Canny Edge Detection
    _l_channel = cv2.morphologyEx(_l_channel, cv2.MORPH_CLOSE, kernel)
    _l_channel = cv2.GaussianBlur(_l_channel, (5, 5), 0)
    edges = cv2.Canny(_l_channel, 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Apply ROI mask
    new_img = cv2.bitwise_and(edges, edges, mask=roi)

    # Hough Line Transform
    lines = cv2.HoughLinesP(new_img, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=200)
    return lines

def exponential_moving_average(data, alpha=0.2):
    ema = []
    s = data[0]
    for point in data:
        s = alpha * point + (1 - alpha) * s
        ema.append(s)
    return np.array(ema)

def main(img):
    global left_buffer, right_buffer, buffer_size
    global kf_left, kf_right

    lines = getLines(img)

    if lines is None:
        M_inv = None  # Ensure M_inv is defined
        robot_position = None  # Ensure robot_position is defined
        return img, robot_position, M_inv

    left_av = []
    right_av = []
    dbscan_left = DBSCAN(50, 2)
    dbscan_right = DBSCAN(50, 2)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 - x1 == 0:
                continue  # Skip vertical lines
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) > 5 or abs(slope) < 0.1:
                continue
            if slope > 0:
                dbscan_right.update(line)
            else:
                dbscan_left.update(line)

    left_classes = dbscan_left.scan()
    left_lines = dbscan_left.return_max(left_classes)
    left_av = fillAvgs(left_lines)

    right_classes = dbscan_right.scan()
    right_lines = dbscan_right.return_max(right_classes)
    right_av = fillAvgs(right_lines)

    if not left_av or not right_av:
        M_inv = None
        robot_position = None
        return img, robot_position, M_inv

    # Average the lines
    left_fitted_av = np.average(left_av, axis=0)
    right_fitted_av = np.average(right_av, axis=0)

    # Update Kalman Filters
    kf_left.predict()
    kf_left.update(left_fitted_av.reshape(2, 1))
    left_state = kf_left.x

    kf_right.predict()
    kf_right.update(right_fitted_av.reshape(2, 1))
    right_state = kf_right.x

    # Use filtered values
    left_fitted_av = left_state.flatten()
    right_fitted_av = right_state.flatten()

    # Drawing averaged lines on original image
    y1 = ORIGINAL_SIZE[1] - 55
    y2 = ORIGINAL_SIZE[1] - 500

    try:
        left_x1 = int((y1 - left_fitted_av[1]) / left_fitted_av[0])
        left_x2 = int((y2 - left_fitted_av[1]) / left_fitted_av[0])
        right_x1 = int((y1 - right_fitted_av[1]) / right_fitted_av[0])
        right_x2 = int((y2 - right_fitted_av[1]) / right_fitted_av[0])
        cv2.line(img, (left_x1, int(y1)), (left_x2, int(y2)), (255, 0, 0), 2)
        cv2.line(img, (right_x1, int(y1)), (right_x2, int(y2)), (255, 0, 0), 2)
    except Exception as e:
        print("Error drawing lines:", e)
        M_inv = None
        robot_position = None
        return img, robot_position, M_inv

    # Perspective Transform
    src_pts = np.float32([
        [left_x2, y2],
        [right_x2, y2],
        [right_x1, y1],
        [left_x1, y1]
    ])
    dst_pts = np.float32([
        [0, 0],
        [WARPED_SIZE[0], 0],
        [WARPED_SIZE[0], WARPED_SIZE[1]],
        [0, WARPED_SIZE[1]]
    ])
    cv2.polylines(img, [src_pts.astype(np.int32)], True, (0, 200, 100), 2)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    f = cv2.warpPerspective(img, M, WARPED_SIZE)

    # Transform the endpoints to warped perspective
    def transformPoints(p):
        return cv2.perspectiveTransform(np.array([[p]], dtype=np.float32), M).squeeze()

    bot_left = np.array([left_x1, y1], dtype="float32")
    top_left = np.array([left_x2, y2], dtype="float32")
    bot_right = np.array([right_x1, y1], dtype="float32")
    top_right = np.array([right_x2, y2], dtype="float32")

    bot_left_warped = transformPoints(bot_left)
    top_left_warped = transformPoints(top_left)
    bot_right_warped = transformPoints(bot_right)
    top_right_warped = transformPoints(top_right)

    # Draw the lane lines in the warped image
    cv2.line(f, tuple(bot_left_warped.astype(int)), tuple(top_left_warped.astype(int)), (0, 255, 0), 3)
    cv2.line(f, tuple(bot_right_warped.astype(int)), tuple(top_right_warped.astype(int)), (0, 255, 0), 3)

    # Calculate and draw the guidance path
    num_points = 50  # Increased for smoother path
    mid_points = []
    for i in range(num_points):
        alpha = i / (num_points - 1)
        left_point = (1 - alpha) * np.array([left_x2, y2]) + alpha * np.array([left_x1, y1])
        right_point = (1 - alpha) * np.array([right_x2, y2]) + alpha * np.array([right_x1, y1])
        mid_point = (left_point + right_point) / 2
        mid_points.append(mid_point)

    # Transform mid-points to warped perspective
    mid_points = np.array([transformPoints(pt) for pt in mid_points])

    # Smooth the path using EMA
    smoothed_mid_points = exponential_moving_average(mid_points, alpha=0.1)

    # Draw the smooth guidance path
    for i in range(len(smoothed_mid_points) - 1):
        pt1 = tuple(smoothed_mid_points[i].astype(int))
        pt2 = tuple(smoothed_mid_points[i + 1].astype(int))
        cv2.line(f, pt1, pt2, (0, 0, 255), 3)  # Red line for the path

    # Optionally, draw points along the path
    for pt in smoothed_mid_points:
        cv2.circle(f, tuple(pt.astype(int)), 3, (0, 0, 255), -1)

    # Draw mid-line between the lanes in the warped image
    mid_top = ((top_left_warped + top_right_warped) / 2).astype(int)
    mid_bot = ((bot_left_warped + bot_right_warped) / 2).astype(int)
    cv2.line(f, tuple(mid_top), tuple(mid_bot), (255, 255, 0), 2)

    # Calculate deviation from the center
    traj_bot = [WARPED_SIZE[0] // 2, WARPED_SIZE[1]]
    cv2.line(f, (traj_bot[0], 0), (traj_bot[0], WARPED_SIZE[1]), (255, 255, 255), 2)

    # Compute deviation
    path_bottom_point = smoothed_mid_points[-1]
    deviation = path_bottom_point[0] - traj_bot[0]
    deviation_text = f"Deviation: {deviation:.2f} pixels"

    # Draw deviation text
    cv2.putText(f, deviation_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Return the position where the robot should go (path_bottom_point)
    robot_position = path_bottom_point

    # Compute the inverse perspective transform
    M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    unwarped = cv2.warpPerspective(f, M_inv, ORIGINAL_SIZE)

    # Combine the original image with the unwarped image
    result = cv2.addWeighted(img, 1, unwarped, 0.6, 0)

    return result, robot_position, M_inv

if __name__ == "__main__":
    # Initialize video capture (0 for webcam or provide video file path)
    cap = cv2.VideoCapture('../agri_videos/urmom.mp4')  # Replace with your video path or use 0 for webcam

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if no frames are returned

        # Optionally resize the frame for faster processing
        frame = cv2.resize(frame, ORIGINAL_SIZE)

        # Record the start time
        start_time = time.time()

        # Process the frame and get robot position and M_inv
        processed_frame, robot_position, M_inv = main(frame)

        # Record the end time
        end_time = time.time()
        processing_time = end_time - start_time  # Time in seconds

        # Measure resource usage
        memory_info = process.memory_info()
        memory_usage_bytes = memory_info.rss  # Resident Set Size
        memory_usage_mb = memory_usage_bytes / (1024 ** 2)  # Convert bytes to MB

        cpu_times = process.cpu_times()
        cpu_time_user = cpu_times.user  # Time spent in user mode
        cpu_time_system = cpu_times.system  # Time spent in kernel mode
        cpu_time_total = cpu_time_user + cpu_time_system  # Total CPU time

        # For GPU usage (if available)
        if gpu_handle:
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
            gpu_memory_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            gpu_memory_used_mb = gpu_memory_info.used / (1024 ** 2)  # Convert bytes to MB
            gpu_memory_total_mb = gpu_memory_info.total / (1024 ** 2)

            # Display GPU usage on the frame
            cv2.putText(processed_frame, f"GPU Usage: {gpu_util.gpu}%", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"GPU Memory: {gpu_memory_used_mb:.2f}/{gpu_memory_total_mb:.2f} MB", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display resource usage on the frame
        cv2.putText(processed_frame, f"RAM Usage: {memory_usage_mb:.2f} MB", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_frame, f"CPU Time: {cpu_time_total:.2f} sec", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_frame, f"Frame Time: {processing_time*1000:.2f} ms", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the robot position on the frame
        if robot_position is not None and M_inv is not None:
            # Transform the robot position back to the original perspective
            robot_position_original = cv2.perspectiveTransform(
                np.array([[robot_position]], dtype=np.float32), M_inv
            ).squeeze()
            robot_position_original = tuple(robot_position_original.astype(int))
            cv2.circle(processed_frame, robot_position_original, 10, (0, 0, 255), -1)
            cv2.putText(
                processed_frame,
                f"Robot Position: {robot_position_original}",
                (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        # Display the processed frame
        cv2.imshow('Processed Frame', processed_frame)

        # Press 'q' to exit the loop and close the video window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()
