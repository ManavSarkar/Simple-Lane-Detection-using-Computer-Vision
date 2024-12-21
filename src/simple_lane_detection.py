import cv2
import numpy as np
import os

# Convert the frame to grayscale
def gray_scale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to the image
def gaussian_blur(frame):
    kernel_size = (15, 15)
    return cv2.GaussianBlur(frame, kernel_size, 0)

# Apply Canny edge detection
def canny_edge(frame):
    return cv2.Canny(frame, 20, 40)

# Apply a trapezoidal mask to the frame
def region_of_interest(frame):
    height, width = frame.shape[:2]
    mask = np.zeros_like(frame)
    polygon = np.array([
        [
            (int(width * 0.2), height),
            (int(width * 0.8), height),
            (int(width * 0.65), int(height * 0.57)),
            (int(width * 0.45), int(height * 0.57))
        ]
    ], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(frame, mask)

# Calculate curvature radius
def calculate_curvature(poly, y_vals):
    first_derivative = 2 * poly[0] * y_vals + poly[1]
    second_derivative = 2 * poly[0]
    return (1 + first_derivative**2)**1.5 / np.abs(second_derivative + 1e-6)

# Fit a polynomial to lane lines and draw them
def fit_polynomial(masked_edge, original_frame):
    lane_pixels = np.column_stack(np.where(masked_edge > 0))
    if len(lane_pixels) == 0:
        return original_frame

    height, width = masked_edge.shape
    midpoint = width // 2 + 100
    left_lane_pixels = lane_pixels[lane_pixels[:, 1] < midpoint]
    right_lane_pixels = lane_pixels[lane_pixels[:, 1] >= midpoint]
    degree = 3

    def fit_and_draw_lane(pixels, color):
        if len(pixels) == 0:
            return
        y = pixels[:, 0]
        x = pixels[:, 1]
        poly = np.polyfit(y, x, degree)
        y_vals = np.linspace(int(height / 1.5), height - 1, num=int(height / 3), dtype=int)
        x_vals = np.polyval(poly, y_vals).astype(int)
        valid_indices = (x_vals >= 0) & (x_vals < width)
        x_vals = x_vals[valid_indices]
        y_vals = y_vals[valid_indices]
        for i in range(len(x_vals) - 1):
            cv2.line(original_frame, (x_vals[i], y_vals[i]), (x_vals[i + 1], y_vals[i + 1]), color, 10)

    fit_and_draw_lane(left_lane_pixels, (255, 0, 0))
    fit_and_draw_lane(right_lane_pixels, (0, 255, 0))
    return original_frame

# Main function for processing the video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Extract video details and output filename
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{base_name}_output.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = gray_scale(frame)
        blur = gaussian_blur(gray)
        edge = canny_edge(blur)
        masked_edge = region_of_interest(edge)
        lanes_frame = fit_polynomial(masked_edge, frame.copy())

        edge_resized = cv2.resize(edge, (200, 150))
        edge_colored = cv2.cvtColor(edge_resized, cv2.COLOR_GRAY2BGR)
        blur_resized = cv2.resize(blur, (200, 150))
        blur_colored = cv2.cvtColor(blur_resized, cv2.COLOR_GRAY2BGR)

        lanes_frame[10:160, 10:210] = blur_colored
        lanes_frame[10:160, 230:430] = edge_colored
        cv2.putText(lanes_frame, "Gaussian Blurred Video", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(lanes_frame, "Canny Edges", (230, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(lanes_frame)

    cap.release()
    out.release()
    print(f"Processed video saved at {output_path}")

if __name__ == "__main__":
    input_video = "sample_videos/Autonomous driving lane detection sample video 1.mp4"
    process_video(input_video)
