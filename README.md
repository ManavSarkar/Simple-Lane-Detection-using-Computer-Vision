# Lane Detection Project

This repository demonstrates a lane detection algorithm for autonomous driving. The algorithm processes video frames to identify lane boundaries, calculate curvature radius, and overlay this information on the original video.

## File Structure
```
Lane-Detection-Project/
|
├── .venv/                    # Virtual environment folder (optional)
├── sample_videos/            # Directory containing sample videos
│   ├── Autonomous driving lane detection sample video 1.mp4
│   ├── Autonomous driving lane detection sample video 2.mp4
│
├── .gitignore                # Git ignore file
├── debug.ipynb               # Jupyter Notebook for lane detection
├── requirements.txt          # Python dependencies
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Install the required Python libraries:

```bash
pip install -r requirements.txt
```

### How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Lane-Detection-Project
   ```

2. Place your video files in the `sample_videos` folder (optional).

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook code.ipynb
   ```

4. Execute the cells in the notebook to run the lane detection pipeline on the sample video.

### Output
- The algorithm overlays detected lanes and curvature radius on video frames.
- Gaussian-blurred and edge-detected video insets are shown for visualization.

## Features
- **Preprocessing:** Converts video frames to grayscale, applies Gaussian blur, and detects edges using the Canny edge detector.
- **Region of Interest:** Focuses on the road area using a trapezoidal mask.
- **Lane Detection:** Uses polynomial fitting and moving average to smooth lane lines.
- **Curvature Radius:** Calculates and displays lane curvature in meters.

## Example Usage
Sample video `Autonomous driving lane detection sample video 1.mp4` demonstrates the pipeline. Replace the video file path in the `video_path` variable to test with other videos.

## Requirements
- OpenCV
- NumPy
- Pillow
- IPython

These dependencies are listed in `requirements.txt`.

## Notes
- For optimal performance, ensure video frames have clear lane markings.
- Adjust the trapezoidal region or parameters as needed for different video resolutions or road conditions.

## Contributing
Contributions are welcome! Please submit a pull request or raise an issue for suggestions and bug reports.

## License
This project is licensed under the MIT License.

# Lane Detection Project

This repository demonstrates a lane detection algorithm for autonomous driving. The algorithm processes video frames to identify lane boundaries, calculate curvature radius, and overlay this information on the original video.

## File Structure
```
Lane-Detection-Project/
|
├── .venv/                    # Virtual environment folder (optional)
├── sample_videos/            # Directory containing sample videos
│   ├── Autonomous driving lane detection sample video 1.mp4
│   ├── Autonomous driving lane detection sample video 2.mp4
│
├── .gitignore                # Git ignore file
├── debug.ipynb               # Jupyter Notebook for lane detection
├── requirements.txt          # Python dependencies
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Install the required Python libraries:

```bash
pip install -r requirements.txt
```

### How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Lane-Detection-Project
   ```

2. Place your video files in the `sample_videos` folder (optional).

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook debug.ipynb
   ```

4. Execute the cells in the notebook to run the lane detection pipeline on the sample video.

### Output
- The algorithm overlays detected lanes and curvature radius on video frames.
- Gaussian-blurred and edge-detected video insets are shown for visualization.

## Features
- **Preprocessing:** Converts video frames to grayscale, applies Gaussian blur, and detects edges using the Canny edge detector.
- **Region of Interest:** Focuses on the road area using a trapezoidal mask.
- **Lane Detection:** Uses polynomial fitting and moving average to smooth lane lines.
- **Curvature Radius:** Calculates and displays lane curvature in meters.

## Example Usage
Sample video `Autonomous driving lane detection sample video 1.mp4` demonstrates the pipeline. Replace the video file path in the `video_path` variable to test with other videos.

## Requirements
- OpenCV
- NumPy
- Pillow
- IPython

These dependencies are listed in `requirements.txt`.

## Notes
- For optimal performance, ensure video frames have clear lane markings.
- Adjust the trapezoidal region or parameters as needed for different video resolutions or road conditions.

## Contributing
Contributions are welcome! Please submit a pull request or raise an issue for suggestions and bug reports.

## License
This project is licensed under the MIT License.

