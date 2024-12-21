# Lane Detection Project

This repository demonstrates two different lane detection algorithms for autonomous driving. 

### Algorithm 1: Lane Detection using Hough Transform and Cany Edge Detection
The algorithm processes video frames to identify lane boundaries, calculate curvature radius, and overlay this information on the original video.


https://github.com/user-attachments/assets/7af18436-cab4-4800-ba86-19982ccbf7dd



### Algorithm 2: Lane Detection using geometric transformations
This algorithm implements a computer vision pipeline to detect road lane boundaries and visualize them on video frames. By using advanced image processing techniques and geometric transformations, the system can estimate lane curvature and the vehicle's position relative to the center.


https://github.com/user-attachments/assets/ecb06c8c-f997-47e3-9040-ed995fa06222


## File Structure
```
Lane-Detection-Project/
|
├── sample_videos/            # Directory containing sample videos
│   ├── Autonomous driving lane detection sample video 1.mp4
|
├── src/
│   ├── __init__.py            # Python package initializer
│   ├── simple_lane_detection.py      # Algorithm 1 lane detection implementation with export to output directory
│   ├── advanced_lane_detection.py      # Algorithm 2 lane detection implementation with export to output directory
│
├── .gitignore                # Git ignore file
├── simple_lane_detection.ipynb               # Jupyter Notebook for algorithm 1 lane detection
├── advanced_lane_detection.ipynb               # Jupyter Notebook for algorithm 2 lane detection

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
