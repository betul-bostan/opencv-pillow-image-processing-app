# OpenCV and Pillow Image Processing App

A Flask-based web application for applying and comparing image-processing operations using **OpenCV** and **Pillow**.

Users can upload an image, select an operation, and view the original image together with the results produced by both image-processing approaches.

## Project Overview

This project demonstrates fundamental computer vision and digital image-processing techniques through an interactive web interface.

For each selected operation, the application generates:

- The original uploaded image
- The result produced with OpenCV
- The result produced with Pillow or Python-based processing

This side-by-side structure makes it easier to compare how different libraries implement similar image-processing operations.

## Supported Operations

### Basic Adjustments

- Grayscale conversion
- Brightness adjustment
- Contrast adjustment
- Contrast stretching
- Image negative
- Horizontal flipping
- Vertical flipping

### Filtering

- Mean filter
- Gaussian blur
- Median filter

### Segmentation and Histogram Operations

- Binary thresholding
- Histogram equalization

### Edge Detection

- Sobel
- Prewitt
- Laplacian

### Morphological Operations

- Erosion
- Dilation
- Opening
- Closing

### Geometric Transformations

- Rotation
- Zoom in
- Zoom out
- Perspective transformation

### Additional Operations

- Convolution and sharpening
- Template matching
- Centroid detection

## Technologies

- Python
- Flask
- OpenCV
- Pillow
- NumPy
- HTML
- CSS
- Jinja2

## Repository Structure

```text
opencv-pillow-image-processing-app/
├── static/
│   └── css/
├── templates/
│   ├── index.html
│   ├── operations.html
│   └── result.html
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

The application automatically creates the following runtime directories when needed:

```text
static/uploads/
static/processed/
```

These directories are excluded from version control because they contain user uploads and generated outputs.

## Installation

Clone the repository:

```bash
git clone https://github.com/betul-bostan/opencv-pillow-image-processing-app.git
cd opencv-pillow-image-processing-app
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Set an optional Flask secret key:

```bash
export SECRET_KEY="your-secret-key"
```

Then run:

```bash
python app.py
```

Open the following address in your browser:

```text
http://127.0.0.1:9000
```

## Application Workflow

1. Upload an image
2. Select an image-processing operation
3. Apply the operation with OpenCV
4. Apply the equivalent operation with Pillow or Python
5. Save the generated outputs temporarily
6. Display the original and processed images side by side

## Implementation Notes

- OpenCV is used for matrix-based computer vision operations.
- Pillow is used for high-level image transformations and filtering.
- Some operations that are not directly available in Pillow are delegated to the OpenCV implementation and converted back into a Pillow image.
- Uploaded and generated files are stored only as runtime artifacts and are excluded from Git.

## Limitations

- Uploaded file type and size validation should be improved.
- Generated files are not automatically cleaned after each session.
- The application currently runs in Flask debug mode during local development.
- Some Pillow results approximate the OpenCV operation rather than reproducing the exact algorithm.
- User sessions may overwrite files when identical filenames are uploaded.

## Future Improvements

- Add file-extension and MIME-type validation
- Add maximum upload-size limits
- Generate unique filenames
- Automatically delete old uploads and processed images
- Add unit tests for image-processing functions
- Add application screenshots
- Add Docker support
- Deploy the Flask application
- Add adjustable filter parameters
- Add downloadable processed results

## Author

**Betül Bostan**

- [GitHub](https://github.com/betul-bostan)
- [LinkedIn](https://www.linkedin.com/in/bet%C3%BCl-bostan-2105942b2/)
