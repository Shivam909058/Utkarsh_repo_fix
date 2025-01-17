## Car Number Plate Detection (Russian Plates)

This project is designed for detecting Russian car number plates using a webcam. It utilizes OpenCV for image capture and plate detection, EasyOCR for optical character recognition (OCR), and Flask for serving a web interface to interact with the system.

## Project Structure 

Car-Number-Plate-Detection


├── model
│   └── haarcascade_russian_plate_number.xml      # Haarcascade classifier for detecting Russian plates

├── static
│   ├── styles.css                               # CSS styles for the web page
│   └── background.jpg                          # Background image for the web page

├── template
│   └── index.html                              # HTML file for the main page

├── number_plates                              # Folder to save captured number plates

├── requirements.txt                            # List of dependencies to install

├── number_plate.py                             # Main script with Flask API for capturing plates and OCR



## Prerequisites

Python 3.x - Ensure you have Python installed on your system.
Virtual Environment - It's recommended to use a virtual environment for managing dependencies.

## Installation and Setup

Follow these steps to get the project running:

**Create a Virtual Environment:**
Open your terminal/command prompt and navigate to the project folder. Then, create and activate a virtual environment:

python -m venv venv      # Create a virtual environment
source venv/bin/activate # For macOS/Linux
venv\Scripts\activate    # For Windows

**Install the Required Packages:**
Use the requirements.txt file to install the necessary dependencies:

pip install -r requirements.txt

**Run the Application:**
After installing the dependencies, you can run the application using the following command:

python number_plate.py

## How to Use
Open the Web Interface:

1) Open a browser and go to http://localhost:5000.

2) Capture a Number Plate:

On the webpage, click the Capture button. This will activate your webcam.
The camera window will open showing the live feed, and any detected number plates will be highlighted with a blue rectangle.
To capture an image, press the 's' key when the desired number plate appears in the camera feed.
The image will be saved to the number_plates/ folder, and the OCR will process the number plate text.

3) View Captured Plates:

After capturing, the web page will display the extracted text (number plate).
You can also view the captured image by clicking on the View Captured Image button on the web page

## File Details

number_plate.py: The main script that runs the Flask application. It handles the webcam capture, plate detection, and OCR processing.

haarcascade_russian_plate_number.xml: The Haarcascade XML file for detecting Russian car number plates. This file is used by OpenCV's CascadeClassifier to detect the number plates.

styles.css: Contains the CSS styles for the webpage.

index.html: The HTML structure for the web interface, allowing users to interact with the system.

## Capturing Number Plates

The system uses a Haar Cascade classifier to detect the number plate in real-time. Once a plate is detected, the user can press 's' to save the image of the detected number plate.
The image is stored in the number_plates folder, and EasyOCR is used to extract text from the plate.

## Notes

The current system is tailored for Russian number plates.
Ensure the camera is connected and working properly before running the application.