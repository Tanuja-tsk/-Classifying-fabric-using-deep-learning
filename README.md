

Pattern Sense: Classifying Fabric Patterns using Deep Learning

Pattern Sense is a deep learning-based project designed to classify fabric patterns such as floral, stripes, geometric, and more. It automates the visual recognition process, saving time and reducing errors in the textile and fashion industries.

Problem Statement

Manual identification of fabric patterns is time-consuming and error-prone. Pattern Sense solves this problem using a CNN-based image classifier that identifies the type of fabric pattern in an image.

Real-World Applications

Fashion Industry

* Quickly sort large collections of fabrics by pattern type
* Helps designers access organized materials for creative workflows

Textile Quality Control

* Detects anomalies or defective prints in fabric
* Ensures consistent product quality during manufacturing

Interior Design

* Helps select matching patterns for upholstery, curtains, and decor
* Reduces time spent on manual matching and catalog review

Technologies Used

TensorFlow 2.14 / Keras 2.14
Transfer Learning with ResNet50
Flask Web Framework
Google Colab for training
HTML and Bootstrap for UI

Project Structure

pattern-sense
app.py                -> Flask application
model/
fabric\_pattern\_model.h5
templates/
home.html
index.html
result.html
data/
dress\_pattern\_data/ (organized folders for training images)
src/
train_model
Steps to Run Locally

1. Download or clone the folder
2. Ensure the trained model file is inside model/fabric\_pattern\_model.h5
3. Create and activate virtual environment
   conda activate pattern-sense
4. Install required packages
   pip install tensorflow==2.14 keras==2.14 flask pandas numpy
5. Run the app
   python app.py
6. Open browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

Results

The model achieves good accuracy using ResNet50 and is able to classify patterns like floral, stripes, polka dots, and geometric with high confidence.

Dataset

Dress Pattern Dataset from Kaggle
Link: [https://www.kaggle.com/datasets/nguyngiabol/dress-pattern-dataset](https://www.kaggle.com/datasets/nguyngiabol/dress-pattern-dataset)
Contains folders of labeled images representing different fabric patterns

Future Scope

* Add real-time webcam-based prediction
* User feedback for wrongly classified images
* Improve with EfficientNet or vision transformers
* Larger dataset support


