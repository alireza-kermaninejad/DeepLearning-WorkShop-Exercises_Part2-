from flask import Flask, render_template, request, jsonify
from tensorflow import tf
import numpy as np
import cv2
from errors_handlers import errors_handlers


app = Flask(__name__)
app.register_blueprint(errors_handlers)

MODEL_PATH = "models/saved_model"
# Loading our trained model
model = tf.keras.models.load_model(MODEL_PATH)

@app.route('/', methods=['GET'])
def home_page():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # load and save image in to a file
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    # checking error
    if type(image) != image:
        return jsonify({"ERROR": "Unsupported Media Type."}), 415

    # read image
    image = cv2.imread(image_path)
    # decode
    image = tf.image.decode_image(image, expand_animations=False)
    # using vgg16 model preprocess input
    image = tf.keras.applications.vgg16.preprocess_input(image)
    # resize
    image = image.resize(image, (224, 224))
    # normalize
    image /= 255.0
    # expand (224, 224, 3) to (1, 224, 224, 3)
    image = tf.expand_dims(image ,axis=0)

    # checking error
    if tf.image.shape != (1, 224, 224, 3):
        return jsonify({"ERROR": "Shape of image  must be (1, 224, 224, 3)."}), 400

    # predict model
    preds = model.predict(image)

    return render_template('index.html')


@app.route("/" ,methods = ["GET"])
def statistic():
    precision = "97.7%"
    recall = "97.7%"
    confusion_matrix = np.array([[214, 16, 0],
                                 [3, 475, 5], 
                                 [0, 2, 391]])

    result = {
        "Confusion_Matrix" : confusion_matrix, "Precision":precision, "Recall":recall}

    return jsonify(result) , 200


# ///////////////////////////////////////////////////////////
# Errors Handling (Some HTTP status codes)
# 4xx client errors
@app.errorhandler(404)
def error_404(error):
    return render_template('errors/404.html'), 404

@app.errorhandler(400)
def error_400(error):
    return render_template('errors/400.html'), 400

@app.errorhandler(403)
def error_403(error):
    return render_template('errors/403.html'), 403

@app.errorhandler(500)
def error_500(error):
    return render_template('errors/500.html'), 500

@app.errorhandler(415)
def error_415(error):
    return render_template('errors/415.html'), 415
# ///////////////////////////////////////////////////////////

if __name__ == '__main__':
    app.run(debug=True)