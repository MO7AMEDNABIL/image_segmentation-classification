from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from classify import classify_image  # Import from your file
from seg import segment_image       # Import from your file

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    task = request.form['task']
    file = request.files['image']
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    if task == 'classification':
        label, confidence = classify_image(filepath)
        result = f"Prediction: {label} (Confidence: {confidence:.2f})"
        display_path = filepath  # Show original image
        return render_template('result copy.html', image_path=display_path, result=result)
    else:
        mask_path = segment_image(filepath)
        result = "Segmentation completed."
        original_path = filepath          # original image path
        segmented_path = mask_path        # mask or segmented image path

        return render_template('result.html', 
                            result=result, 
                            original_path=original_path, 
                            segmented_path=segmented_path)

    

if __name__ == '__main__':
    app.run(debug=True)
