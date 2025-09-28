from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # TODO: Run model inference and create keypoints image & measurements
            # For demo, use dummy data for now:
            measurements = {
                'body_length': 24.33,
                'height_at_withers': 196,
                'chest_width': 211.8,
                'rump_angle': 'N/A'
            }
            atc_pred = "Buffalo"

            keypoints_img = url_for('static', filename='uploads/' + filename)  # Replace with overlay image if available

            return render_template('result.html',
                                   filename=filename,
                                   measurements=measurements,
                                   atc_pred=atc_pred,
                                   keypoints_img=keypoints_img)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
