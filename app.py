from flask import Flask, render_template, request, session, flash, redirect, url_for
from PIL import Image
from io import BytesIO
import base64
from parallel import parallel
from sequential import sequential
import cv2
import numpy as np

app = Flask(__name__)
# app.secret_key = os.urandom(24)
@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        code = request.form['code']
        if 'image' not in request.files:
            flash('You must upload an image')
            return redirect(request.url)
        image = request.files['image']
        if image.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        img = Image.open(BytesIO(image.read()))
        img = np.array(img)
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if code == 'p':
            img = parallel(img)
        else:
            img = sequential(img)

        img = Image.fromarray(img)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img = base64.b64encode(buffered.getvalue())
        return render_template('index.html', brain=img.decode('utf-8'))
    else:
        return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True, port=5000)
