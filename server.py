from flask import Flask, render_template, request
from werkzeug import secure_filename
import os
app = Flask(__name__)
	
# @app.route('/uploader', methods = ['GET', 'POST'])
# def upload_file():
#    if request.method == 'POST':
#       f = request.files['file']
#       f.save(secure_filename(f.filename))
#       os.system("python3 label_image.py --graph=/home/sasi-jana/Downloads/example_code/graph.pb --labels=/home/sasi-jana/Downloads/example_code/label.txt --input_layer=Placeholder --output_layer=final_result --image=$HOME/Downloads/rose.jpeg")
#       return 'file uploaded successfully'
		
# if __name__ == '__main__':
#    app.run(debug = True)

@app.route('/signUpUser', methods=['POST'])
def signUpUser():
    user =  request.form['name'];
    return json.dumps({'data':'OK'});