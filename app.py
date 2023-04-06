from flask import Flask, render_template, request
from werkzeug.utils  import secure_filename
from keras.preprocessing.image import ImageDataGenerator
from explanation import _compare_an_image
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np
import os

try:
	import shutil
	# % cd uploaded % mkdir image % cd ..
	print()
except:
	pass

model = tf.keras.models.load_model('models\\VGG19.h5')
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploaded\\image'

@app.route('/',methods = ['GET'])
def home():
	return render_template('index.html')

@app.route('/generate',methods = ['GET', 'POST'])
def upload_f():
	if request.method == 'POST':
		with os.scandir(app.config['UPLOAD_FOLDER']) as entries:
			for entry in entries:
				if entry.is_dir() and not entry.is_symlink():
					shutil.rmtree(entry.path)
				else:
					os.remove(entry.path)
	return render_template('upload.html')

def finds():
	test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
	vals = ['Melanoma', 'Nevus'] # change this according to what you've trained your model to do
	test_dir = 'uploaded'
	test_generator = test_datagen.flow_from_directory(
			test_dir,
			shuffle=False,
			batch_size = 1)

	pred = model.predict_generator(test_generator)
	
	print(pred)
	return str(vals[np.argmax(pred)])

@app.route('/result', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		
		f = request.files['file']
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
		val = finds()

		pathf1= os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
		img = np.array(load_img(pathf1,target_size=(224,224,3)),dtype=np.float64)
		# grad_cam=_compare_an_image(model,img,model.layers[-5].name)
		# f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))

		return render_template('pred.html', ss = val)

if __name__ == '__main__':
	from dotenv import load_dotenv
	dotenv_path = '.env' # Path to .env file
	load_dotenv(dotenv_path)
	app.run(debug=True)
