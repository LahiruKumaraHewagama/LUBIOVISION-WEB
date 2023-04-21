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

app.config['UPLOAD_FOLDER'] = 'static\\uploaded'

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
	return render_template('upload.html',buttonClicked=True)

def finds():
	test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
	vals = ['Melanoma', 'Nevus'] # change this according to what you've trained your model to do
	test_dir = 'static'
	test_generator = test_datagen.flow_from_directory(
			test_dir,
			shuffle=False,
			batch_size = 1)

	pred = model.predict_generator(test_generator)
	
	print(pred)
	return str(vals[np.argmax(pred)])

def heatmap(path):
	import argparse
	import cv2
	import numpy as np
	import torch
	from torchvision import models
	from pytorch_grad_cam import GradCAM, \
		HiResCAM, \
		ScoreCAM, \
		GradCAMPlusPlus, \
		AblationCAM, \
		XGradCAM, \
		EigenCAM, \
		EigenGradCAM, \
		LayerCAM, \
		FullGrad, \
		GradCAMElementWise


	from pytorch_grad_cam import GuidedBackpropReLUModel
	from pytorch_grad_cam.utils.image import show_cam_on_image, \
		deprocess_image, \
		preprocess_image
	from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


	methods = \
		{"gradcam": GradCAM,
		"hirescam": HiResCAM,
		"scorecam": ScoreCAM,
		"gradcam++": GradCAMPlusPlus,
		"ablationcam": AblationCAM,
		"xgradcam": XGradCAM,
		"eigencam": EigenCAM,
		"eigengradcam": EigenGradCAM,
		"layercam": LayerCAM,
		"fullgrad": FullGrad,
		"gradcamelementwise": GradCAMElementWise}

	model = models.resnet50(pretrained=True)
	# import tensorflow as tf
	# model= tf.keras.models.load_model(r'D:\CSE\6,7, 8 SEMESTERS\FYP\IMPLEMENT - NEW\CNN + Segmentation\new implementaion cnn\\Xception.h5')

	# Choose the target layer you want to compute the visualization for.
	# Usually this will be the last convolutional layer in the model.
	# Some common choices can be:
	# Resnet18 and 50: model.layer4
	# VGG, densenet161: model.features[-1]
	# mnasnet1_0: model.layers[-1]
	# You can print the model to help chose the layer
	# You can pass a list with several target layers,
	# in that case the CAMs will be computed per layer and then aggregated.
	# You can also try selecting all layers of a certain type, with e.g:
	# from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
	# find_layer_types_recursive(model, [torch.nn.ReLU])

	target_layers = [model.layer4]
	# target_layers = model.layers[-5].name

	rgb_img = cv2.imread(path, 1)[:, :, ::-1]
	rgb_img = np.float32(rgb_img) / 255
	input_tensor = preprocess_image(rgb_img,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

	# We have to specify the target we want to generate
	# the Class Activation Maps for.
	# If targets is None, the highest scoring category (for every member in the batch) will be used.
	# You can target specific categories by
	# targets = [e.g ClassifierOutputTarget(281)]
	targets = None

	# Using the with statement ensures the context is freed, and you can
	# recreate different CAM objects in a loop.
	cam_algorithm = methods["gradcam++"]
	with cam_algorithm(model=model,
					target_layers=target_layers,
					use_cuda=False) as cam:

		# AblationCAM and ScoreCAM have batched implementations.
		# You can override the internal batch size for faster computation.
		cam.batch_size = 32
		grayscale_cam = cam(input_tensor=input_tensor,
							targets=targets)

		# Here grayscale_cam has only one image in the batch
		grayscale_cam = grayscale_cam[0, :]

		cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

		# cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
		cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

	gb_model = GuidedBackpropReLUModel(model=model, use_cuda=False)
	gb = gb_model(input_tensor, target_category=None)

	cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
	cam_gb = deprocess_image(cam_mask * gb)
	gb = deprocess_image(gb)

	cv2.imwrite('static\Grad_cam++.jpg', cam_image)
	# Grad cam 
	cam_algorithm = methods["gradcam"]
	with cam_algorithm(model=model,
					target_layers=target_layers,
					use_cuda=False) as cam:

		# AblationCAM and ScoreCAM have batched implementations.
		# You can override the internal batch size for faster computation.
		cam.batch_size = 32
		grayscale_cam = cam(input_tensor=input_tensor,
							targets=targets)

		# Here grayscale_cam has only one image in the batch
		grayscale_cam = grayscale_cam[0, :]

		cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

		# cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
		cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

	gb_model = GuidedBackpropReLUModel(model=model, use_cuda=False)
	gb = gb_model(input_tensor, target_category=None)

	cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
	cam_gb = deprocess_image(cam_mask * gb)
	gb = deprocess_image(gb)

	cv2.imwrite('static\Grad_cam.jpg', cam_image)
	# cv2.imwrite('B_gb.jpg', gb)
	# cv2.imwrite('C_cam_gb.jpg', cam_gb)

	return 

@app.route('/result', methods = ['GET', 'POST'])
def upload_file():

	if request.method == 'POST':
		
		f = request.files['file']
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
		val = finds()

		pathf1= os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
		img = np.array(load_img(pathf1,target_size=(224,224,3)),dtype=np.float64)

		heatmap(pathf1)
		# grad_cam=_compare_an_image(model,img,model.layers[-5].name)
		# f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))

		return render_template('pred.html', ss = val,image=pathf1)

if __name__ == '__main__':
	from dotenv import load_dotenv
	dotenv_path = '.env' # Path to .env file
	load_dotenv(dotenv_path)
	app.run(debug=True)
