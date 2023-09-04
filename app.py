from flask import Flask, render_template, request
from werkzeug.utils  import secure_filename
from keras.preprocessing.image import ImageDataGenerator
# from explanation import _compare_an_image
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np
import os
from uuid import uuid4
from datetime import date
import cv2
import numpy as np
import torch
from vit_pytorch import ViT
import torch
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image,preprocess_image
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


try:
	import shutil
	# % cd uploaded % mkdir image % cd ..
	print()
except:
	pass

model = tf.keras.models.load_model('models/Xception.h5')
vit_model = torch.load('models/SMVIT.pt',map_location=torch.device('cpu'))
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploaded'

@app.route('/',methods = ['GET'])
def home():
	# return render_template('edit.html')
	return render_template('index.html')

@app.route('/pdf',methods = ['POST'])
def pdf():
	# return render_template('edit.html')
# document.getElementById("patientName").value =document.getElementById("contact-name").value;
# 			document.getElementById("uuid").value =id;
# 			document.getElementById("date").value =date;


# 			document.getElementById("val").value =ss;
# 			document.getElementById("acc").value =accuracy;
# 			document.getElementById("approval").value =document.getElementById("Approve").checked;
# 			document.getElementById("feedback").value =document.getElementById("contact-message").value;
# 			document.getElementById("imgname").value =name;
# 			document.getElementById("imgurl").value =image;
	if request.method == 'POST':

			try:
				patientName=request.form.get('patientName')
				uuid=request.form.get('uuid')
				dates=request.form.get('date')
				val=request.form.get('val')
				acc=request.form.get('acc')
				approval=request.form.get('approval')
				feedback=request.form.get('feedback')
				imgname=request.form.get('imgname')			
				imgurl=request.form.get('imgurl')

			except:
				pass


	return render_template('pdf.html',patientName=patientName,uuid=uuid,date=dates,accuracy=acc,val=val,approval=approval,feedback=feedback,imgname=imgname,imgurl=imgurl)

@app.route('/generate',methods = ['GET', 'POST'])
def generate():
	if request.method == 'POST':

		try:
			url=request.form.get('url')
			os.remove(url)
			os.remove(url.split('.')[0]+'_gradcam++.jpg')
			os.remove(url.split('.')[0]+'_gradcam.jpg')
		except:
			pass

	return render_template('upload.html',buttonClicked=True)

@app.route('/usermanual',methods = ['GET', 'POST'])
def usermanual():
	if request.method == 'POST':
		pass
	return render_template('usermanual.html')

@app.route('/aboutus',methods = ['GET', 'POST'])
def aboutus():
	if request.method == 'POST':
		pass
	return render_template('aboutus.html')

def accurygetting(mel_score,nv_score):
	
	mel_fac = 0
	nv_fac = 0

	if nv_score == mel_score:
		nv_fac = abs(nv_score)
		mel_fac = abs(mel_score)
	elif mel_score > 0 and nv_score > 0:
		if mel_score > nv_score:
			mel_fac = abs(mel_score)+abs(nv_score)
			nv_fac = abs(nv_score)
		else:
			nv_fac = abs(mel_score)+abs(nv_score)
			mel_fac = abs(mel_score)
	elif mel_score < 0 and nv_score < 0:
		if mel_score > nv_score:
			mel_fac = abs(mel_score)+abs(nv_score)
			nv_fac = abs(nv_score)
		else:
			nv_fac = abs(mel_score)+abs(nv_score)
			mel_fac = abs(mel_score)
	elif mel_score > 0:
		mel_fac = abs(mel_score)+abs(nv_score)
		nv_fac = abs(nv_score)    
	elif nv_score > 0:
		nv_fac = abs(mel_score)+abs(nv_score)
		mel_fac = abs(mel_score)
	else:
		nv_fac = abs(nv_score)
		mel_fac = abs(mel_score)
		
	mel_prsent = (mel_fac * 100)/ (mel_fac + nv_fac)
	nv_prsent = (nv_fac * 100)/ (mel_fac + nv_fac)

	return max([mel_prsent,nv_prsent])



def finds(path,model_type):

	skin_class = ['Melanoma', 'Non-Melanoma'] # change this according to what you've trained your model to do
	if(model_type=='cnn'):
		test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
				
		#load the image
		inputs_image = load_img(path, target_size=(224, 224))

		#preprocess the image
		inputs_image = img_to_array(inputs_image)
		inputs_image = inputs_image.reshape((1, inputs_image.shape[0], inputs_image.shape[1], inputs_image.shape[2]))
		inputs_image = preprocess_input(inputs_image)

		#make the prediction
		pred = model.predict(inputs_image)
		# acc=pred[0][np.argmax(pred)]/sum(pred[0])	
		acc=accurygetting(pred[0][0],pred[0][1])	
		return str(skin_class[np.argmax(pred)]),str(round(acc,2))+'%'

	else:
		def reshape_transform(tensor, height=14, width=14):
			result = tensor[:, 1:, :].reshape(tensor.size(0),
											height, width, tensor.size(2))

			# Bring the channels to the first dimension,
			# like in CNNs.
			result = result.transpose(2, 3).transpose(1, 2)
			return result

		# target_layers = [vit_model.transformer.encoder.layer[0].attention_norm]
		# target_layers = [vit_model.transformer.encoder.layer[1].attention_norm]
		target_layers = [vit_model.transformer.encoder.layer[-1].attention_norm]


		url=path
		rgb_img = cv2.imread(url, 1)[:, :, ::-1]
		rgb_img = cv2.resize(rgb_img, (224, 224))
		rgb_img = np.float32(rgb_img) / 255
		input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

		methods = {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus }


		# method='gradcam'
		method='gradcam'

		cam = methods[method](model=vit_model,target_layers=target_layers,use_cuda=False,reshape_transform=reshape_transform)

		# If None, returns the map for the highest scoring category.
		# Otherwise, targets the requested category.
		targets = None

		# input_tensor = input_tensor.cuda()
		# AblationCAM and ScoreCAM have batched implementations.
		# You can override the internal batch size for faster computation.
		cam.batch_size = 32

		outputs = cam.activations_and_grads(input_tensor)[0]
		print(outputs)

		if targets is None:
			target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
			print(target_categories)
			targets = [ClassifierOutputTarget(category) for category in target_categories]

		pred = outputs.cpu().data.numpy()
		# print(pred)
		# acc=pred[0][target_categories[0]]/(abs(pred[0][0])+abs(pred[0][1]))
		acc=accurygetting(pred[0][0],pred[0][1])
		return skin_class[target_categories[0]],str(round(acc,2))+'%'
	

def heatmap(path,model_type,name):
	import argparse
	import cv2
	import numpy as np
	import torch
	from torchvision import models
	
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

	print(model_type)

	if(model_type=='cnn'):
		model = models.resnet50(pretrained=True)

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
		method="gradcam++"
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

		# cv2.imwrite('static\Grad_cam++.jpg', cam_image)
		# name=path.split('.')[0].split('/')[-1]
		cv2.imwrite(f'static//uploaded//'+name+"_"+method+'.jpg', cam_image)		


		# Grad cam 
		cam_algorithm = methods["gradcam"]
		method="gradcam"
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

		# cv2.imwrite('static\Grad_cam.jpg', cam_image)
		# name=path.split('.')[0].split('/')[-1]
		cv2.imwrite(f'static//uploaded//'+name+"_"+method+'.jpg', cam_image)		
		# cv2.imwrite('B_gb.jpg', gb)
		# cv2.imwrite('C_cam_gb.jpg', cam_gb)

	else:

		def reshape_transform(tensor, height=14, width=14):
			result = tensor[:, 1:, :].reshape(tensor.size(0),
											height, width, tensor.size(2))

			# Bring the channels to the first dimension,
			# like in CNNs.
			result = result.transpose(2, 3).transpose(1, 2)
			return result


		methods = {"gradcam": GradCAM,
				"scorecam": ScoreCAM,
				"gradcam++": GradCAMPlusPlus }


		# target_layers = [vit_model.transformer.encoder.layer[0].attention_norm]
		target_layers = [vit_model.transformer.encoder.layer[1].attention_norm]
		# target_layers = [vit_model.transformer.encoder.layer[-1].attention_norm]

		# method='gradcam'
		method='gradcam'

		cam = methods[method](model=vit_model,target_layers=target_layers,use_cuda=False,reshape_transform=reshape_transform)

		url=path
		rgb_img = cv2.imread(url, 1)[:, :, ::-1]
		rgb_img = cv2.resize(rgb_img, (224, 224))
		rgb_img = np.float32(rgb_img) / 255
		input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

		# If None, returns the map for the highest scoring category.
		# Otherwise, targets the requested category.
		targets = None
		# input_tensor = input_tensor.cuda()
		# AblationCAM and ScoreCAM have batched implementations.
		# You can override the internal batch size for faster computation.
		cam.batch_size = 32

		outputs = cam.activations_and_grads(input_tensor)[0]
		if targets is None:
			target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
		
			targets = [ClassifierOutputTarget(category) for category in target_categories]

		grayscale_cam = cam(input_tensor=input_tensor,aug_smooth =False,targets=targets)

		# Here grayscale_cam has only one image in the batch
		grayscale_cam = grayscale_cam[0, :]

		cam_image = show_cam_on_image(rgb_img, grayscale_cam)

		# name=url.split('.')[0].split('/')[-1]
		cv2.imwrite(f'static//uploaded//'+name+"_"+method+'.jpg', cam_image)		



		# method='gradcam'
		method='gradcam++'

		cam = methods[method](model=vit_model,target_layers=target_layers,use_cuda=False,reshape_transform=reshape_transform)

		url=path
		rgb_img = cv2.imread(url, 1)[:, :, ::-1]
		rgb_img = cv2.resize(rgb_img, (224, 224))
		rgb_img = np.float32(rgb_img) / 255
		input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

		# If None, returns the map for the highest scoring category.
		# Otherwise, targets the requested category.
		targets = None
		# input_tensor = input_tensor.cuda()
		# You can override the internal batch size for faster computation.
		cam.batch_size = 32

		outputs = cam.activations_and_grads(input_tensor)[0]
		if targets is None:
			target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
		
			targets = [ClassifierOutputTarget(category) for category in target_categories]

		grayscale_cam = cam(input_tensor=input_tensor,aug_smooth =False,targets=targets)

		# Here grayscale_cam has only one image in the batch
		grayscale_cam = grayscale_cam[0, :]

		cam_image = show_cam_on_image(rgb_img, grayscale_cam)

		# name=url.split('.')[0].split('/')[-1]
		cv2.imwrite(f'static//uploaded//'+name+"_"+method+'.jpg', cam_image)		
		# cv2.imwrite('static\\'+name+'_Grad_cam++.jpg', cam_image)

	return


@app.route('/result', methods = ['GET', 'POST'])
def upload_file():

	if request.method == 'POST':
	
		# print(request.form.get('model'))
		model_type=request.form.get('model')
		f = request.files['file']
		f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
		name=f.filename.split('.')[0]

		pathf1= os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))

		val,acc = finds(pathf1,model_type)
				
		img = np.array(load_img(pathf1,target_size=(224,224,3)),dtype=np.float64)

		heatmap(pathf1,model_type,name)
		# grad_cam=_compare_an_image(model,img,model.layers[-5].name)
		# f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
		id=str(uuid4())
		return render_template('pred.html', ss = val,image=pathf1,accuracy=acc,name=name,id=id,date=str(date.today()))

if __name__ == '__main__':
	from dotenv import load_dotenv
	dotenv_path = '.env' # Path to .env file
	load_dotenv(dotenv_path)
	app.run(debug=False)
	app.run(use_reloader=True)

