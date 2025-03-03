#Importing the required Libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for, send_file
from ultralytics import YOLO
import os
import logging
import mimetypes
from pathlib import Path
import re
import urllib.parse


logging.basicConfig(level=logging.DEBUG)

# Initialize the Flask App
app = Flask(__name__)

#Configuring the directories for saving original and predicted Images and Videos
app.config['IMAGE_UPLOAD_FOLDER'] = os.path.join('Uploads', 'Uploaded_Images')
app.config['VIDEO_UPLOAD_FOLDER'] = os.path.join('Uploads' ,'Uploaded_Videos')
app.config['IMAGE_SAVE_FOLDER'] = os.path.join('Predictions', 'Predicted_Images')
app.config['VIDEO_SAVE_FOLDER'] = os.path.join('Predictions', 'Predicted_Videos')

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB max file size

#Creating the directories if they do not exist
os.makedirs(app.config['IMAGE_UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['VIDEO_UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMAGE_SAVE_FOLDER'], exist_ok=True)
os.makedirs(app.config['VIDEO_SAVE_FOLDER'], exist_ok=True)

# Load the Pre-Trained Model/Weights
model_path = os.getenv('MODEL_PATH', "best.pt")
model = YOLO(model_path)

def sanitize_name(filename):
    """Remove special characters and spaces from filename"""
    # Remove special characters and replace spaces with underscores
    clean_name = re.sub(r'[^\w\-_.!]', ' ', filename)
    return clean_name


#Rendering the Home Page through Flask
@app.route('/')
def home():
    return render_template('index.html')

#Predicting on the Uploaded Image
@app.route('/predict', methods=['POST'])
def predict():
    try:
        #Checking if the request has the file
        if 'test_file' not in request.files:
            #return jsonify({'Error': 'No test_image key in request files'}), 400
            return jsonify({'Error': 'No File Uploaded'})
        
        #Alloting the file from request in a variable
        file = request.files['test_file']

        #Checking for filename
        if not file.filename:
            return jsonify({'Error': 'No File Found'}), 400

        #Getting the extension of the file
        ext = file.filename.split('.')[-1]

        #Verifying if the uploaded file is an Image or a Video
        if ext in ['bmp', 'dng', 'jpg', 'jpeg', 'png', 'mpo', 'webp', 'tif', 'tiff', 'pfm', 'HEIC']:
            
            #Providing the path to save the Original Image (without predictions)
            img_path = os.path.join(app.config['IMAGE_UPLOAD_FOLDER'], file.filename)

            #Saving the Original Image
            app.logger.info(f'Saving the Uploaded Image to: {img_path}')
            file.save(img_path)

            #Getting the Image file name without extension
            img_name = os.path.basename(file.filename).split('.')[0] 
        
            #Opening the Image and converting it to BGR format
            image = Image.open(file)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            #Predicting the Image using the pre-trained model
            results = model.predict(image, save=False) #Saving is Flase so as to avoid the default the saving features of the model.

            #Providing the path to save the Predicted Image with suffix '_pred'
            pred_img_name = f'{img_name}_pred.jpg'
            save_img_path = os.path.join(app.config['IMAGE_SAVE_FOLDER'], pred_img_name)
            app.logger.info(f'Saving the Predicted Image to: {save_img_path}')

            # Process results list
            for result in results:
                arr = result.plot() #To plot the predicted image with bounding boxes
                cv2.imwrite(save_img_path, arr) #Saving the predicted image
            
            #Clean up the Uploads
            try:
                os.remove(img_path)
            except Exception as error:
                app.logger.warning(f'Could not remove the Uploaded file: {error}')

            return jsonify({'image': url_for('predict_file', filename=pred_img_name)})

        elif ext in ['asf', 'm4v', 'mpeg', 'mpg', 'ts', 'mp4', 'avi', 'mov', 'flv', 'wmv', 'webm', 'mkv']:
                
            #Providing the path to save the Original Video (without predictions)
            clean_filename = sanitize_name(file.filename)
            vid_path = os.path.join(app.config['VIDEO_UPLOAD_FOLDER'], clean_filename)

            #Saving the Original Video File
            app.logger.info(f'Saving the Uploaded Video to: {vid_path}')
            file.save(vid_path)

            #Getting the video file name without extension
            vid_name = os.path.basename(clean_filename).split('.')[0]

            try:
                #Reading the video file
                vid = cv2.VideoCapture(vid_path)
                if not vid.isOpened():
                    raise ValueError('Video File could not be opened')

                #Getting the video properties
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                fps = vid.get(cv2.CAP_PROP_FPS)
                frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

                pred_vid_name = f'{vid_name}_pred.mp4'
                save_video_path = os.path.join(app.config['VIDEO_SAVE_FOLDER'], pred_vid_name)

                out_video = cv2.VideoWriter(save_video_path, fourcc, fps, (frame_width, frame_height))
                app.logger.info(f'Saving the Predicted Video to: {save_video_path}')

                #Processing the frames of the video
                while True:
                    ret, frame = vid.read()
                        
                    if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    #Predicting the Frames using the pre-trained model
                    results = model.predict(frame, save=False)
                    for result in results:
                        pred_frame = result.plot()
                        out_video.write(pred_frame)
                    
                #Release rosources
                vid.release()
                out_video.release()
                cv2.destroyAllWindows()

                #Clean up uploads
                try:
                    os.remove(vid_path)
                except Exception as error:
                    app.logger.warning(f'Could not remove the uploaded file: {error}')

                if os.path.exists(save_video_path) and os.path.getsize(save_video_path) > 0:
                    app.logger.info(f'Video saved successfully to: {save_video_path}')
                    return jsonify({'video': url_for('predict_file', filename = pred_vid_name)})
                else:
                    return jsonify({'Error: Video could not be saved'}), 500

            except Exception as error:
                app.logger.error(f'Error processing the video: {str(error)}')
                return jsonify({'Error': f'Video could not be processed {str(error)}'}), 500
                
            finally:
                if 'vid' in locals():
                    vid.release()
                if 'out_video' in locals():
                    out_video.release()
                cv2.destroyAllWindows()
        
        else:
            return jsonify({'Error: Invalid file type'}), 400

    except Exception as error:
        app.logger.error(f'Error: {str(error)}')
        return jsonify({'File Prediction Error': str(error)}), 500

@app.route('/Predictions<path:filename>')
def predict_file(filename):
    
    try:
        ext = filename.split('.')[-1]
        #Determing the file is Image or Video
        if ext in ['mp4']:
            folder = app.config['VIDEO_SAVE_FOLDER']
            
            print('File Name:', filename)

            clean_filename = sanitize_name(urllib.parse.unquote(filename))
            print('Clean File Name:', clean_filename)

            file_path = os.path.join(folder, clean_filename)
            print('File Path:', file_path)
            
            if not file_path:
                return jsonify({'Error: File not Found'}), 404
            
            file_size = os.path.getsize(file_path)
            app.logger.info(f'Predicting Video File:{clean_filename}, Size: {file_size} bytes')

            #Setting the mimetype
            mime_types = {'.mp4': 'video/mp4'}
            mime_type = mime_types.get(ext, 'video/mp4')

            response = send_from_directory(folder, clean_filename, mimetype=mime_type, as_attachment=False, conditional=True)

            #Setting additional headers
            response.headers['Accept-Ranges'] = 'bytes'
            response.headers['Cache-Control'] = 'no-cache'

            return response
 
        elif ext in ['jpg']:
            folder = app.config['IMAGE_SAVE_FOLDER']
            return send_from_directory(folder, filename)
        
        else:
            app.logger.error('Invalid File Type')
            return jsonify(['Error: Invalid File Type']), 500
        
    except Exception as error:
        app.logger.error(f'Error while reading file path of Predicted File: {str(error)}')
        return jsonify({'Error': str(error)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))