import argparse
import base64
from datetime import datetime
import os
import shutil
import sys

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from model import *

sio = socketio.Server()
app = Flask(__name__)
speed_limit = 5
c = 0
image1 = None
image2 = None
image3 = None
steering_angle = 0
steering_angle_prev = 0
throttle = 0
throttle_prev = 0
speed = 0
#prev_image_array = None

transformations = transforms.Compose([transforms.Lambda(lambda x: (x / 255.0))])

@sio.on('telemetry')
def telemetry(sid, data):
    global speed_limit,c,image1,image2,image3,steering_angle,throttle,steering_angle_prev,throttle_prev,speed
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        image = image_array[65:-25, :, :]
        image = transformations(image)
        image = torch.Tensor(image)
        #print(image.shape)
        image = image.view(1, 3, 70, 320)
        image = Variable(image)
        #print(image.shape)
        if c == 0:
            image1 = image
            c = 1
            send_control(0,0)
        elif c == 1:
            image2 = image
            c = 2
            send_control(0,0)
        else:
            c=0
            image3 = image
            image_stack = torch.cat((image1,image2,image3),axis=0)
            image_stack = torch.unsqueeze(image_stack,0)
            steering_angle = float(model(image_stack).view(-1).data.numpy()[0])
            steering_angle_prev = steering_angle
            throttle = 1-(speed/speed_limit)
            throttle_prev = throttle
            print(steering_angle, throttle)
            send_control(steering_angle, throttle)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':
    ### Testing phase ###
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()
    model = Combine()


    try:
        checkpoint = torch.load(
            args.model, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])

    except KeyError:
        checkpoint = torch.load(
            args.model, map_location=lambda storage, loc: storage)
        model = checkpoint['model']

    except RuntimeError:
        print("==> Please check using the same model as the checkpoint")
        sys.exit()

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
