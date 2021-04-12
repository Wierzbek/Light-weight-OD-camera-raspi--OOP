# USAGE
# python pi_face_recognition.py --cascade haarcascade_frontalface_default.xml --encodings encodings.pickle

# import the necessary packages
from imutils.video import VideoStream
import face_recognition
import argparse
import pickle
import time
import cv2
import pantilthat
from simple_pid import PID

class VideoCamera(object):
    def __init__(self):
        # construct the argument parser and parse the arguments
        ap = argparse.ArgumentParser()
        ap.add_argument("-c", "--cascade", required=False, default="haarcascade_frontalface_default.xml",
                        help="path to where the face cascade resides")
        ap.add_argument("-e", "--encodings", required=False, default="encodings.pickle",
                        help="path to serialized db of facial encodings")
        args = vars(ap.parse_args())

        # load the known faces and embeddings along with OpenCV's Haar
        # cascade for face detection
        print("[INFO] loading encodings + face detector...")
        self.data = pickle.loads(open(args["encodings"], "rb").read())
        self.detector = cv2.CascadeClassifier(args["cascade"])

        #capturing video
        print("[INFO] starting video stream...")
        self.video = cv2.VideoCapture(0)
        #let the camera warm up
        time.sleep(2.0)

    def __del__(self):
        #releasing camera
        self.video.release()

    def PID_Controller(self, rects, name):
        pid = PID(0.2, 0.5, 1, setpoint=2)
        if True:
            for (x, y, w, h) in rects:
                # Value of the central x in the bounding box
                pan = x + w / 2
                # Pixels per one degree
                dg = 500 / 180
                panLeft = (pan / dg) - 90
                panRight = (pan - 250) // dg
                tmp = 0;
                if name != "Unknown":
                    if pan > 250:
                        outp = pid(panRight // 1)
                        pantilthat.pan(outp)
                    elif pan <= 250:
                        outp = pid(panLeft // 1)
                        pantilthat.pan(outp)
           	

    def get_frame(self):
        # grab the frame from the threaded video stream and resize it
        # to 500px (to speedup processing)
        #extracting frames
        ds_factor=0.6
        ret, frame = self.video.read()
        frame=cv2.resize(frame,None,fx=ds_factor,fy=ds_factor,
        interpolation=cv2.INTER_AREA)

        # convert the input frame from (1) BGR to grayscale (for face
        # detection) and (2) from BGR to RGB (for face recognition)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # detect faces in the grayscale frame
        rects = self.detector.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

        # OpenCV returns bounding box coordinates in (x, y, w, h) order
        # but we need them in (top, right, bottom, left) order, so we
        # need to do a bit of reordering
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

        # compute the facial embeddings for each face bounding box
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(self.data["encodings"],
                                                     encoding, tolerance=0.46)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = self.data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)

            self.PID_Controller(rects, name)

        # encode OpenCV raw frame to jpg and displaying it
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

