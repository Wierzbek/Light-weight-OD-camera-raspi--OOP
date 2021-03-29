# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2


# Load the Haarcascade and the HOG detector files
class loadFiles:
    def __init__(self):

        # Try to load the files, Throw an exception when file not loaded
        try:
            # construct the argument parser
            self.ap = argparse.ArgumentParser()
            # parse the Haarcascade frontal face detector as an argument
            self.ap.add_argument("-c", "--cascade", required=False, default="haarcascade_frontalface_default.xml",
                            help="Path to the face cascade directory")
            # parse the pickle file with the HOG encodings of the face
            self.ap.add_argument("-e", "--encodings", required=False, default="encodings.pickle",
                            help="Path to the serialized database file of facial encodings")
            args = vars(self.ap.parse_args())

            # load the pickle file with faces and embeddings
            print("[INFO] Loading the encodings")
            self.data = pickle.loads(open(args["encodings"], "rb").read())

            # open the OpenCV's Haar cascade for face detection
            print("[INFO] Loading the face detector...")
            self.detector = cv2.CascadeClassifier(args["cascade"])
        except Exception:
            print("Error: Could not load the haarcascade or the .pickle file!")
            quit()

    def returnEncodings(self):
        return self.data

    def returnHaarDetector(self):
        return self.detector

class videoStream:
    def __init__(self):
        # initialize the video stream from the RasPi camera module
        print("[INFO] starting video stream...")
        self.vs = VideoStream(usePiCamera=True).start()

        # let the camera module warm-up for 2 seconds
        time.sleep(2.0)

    def stopVideoStream(self):
        self.vs.stop()

    def getFrame(self):
        # grab the frame from the threaded video stream and resize it
        # to 500px (to speedup processing)
        self.frame = self.vs.read()
        self.frame = imutils.resize(self.frame, width=500)
        return self.frame

    def getGreyScaleFrame(self):
        # convert the input frame from BGR to grayscale (for face detection)
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        return gray

    def getRGBFrame(self):
        # convert the input frame from BGR to RGB (for face recognition)
        rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        return rgb

    def startFPScounter(self):
        # start the FPS counter
        self.fps = FPS().start()

    def updateFPScounter(self):
        # update the FPS counter
        self.fps.update()

    def stopFPScounter(self):
        self.fps.stop()
        print("[INFO] elasped time: {:.2f}".format(self.fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))

class faceDetector:
    def __init__(self, vidStream, files):
        self.frame = vidStream.getFrame()
        gray = vidStream.getGreyScaleFrame()
        rgb = vidStream.getRGBFrame()
        # detect faces in the grayscale frame
        rects = files.returnHaarDetector().detectMultiScale(gray, scaleFactor=1.1,
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
            matches = face_recognition.compare_faces(files.returnEncodings()["encodings"],
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
                    name = files.returnEncodings()["names"][i]
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
            cv2.rectangle(self.frame, (left, top), (right, bottom),
                          (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(self.frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)

    def detectFace(self):

        return self.frame

def main():
    # load the object detection files (haarcascade and HOG detector)
    files = loadFiles()
    # instantiate the video stream object
    vidStream = videoStream()

    # start the FPS counter
    vidStream.startFPScounter()

    # loop over frames from the video stream
    while True:

        frame = faceDetector(vidStream, files).detectFace()

        # display the image to our screen
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # update the FPS counter
        vidStream.updateFPScounter()

    # stop the timer and display FPS information
    vidStream.stopFPScounter()

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vidStream.stopVideoStream()

if __name__ == "__main__":
    main()