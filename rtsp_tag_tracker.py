import cv2
import numpy as np
#from apriltags import apriltag
from pyapriltags import Detector
#from dt_apriltags import Detector
import os, sys, threading, queue

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
os.environ["QT_QPA_PLATFORM"] = "xcb"

# we need a class to read the video stream and provide only the latest frame, not all of them in series
class UnbufferedVideoCapture:
    def __init__(self, *args, **kwargs):
        self.cap = cv2.VideoCapture(*args, **kwargs)
        if not self.cap.isOpened():
            print("Error: Could not open RTSP stream")
            sys.exit(1)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

def main(rtsp_url, tag_family='tag36h11', missing_threshold=5):
    cap = UnbufferedVideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    frame = cap.read()
    #Map of tag_id to frames-since-last-seen count
    tag_tracker = {}
    detector = Detector(
        families=tag_family,
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0)

    while True:
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = detector.detect(gray)

        # Report newly detected tags and reset/add their counters
        for tag in results:
            if tag.tag_id not in tag_tracker:
                print(f"Found AprilTag ID: {tag.tag_id}")
            tag_tracker[tag.tag_id] = 0
            for idx in range(len(tag.corners)):
                cv2.line(frame, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))

            cv2.putText(frame, str(tag.tag_id),
                org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 0, 255))
        cv2.imshow('frame', frame)

        # Update all tag counters and remove stale ones
        for tag_id in list(tag_tracker.keys()):
            if tag_tracker[tag_id] != 0:
                tag_tracker[tag_id] += 1
                if tag_tracker[tag_id] >= missing_threshold:
                    print(f"AprilTag ID {tag_id} is no longer visible")
                    del tag_tracker[tag_id]
        frame = cap.read()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Monitor RTSP stream for AprilTags')
    parser.add_argument('rtsp_url', help='RTSP stream URL')
    parser.add_argument('--tag_family', default='tag36h11', help='AprilTag family to monitor')
    parser.add_argument('--missing-threshold', type=int, default=5,
                        help='Number of frames before declaring a tag as missing (default: 5)')

    args = parser.parse_args()
    main(args.rtsp_url, args.tag_family, args.missing_threshold)
