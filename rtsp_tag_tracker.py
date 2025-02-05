import cv2
import numpy as np
#from apriltag import apriltag # doesn't work for reasons? wrong pip package?
from dt_apriltags import Detector
import sys
from urllib.parse import urlparse


def main(rtsp_url, tag_family, missing_threshold=5):
    # Initialize video capture
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Could not open RTSP stream")
        sys.exit(1)

    # Initialize AprilTag detector
    #detector = apriltag(tag_family)
    detector = Detector(
        families=tag_family,
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0)

    # Map of tag_id to frames-since-last-seen count
    tag_tracker = {}

    print("Reporting visible AprilTags. Press Ctrl+C to exit.")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect AprilTags
            results = detector.detect(gray)
            detected_tags = {r.tag_id for r in results}

            # Report newly detected tags and reset/add their counters
            for tag_id in detected_tags:
                if tag_id not in tag_tracker:
                    print(f"Found AprilTag ID: {tag_id}")
                tag_tracker[tag_id] = 0

            # Update all tag counters and remove stale ones
            for tag_id in list(tag_tracker.keys()):
                if tag_id not in detected_tags:
                    tag_tracker[tag_id] += 1
                    if tag_tracker[tag_id] >= missing_threshold:
                        print(f"AprilTag ID {tag_id} is no longer visible")
                        del tag_tracker[tag_id]

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        cap.release()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Monitor RTSP stream for AprilTags')
    parser.add_argument('rtsp_url', help='RTSP stream URL')
    parser.add_argument('tag_family', default='tagStandard41h12', help='AprilTag family to monitor')
    parser.add_argument('--missing-threshold', type=int, default=5,
                        help='Number of frames before declaring a tag as missing (default: 5)')
    
    args = parser.parse_args()
    main(args.rtsp_url, args.tag_family, args.missing_threshold)
