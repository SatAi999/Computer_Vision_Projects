import cv2
import numpy as np
import time
import math
from ultralytics import YOLO
import easyocr
from collections import defaultdict, deque
import argparse

class VehicleSpeedPlateDetector:
    def __init__(self, video_path, output_path=None):
        self.video_path = video_path
        self.output_path = output_path
        
        # Load YOLO models
        print("Loading YOLO models...")
        self.vehicle_model = YOLO('yolov8n.pt')  # Pre-trained for vehicles
        # For plate detection, you can use a custom trained model or the general model
        self.plate_model = YOLO('yolov8n.pt')  # Replace with custom plate model if available
        
        # Initialize OCR reader
        print("Initializing OCR reader...")
        self.reader = easyocr.Reader(['en'])
        
        # Tracking variables
        self.vehicle_tracks = {}
        self.next_track_id = 0
        self.max_disappeared = 30
        self.max_distance = 100
        
        # Speed calculation parameters
        self.fps = 30
        self.pixels_per_meter = 10  # Calibration parameter - adjust based on your video
        self.coordinates_buffer = defaultdict(lambda: deque(maxlen=30))  # Store last 1 second of coordinates
        
        # Video properties
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer for output
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        else:
            self.out = None
            
        print(f"Video FPS: {self.fps}, Resolution: {self.width}x{self.height}")

    def detect_vehicles(self, frame):
        """Detect vehicles in the frame"""
        results = self.vehicle_model(frame, classes=[2, 3, 5, 7])  # car, motorcycle, bus, truck
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = box.cls[0].cpu().numpy()
                    
                    if conf > 0.5:  # Confidence threshold
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'class': int(cls)
                        })
        
        return detections

    def detect_license_plates(self, frame, vehicle_bbox):
        """Detect license plates within vehicle bounding box"""
        x1, y1, x2, y2 = vehicle_bbox
        # Expand the bounding box slightly for better plate detection
        margin = 20
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame.shape[1], x2 + margin)
        y2 = min(frame.shape[0], y2 + margin)
        
        vehicle_crop = frame[y1:y2, x1:x2]
        
        # Use YOLO to detect objects that might be license plates
        results = self.plate_model(vehicle_crop, classes=[0])  # Detect all objects, filter later
        
        plates = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    px1, py1, px2, py2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # Convert coordinates back to original frame
                    abs_x1 = int(px1 + x1)
                    abs_y1 = int(py1 + y1)
                    abs_x2 = int(px2 + x1)
                    abs_y2 = int(py2 + y1)
                    
                    # Filter based on aspect ratio (license plates are typically rectangular)
                    width = abs_x2 - abs_x1
                    height = abs_y2 - abs_y1
                    aspect_ratio = width / height if height > 0 else 0
                    
                    if 2 < aspect_ratio < 6 and width > 50 and height > 15:  # Typical plate dimensions
                        plates.append({
                            'bbox': [abs_x1, abs_y1, abs_x2, abs_y2],
                            'confidence': float(conf)
                        })
        
        return plates

    def read_license_plate(self, frame, plate_bbox):
        """Extract text from license plate using OCR"""
        x1, y1, x2, y2 = plate_bbox
        plate_crop = frame[y1:y2, x1:x2]
        
        # Preprocessing for better OCR
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        try:
            results = self.reader.readtext(thresh)
            if results:
                # Get the text with highest confidence
                best_result = max(results, key=lambda x: x[2])
                text = best_result[1].strip()
                confidence = best_result[2]
                
                # Basic filtering for license plate format
                if len(text) >= 4 and confidence > 0.3:
                    # Remove spaces and special characters, keep alphanumeric
                    text = ''.join(c for c in text if c.isalnum())
                    return text
        except:
            pass
        
        return ""

    def calculate_centroid(self, bbox):
        """Calculate centroid of bounding box"""
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) // 2, (y1 + y2) // 2]

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def track_vehicles(self, detections):
        """Simple centroid-based tracking"""
        if not detections:
            return {}

        # If no existing tracks, create new ones
        if not self.vehicle_tracks:
            for detection in detections:
                centroid = self.calculate_centroid(detection['bbox'])
                self.vehicle_tracks[self.next_track_id] = {
                    'centroid': centroid,
                    'bbox': detection['bbox'],
                    'disappeared': 0,
                    'detection': detection
                }
                self.next_track_id += 1
            return self.vehicle_tracks

        # Calculate centroids for new detections
        input_centroids = [self.calculate_centroid(det['bbox']) for det in detections]

        # If we have existing tracks but no detections, mark as disappeared
        if not input_centroids:
            for track_id in list(self.vehicle_tracks.keys()):
                self.vehicle_tracks[track_id]['disappeared'] += 1
                if self.vehicle_tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.vehicle_tracks[track_id]
            return self.vehicle_tracks

        # Match existing tracks with new detections
        track_ids = list(self.vehicle_tracks.keys())
        track_centroids = [self.vehicle_tracks[tid]['centroid'] for tid in track_ids]

        # Compute distance matrix
        distances = np.zeros((len(track_centroids), len(input_centroids)))
        for i, track_centroid in enumerate(track_centroids):
            for j, input_centroid in enumerate(input_centroids):
                distances[i][j] = self.calculate_distance(track_centroid, input_centroid)

        # Find minimum distances for assignment
        used_detection_indices = set()
        used_track_indices = set()

        for i in range(len(track_centroids)):
            if i in used_track_indices:
                continue
            
            min_dist = float('inf')
            min_j = -1
            
            for j in range(len(input_centroids)):
                if j in used_detection_indices:
                    continue
                if distances[i][j] < min_dist:
                    min_dist = distances[i][j]
                    min_j = j
            
            if min_j != -1 and min_dist < self.max_distance:
                track_id = track_ids[i]
                self.vehicle_tracks[track_id]['centroid'] = input_centroids[min_j]
                self.vehicle_tracks[track_id]['bbox'] = detections[min_j]['bbox']
                self.vehicle_tracks[track_id]['disappeared'] = 0
                self.vehicle_tracks[track_id]['detection'] = detections[min_j]
                used_detection_indices.add(min_j)
                used_track_indices.add(i)

        # Handle unmatched detections (create new tracks)
        for j in range(len(input_centroids)):
            if j not in used_detection_indices:
                self.vehicle_tracks[self.next_track_id] = {
                    'centroid': input_centroids[j],
                    'bbox': detections[j]['bbox'],
                    'disappeared': 0,
                    'detection': detections[j]
                }
                self.next_track_id += 1

        # Handle unmatched tracks (mark as disappeared)
        for i in range(len(track_centroids)):
            if i not in used_track_indices:
                track_id = track_ids[i]
                self.vehicle_tracks[track_id]['disappeared'] += 1
                if self.vehicle_tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.vehicle_tracks[track_id]

        return self.vehicle_tracks

    def calculate_speed(self, track_id, current_centroid):
        """Calculate speed based on centroid movement"""
        self.coordinates_buffer[track_id].append(current_centroid)
        
        if len(self.coordinates_buffer[track_id]) < 2:
            return 0.0
        
        # Calculate speed over the last second
        if len(self.coordinates_buffer[track_id]) >= self.fps // 2:  # At least half second of data
            start_point = self.coordinates_buffer[track_id][0]
            end_point = self.coordinates_buffer[track_id][-1]
            
            # Calculate distance in pixels
            pixel_distance = self.calculate_distance(start_point, end_point)
            
            # Convert to meters
            distance_meters = pixel_distance / self.pixels_per_meter
            
            # Calculate time
            time_seconds = len(self.coordinates_buffer[track_id]) / self.fps
            
            # Calculate speed in m/s and convert to km/h
            if time_seconds > 0:
                speed_ms = distance_meters / time_seconds
                speed_kmh = speed_ms * 3.6
                return speed_kmh
        
        return 0.0

    def process_video(self):
        """Process the entire video"""
        frame_count = 0
        
        print("Processing video...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect vehicles
            vehicle_detections = self.detect_vehicles(frame)
            
            # Track vehicles
            tracked_vehicles = self.track_vehicles(vehicle_detections)
            
            # Process each tracked vehicle
            for track_id, track_info in tracked_vehicles.items():
                centroid = track_info['centroid']
                bbox = track_info['bbox']
                
                # Calculate speed
                speed = self.calculate_speed(track_id, centroid)
                
                # Detect license plates in vehicle region
                plates = self.detect_license_plates(frame, bbox)
                
                plate_text = ""
                if plates:
                    # Use the first detected plate
                    plate_bbox = plates[0]['bbox']
                    plate_text = self.read_license_plate(frame, plate_bbox)
                    
                    # Draw plate bounding box
                    cv2.rectangle(frame, (plate_bbox[0], plate_bbox[1]), 
                                (plate_bbox[2], plate_bbox[3]), (0, 255, 255), 2)
                
                # Draw vehicle bounding box
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # Draw centroid
                cv2.circle(frame, tuple(centroid), 5, (255, 0, 0), -1)
                
                # Display information
                info_text = f"ID: {track_id}"
                if speed > 0:
                    info_text += f" Speed: {speed:.1f} km/h"
                if plate_text:
                    info_text += f" Plate: {plate_text}"
                
                cv2.putText(frame, info_text, (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Speed limit warning (example: 60 km/h)
                if speed > 60:
                    cv2.putText(frame, "OVER SPEED!", (bbox[0], bbox[1] - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display frame info
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Vehicle Speed and Plate Detection', frame)
            
            # Write frame to output video
            if self.out:
                self.out.write(frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
        
        # Cleanup
        self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()
        
        print(f"Processing complete! Total frames: {frame_count}")

def main():
    parser = argparse.ArgumentParser(description='Vehicle Speed and License Plate Detection')
    parser.add_argument('--input', '-i', required=True, help='Input video path')
    parser.add_argument('--output', '-o', help='Output video path (optional)')
    parser.add_argument('--ppm', type=float, default=10, help='Pixels per meter calibration')
    
    args = parser.parse_args()
    
    # Create detector instance
    detector = VehicleSpeedPlateDetector(args.input, args.output)
    detector.pixels_per_meter = args.ppm
    
    # Process video
    detector.process_video()

if __name__ == "__main__":
    main()
