import cv2
import numpy as np
from collections import deque
import time

class StaticHoopTracker:
    def __init__(self):
        # Color ranges for basketball detection
        self.orange_ranges = [
            (np.array([5, 100, 100]), np.array([15, 255, 255])),
            (np.array([10, 50, 50]), np.array([20, 255, 255])),
            (np.array([0, 50, 100]), np.array([10, 255, 255]))
        ]
        
        # Basketball tracking
        self.ball_positions = deque(maxlen=30)
        self.ball_sizes = deque(maxlen=30)
        
        # Initialize
        self.hoop_center = None
        self.hoop_radius = 40  # Default radius
        self.hoop_detected = False
        
        # Shot detection
        self.shot_count = 0
        self.last_shot_time = 0
        self.shot_cooldown = 1.0
        
        # Depth calibration
        self.calibration_mode = False
        self.near_ball_size = None  # Ball size when close
        self.far_ball_size = None   # Ball size when far
        self.near_distance = 100    # cm
        self.far_distance = 300     # cm
        
        # Shot detection zones
        self.approach_zone = 150    # pixels around hoop (increased)
        self.scoring_zone = 80      # pixels for made shot (increased)
        
        # Visual settings
        self.show_hoop_guide = True
        
        # Trajectory tracking
        self.trajectory = deque(maxlen=60)
        self.shot_in_progress = False
        
    def detect_basketball(self, frame, hsv):
        """Detect basketball with size tracking for depth"""
        h, w = frame.shape[:2]
        
        # Create combined mask
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        for lower, upper in self.orange_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Morphological operations
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None, combined_mask
        
        # Find the largest circular contour
        best_ball = None
        best_circularity = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200:  # Minimum area
                continue
            
            # Get enclosing circle
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > 0.7 and circularity > best_circularity:
                    best_circularity = circularity
                    best_ball = ((int(x), int(y)), int(radius))
        
        if best_ball:
            return best_ball[0], best_ball[1], combined_mask
        
        return None, None, combined_mask
    
    def estimate_distance(self, ball_radius):
        """Estimate distance based on ball size"""
        if not self.near_ball_size or not self.far_ball_size:
            return None
        
        # Linear interpolation between near and far
        size_range = self.near_ball_size - self.far_ball_size
        if size_range <= 0:
            return None
        
        # Calculate distance
        size_ratio = (ball_radius - self.far_ball_size) / size_range
        size_ratio = np.clip(size_ratio, 0, 1)
        
        distance = self.far_distance - (self.far_distance - self.near_distance) * size_ratio
        return distance
    
    def check_shot(self, ball_pos, ball_radius):
        """Simple shot detection for static hoop"""
        if not self.hoop_detected or not ball_pos:
            return False
        
        current_time = time.time()
        if current_time - self.last_shot_time < self.shot_cooldown:
            return False
        
        # Add to trajectory
        self.trajectory.append((ball_pos, ball_radius, current_time))
        
        if len(self.trajectory) < 20:
            return False
        
        # Calculate distance to hoop
        hoop_x, hoop_y = self.hoop_center
        ball_x, ball_y = ball_pos
        
        # Avoid overflow by checking values first
        if abs(ball_x - hoop_x) > 1000 or abs(ball_y - hoop_y) > 1000:
            return False
        
        distance = np.sqrt((ball_x - hoop_x)**2 + (ball_y - hoop_y)**2)
        
        # Check if ball passed through hoop area
        if distance < self.scoring_zone:
            # Analyze trajectory - did ball come from above?
            positions = [t[0] for t in self.trajectory[-20:]]
            y_values = [p[1] for p in positions]
            
            # Find peak (minimum y)
            min_y = min(y_values)
            min_idx = y_values.index(min_y)
            
            # Check if trajectory shows downward motion through hoop
            if min_idx < len(y_values) - 5:  # Peak not at the end
                if min_y < hoop_y - 50:  # Came from above
                    if y_values[-1] > hoop_y:  # Now below hoop
                        self.shot_count += 1
                        self.last_shot_time = current_time
                        self.trajectory.clear()
                        return True
        
        return False
    
    def draw_interface(self, frame, ball_pos, ball_radius, mask):
        """Draw UI with depth visualization"""
        h, w = frame.shape[:2]
        
        # Draw calibration instructions
        if self.calibration_mode:
            cv2.putText(frame, "CALIBRATION MODE", (w//2 - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, "1: Set NEAR position | 2: Set FAR position | C: Complete", 
                       (w//2 - 250, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            if ball_radius:
                cv2.putText(frame, f"Current ball size: {ball_radius}", (w//2 - 100, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Draw shot counter and missed shots
        cv2.putText(frame, f"Shots Made: {self.shot_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Shots Missed: 0", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw basketball and depth
        if ball_pos and ball_radius:
            # Ball with color based on distance
            distance = self.estimate_distance(ball_radius)
            if distance:
                # Color gradient: green (close) to red (far)
                color_ratio = (distance - self.near_distance) / (self.far_distance - self.near_distance)
                color_ratio = np.clip(color_ratio, 0, 1)
                color = (0, int(255 * (1 - color_ratio)), int(255 * color_ratio))
            else:
                color = (0, 255, 0)
            
            cv2.circle(frame, ball_pos, ball_radius, color, 2)
            cv2.circle(frame, ball_pos, 2, color, -1)
            
            # Distance text
            if distance:
                cv2.putText(frame, f"{distance:.0f}cm", 
                          (ball_pos[0] - 30, ball_pos[1] - ball_radius - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Trajectory trail
            self.ball_positions.append(ball_pos)
            self.ball_sizes.append(ball_radius)
            
            if len(self.ball_positions) > 1:
                for i in range(1, len(self.ball_positions)):
                    if self.ball_positions[i-1] and self.ball_positions[i]:
                        # Thickness based on ball size (depth)
                        thickness = max(1, int(self.ball_sizes[i] / 10))
                        cv2.line(frame, self.ball_positions[i-1], 
                               self.ball_positions[i], (0, 255, 255), thickness)
        
        # Draw hoop
        if self.hoop_detected and self.hoop_center:
            hx, hy = self.hoop_center
            
            # Draw crosshair at hoop center for visibility
            cv2.line(frame, (hx - 20, hy), (hx + 20, hy), (0, 255, 255), 2)
            cv2.line(frame, (hx, hy - 20), (hx, hy + 20), (0, 255, 255), 2)
            
            # Main hoop - thick and bright
            cv2.circle(frame, (hx, hy), self.hoop_radius, (0, 0, 255), 4)
            cv2.circle(frame, (hx, hy), self.hoop_radius - 5, (0, 255, 255), 3)
            cv2.circle(frame, (hx, hy), self.hoop_radius + 5, (255, 255, 255), 2)
            
            # Backboard
            board_width = int(self.hoop_radius * 2.5)
            board_height = int(self.hoop_radius * 1.2)
            cv2.rectangle(frame, 
                         (hx - board_width, hy - board_height),
                         (hx + board_width, hy),
                         (255, 255, 255), 3)
            
            # Inner square on backboard
            inner_width = int(self.hoop_radius * 0.8)
            inner_height = int(self.hoop_radius * 0.6)
            cv2.rectangle(frame,
                         (hx - inner_width, hy - inner_height),
                         (hx + inner_width, hy),
                         (0, 0, 0), 2)
            
            # Scoring zones with labels
            cv2.circle(frame, (hx, hy), self.scoring_zone, 
                      (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, "SCORE", (hx - 30, hy - self.scoring_zone - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.circle(frame, (hx, hy), self.approach_zone, 
                      (255, 255, 0), 2, cv2.LINE_AA)
            
            # Net visualization
            net_lines = 8
            for i in range(net_lines):
                y_offset = hy + (i + 1) * 8
                x_width = int(self.hoop_radius * (1 - i / (net_lines * 1.5)))
                if x_width > 5:
                    cv2.line(frame, 
                            (hx - x_width, y_offset),
                            (hx + x_width, y_offset),
                            (200, 200, 200), 1)
            
            # Hoop info text
            cv2.putText(frame, f"Hoop: ({hx}, {hy}) R:{self.hoop_radius}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Mask preview
        mask_small = cv2.resize(mask, (160, 120))
        mask_colored = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
        frame[h-130:h-10, w-170:w-10] = mask_colored
        
        # Instructions
        instructions = "H: Set Hoop | C: Calibrate Depth | R: Reset | SPACE: Pause | Q: Quit"
        cv2.putText(frame, instructions, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def process_frame(self, frame):
        """Process single frame"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect basketball
        ball_pos, ball_radius, mask = self.detect_basketball(frame, hsv)
        
        # Check for shot
        if ball_pos and ball_radius:
            if self.check_shot(ball_pos, ball_radius):
                print(f"SHOT MADE! Total: {self.shot_count}")
        
        # Draw interface
        output = self.draw_interface(frame, ball_pos, ball_radius, mask)
        
        return output, ball_radius

def main():
    # Initialize tracker
    tracker = StaticHoopTracker()
    
    # Video setup
    video_path = "input.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video loaded: {fps} FPS, {total_frames} total frames")
    
    # Window setup
    cv2.namedWindow('Static Hoop Tracker', cv2.WINDOW_NORMAL)
    
    print("\n=== STATIC HOOP BASKETBALL TRACKER ===")
    print("SETUP:")
    print("1. LEFT CLICK to set hoop center")
    print("2. RIGHT CLICK to adjust hoop size")
    print("3. Press 'C' to calibrate depth:")
    print("   - Stand close and press '1'")
    print("   - Stand far and press '2'")
    print("\nCONTROLS:")
    print("H - Re-select hoop | R - Reset counter")
    print("SPACE - Pause | Q - Quit")
    print("=====================================\n")
    
    paused = False
    frame_count = 0
    current_ball_radius = None
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            tracker.hoop_center = (x, y)
            tracker.hoop_detected = True
            print(f"Hoop set at: {x}, {y}")
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click to adjust hoop size
            if tracker.hoop_detected:
                dist = np.sqrt((x - tracker.hoop_center[0])**2 + (y - tracker.hoop_center[1])**2)
                tracker.hoop_radius = int(dist)
                print(f"Hoop radius set to: {tracker.hoop_radius}")
    
    cv2.setMouseCallback('Static Hoop Tracker', mouse_callback)
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break
            frame_count += 1
        
        # Process frame
        output, ball_radius = tracker.process_frame(frame)
        if ball_radius:
            current_ball_radius = ball_radius
        
        # Frame counter
        cv2.putText(output, f"Frame: {frame_count}/{total_frames}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow('Static Hoop Tracker', output)
        
        # Handle keys
        key = cv2.waitKey(30 if not paused else 0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('r'):
            tracker.shot_count = 0
            print("Counter reset")
        elif key == ord('h'):
            print("Click hoop location")
        elif key == ord('+') or key == ord('='):
            tracker.hoop_radius += 5
            print(f"Hoop radius increased to: {tracker.hoop_radius}")
        elif key == ord('-'):
            tracker.hoop_radius = max(20, tracker.hoop_radius - 5)
            print(f"Hoop radius decreased to: {tracker.hoop_radius}")
        elif key == ord('c'):
            tracker.calibration_mode = not tracker.calibration_mode
            print("Calibration mode:", "ON" if tracker.calibration_mode else "OFF")
        elif key == ord('1') and tracker.calibration_mode and current_ball_radius:
            tracker.near_ball_size = current_ball_radius
            print(f"Near position set: ball size = {current_ball_radius}")
        elif key == ord('2') and tracker.calibration_mode and current_ball_radius:
            tracker.far_ball_size = current_ball_radius
            print(f"Far position set: ball size = {current_ball_radius}")
            if tracker.near_ball_size and tracker.far_ball_size:
                tracker.calibration_mode = False
                print("Calibration complete!")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
