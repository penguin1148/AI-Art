#!/usr/bin/env python3
"""
Hand Motion Drawing Application
Tracks hand movements via webcam and draws colorful trails that fade over time.
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time


class HandDrawing:
    """Main application class for hand tracking and drawing."""

    def __init__(self, fade_duration=3.0, trail_thickness=10):
        """
        Initialize the hand drawing application.

        Args:
            fade_duration: Time in seconds before trails completely fade
            trail_thickness: Thickness of the drawing trails
        """
        # MediaPipe hand tracking setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Drawing parameters
        self.fade_duration = fade_duration
        self.trail_thickness = trail_thickness

        # Canvas and trail storage
        self.canvas = None
        self.trails = []  # List of trails: [(points, timestamp, color)]

        # Color cycling
        self.current_color_index = 0
        self.colors = self.generate_rainbow_colors(6)

        # State
        self.drawing_enabled = True
        self.last_position = {}  # Track last position per hand

    def generate_rainbow_colors(self, n=6):
        """Generate rainbow colors in BGR format for OpenCV."""
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            hsv = np.uint8([[[hue, 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            colors.append(tuple(map(int, bgr[0][0])))
        return colors

    def get_next_color(self):
        """Get the next color in the rainbow cycle."""
        color = self.colors[self.current_color_index]
        self.current_color_index = (self.current_color_index + 1) % len(self.colors)
        return color

    def add_trail_point(self, x, y, hand_id=0):
        """
        Add a point to the drawing trail.

        Args:
            x, y: Coordinates of the point
            hand_id: Identifier for which hand (0 or 1)
        """
        current_time = time.time()

        # Get or create trail for this hand
        if hand_id not in self.last_position:
            # New trail for this hand
            color = self.get_next_color()
            self.trails.append({
                'points': deque(maxlen=1000),
                'timestamps': deque(maxlen=1000),
                'color': color,
                'hand_id': hand_id
            })
            self.last_position[hand_id] = len(self.trails) - 1

        trail_idx = self.last_position[hand_id]
        trail = self.trails[trail_idx]

        # Add point to trail
        trail['points'].append((x, y))
        trail['timestamps'].append(current_time)

    def draw_trails(self, frame):
        """
        Draw all trails on the frame with fading effect.

        Args:
            frame: The video frame to draw on
        """
        current_time = time.time()
        trails_to_remove = []

        for idx, trail in enumerate(self.trails):
            points = list(trail['points'])
            timestamps = list(trail['timestamps'])
            color = trail['color']

            if len(points) < 2:
                continue

            # Draw trail segments with fading
            for i in range(1, len(points)):
                age = current_time - timestamps[i]

                # Skip if too old
                if age > self.fade_duration:
                    continue

                # Calculate alpha (transparency) based on age
                alpha = 1.0 - (age / self.fade_duration)
                alpha = max(0.0, min(1.0, alpha))

                # Draw line with fading effect
                if alpha > 0:
                    # Scale color by alpha
                    faded_color = tuple(int(c * alpha) for c in color)
                    thickness = max(1, int(self.trail_thickness * alpha))

                    cv2.line(frame, points[i-1], points[i], faded_color, thickness)

            # Mark trail for removal if all points are too old
            if timestamps and current_time - timestamps[-1] > self.fade_duration:
                trails_to_remove.append(idx)

        # Remove old trails
        for idx in reversed(trails_to_remove):
            trail = self.trails[idx]
            if trail['hand_id'] in self.last_position:
                del self.last_position[trail['hand_id']]
            del self.trails[idx]

    def clean_old_points(self):
        """Remove points that are older than fade_duration."""
        current_time = time.time()

        for trail in self.trails:
            while trail['timestamps'] and current_time - trail['timestamps'][0] > self.fade_duration:
                trail['points'].popleft()
                trail['timestamps'].popleft()

    def process_hand_landmarks(self, landmarks, frame_shape):
        """
        Extract index finger tip position from hand landmarks.

        Args:
            landmarks: MediaPipe hand landmarks
            frame_shape: Shape of the frame (height, width, channels)

        Returns:
            Tuple of (x, y) coordinates
        """
        h, w, _ = frame_shape
        # Index finger tip is landmark 8
        index_tip = landmarks.landmark[8]
        x = int(index_tip.x * w)
        y = int(index_tip.y * h)
        return x, y

    def draw_ui(self, frame):
        """Draw UI elements on the frame."""
        h, w = frame.shape[:2]

        # Status text
        status = "Drawing: ON" if self.drawing_enabled else "Drawing: OFF"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 0) if self.drawing_enabled else (0, 0, 255), 2)

        # Instructions
        instructions = [
            "SPACE - Toggle drawing",
            "C - Clear canvas",
            "Q - Quit"
        ]

        y_offset = h - 90
        for instruction in instructions:
            cv2.putText(frame, instruction, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25

    def run(self):
        """Main application loop."""
        # Initialize video capture
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        print("Hand Motion Drawing Started!")
        print("- Point with your index finger to draw")
        print("- SPACE: Toggle drawing on/off")
        print("- C: Clear canvas")
        print("- Q: Quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Initialize canvas if needed
            if self.canvas is None:
                self.canvas = np.zeros_like(frame)

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process hands
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Draw hand skeleton
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                    # Get index finger tip position
                    x, y = self.process_hand_landmarks(hand_landmarks, frame.shape)

                    # Draw a marker at finger tip
                    cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)

                    # Add to trail if drawing is enabled
                    if self.drawing_enabled:
                        self.add_trail_point(x, y, hand_idx)

            # Clean old points periodically
            self.clean_old_points()

            # Draw trails with fading effect
            self.draw_trails(frame)

            # Draw UI
            self.draw_ui(frame)

            # Display
            cv2.imshow('Hand Motion Drawing', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                self.drawing_enabled = not self.drawing_enabled
                print(f"Drawing {'enabled' if self.drawing_enabled else 'disabled'}")
            elif key == ord('c'):
                self.trails.clear()
                self.last_position.clear()
                print("Canvas cleared")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Entry point for the application."""
    app = HandDrawing(fade_duration=3.0, trail_thickness=10)
    app.run()


if __name__ == "__main__":
    main()
