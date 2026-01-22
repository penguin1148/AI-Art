#!/usr/bin/env python3
"""
Hand Motion Drawing Application
Tracks hand movements via webcam and creates fire-like particle effects that fade over time.
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import time


class Particle:
    """Represents a single particle in the fire effect."""

    def __init__(self, x, y, vx, vy, color, size, lifetime):
        self.x = x
        self.y = y
        self.vx = vx  # Velocity x
        self.vy = vy  # Velocity y
        self.color = color
        self.size = size
        self.lifetime = lifetime
        self.birth_time = time.time()

    def update(self):
        """Update particle position and apply gravity/drag."""
        self.x += self.vx
        self.y += self.vy
        # Apply slight upward motion (fire rises)
        self.vy -= 0.3
        # Apply drag
        self.vx *= 0.98
        self.vy *= 0.98

    def get_age(self):
        """Get particle age in seconds."""
        return time.time() - self.birth_time

    def is_alive(self):
        """Check if particle is still alive."""
        return self.get_age() < self.lifetime

    def get_alpha(self):
        """Get current alpha based on age."""
        age = self.get_age()
        return max(0.0, 1.0 - (age / self.lifetime))


class HandDrawing:
    """Main application class for hand tracking and drawing."""

    def __init__(self, fade_duration=2.0, particles_per_frame=15):
        """
        Initialize the hand drawing application.

        Args:
            fade_duration: Time in seconds before particles completely fade
            particles_per_frame: Number of particles to spawn per frame
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

        # Particle parameters
        self.fade_duration = fade_duration
        self.particles_per_frame = particles_per_frame

        # Particle storage
        self.particles = []

        # State
        self.drawing_enabled = True

    def generate_fire_color(self):
        """Generate a fire-like color (red, orange, yellow)."""
        # Random fire color
        choice = random.random()
        if choice < 0.4:
            # Red-orange
            return (0, int(random.uniform(50, 150)), int(random.uniform(200, 255)))
        elif choice < 0.7:
            # Orange
            return (0, int(random.uniform(100, 200)), int(random.uniform(200, 255)))
        else:
            # Yellow-white
            return (int(random.uniform(100, 200)), int(random.uniform(200, 255)), int(random.uniform(240, 255)))

    def spawn_particles(self, x, y):
        """
        Spawn particles at the given position with splash effect.

        Args:
            x, y: Coordinates where particles should spawn
        """
        for _ in range(self.particles_per_frame):
            # Random velocity for splash effect
            angle = random.uniform(0, 2 * np.pi)
            speed = random.uniform(2, 8)
            vx = np.cos(angle) * speed
            vy = np.sin(angle) * speed

            # Random fire color
            color = self.generate_fire_color()

            # Random size
            size = random.uniform(3, 10)

            # Random lifetime variation
            lifetime = self.fade_duration * random.uniform(0.7, 1.3)

            particle = Particle(x, y, vx, vy, color, size, lifetime)
            self.particles.append(particle)

    def update_and_draw_particles(self, frame):
        """
        Update all particles and draw them with fading effect.

        Args:
            frame: The video frame to draw on
        """
        particles_to_keep = []

        for particle in self.particles:
            if particle.is_alive():
                # Update particle position
                particle.update()

                # Get alpha for fading
                alpha = particle.get_alpha()

                if alpha > 0:
                    # Calculate faded color
                    faded_color = tuple(int(c * alpha) for c in particle.color)

                    # Calculate size based on alpha (particles shrink as they fade)
                    current_size = max(1, int(particle.size * alpha))

                    # Draw particle as a circle
                    cv2.circle(frame,
                             (int(particle.x), int(particle.y)),
                             current_size,
                             faded_color,
                             -1)

                    # Add glow effect (larger, more transparent circle)
                    glow_size = current_size + 3
                    glow_alpha = alpha * 0.3
                    glow_color = tuple(int(c * glow_alpha) for c in particle.color)
                    cv2.circle(frame,
                             (int(particle.x), int(particle.y)),
                             glow_size,
                             glow_color,
                             -1)

                    particles_to_keep.append(particle)

        self.particles = particles_to_keep

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

        # Particle count
        particle_count = f"Particles: {len(self.particles)}"
        cv2.putText(frame, particle_count, (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2)

        # Instructions
        instructions = [
            "SPACE - Toggle drawing",
            "C - Clear particles",
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

        print("Hand Motion Fire Drawing Started!")
        print("- Point with your index finger to create fire particles")
        print("- SPACE: Toggle drawing on/off")
        print("- C: Clear all particles")
        print("- Q: Quit")

        # Get frame dimensions
        ret, temp_frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            return
        h, w, _ = temp_frame.shape

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process hands
            results = self.hands.process(rgb_frame)

            # Create black canvas for particles
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

            # Track hands and spawn particles
            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Get index finger tip position
                    x, y = self.process_hand_landmarks(hand_landmarks, frame.shape)

                    # Spawn particles if drawing is enabled
                    if self.drawing_enabled:
                        self.spawn_particles(x, y)

            # Update and draw all particles on the black canvas
            self.update_and_draw_particles(canvas)

            # Draw UI on the canvas
            self.draw_ui(canvas)

            # Display
            cv2.imshow('Hand Motion Fire Drawing', canvas)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                self.drawing_enabled = not self.drawing_enabled
                print(f"Drawing {'enabled' if self.drawing_enabled else 'disabled'}")
            elif key == ord('c'):
                self.particles.clear()
                print("Particles cleared")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Entry point for the application."""
    app = HandDrawing(fade_duration=2.0, particles_per_frame=15)
    app.run()


if __name__ == "__main__":
    main()
