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
import pygame
import math


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
        self.sound_enabled = True
        self.fullscreen = True  # Start in fullscreen mode

        # Motion tracking
        self.last_hand_positions = {}  # Track last position per hand for speed calculation
        self.hand_speeds = {}  # Current speed per hand

        # Flash effect
        self.flash_intensity = 0.0  # Current flash brightness (0-1)

        # Initialize audio
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            self.sound = self.generate_fire_sound()
            self.sound_channel = None
        except Exception as e:
            print(f"Warning: Could not initialize audio: {e}")
            self.sound_enabled = False

    def generate_fire_sound(self):
        """Generate a fire/whoosh sound effect dynamically."""
        sample_rate = 22050
        duration = 0.3
        num_samples = int(duration * sample_rate)

        # Generate pink noise
        noise = np.random.uniform(-0.5, 0.5, num_samples)

        # Create envelope (quick attack, slow decay)
        attack_samples = int(0.05 * sample_rate)
        decay_samples = num_samples - attack_samples

        envelope = np.zeros(num_samples)
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        envelope[attack_samples:] = np.linspace(1, 0, decay_samples) ** 2

        # Apply envelope
        audio = noise * envelope

        # Normalize and convert to 16-bit
        audio = audio / np.max(np.abs(audio))
        audio_int = (audio * 32767).astype(np.int16)

        # Create pygame Sound from numpy array
        sound = pygame.sndarray.make_sound(audio_int)
        return sound

    def generate_fire_color(self, brightness_multiplier=1.0):
        """
        Generate a fire-like color (red, orange, yellow).

        Args:
            brightness_multiplier: Multiplier for brightness (for flash effect)
        """
        # Random fire color
        choice = random.random()
        if choice < 0.4:
            # Red-orange
            base_color = (0, int(random.uniform(50, 150)), int(random.uniform(200, 255)))
        elif choice < 0.7:
            # Orange
            base_color = (0, int(random.uniform(100, 200)), int(random.uniform(200, 255)))
        else:
            # Yellow-white
            base_color = (int(random.uniform(100, 200)), int(random.uniform(200, 255)), int(random.uniform(240, 255)))

        # Apply brightness multiplier and clamp to 0-255
        return tuple(min(255, int(c * brightness_multiplier)) for c in base_color)

    def spawn_particles(self, x, y, count=None, brightness_multiplier=1.0):
        """
        Spawn particles at the given position with splash effect.

        Args:
            x, y: Coordinates where particles should spawn
            count: Number of particles to spawn (default: particles_per_frame)
            brightness_multiplier: Brightness multiplier for flash effect
        """
        if count is None:
            count = self.particles_per_frame

        for _ in range(count):
            # Random velocity for splash effect
            angle = random.uniform(0, 2 * np.pi)
            speed = random.uniform(2, 8) * (1 + (brightness_multiplier - 1) * 0.5)
            vx = np.cos(angle) * speed
            vy = np.sin(angle) * speed

            # Random fire color with brightness
            color = self.generate_fire_color(brightness_multiplier)

            # Random size (bigger for flash)
            size = random.uniform(3, 10) * (1 + (brightness_multiplier - 1) * 0.3)

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

    def calculate_hand_spread(self, landmarks, frame_shape):
        """
        Calculate how spread out the hand is (thumb to pinky distance).

        Args:
            landmarks: MediaPipe hand landmarks
            frame_shape: Shape of the frame (height, width, channels)

        Returns:
            Distance between thumb tip and pinky tip in pixels
        """
        h, w, _ = frame_shape

        # Thumb tip is landmark 4, Pinky tip is landmark 20
        thumb_tip = landmarks.landmark[4]
        pinky_tip = landmarks.landmark[20]

        # Convert to pixel coordinates
        thumb_x = thumb_tip.x * w
        thumb_y = thumb_tip.y * h
        pinky_x = pinky_tip.x * w
        pinky_y = pinky_tip.y * h

        # Calculate Euclidean distance
        distance = math.sqrt((thumb_x - pinky_x)**2 + (thumb_y - pinky_y)**2)
        return distance

    def calculate_speed(self, current_pos, hand_id):
        """
        Calculate movement speed of the hand.

        Args:
            current_pos: Current (x, y) position
            hand_id: Hand identifier

        Returns:
            Speed in pixels per frame
        """
        if hand_id in self.last_hand_positions:
            last_pos = self.last_hand_positions[hand_id]
            distance = math.sqrt((current_pos[0] - last_pos[0])**2 +
                               (current_pos[1] - last_pos[1])**2)
            return distance
        return 0

    def play_sound_with_speed(self, speed):
        """
        Play drawing sound with volume based on speed.

        Args:
            speed: Movement speed in pixels per frame
        """
        if not self.sound_enabled:
            return

        # Map speed to volume (0.0 - 1.0)
        # Speed range roughly 0-100 pixels per frame
        volume = min(1.0, speed / 50.0)

        # Only play if volume is significant
        if volume > 0.1:
            # Stop previous sound if playing
            if self.sound_channel and self.sound_channel.get_busy():
                pass  # Let it continue for overlap effect
            else:
                # Play new sound
                self.sound.set_volume(volume)
                self.sound_channel = self.sound.play()

    def draw_ui(self, frame):
        """Draw UI elements on the frame."""
        h, w = frame.shape[:2]

        # Status text
        status = "Drawing: ON" if self.drawing_enabled else "Drawing: OFF"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 0) if self.drawing_enabled else (0, 0, 255), 2)

        # Sound status
        sound_status = f"Sound: {'ON' if self.sound_enabled else 'OFF'}"
        cv2.putText(frame, sound_status, (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 255) if self.sound_enabled else (128, 128, 128), 2)

        # Particle count
        particle_count = f"Particles: {len(self.particles)}"
        cv2.putText(frame, particle_count, (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2)

        # Flash indicator
        if self.flash_intensity > 0.1:
            flash_text = "FLASH!"
            cv2.putText(frame, flash_text, (w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, (255, 255, 255), 3)

        # Instructions
        instructions = [
            "SPACE - Toggle drawing",
            "S - Toggle sound",
            "F - Toggle fullscreen",
            "C - Clear particles",
            "Spread hand wide for FLASH!",
            "Q - Quit"
        ]

        y_offset = h - 165
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
        print("- Move faster for louder sound effects")
        print("- Spread your hand wide for a bright FLASH!")
        print("- SPACE: Toggle drawing on/off")
        print("- S: Toggle sound on/off")
        print("- F: Toggle fullscreen on/off")
        print("- C: Clear all particles")
        print("- Q: Quit")

        # Get frame dimensions
        ret, temp_frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            return
        cam_h, cam_w, _ = temp_frame.shape

        # Create window and set to fullscreen
        window_name = 'Hand Motion Fire Drawing'
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)

        # Get screen dimensions
        if self.fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            # Get actual screen size after fullscreen is set
            screen_info = cv2.getWindowImageRect(window_name)
            if screen_info[2] > 0 and screen_info[3] > 0:
                screen_w, screen_h = screen_info[2], screen_info[3]
            else:
                # Fallback to common screen size
                screen_w, screen_h = 1920, 1080
        else:
            screen_w, screen_h = cam_w, cam_h
            cv2.resizeWindow(window_name, screen_w, screen_h)

        print(f"Display resolution: {screen_w}x{screen_h}")
        print(f"Camera resolution: {cam_w}x{cam_h}")

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

            # Create black canvas for particles (use screen dimensions)
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

            # Decay flash intensity
            self.flash_intensity *= 0.8

            # Track hands and spawn particles
            max_speed = 0  # Track maximum speed across all hands
            if results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Get index finger tip position (in camera coordinates)
                    cam_x, cam_y = self.process_hand_landmarks(hand_landmarks, frame.shape)

                    # Scale to screen coordinates
                    x = int(cam_x * screen_w / cam_w)
                    y = int(cam_y * screen_h / cam_h)

                    # Calculate movement speed
                    speed = self.calculate_speed((x, y), hand_idx)
                    max_speed = max(max_speed, speed)

                    # Update last position
                    self.last_hand_positions[hand_idx] = (x, y)

                    # Check hand spread for flash effect
                    hand_spread = self.calculate_hand_spread(hand_landmarks, frame.shape)

                    # Flash threshold (typical spread is 100-200, wide spread is 200+)
                    flash_threshold = 180
                    is_flash = hand_spread > flash_threshold

                    if is_flash:
                        # Trigger flash effect
                        self.flash_intensity = 1.0

                    # Spawn particles if drawing is enabled
                    if self.drawing_enabled:
                        if is_flash:
                            # Flash mode: more particles, brighter, bigger
                            brightness = 1.5 + self.flash_intensity * 0.5
                            particle_count = int(self.particles_per_frame * 3)
                            self.spawn_particles(x, y, count=particle_count,
                                               brightness_multiplier=brightness)
                        else:
                            # Normal mode
                            self.spawn_particles(x, y)

            # Play sound based on maximum speed
            if max_speed > 0:
                self.play_sound_with_speed(max_speed)

            # Apply flash effect to canvas if active
            if self.flash_intensity > 0.1:
                # Add white overlay for flash
                flash_overlay = np.ones((screen_h, screen_w, 3), dtype=np.uint8) * int(50 * self.flash_intensity)
                canvas = cv2.add(canvas, flash_overlay)

            # Update and draw all particles on the black canvas
            self.update_and_draw_particles(canvas)

            # Draw UI on the canvas
            self.draw_ui(canvas)

            # Display
            cv2.imshow(window_name, canvas)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                self.drawing_enabled = not self.drawing_enabled
                print(f"Drawing {'enabled' if self.drawing_enabled else 'disabled'}")
            elif key == ord('s') or key == ord('S'):
                self.sound_enabled = not self.sound_enabled
                print(f"Sound {'enabled' if self.sound_enabled else 'disabled'}")
            elif key == ord('f') or key == ord('F'):
                self.fullscreen = not self.fullscreen
                if self.fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    # Update screen dimensions
                    screen_info = cv2.getWindowImageRect(window_name)
                    if screen_info[2] > 0 and screen_info[3] > 0:
                        screen_w, screen_h = screen_info[2], screen_info[3]
                    else:
                        screen_w, screen_h = 1920, 1080
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    screen_w, screen_h = cam_w, cam_h
                    cv2.resizeWindow(window_name, screen_w, screen_h)
                print(f"Fullscreen {'enabled' if self.fullscreen else 'disabled'} - Resolution: {screen_w}x{screen_h}")
            elif key == ord('c'):
                self.particles.clear()
                print("Particles cleared")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        if self.sound_enabled:
            pygame.mixer.quit()


def main():
    """Entry point for the application."""
    app = HandDrawing(fade_duration=2.0, particles_per_frame=15)
    app.run()


if __name__ == "__main__":
    main()
