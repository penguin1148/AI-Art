# AI-Art: Hand Motion Fire Drawing

Transform your hand movements into stunning fire particles! This interactive application uses your webcam to track hand motions and creates beautiful fire-like particle effects that splash and fade over time.

## Features

- **Real-time Hand Tracking**: Uses MediaPipe for accurate hand detection and tracking
- **Fire Particle Effects**: Creates dynamic particle splashes with fire colors (red, orange, yellow)
- **Physics-Based Motion**: Particles have velocity, gravity, and drag for realistic movement
- **Fading Effect**: Particles gradually fade and shrink over 2 seconds
- **Multi-hand Support**: Track up to 2 hands simultaneously
- **Invisible Hand**: Only the particle effects are visible - your hand stays hidden
- **Speed-Based Sound Effects**: Dynamic audio that increases in volume as you move faster
- **Flash Effect**: Spread your hand wide to trigger bright particle bursts with enhanced effects
- **Interactive Controls**: Toggle drawing, sound, clear particles, and more

## Demo

Point with your index finger to create explosive fire particle effects. Particles spray out in all directions, rise upward like flames, and gradually fade away, creating a mesmerizing visual experience on a black canvas.

**Interactive Features:**
- Move your hand **slowly** for quiet, gentle particles
- Move your hand **quickly** for loud whooshing sounds and intense effects
- **Spread your hand wide** (thumb to pinky) to trigger a FLASH - brighter, bigger, more particles!

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd AI-Art
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8 or higher
- Webcam
- Audio output (speakers/headphones for sound effects)
- Dependencies:
  - opencv-python
  - mediapipe
  - numpy
  - pygame

## Usage

Run the application:
```bash
python hand_drawing.py
```

### Controls

- **Point with your index finger** to create fire particles
- **Move faster** to increase sound volume
- **Spread your hand wide** (open palm) to trigger FLASH effect
- **SPACE** - Toggle drawing on/off
- **S** - Toggle sound effects on/off
- **F** - Toggle fullscreen mode on/off
- **C** - Clear all particles from the canvas
- **Q** - Quit the application

## How It Works

1. The application captures video from your webcam
2. MediaPipe detects and tracks your hand(s) in real-time
3. The position of your index finger tip is used as the particle spawn point
4. **Movement tracking**: The app calculates your hand movement speed between frames
5. **Sound feedback**: Faster movement = louder whoosh sound effects
6. **Hand spread detection**: Measures distance between thumb and pinky tips
7. As you move your hand, particles are spawned (15 per frame normally, 45 during flash)
8. Each particle has:
   - Random velocity for splash effect (360° spray pattern)
   - Random fire color (red, orange, or yellow)
   - Random size (3-10 pixels, larger during flash)
   - Upward motion (simulating rising flames)
   - Gravity and drag physics
9. **Flash effect**: When hand spread > 180 pixels:
   - 3x more particles spawn
   - 1.5-2x brighter colors
   - Larger particle sizes
   - White screen flash overlay
   - "FLASH!" indicator appears
10. Particles automatically fade and shrink over 2 seconds
11. Your hand is completely invisible - only particles are shown on a black background

## Customization

You can customize the application by modifying parameters in `hand_drawing.py`:

```python
app = HandDrawing(
    fade_duration=2.0,        # Time in seconds before particles fade
    particles_per_frame=15    # Number of particles spawned per frame
)
```

You can also modify particle physics in the `Particle.update()` method:
- Adjust upward velocity: Change `self.vy -= 0.3` for more/less rising motion
- Adjust drag: Change `0.98` multiplier for faster/slower deceleration
- Modify colors: Edit the `generate_fire_color()` method for different color schemes

## Technical Details

- **Hand Tracking**: Uses MediaPipe Hands solution for robust hand landmark detection (21 landmarks per hand)
- **Particle System**: Object-oriented design with Particle class for each fire particle
- **Physics Simulation**: Implements velocity, gravity, and drag for realistic particle motion
- **Fading Algorithm**: Time-based alpha blending for smooth fade effects with particle shrinking
- **Color System**: Randomized BGR colors in fire spectrum (red/orange/yellow) with brightness multiplier
- **Glow Effect**: Each particle has a larger, semi-transparent glow halo for visual enhancement
- **Performance**: Efficient particle lifecycle management - automatic cleanup of expired particles
- **Audio System**:
  - Pygame mixer for sound playback
  - Procedurally generated whoosh/fire sound using numpy
  - Volume mapped to movement speed (0-100 pixels/frame → 0-100% volume)
  - 0.3 second sound duration with envelope (attack/decay)
- **Motion Tracking**: Euclidean distance calculation between consecutive hand positions
- **Hand Gesture Recognition**:
  - Measures thumb tip (landmark 4) to pinky tip (landmark 20) distance
  - Flash threshold: 180 pixels
  - Typical closed hand: ~100px, open hand: ~200+px
- **Flash Effect**:
  - Exponential decay (0.8x per frame) for smooth fade
  - White overlay proportional to flash intensity
  - Enhanced particle parameters during flash

## Troubleshooting

**Webcam not detected:**
- Make sure your webcam is connected and not being used by another application
- Try changing the camera index in the code: `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

**Hand tracking not working:**
- Ensure good lighting conditions
- Keep your hand within the camera frame
- Try adjusting the `min_detection_confidence` parameter

**Performance issues:**
- Reduce the `particles_per_frame` value (try 10 or 8)
- Decrease the `fade_duration` (particles will disappear faster)
- Lower your webcam resolution

**Particles not visible:**
- Make sure drawing is enabled (check status in top-left corner)
- Move your hand more - particles spawn continuously while your finger is tracked
- Ensure your hand is detected by MediaPipe (good lighting helps)

**Sound not working:**
- Make sure sound is enabled (press 'S' to toggle)
- Check that your system audio is not muted
- Move your hand faster - sound only plays above a minimum speed threshold
- If you see "Warning: Could not initialize audio" on startup, pygame/audio drivers may not be available

**Flash not triggering:**
- Spread your hand as wide as possible (thumb to pinky)
- Make sure all fingers are visible to the camera
- The flash threshold is 180 pixels - try moving closer to the camera for a wider hand spread
- Watch for the "FLASH!" indicator in the top-right when it triggers

## License

MIT License

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## Acknowledgments

- Built with [MediaPipe](https://google.github.io/mediapipe/)
- Uses [OpenCV](https://opencv.org/) for video processing
