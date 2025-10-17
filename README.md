# Slam Dunk EQ - Screen Monitor Application

A powerful GUI application that monitors screen areas for changes using two detection modes with customizable trigger actions. Perfect for gaming, monitoring applications, or automating responses to screen events.

## Warning!!

I cannot recommend using this application on servers that do not allow for automation, especially the Project 1999 Classic Everquest Server which has the best meta ever that you would only be cheating yourself out of if you use this.

![Quick Start Guide](Quick%20Start.png)

## Features

### 🔍 COTH Mode (Change Over Threshold)

- Monitors a selected screen area for pixel color changes
- Triggers when more than 50% of pixels change by more than 5%
- Perfect for detecting visual changes in applications
- Lightweight and extremely fast (20 checks per second)

### 📝 TRACK Mode (Text Detection)

- Uses OCR (Optical Character Recognition) to detect specific words
- Monitors a selected screen area for appearance of watch words
- Great for monitoring text-based notifications
- Powered by Tesseract OCR for fast and accurate text recognition
- Automatically checks for Tesseract and guides installation if needed

### ⚡ Customizable Trigger Actions

When a trigger is detected, execute multiple actions in sequence:

- **🔊 Alarm Sound** - Plays audible beep (can be enabled/disabled)
- **🪟 Focus Window** - Automatically brings a specific window to foreground
- **⌨️ Key Spam** - Sends keyboard input multiple times (configurable key and repeat count)
- **⏱️ Delay** - Adds timed delays between actions

**Actions are:**

- Fully customizable and can be reordered
- Executed sequentially in the order you define
- Can be enabled/disabled individually
- Great for gaming automation or workflow optimization

### 🎯 Smart Optimization

- Only runs OCR when screen content changes (TRACK mode)
- Minimal CPU usage when screen is static
- Efficient change detection algorithms

## Installation

### Option 1: Standalone Executable (Recommended for Most Users)

**Download the pre-built executable for instant use - no Python installation required!**

1. Download `ScreenMonitor.exe` from the `dist` folder (82 MB)
2. **Install Tesseract OCR** (required for TRACK mode - see instructions below)
3. Double-click `ScreenMonitor.exe` to run

**Why install Tesseract separately?**

- Keeps the executable small (82 MB vs 500+ MB)
- Tesseract is maintained independently with regular updates
- You can use the same Tesseract installation for other applications
- Installation is quick and easy (~80 MB download)

### Option 2: Python Installation (For Developers)

**If you want to modify the code or prefer a smaller installation:**

1. Install Python 3.8 or higher

2. Install Tesseract OCR (see installation instructions below)

3. Clone or download this repository

4. Install required packages:

```bash
pip install -r requirements.txt
```

5. Run the application:

```bash
python screen_monitor.py
```

### Installing Tesseract OCR (Required for TRACK Mode)

**Important:** TRACK mode requires Tesseract OCR. The application will automatically detect if it's missing and guide you through installation.

#### Windows Installation:

1. **Download Tesseract:**

   - Visit: https://github.com/UB-Mannheim/tesseract/wiki
   - Download the latest Windows installer (recommended: 64-bit version)
   - File size: ~80 MB

2. **Run the installer:**

   - Double-click the downloaded installer
   - Follow the installation wizard
   - Use the default installation location for automatic detection

3. **Verify installation:**
   - Open Command Prompt
   - Run: `tesseract --version`
   - You should see version information

**Automatic Detection:**
The application automatically detects Tesseract in these locations:

- `C:\Program Files\Tesseract-OCR\tesseract.exe`
- `C:\Program Files (x86)\Tesseract-OCR\tesseract.exe`

**What happens if Tesseract is not installed?**

- COTH mode works without it
- When you try to use TRACK mode, a friendly dialog will:
  - Explain what Tesseract is
  - Offer to open the download page
  - Provide installation instructions

**Note:** Tesseract is a free, open-source OCR engine with excellent performance and minimal overhead.

## Usage

### Basic Setup

1. **Start the application:**

   - **Executable:** Double-click `ScreenMonitor.exe`
   - **Python:** Run `python screen_monitor.py`

2. **Select a monitoring mode:**

   - **COTH Mode**: For detecting visual changes (color/pixel changes)
   - **TRACK Mode**: For detecting specific text (requires Tesseract OCR)

3. **Define watch area:**

   - Click and drag to select the area on your screen you want to monitor
   - Release to confirm selection
   - You can edit the watch area later if needed

4. **For TRACK mode only:**
   - Enter comma-separated watch words when prompted
   - Example: `error, warning, failed, complete`
   - The monitor will trigger when any of these words appear
   - Words are case-insensitive

### Configuring Trigger Actions

Once monitoring starts, you can configure what happens when a trigger is detected:

1. **View Current Actions:**

   - The "Trigger Actions" section shows all configured actions
   - Default: Alarm sound is enabled

2. **Add Actions:**

   - Click **"+ Add Action"** button
   - Choose from:
     - **Focus Window**: Select a window to bring to foreground
     - **Key Spam**: Configure a key to press multiple times
     - **Delay**: Add a timed pause (in milliseconds)

3. **Manage Actions:**

   - **Reorder**: Use ▲▼ arrows to change execution order
   - **Enable/Disable**: Check/uncheck to toggle individual actions
   - **Delete**: Click ✕ to remove an action

4. **Action Examples:**
   - Gaming: Focus game window → Delay 150ms → Press 'E' 5 times
   - Monitoring: Focus alert window → Sound alarm
   - Automation: Delay 500ms → Press 'Enter' 1 time

### During Monitoring

- **Status Display**: Shows real-time detection information

  - COTH: Shows pixel change percentage
  - TRACK: Shows detected words and OCR processing time

- **Edit Settings**: While monitoring, you can:

  - Edit watch area (Show/Edit button)
  - Edit watch words (TRACK mode only)
  - Modify trigger actions

- **Stop Monitoring**:
  - Click "Stop Monitoring" button
  - Or close the application window

### When Triggered

- Configured actions execute sequentially
- Alarm (if enabled) plays for 5 seconds or until you press any key
- Actions run once per trigger detection
- Monitor continues running after actions complete

## System Requirements

- **Operating System:** Windows 10 or higher
- **RAM:**
  - COTH mode: 1GB minimum
  - TRACK mode: 2GB minimum, 4GB recommended
- **Disk Space:**
  - ScreenMonitor.exe: 82 MB
  - Tesseract OCR: ~80 MB (required for TRACK mode)
  - Python installation (developers): ~200 MB
- **CPU:** Any modern processor (OCR is optimized for minimal usage)
- **Display:** Any resolution supported
- **Dependencies:**
  - Tesseract OCR (required for TRACK mode only)
  - COTH mode has no external dependencies

**File Sizes:**

- `ScreenMonitor.exe`: 82 MB
- Tesseract OCR installer: ~80 MB

## Technical Details

### COTH Mode

- **Detection Method:** Pixel-based comparison
- Captures screenshots at 20Hz (20 times per second)
- Compares pixel values between consecutive frames
- Triggers when >50% of pixels change by >5% brightness
- Extremely lightweight and fast
- No external dependencies required

### TRACK Mode

- **OCR Engine:** Tesseract OCR (open-source)
- **Optimization:** Only runs OCR when screen content changes (>2% pixel change)
- Saves CPU by skipping OCR on static screens
- Captures screenshots and runs OCR analysis
- Searches for user-defined watch words
- Supports multiple languages (English by default)
- Case-insensitive word matching
- Fast and lightweight compared to deep learning models
- Typical OCR speed: 50-200ms depending on area size

### Trigger Actions System

- **Sequential Execution:** Actions run in order from top to bottom
- **Window Focus:** Uses Windows API to bring windows to foreground
- **Key Spam:** Simulates keyboard input using pynput library
  - Supports special keys (Enter, Space, F1-F12, etc.)
  - Random timing between presses (10-30ms) for natural behavior
  - Can target specific windows
- **Delays:** Precise millisecond timing using Python's time module
- **Thread-Safe:** All actions execute in controlled threads

### Architecture

- **GUI Framework:** Python and Tkinter with modern styling
- **Image Processing:** PIL/Pillow for screenshots and basic operations
- **OCR:** Tesseract OCR for text recognition
- **Computer Vision:** OpenCV for image preprocessing
- **Input Simulation:** pynput for keyboard/mouse control
- **Windows Integration:** ctypes for native Windows API calls
- **Build System:** PyInstaller for standalone executable creation
- **Executable Size:** 82 MB (84% smaller than previous version)

## Building from Source

If you want to create your own executable:

```bash
# Install PyInstaller
pip install pyinstaller

# Build the executable
pyinstaller slam_dunk_eq.spec --clean
```

The executable will be created in the `dist` folder.

## Troubleshooting

### Installation Issues

**Executable won't start:**

- Ensure you're running Windows 10 or higher
- Check that your antivirus isn't blocking the exe
- Try running as administrator
- Make sure you have the Visual C++ Redistributable installed

**Tesseract installation fails:**

- Download from the official source: https://github.com/UB-Mannheim/tesseract/wiki
- Choose the 64-bit installer for modern systems
- Disable antivirus temporarily during installation
- Verify with: `tesseract --version` in Command Prompt

### Detection Issues

**TRACK mode not detecting text:**

- **First, check Tesseract:**
  - Verify installation: `tesseract --version`
  - Ensure it's in default location or added to PATH
- **Then, check your setup:**
  - Ensure watch area contains clear, readable text
  - Try increasing the watch area size
  - Make sure text isn't obscured or changing too quickly
  - Check that text color contrasts with background
  - Test with simple, large text first

**COTH mode triggering too often:**

- The area might include animated elements
- Try selecting a more specific area
- Consider using TRACK mode instead if monitoring text

**COTH mode not triggering:**

- Ensure the watched area actually changes
- Check that changes are significant (>50% pixels, >5% brightness)
- Try a test with a dramatic visual change first

### Trigger Action Issues

**Key presses not working:**

- Make sure the target window has focus (use Focus Window action first)
- Some games/applications block simulated input
- Try adding a delay (150-300ms) before key presses
- Run the application as administrator for better input compatibility

**Window focus not working:**

- The target window must be visible (not minimized)
- Some fullscreen applications can't be focused externally
- Try running Slam Dunk EQ as administrator
- Refresh the window list if a new window appeared

**Actions not executing in correct order:**

- Check the arrow buttons - order matters!
- Use delays between actions if timing is critical
- Verify all actions are enabled (checkboxes)

### Performance Issues

**High CPU usage:**

- Normal in TRACK mode when screen changes frequently
- OCR only runs when content changes (check status display)
- Consider using a smaller watch area
- COTH mode uses minimal CPU

**OCR is slow:**

- Reduce watch area size for faster processing
- Close unnecessary applications
- Ensure Tesseract is installed (not using fallback)
- Check status display for actual OCR timing

### Other Issues

**Alarm not playing:**

- Check your system volume
- Ensure Windows sound services are running
- Some systems don't support the beep function
- Try enabling/disabling in trigger actions

**Application crashes:**

- Check if Tesseract is properly installed (for TRACK mode)
- Try running as administrator
- Check Windows Event Viewer for error details
- Report bugs with full error messages

**Can't select certain screen areas:**

- Some applications block screen capture (DRM content)
- Try windowed mode instead of fullscreen
- Some games require admin privileges to capture

### Getting Help

If you encounter issues not listed here:

1. Check that you're using the latest version
2. Verify all prerequisites are installed
3. Try with a simple test case first
4. Report bugs with:
   - Detailed steps to reproduce
   - Error messages (if any)
   - System information (Windows version, etc.)

## Use Cases & Examples

### Gaming

**Auto-loot in EverQuest:**

- Use TRACK mode to watch for "corpse" text
- Actions: Focus game → Delay 150ms → Press 'E' 3 times
- Never miss loot opportunities!

**Boss respawn alerts:**

- Use COTH mode on spawn location
- Actions: Sound alarm → Focus game window
- Get notified immediately when boss appears

**Chat monitoring:**

- Use TRACK mode on chat window for keywords
- Watch for: "raid", "group", "invite", your character name
- Actions: Sound alarm to get your attention

### Application Monitoring

**Build completion:**

- Use TRACK mode on build status area
- Watch for: "success", "failed", "completed"
- Actions: Sound alarm → Focus IDE window

**Error detection:**

- Use TRACK mode on log window
- Watch for: "error", "exception", "failed"
- Actions: Alarm + focus window for immediate attention

**System notifications:**

- Use COTH mode on notification area
- Detect when new notifications appear
- Actions: Custom response based on your workflow

### Automation

**Form submission:**

- Use TRACK mode to detect success message
- Actions: Delay 1000ms → Press 'Enter' → Focus next app
- Chain actions together

**Multi-window workflows:**

- Detect completion in one app
- Actions: Focus another app → Send keyboard commands
- Automate repetitive multi-app tasks

## Version History

### Current Version

- **Size:** 82 MB (84% smaller than v1.0)
- **OCR Engine:** Tesseract (replaced PaddleOCR)
- **New Features:**
  - Customizable trigger actions system
  - Window focus automation
  - Key spam functionality
  - Configurable delays
  - Smart Tesseract detection with guided installation
  - Optimized OCR (only runs when content changes)
- **Improvements:**
  - Much smaller executable size
  - Faster OCR performance
  - Modern UI design
  - Better error handling

## License

This project is provided as-is for personal and educational use.

## Credits

- OCR powered by [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- Icon: Shaq Dunk theme
- Developed with Python, Tkinter, OpenCV, and PIL
