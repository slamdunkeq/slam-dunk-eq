"""
Screen Monitor Application
Monitors screen areas for pixel changes (COTH mode) or text appearance (TRACK mode)
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import threading
import time
import numpy as np
from PIL import ImageGrab
import winsound
from pynput import mouse, keyboard
import pytesseract
import cv2
import ctypes
import ctypes.wintypes


class WindowManager:
    """Helper class to enumerate and focus windows"""

    @staticmethod
    def get_running_windows():
        """Get list of visible windows with titles"""
        windows = []

        def callback(hwnd, lParam):
            if ctypes.windll.user32.IsWindowVisible(hwnd):
                length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buff = ctypes.create_unicode_buffer(length + 1)
                    ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
                    title = buff.value
                    if title and title.strip():  # Only add non-empty titles
                        windows.append((hwnd, title))
            return True

        # Define callback type
        EnumWindowsProc = ctypes.WINFUNCTYPE(
            ctypes.c_bool,
            ctypes.wintypes.HWND,
            ctypes.wintypes.LPARAM
        )

        # Enumerate windows
        ctypes.windll.user32.EnumWindows(EnumWindowsProc(callback), 0)

        return windows

    @staticmethod
    def focus_window(hwnd):
        """Bring window to foreground"""
        try:
            # Restore if minimized
            SW_RESTORE = 9
            ctypes.windll.user32.ShowWindow(hwnd, SW_RESTORE)

            # Bring to foreground
            ctypes.windll.user32.SetForegroundWindow(hwnd)
            return True
        except Exception as e:
            print(f"Error focusing window: {e}")
            return False


class ScreenAreaSelector:
    """Allows user to select a rectangular area on screen"""

    def __init__(self):
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.rect = None
        self.canvas = None
        self.root = None

    def select_area(self):
        """Create transparent window for area selection"""
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-alpha', 0.3)
        self.root.attributes('-topmost', True)

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        self.canvas = tk.Canvas(self.root, bg='gray', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        instruction = self.canvas.create_text(
            screen_width // 2, 50,
            text="Click and drag to select watch area. Release to confirm.",
            font=('Arial', 16, 'bold'),
            fill='white'
        )

        self.canvas.bind('<Button-1>', self.on_press)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)

        self.root.mainloop()

        if self.start_x and self.end_x:
            # Ensure coordinates are ordered correctly
            x1 = min(self.start_x, self.end_x)
            x2 = max(self.start_x, self.end_x)
            y1 = min(self.start_y, self.end_y)
            y2 = max(self.start_y, self.end_y)
            return (x1, y1, x2, y2)
        return None

    def on_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline='red', width=3
        )

    def on_drag(self, event):
        if self.rect:
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        self.end_x = event.x
        self.end_y = event.y
        self.root.quit()
        self.root.destroy()


class TriggerAction:
    """Base class for trigger actions"""
    def __init__(self, enabled=True):
        self.enabled = enabled

    def execute(self):
        """Execute the action"""
        pass

    def get_description(self):
        """Get a description of the action for UI display"""
        return "Unknown action"


class AlarmAction(TriggerAction):
    """Play alarm sound"""
    def __init__(self, enabled=True):
        super().__init__(enabled)

    def execute(self):
        if not self.enabled:
            return
        # Play a beep sound (frequency=1000Hz, duration=500ms)
        try:
            winsound.Beep(1000, 500)
        except:
            pass

    def get_description(self):
        return "Alarm Sound" if self.enabled else "Alarm Sound (Disabled)"


class WindowFocusAction(TriggerAction):
    """Focus a specific window"""
    def __init__(self, window_hwnd, window_title, enabled=True):
        super().__init__(enabled)
        self.window_hwnd = window_hwnd
        self.window_title = window_title

    def execute(self):
        if not self.enabled:
            return
        WindowManager.focus_window(self.window_hwnd)

    def get_description(self):
        return f"Focus: {self.window_title}"


class KeySpamAction(TriggerAction):
    """Spam a key multiple times"""
    def __init__(self, key, repeat_count, enabled=True, target_window_hwnd=None):
        super().__init__(enabled)
        self.key = key
        self.repeat_count = repeat_count
        self.target_window_hwnd = target_window_hwnd

    def execute(self):
        if not self.enabled:
            return
        import random

        # Get the virtual key code
        vk_code = self._get_virtual_key_code(self.key)

        if vk_code is None:
            print(f"Warning: Could not find virtual key code for '{self.key}'")
            return

        # If target window specified, ensure it's focused first
        if self.target_window_hwnd:
            # Make sure the target window is in foreground
            ctypes.windll.user32.SetForegroundWindow(self.target_window_hwnd)
            time.sleep(0.15)  # Delay to ensure window is focused

        # Use pynput for key sending (works reliably across applications)
        from pynput.keyboard import Controller
        kbd = Controller()
        key_to_press = self._parse_key(self.key)

        for _ in range(self.repeat_count):
            kbd.press(key_to_press)
            kbd.release(key_to_press)
            time.sleep(random.uniform(0.01, 0.03))  # 10-30ms delay

    def _send_key_with_sendinput(self, vk_code):
        """Send key using SendInput API (more reliable for games)"""
        # Define INPUT structure correctly with union
        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
            ]

        class INPUT_UNION(ctypes.Union):
            _fields_ = [("ki", KEYBDINPUT)]

        class INPUT(ctypes.Structure):
            _fields_ = [
                ("type", ctypes.c_ulong),
                ("union", INPUT_UNION)
            ]

        # Constants
        INPUT_KEYBOARD = 1
        KEYEVENTF_KEYUP = 0x0002

        # Key down
        key_down = INPUT()
        key_down.type = INPUT_KEYBOARD
        key_down.union.ki.wVk = vk_code
        key_down.union.ki.wScan = 0
        key_down.union.ki.dwFlags = 0
        key_down.union.ki.time = 0
        key_down.union.ki.dwExtraInfo = None

        # Key up
        key_up = INPUT()
        key_up.type = INPUT_KEYBOARD
        key_up.union.ki.wVk = vk_code
        key_up.union.ki.wScan = 0
        key_up.union.ki.dwFlags = KEYEVENTF_KEYUP
        key_up.union.ki.time = 0
        key_up.union.ki.dwExtraInfo = None

        # Send the inputs
        result_down = ctypes.windll.user32.SendInput(1, ctypes.byref(key_down), ctypes.sizeof(INPUT))
        time.sleep(0.01)  # Small delay between down and up
        result_up = ctypes.windll.user32.SendInput(1, ctypes.byref(key_up), ctypes.sizeof(INPUT))

        # Debug output if sending fails
        if result_down == 0 or result_up == 0:
            print(f"Warning: SendInput failed - down: {result_down}, up: {result_up}")

    def _get_virtual_key_code(self, key_str):
        """Get Windows virtual key code for a key string"""
        key_lower = key_str.lower().strip()

        # Map of key names to virtual key codes
        vk_codes = {
            # Letters
            'a': 0x41, 'b': 0x42, 'c': 0x43, 'd': 0x44, 'e': 0x45,
            'f': 0x46, 'g': 0x47, 'h': 0x48, 'i': 0x49, 'j': 0x4A,
            'k': 0x4B, 'l': 0x4C, 'm': 0x4D, 'n': 0x4E, 'o': 0x4F,
            'p': 0x50, 'q': 0x51, 'r': 0x52, 's': 0x53, 't': 0x54,
            'u': 0x55, 'v': 0x56, 'w': 0x57, 'x': 0x58, 'y': 0x59, 'z': 0x5A,
            # Numbers
            '0': 0x30, '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34,
            '5': 0x35, '6': 0x36, '7': 0x37, '8': 0x38, '9': 0x39,
            # Special keys
            'space': 0x20, 'enter': 0x0D, 'tab': 0x09, 'esc': 0x1B, 'escape': 0x1B,
            'backspace': 0x08, 'delete': 0x2E,
            'shift': 0x10, 'ctrl': 0x11, 'alt': 0x12,
            'up': 0x26, 'down': 0x28, 'left': 0x25, 'right': 0x27,
            # Function keys
            'f1': 0x70, 'f2': 0x71, 'f3': 0x72, 'f4': 0x73,
            'f5': 0x74, 'f6': 0x75, 'f7': 0x76, 'f8': 0x77,
            'f9': 0x78, 'f10': 0x79, 'f11': 0x7A, 'f12': 0x7B,
        }

        # Check if it's in our map
        if key_lower in vk_codes:
            return vk_codes[key_lower]

        # For single character, convert to uppercase and get ASCII code
        if len(key_str) == 1:
            return ord(key_str.upper())

        return None

    def _parse_key(self, key_str):
        """Parse key string to pynput Key object or character"""
        from pynput.keyboard import Key

        # Map common key names to Key enum values
        special_keys = {
            'space': Key.space,
            'enter': Key.enter,
            'tab': Key.tab,
            'esc': Key.esc,
            'escape': Key.esc,
            'backspace': Key.backspace,
            'delete': Key.delete,
            'shift': Key.shift,
            'ctrl': Key.ctrl,
            'alt': Key.alt,
            'up': Key.up,
            'down': Key.down,
            'left': Key.left,
            'right': Key.right,
            'f1': Key.f1,
            'f2': Key.f2,
            'f3': Key.f3,
            'f4': Key.f4,
            'f5': Key.f5,
            'f6': Key.f6,
            'f7': Key.f7,
            'f8': Key.f8,
            'f9': Key.f9,
            'f10': Key.f10,
            'f11': Key.f11,
            'f12': Key.f12,
        }

        key_lower = key_str.lower().strip()

        # Check if it's a special key
        if key_lower in special_keys:
            return special_keys[key_lower]

        # Otherwise, return the character as-is
        # For single characters, pynput accepts them directly
        return key_str if len(key_str) == 1 else key_str[0]

    def get_description(self):
        return f"Press '{self.key}' {self.repeat_count}x"


class DelayAction(TriggerAction):
    """Delay for a specified number of milliseconds"""
    def __init__(self, delay_ms, enabled=True):
        super().__init__(enabled)
        self.delay_ms = delay_ms

    def execute(self):
        if not self.enabled:
            return
        time.sleep(self.delay_ms / 1000.0)

    def get_description(self):
        return f"Delay {self.delay_ms}ms"


class AlarmController:
    """Controls alarm and trigger actions"""

    def __init__(self):
        self.alarm_active = False
        self.alarm_thread = None
        self.stop_alarm_flag = False
        self.keyboard_listener = None
        self.trigger_actions = [AlarmAction(enabled=True)]  # Default alarm enabled

    def add_action(self, action):
        """Add a trigger action"""
        self.trigger_actions.append(action)

    def remove_action(self, action):
        """Remove a trigger action"""
        if action in self.trigger_actions:
            self.trigger_actions.remove(action)

    def move_action_up(self, action):
        """Move action up in the list"""
        if action not in self.trigger_actions:
            return
        idx = self.trigger_actions.index(action)
        if idx > 0:
            self.trigger_actions[idx], self.trigger_actions[idx - 1] = \
                self.trigger_actions[idx - 1], self.trigger_actions[idx]

    def move_action_down(self, action):
        """Move action down in the list"""
        if action not in self.trigger_actions:
            return
        idx = self.trigger_actions.index(action)
        if idx < len(self.trigger_actions) - 1:
            self.trigger_actions[idx], self.trigger_actions[idx + 1] = \
                self.trigger_actions[idx + 1], self.trigger_actions[idx]

    def get_actions(self):
        """Get list of trigger actions"""
        return self.trigger_actions

    def clear_actions(self):
        """Clear all actions except default alarm"""
        self.trigger_actions = [AlarmAction(enabled=True)]

    def start_alarm(self):
        """Start alarm with 5-second timeout and keyboard monitoring"""
        if self.alarm_active:
            return

        self.alarm_active = True
        self.stop_alarm_flag = False

        # Start keyboard listener only (ignore mouse input)
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_keyboard_input
        )

        self.keyboard_listener.start()

        # Start alarm sound in separate thread (continuous beeping) - runs concurrently
        self.alarm_thread = threading.Thread(target=self._play_alarm, daemon=True)
        self.alarm_thread.start()

        # Get the focused window handle from WindowFocusAction if present
        target_hwnd = None
        for action in self.trigger_actions:
            if isinstance(action, WindowFocusAction) and action.enabled:
                target_hwnd = action.window_hwnd
                break

        # Execute all non-alarm trigger actions sequentially
        # (alarm is already playing in background)
        for action in self.trigger_actions:
            if not isinstance(action, AlarmAction):
                try:
                    # Pass target window to KeySpamAction
                    if isinstance(action, KeySpamAction) and target_hwnd:
                        action.target_window_hwnd = target_hwnd
                    action.execute()
                except Exception as e:
                    print(f"Error executing action: {e}")

        # Start timeout timer
        timeout_thread = threading.Thread(target=self._timeout_timer, daemon=True)
        timeout_thread.start()

    def on_keyboard_input(self, *args):
        """Called when user presses any key"""
        if self.alarm_active:
            self.stop_alarm()

    def stop_alarm(self):
        """Stop the alarm and cleanup listeners"""
        self.stop_alarm_flag = True
        self.alarm_active = False

        if self.keyboard_listener:
            self.keyboard_listener.stop()

    def _play_alarm(self):
        """Play alarm sound continuously until stopped"""
        while self.alarm_active and not self.stop_alarm_flag:
            try:
                # Play a beep sound (frequency=1000Hz, duration=500ms)
                winsound.Beep(1000, 500)
                time.sleep(0.1)
            except:
                pass

    def _timeout_timer(self):
        """Stop alarm after 5 seconds"""
        time.sleep(5)
        if self.alarm_active:
            self.stop_alarm()


class COTHMonitor:
    """Change Over Threshold (COTH) - Monitors for pixel color changes"""

    def __init__(self, watch_area, alarm_controller, status_callback=None):
        self.watch_area = watch_area
        self.alarm_controller = alarm_controller
        self.running = False
        self.previous_image = None
        self.monitor_thread = None
        self.status_callback = status_callback
        self.current_change_percent = 0.0

    def start(self):
        """Start monitoring in background thread"""
        # Capture initial image
        self.previous_image = self._capture_area()
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop monitoring"""
        self.running = False
        # Wait for thread to finish (with timeout)
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)

    def _capture_area(self):
        """Capture the watch area as numpy array"""
        screenshot = ImageGrab.grab(bbox=self.watch_area)
        return np.array(screenshot)

    def _monitor_loop(self):
        """Continuously monitor for changes"""
        while self.running:
            current_image = self._capture_area()

            change_detected, change_percent = self._detect_change(current_image)
            self.current_change_percent = change_percent

            # Update status callback with current percentage
            if self.status_callback:
                self.status_callback(f"Pixel change: {change_percent:.1f}%")

            if change_detected:
                self.alarm_controller.start_alarm()
                # Wait a bit before checking again to avoid repeated alarms
                time.sleep(2)

            # Update previous image for next comparison
            self.previous_image = current_image

            time.sleep(0.05)  # Check 20 times per second

    def _detect_change(self, current_image):
        """Detect if more than 50% of pixels changed by more than 5% from previous frame"""
        if self.previous_image is None:
            return False, 0.0

        # Calculate absolute difference from previous frame
        diff = np.abs(current_image.astype(float) - self.previous_image.astype(float))

        # Calculate percentage change per pixel (average across RGB channels)
        percent_change = np.mean(diff, axis=2) / 255.0 * 100

        # Count pixels with more than 5% change
        changed_pixels = np.sum(percent_change > 5)
        total_pixels = percent_change.shape[0] * percent_change.shape[1]

        change_ratio = changed_pixels / total_pixels
        change_percent = change_ratio * 100

        return change_ratio > 0.5, change_percent


class TRACKMonitor:
    """TRACK mode - Monitors for specific words using OCR"""

    @staticmethod
    def check_tesseract_available():
        """Check if Tesseract OCR is installed and available"""
        # First, try to set Tesseract path if in default location
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break

        # Try to get Tesseract version to verify it's available
        try:
            pytesseract.get_tesseract_version()
            return True, None
        except Exception as e:
            return False, str(e)

    def __init__(self, watch_area, watch_words, alarm_controller, status_callback=None):
        self.watch_area = watch_area
        self.watch_words = [word.lower() for word in watch_words]
        self.alarm_controller = alarm_controller
        self.running = False
        self.monitor_thread = None
        self.status_callback = status_callback
        self.detected_words = []
        self.previous_image = None

        # Set Tesseract path if in default location
        self._setup_tesseract_path()

    def _setup_tesseract_path(self):
        """Configure Tesseract path if in default location"""
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break

    def start(self):
        """Start monitoring in background thread"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop monitoring"""
        self.running = False
        # Wait for thread to finish (with timeout)
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)

    def _capture_area(self):
        """Capture the watch area"""
        screenshot = ImageGrab.grab(bbox=self.watch_area)
        return screenshot

    def _monitor_loop(self):
        """Continuously monitor for watch words"""
        while self.running:
            loop_start = time.time()

            screenshot = self._capture_area()
            capture_time = time.time() - loop_start

            # Check if image has changed before running expensive OCR
            if self._has_image_changed(screenshot):
                change_check_time = time.time() - loop_start - capture_time

                ocr_start = time.time()
                detected, found_words = self._detect_words(screenshot)
                ocr_time = time.time() - ocr_start

                self.detected_words = found_words

                # Update status callback with detected words
                if self.status_callback:
                    if found_words:
                        words_str = ", ".join(found_words)
                        self.status_callback(f"Detected: {words_str} (OCR: {ocr_time*1000:.0f}ms)")
                    else:
                        self.status_callback(f"Scanning... (OCR: {ocr_time*1000:.0f}ms)")

                if detected:
                    total_time = time.time() - loop_start
                    print(f"TIMING: Capture={capture_time*1000:.0f}ms, Change={change_check_time*1000:.0f}ms, OCR={ocr_time*1000:.0f}ms, Total={total_time*1000:.0f}ms")
                    self.alarm_controller.start_alarm()
                    # Wait a bit before checking again to avoid repeated alarms
                    time.sleep(2)

                # Store current image for next comparison
                self.previous_image = np.array(screenshot)
            # else: Image hasn't changed, skip OCR to save CPU

            time.sleep(0.05)  # Check 20 times per second - OCR will only run when image changes more than 0.1%

    def _has_image_changed(self, current_screenshot):
        """Check if the image has changed since last check"""
        if self.previous_image is None:
            return True  # First run, always process

        current_array = np.array(current_screenshot)

        # Quick comparison - if images are identical, skip OCR
        if np.array_equal(current_array, self.previous_image):
            return False

        # Calculate difference - ANY change triggers OCR for fastest detection
        # We want to catch text as soon as it appears, even if faint
        diff = np.abs(current_array.astype(float) - self.previous_image.astype(float))
        max_change = np.max(diff)

        # If ANY pixel changed by more than a small amount, run OCR
        # This catches text appearing even if it's just a few pixels
        return max_change > 5  # Any pixel changed by more than 5/255 (2%)

    def _preprocess_image(self, image):
        """Enhance image for multi-color text on black background (green, blue, red)"""
        # Convert PIL to numpy array
        img_array = np.array(image)

        # CRITICAL: Scale up the image 3x - OCR works MUCH better on larger text
        scale_factor = 3
        height, width = img_array.shape[:2]
        img_array = cv2.resize(img_array, (width * scale_factor, height * scale_factor),
                              interpolation=cv2.INTER_CUBIC)

        # Process each color channel separately (for green, blue, red text)
        red_channel = img_array[:, :, 0]
        green_channel = img_array[:, :, 1]
        blue_channel = img_array[:, :, 2]

        processed_channels = []

        for channel in [red_channel, green_channel, blue_channel]:
            # Apply extreme CLAHE
            clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(4, 4))
            enhanced = clahe.apply(channel)

            # Apply very aggressive gamma correction
            gamma = 3.0
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                             for i in np.arange(0, 256)]).astype("uint8")
            gamma_corrected = cv2.LUT(enhanced, table)

            # Normalize to full range
            normalized = cv2.normalize(gamma_corrected, None, 0, 255, cv2.NORM_MINMAX)

            # Apply very aggressive threshold
            # Use a low fixed threshold to catch dim text
            _, binary = cv2.threshold(normalized, 15, 255, cv2.THRESH_BINARY)

            processed_channels.append(binary)

        # Combine all channels - text will appear in at least one channel
        combined = cv2.bitwise_or(processed_channels[0], processed_channels[1])
        combined = cv2.bitwise_or(combined, processed_channels[2])

        # Apply strong sharpening to the combined result
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(combined, -1, kernel)

        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)

        # Remove small noise
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # Convert grayscale back to 3-channel BGR for PaddleOCR
        # PaddleOCR requires 3-channel images
        cleaned_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

        return cleaned_bgr

    def _detect_words(self, image):
        """Use Tesseract OCR to detect watch words in image"""
        try:
            # Convert PIL to numpy if needed
            img_array = np.array(image)

            # Tesseract works with PIL images or numpy arrays
            # Use minimal config for speed: --psm 6 (single block)
            # --oem 3 uses default OCR engine mode (LSTM + legacy)
            custom_config = r'--oem 3 --psm 6'

            # Run Tesseract
            detected_text = pytesseract.image_to_string(img_array, config=custom_config).lower()

            # Check if any watch words are present
            found_words = []
            for word in self.watch_words:
                if word in detected_text:
                    found_words.append(word)

            return len(found_words) > 0, found_words

        except Exception as e:
            print(f"OCR Error: {e}")
            import traceback
            traceback.print_exc()
            return False, []


class ScreenMonitorApp:
    """Main application"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Slam Dunk EQ")
        self.root.geometry("500x850")
        self.root.resizable(False, False)

        # Modern color scheme
        self.colors = {
            'bg': '#1e1e2e',           # Dark background
            'surface': '#2a2a3e',      # Slightly lighter surface
            'surface_light': '#363654', # Even lighter surface
            'primary': '#7289da',      # Discord-like blue
            'primary_hover': '#5b6eae',
            'success': '#43b581',      # Green
            'success_hover': '#3a9b6d',
            'danger': '#f04747',       # Red
            'danger_hover': '#d83c3c',
            'warning': '#faa61a',      # Orange
            'warning_hover': '#e89512',
            'text': '#dcddde',         # Light text
            'text_muted': '#72767d',   # Muted text
            'accent': '#7289da'        # Accent color
        }

        # Configure root window
        self.root.configure(bg=self.colors['bg'])

        # Configure ttk style for modern look
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Use clam as base theme

        self.monitor = None
        self.alarm_controller = AlarmController()
        self.highlight_window = None
        self.watch_area = None
        self.bg_photo = None
        self.watch_words_label = None
        self.edit_words_button = None

        # Load background image
        self._load_background_image()

        self.create_widgets()

    def _get_resource_path(self, relative_path):
        """Get absolute path to resource, works for dev and for PyInstaller"""
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    def _load_background_image(self):
        """Background image loading disabled"""
        self.bg_photo = None

    def _create_modern_button(self, parent, text, command, style='primary', width=20, height=2):
        """Create a modern styled button with hover effects"""
        if style == 'primary':
            bg_color = self.colors['primary']
            hover_color = self.colors['primary_hover']
        elif style == 'success':
            bg_color = self.colors['success']
            hover_color = self.colors['success_hover']
        elif style == 'danger':
            bg_color = self.colors['danger']
            hover_color = self.colors['danger_hover']
        elif style == 'warning':
            bg_color = self.colors['warning']
            hover_color = self.colors['warning_hover']
        else:
            bg_color = self.colors['surface']
            hover_color = self.colors['surface_light']

        button = tk.Button(
            parent,
            text=text,
            command=command,
            font=('Segoe UI', 10, 'bold'),
            bg=bg_color,
            fg=self.colors['text'],
            activebackground=hover_color,
            activeforeground=self.colors['text'],
            relief=tk.FLAT,
            bd=0,
            padx=15,
            pady=10,
            width=width,
            height=height,
            cursor='hand2'
        )

        # Add hover effects
        def on_enter(e):
            button['background'] = hover_color

        def on_leave(e):
            button['background'] = bg_color

        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)

        return button

    def create_widgets(self):
        """Create the main GUI"""
        # Create main container with modern background
        main_container = tk.Frame(self.root, bg=self.colors['bg'])
        main_container.pack(fill=tk.BOTH, expand=True)

        # Header section with title
        header_frame = tk.Frame(main_container, bg=self.colors['surface'], height=80)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)

        # Title
        title_label = tk.Label(
            header_frame,
            text="⚡ Slam Dunk EQ",
            font=('Segoe UI', 20, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['text']
        )
        title_label.pack(pady=20)

        # Create a frame for UI elements with modern styling
        self.ui_frame = tk.Frame(main_container, bg=self.colors['bg'])
        self.ui_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Instructions
        self.instructions = tk.Label(
            self.ui_frame,
            text="Select monitoring mode:",
            font=('Segoe UI', 14, 'bold'),
            bg=self.colors['bg'],
            fg=self.colors['text']
        )
        self.instructions.pack(pady=(10, 20), padx=20)

        # COTH Button
        self.coth_button = self._create_modern_button(
            self.ui_frame,
            text="🔍 COTH Mode\nColor Change Detection",
            command=self.start_coth_mode,
            style='primary',
            width=25,
            height=3
        )
        self.coth_button.pack(pady=10, padx=20)

        # TRACK Button
        self.track_button = self._create_modern_button(
            self.ui_frame,
            text="📝 TRACK Mode\nText Detection",
            command=self.start_track_mode,
            style='primary',
            width=25,
            height=3
        )
        self.track_button.pack(pady=10, padx=20)

        # Status label (initially empty)
        self.status_label = tk.Label(
            self.ui_frame,
            text="",
            font=('Segoe UI', 10),
            fg=self.colors['success'],
            bg=self.colors['bg']
        )
        self.status_label.pack(pady=15, padx=20)

    def start_coth_mode(self):
        """Start COTH monitoring mode"""
        # If already running, stop it
        if self.monitor:
            self.stop_monitoring()
            return

        # Hide main window
        self.root.withdraw()

        # Select area
        selector = ScreenAreaSelector()
        watch_area = selector.select_area()

        if watch_area:
            # Store watch area
            self.watch_area = watch_area

            # Show main window
            self.root.deiconify()

            # Start monitoring
            self.monitor = COTHMonitor(watch_area, self.alarm_controller, self._update_status)
            self.monitor.start()

            # Update UI to active mode
            self._set_active_mode('COTH')
        else:
            self.root.deiconify()
            messagebox.showwarning("Cancelled", "Area selection cancelled")

    def start_track_mode(self):
        """Start TRACK monitoring mode"""
        # If already running, stop it
        if self.monitor:
            self.stop_monitoring()
            return

        # Check if Tesseract is available before proceeding
        tesseract_available, error_msg = TRACKMonitor.check_tesseract_available()

        if not tesseract_available:
            # Show user-friendly installation dialog
            response = messagebox.askyesno(
                "Tesseract OCR Not Found",
                "Tesseract OCR is required for TRACK mode but was not found on your system.\n\n"
                "Tesseract is a free, open-source OCR engine that enables text detection.\n\n"
                "Would you like to visit the download page to install it?\n\n"
                "Installation is quick and easy (~80 MB download).",
                icon='warning'
            )

            if response:
                # Open the Tesseract download page in the default browser
                import webbrowser
                webbrowser.open('https://github.com/UB-Mannheim/tesseract/wiki')
                messagebox.showinfo(
                    "Installation Instructions",
                    "After installing Tesseract:\n\n"
                    "1. Restart Slam Dunk EQ\n"
                    "2. Select TRACK mode again\n\n"
                    "The application will automatically detect Tesseract if installed in the default location:\n"
                    "  • C:\\Program Files\\Tesseract-OCR\\\n"
                    "  • C:\\Program Files (x86)\\Tesseract-OCR\\\n\n"
                    "Or add Tesseract to your system PATH."
                )
            return

        # Hide main window
        self.root.withdraw()

        # Select area
        selector = ScreenAreaSelector()
        watch_area = selector.select_area()

        if watch_area:
            # Show main window temporarily for input
            self.root.deiconify()

            # Get watch words
            watch_words_input = simpledialog.askstring(
                "Watch Words",
                "Enter watch words (comma-separated):",
                parent=self.root
            )

            if watch_words_input:
                watch_words = [word.strip() for word in watch_words_input.split(',')]

                # Store watch area
                self.watch_area = watch_area

                # Start monitoring
                self.monitor = TRACKMonitor(watch_area, watch_words, self.alarm_controller, self._update_status)
                self.monitor.start()

                # Update UI to active mode
                self._set_active_mode('TRACK')
            else:
                messagebox.showwarning("Cancelled", "No watch words provided")
        else:
            self.root.deiconify()
            messagebox.showwarning("Cancelled", "Area selection cancelled")

    def stop_monitoring(self):
        """Stop the current monitoring"""
        if self.monitor:
            self.monitor.stop()
            self.monitor = None

        self.alarm_controller.stop_alarm()
        self.alarm_controller.clear_actions()  # Clear all trigger actions
        self.watch_area = None
        self._hide_highlight()
        self._reset_ui()

    def _update_status(self, status_text):
        """Update status label (called from monitor threads)"""
        # Schedule UI update in main thread
        # Check if status_label still exists (it may have been destroyed during UI changes)
        def update_label():
            if self.status_label and self.status_label.winfo_exists():
                self.status_label.config(text=status_text, fg='blue')
        self.root.after(0, update_label)

    def _show_highlight(self, event=None):
        """Show highlight overlay over watch area"""
        if not self.watch_area:
            return

        if self.highlight_window:
            return  # Already showing

        # Create transparent overlay window
        self.highlight_window = tk.Toplevel(self.root)
        self.highlight_window.attributes('-topmost', True)
        self.highlight_window.attributes('-alpha', 0.3)
        self.highlight_window.overrideredirect(True)

        x1, y1, x2, y2 = self.watch_area
        width = x2 - x1
        height = y2 - y1

        self.highlight_window.geometry(f"{width}x{height}+{x1}+{y1}")

        # Create yellow highlight
        canvas = tk.Canvas(self.highlight_window, bg='yellow', highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)

    def _hide_highlight(self, event=None):
        """Hide highlight overlay"""
        if self.highlight_window:
            self.highlight_window.destroy()
            self.highlight_window = None

    def _edit_watch_words(self):
        """Allow user to edit watch words while monitoring is active"""
        if not self.monitor or not isinstance(self.monitor, TRACKMonitor):
            return

        # Get current watch words
        current_words = ', '.join(self.monitor.watch_words)

        # Show dialog to edit
        new_words_input = simpledialog.askstring(
            "Edit Watch Words",
            "Enter watch words (comma-separated):",
            parent=self.root,
            initialvalue=current_words
        )

        if new_words_input:
            # Update the monitor's watch words
            new_words = [word.strip().lower() for word in new_words_input.split(',')]
            self.monitor.watch_words = new_words

            # Reset previous image to force OCR re-read on next check
            self.monitor.previous_image = None

            # Update the display
            self._update_watch_words_display(new_words)

    def _update_watch_words_display(self, watch_words):
        """Update the watch words label"""
        if self.watch_words_label:
            words_text = ', '.join(watch_words)
            self.watch_words_label.config(text=f"Watching: {words_text}")

    def _set_active_mode(self, mode):
        """Update UI to show active mode with details and trigger actions"""
        # Clear the UI
        for widget in self.ui_frame.winfo_children():
            if widget not in (self.instructions, self.status_label):
                widget.destroy()

        self.instructions.config(
            text=f"{'🔍' if mode == 'COTH' else '📝'} {mode} Mode Active",
            font=('Segoe UI', 14, 'bold'),
            bg=self.colors['bg'],
            fg=self.colors['text']
        )

        # Mode details section with modern card style
        details_frame = tk.Frame(
            self.ui_frame,
            bg=self.colors['surface'],
            relief=tk.FLAT,
            bd=0
        )
        details_frame.pack(pady=10, padx=15, fill=tk.X, before=self.status_label)

        # Add subtle padding inside the card
        details_inner = tk.Frame(details_frame, bg=self.colors['surface'])
        details_inner.pack(padx=15, pady=12, fill=tk.X)

        tk.Label(
            details_inner,
            text="📋 Mode Details",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['primary'],
            anchor='w'
        ).pack(pady=(0, 8), fill=tk.X)

        # Mode type
        mode_desc = "Color Change Detection" if mode == 'COTH' else "Text Detection"
        tk.Label(
            details_inner,
            text=f"Type: {mode_desc}",
            font=('Segoe UI', 9),
            bg=self.colors['surface'],
            fg=self.colors['text'],
            anchor='w'
        ).pack(pady=2, fill=tk.X)

        # Watch area with edit button
        area_frame = tk.Frame(details_inner, bg=self.colors['surface'])
        area_frame.pack(pady=5, fill=tk.X)

        tk.Label(
            area_frame,
            text="Watch Area: ",
            font=('Segoe UI', 9),
            bg=self.colors['surface'],
            fg=self.colors['text_muted']
        ).pack(side=tk.LEFT)

        show_edit_btn = tk.Button(
            area_frame,
            text="Show/Edit",
            font=('Segoe UI', 8),
            command=self._edit_watch_area,
            bg=self.colors['success'],
            fg=self.colors['text'],
            activebackground=self.colors['success_hover'],
            activeforeground=self.colors['text'],
            bd=0,
            relief=tk.FLAT,
            cursor='hand2',
            padx=8,
            pady=2
        )
        show_edit_btn.pack(side=tk.LEFT, padx=5)

        # Add hover effects
        def on_enter_show(e):
            show_edit_btn['background'] = self.colors['success_hover']
        def on_leave_show(e):
            show_edit_btn['background'] = self.colors['success']

        show_edit_btn.bind('<Enter>', lambda e: (on_enter_show(e), self._show_highlight(e)))
        show_edit_btn.bind('<Leave>', lambda e: (on_leave_show(e), self._hide_highlight(e)))

        # Watch words for TRACK mode
        if mode == 'TRACK' and isinstance(self.monitor, TRACKMonitor):
            words_frame = tk.Frame(details_inner, bg=self.colors['surface'])
            words_frame.pack(pady=5, fill=tk.X)

            tk.Label(
                words_frame,
                text="Watch Words: ",
                font=('Segoe UI', 9),
                bg=self.colors['surface'],
                fg=self.colors['text_muted']
            ).pack(side=tk.LEFT)

            self.watch_words_label = tk.Label(
                words_frame,
                text=', '.join(self.monitor.watch_words),
                font=('Segoe UI', 9),
                bg=self.colors['surface'],
                fg=self.colors['warning']
            )
            self.watch_words_label.pack(side=tk.LEFT, padx=5)

            edit_words_btn = tk.Button(
                words_frame,
                text="Edit",
                font=('Segoe UI', 8),
                command=self._edit_watch_words,
                bg=self.colors['success'],
                fg=self.colors['text'],
                activebackground=self.colors['success_hover'],
                activeforeground=self.colors['text'],
                bd=0,
                relief=tk.FLAT,
                cursor='hand2',
                padx=8,
                pady=2
            )
            edit_words_btn.pack(side=tk.LEFT, padx=5)

            # Add hover effects
            def on_enter_edit(e):
                edit_words_btn['background'] = self.colors['success_hover']
            def on_leave_edit(e):
                edit_words_btn['background'] = self.colors['success']

            edit_words_btn.bind('<Enter>', on_enter_edit)
            edit_words_btn.bind('<Leave>', on_leave_edit)

        # Trigger Actions section
        actions_frame = tk.Frame(self.ui_frame, bg=self.colors['surface'], relief=tk.FLAT, bd=0)
        actions_frame.pack(pady=10, padx=15, fill=tk.X, before=self.status_label)

        # Actions header
        actions_header = tk.Frame(actions_frame, bg=self.colors['surface'])
        actions_header.pack(padx=15, pady=(12, 8), fill=tk.X)

        tk.Label(
            actions_header,
            text="⚡ Trigger Actions",
            font=('Segoe UI', 11, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['primary'],
            anchor='w'
        ).pack(fill=tk.X)

        # Create canvas with scrollbar for actions list
        canvas_frame = tk.Frame(actions_frame, bg=self.colors['surface'])
        canvas_frame.pack(padx=15, pady=5, fill=tk.BOTH)

        # Limit height to prevent overflow
        canvas = tk.Canvas(canvas_frame, bg=self.colors['surface'], height=150, highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview,
                                bg=self.colors['surface_light'], troughcolor=self.colors['surface'])
        self.actions_list_frame = tk.Frame(canvas, bg=self.colors['surface'])

        self.actions_list_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.actions_list_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._refresh_actions_list()

        # Add Action button
        add_action_btn = self._create_modern_button(
            actions_frame,
            text="+ Add Action",
            command=self._show_add_action_menu,
            style='warning',
            width=15,
            height=1
        )
        add_action_btn.pack(pady=(5, 12), padx=15)

        # Stop button
        stop_btn = self._create_modern_button(
            self.ui_frame,
            text="■ Stop Monitoring",
            command=self.stop_monitoring,
            style='danger',
            width=20,
            height=2
        )
        stop_btn.pack(pady=8, padx=20, before=self.status_label)

        self.status_label.config(
            text="Initializing...",
            fg=self.colors['primary'],
            bg=self.colors['bg'],
            font=('Segoe UI', 9)
        )

    def _refresh_actions_list(self):
        """Refresh the trigger actions list display"""
        # Clear current list
        for widget in self.actions_list_frame.winfo_children():
            widget.destroy()

        actions = self.alarm_controller.get_actions()

        # Display each action
        for idx, action in enumerate(actions):
            action_item_frame = tk.Frame(self.actions_list_frame, bg=self.colors['surface_light'],
                                        relief=tk.FLAT, bd=0)
            action_item_frame.pack(pady=3, padx=5, fill=tk.X)

            # Up/Down arrow buttons (only for non-alarm actions)
            if not isinstance(action, AlarmAction):
                arrow_frame = tk.Frame(action_item_frame, bg=self.colors['surface_light'])
                arrow_frame.pack(side=tk.LEFT, padx=5, pady=3)

                # Up arrow
                up_btn = tk.Button(
                    arrow_frame,
                    text="▲",
                    font=('Segoe UI', 7),
                    command=lambda a=action: self._move_action_up(a),
                    bg=self.colors['surface'],
                    fg=self.colors['text'],
                    activebackground=self.colors['primary'],
                    activeforeground=self.colors['text'],
                    bd=0,
                    relief=tk.FLAT,
                    width=2,
                    height=1,
                    cursor='hand2' if idx > 0 else 'arrow',
                    state=tk.NORMAL if idx > 0 else tk.DISABLED
                )
                up_btn.pack()

                # Down arrow
                down_btn = tk.Button(
                    arrow_frame,
                    text="▼",
                    font=('Segoe UI', 7),
                    command=lambda a=action: self._move_action_down(a),
                    bg=self.colors['surface'],
                    fg=self.colors['text'],
                    activebackground=self.colors['primary'],
                    activeforeground=self.colors['text'],
                    bd=0,
                    relief=tk.FLAT,
                    width=2,
                    height=1,
                    cursor='hand2' if idx < len(actions) - 1 else 'arrow',
                    state=tk.NORMAL if idx < len(actions) - 1 else tk.DISABLED
                )
                down_btn.pack()
            else:
                # Add spacer for alarm action to align with other actions
                spacer = tk.Frame(action_item_frame, bg=self.colors['surface_light'], width=30)
                spacer.pack(side=tk.LEFT, padx=5)

            # Checkbox for enable/disable
            enabled_var = tk.BooleanVar(value=action.enabled)

            def toggle_action(act=action, var=enabled_var):
                act.enabled = var.get()

            tk.Checkbutton(
                action_item_frame,
                variable=enabled_var,
                command=toggle_action,
                bg=self.colors['surface_light'],
                activebackground=self.colors['surface_light'],
                selectcolor='white',  # White background when checked
                fg='black',  # Black checkmark
                activeforeground='black'
            ).pack(side=tk.LEFT, padx=5)

            # Action description
            tk.Label(
                action_item_frame,
                text=action.get_description(),
                font=('Segoe UI', 9),
                bg=self.colors['surface_light'],
                fg=self.colors['text'],
                anchor='w'
            ).pack(side=tk.LEFT, padx=8, pady=5, fill=tk.X, expand=True)

            # Delete button (only for non-alarm actions)
            if not isinstance(action, AlarmAction):
                delete_btn = tk.Button(
                    action_item_frame,
                    text="✕",
                    font=('Segoe UI', 9, 'bold'),
                    command=lambda a=action: self._delete_action(a),
                    bg=self.colors['danger'],
                    fg=self.colors['text'],
                    activebackground=self.colors['danger_hover'],
                    activeforeground=self.colors['text'],
                    bd=0,
                    relief=tk.FLAT,
                    width=3,
                    cursor='hand2'
                )
                delete_btn.pack(side=tk.RIGHT, padx=5, pady=3)

                # Add hover effect
                def on_enter_delete(e, btn=delete_btn):
                    btn['background'] = self.colors['danger_hover']
                def on_leave_delete(e, btn=delete_btn):
                    btn['background'] = self.colors['danger']

                delete_btn.bind('<Enter>', on_enter_delete)
                delete_btn.bind('<Leave>', on_leave_delete)

    def _move_action_up(self, action):
        """Move action up in the list"""
        self.alarm_controller.move_action_up(action)
        self._refresh_actions_list()

    def _move_action_down(self, action):
        """Move action down in the list"""
        self.alarm_controller.move_action_down(action)
        self._refresh_actions_list()

    def _delete_action(self, action):
        """Delete a trigger action"""
        self.alarm_controller.remove_action(action)
        self._refresh_actions_list()

    def _show_add_action_menu(self):
        """Show menu to add new actions"""
        menu = tk.Menu(self.root, tearoff=0,
                      bg=self.colors['surface'],
                      fg=self.colors['text'],
                      activebackground=self.colors['primary'],
                      activeforeground=self.colors['text'],
                      relief=tk.FLAT,
                      bd=0)
        menu.add_command(label="🪟 Focus Window", command=self._add_window_focus_action)
        menu.add_command(label="⌨ Key Spam", command=self._add_key_spam_action)
        menu.add_command(label="⏱ Delay", command=self._add_delay_action)

        # Show menu at mouse position
        try:
            menu.tk_popup(self.root.winfo_pointerx(), self.root.winfo_pointery())
        finally:
            menu.grab_release()

    def _add_window_focus_action(self):
        """Add window focus action"""
        # Get running windows
        windows = WindowManager.get_running_windows()

        if not windows:
            messagebox.showinfo("No Windows", "No windows available to select")
            return

        # Create selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Window to Focus")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(
            dialog,
            text="Select window to focus when alarm triggers:",
            font=('Arial', 10),
            pady=10
        ).pack()

        # Listbox with scrollbar
        list_frame = tk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, font=('Arial', 9))
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)

        for _, title in windows:
            listbox.insert(tk.END, title)

        def on_select():
            selection = listbox.curselection()
            if selection:
                idx = selection[0]
                hwnd, title = windows[idx]
                action = WindowFocusAction(hwnd, title)
                self.alarm_controller.add_action(action)
                self._refresh_actions_list()
            dialog.destroy()

        tk.Button(
            dialog,
            text="Add",
            command=on_select,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 10, 'bold'),
            width=10
        ).pack(pady=10)

    def _add_key_spam_action(self):
        """Add key spam action"""
        # Dialog for key input
        key = simpledialog.askstring(
            "Key Spam",
            "Enter key to spam (e.g., 'a', '1', 'space'):",
            parent=self.root
        )

        if not key:
            return

        # Dialog for repeat count
        repeat_str = simpledialog.askstring(
            "Key Spam",
            f"How many times to press '{key}'?",
            parent=self.root,
            initialvalue="5"
        )

        if not repeat_str:
            return

        try:
            repeat_count = int(repeat_str)
            if repeat_count < 1 or repeat_count > 100:
                messagebox.showerror("Invalid Input", "Repeat count must be between 1 and 100")
                return

            action = KeySpamAction(key, repeat_count)
            self.alarm_controller.add_action(action)
            self._refresh_actions_list()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number")

    def _add_delay_action(self):
        """Add delay action"""
        # Dialog for delay input
        delay_str = simpledialog.askstring(
            "Delay Action",
            "Enter delay in milliseconds (e.g., 500 for half second):",
            parent=self.root,
            initialvalue="500"
        )

        if not delay_str:
            return

        try:
            delay_ms = int(delay_str)
            if delay_ms < 1 or delay_ms > 10000:
                messagebox.showerror("Invalid Input", "Delay must be between 1 and 10000 ms")
                return

            action = DelayAction(delay_ms)
            self.alarm_controller.add_action(action)
            self._refresh_actions_list()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number")

    def _edit_watch_area(self):
        """Edit the watch area"""
        # Hide main window
        self.root.withdraw()

        # Select new area
        selector = ScreenAreaSelector()
        watch_area = selector.select_area()

        # Show main window
        self.root.deiconify()

        if watch_area:
            # Update watch area
            self.watch_area = watch_area

            # Restart monitor with new area
            if self.monitor:
                current_mode = 'COTH' if isinstance(self.monitor, COTHMonitor) else 'TRACK'

                # Stop current monitor
                self.monitor.stop()

                # Start new monitor
                if current_mode == 'COTH':
                    self.monitor = COTHMonitor(watch_area, self.alarm_controller, self._update_status)
                else:
                    watch_words = self.monitor.watch_words
                    self.monitor = TRACKMonitor(watch_area, watch_words, self.alarm_controller, self._update_status)

                self.monitor.start()
                messagebox.showinfo("Success", "Watch area updated")
        else:
            messagebox.showinfo("Cancelled", "Watch area not changed")

    def _reset_ui(self):
        """Reset UI to show both mode buttons"""
        # Clear all widgets except instructions and status_label
        for widget in self.ui_frame.winfo_children():
            if widget not in (self.instructions, self.status_label):
                widget.destroy()

        self.instructions.config(
            text="Select monitoring mode:",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['bg'],
            fg=self.colors['text']
        )

        # Clear references
        self.watch_words_label = None
        self.edit_words_button = None

        # Recreate COTH button
        self.coth_button = self._create_modern_button(
            self.ui_frame,
            text="🔍 COTH Mode\n(Color Change Detection)",
            command=self.start_coth_mode,
            style='primary',
            width=25,
            height=3
        )
        self.coth_button.pack(pady=8, padx=20, before=self.status_label)

        # Recreate TRACK button
        self.track_button = self._create_modern_button(
            self.ui_frame,
            text="📝 TRACK Mode\n(Text Detection)",
            command=self.start_track_mode,
            style='primary',
            width=25,
            height=3
        )
        self.track_button.pack(pady=8, padx=20, before=self.status_label)

        self.status_label.config(
            text="",
            fg=self.colors['success'],
            bg=self.colors['bg']
        )

    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        """Clean up before closing"""
        self.stop_monitoring()
        self.root.destroy()


if __name__ == "__main__":
    app = ScreenMonitorApp()
    app.run()
