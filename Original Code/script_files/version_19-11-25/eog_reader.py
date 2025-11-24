import threading
import collections
import numpy as np
import time
from pylsl import StreamInlet, resolve_byprop
from signal_processing_wavelet import process_eog_signals, process_eog_signals_with_blinks
import config
from config import DETECT_PERIOD, H_VELOCITY_THRESHOLD, V_VELOCITY_THRESHOLD
from dataclasses import dataclass

@dataclass
class Detection:
    ts: float
    direction: str
    is_horizontal: bool
    is_blink: bool = False  # Add blink flag
    is_final: bool = False  # Flag to indicate final detection in a window
    blink_duration: float = 0.0
    h_value: float = 0.0
    v_value: float = 0.0
    h_velocity: float = 0.0
    v_velocity: float = 0.0

class EOGReader(threading.Thread):
    def __init__(self, out_queue, max_queue=50, calibration_params=None):
        super().__init__()
        self.record_raw = False
        self.raw_log = []
        self.start_time = None
        self.out_queue = out_queue
        self.max_queue = max_queue
        self.calibration_params = calibration_params or {
            "baselines": {"H": 0, "V": 0},
            "thresholds": {"left": 0.3, "right": 0.3, "up": 0.3, "down": 0.3},
            "blink_threshold": config.BLINK_THRESHOLD,
            "channel_norm_factors": {"ch1": 1, "ch2": 1, "ch3": 1, "ch8": 1},
            "alpha": 0.0
        }
        self.last_blink_time = -1e9
        self.running = True
        self.in_blink_cooldown = False  # Flag to track blink cooldown period

        # Buffers for plotting (5 seconds) and detection (1 second)
        self.channel_buffers = [collections.deque(maxlen=config.PLOT_MAX_SAMPLES) for _ in range(config.TOTAL_CHANNELS)]
        self.time_buffer = collections.deque(maxlen=config.PLOT_MAX_SAMPLES)
        self.detect_channel_buffers = [collections.deque(maxlen=config.DETECT_MAX_SAMPLES) for _ in range(config.TOTAL_CHANNELS)]
        self.detect_time_buffer = collections.deque(maxlen=config.DETECT_MAX_SAMPLES)
        self.last_detection_time = 0.0
        self.last_h_movement_time = -1e9
        self.last_v_movement_time = -1e9
        self.last_blink_time = -1e9

        #store last detections for final selection
        self.pending_h = None
        self.pending_v = None
        self.last_final_detection = None

        print("Looking for LSL stream...")
        streams = resolve_byprop('name', config.LSL_STREAM_NAME, timeout=5.0)
        if not streams:
            raise RuntimeError(f"{config.LSL_STREAM_NAME} stream not found")
        self.inlet = StreamInlet(streams[0])
        print("Connected to stream.")
        self.start_time = time.time()

        # For plotting and debugging - store full buffer data
        self.full_H = np.array([])
        self.full_V = np.array([])
        self.full_times = np.array([])

        # For detection - store detection window data
        self.latest_H = np.array([])
        self.latest_V = np.array([])
        self.latest_times = np.array([])
        self.debug_counter = 0
        self.last_plot_update = time.time()

    def save_raw_data(self, filename):
        if len(self.raw_log) == 0:
            print(f"[EOGReader] No raw data to save for {filename}")
            return
        import csv
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time"] + [f"ch{i}" for i in range(1, len(self.raw_log[0]))])
            writer.writerows(self.raw_log)
        print(f"[EOGReader] Saved raw data: {filename}")

    def _push_horizontal(self, det: Detection):
        """Push horizontal detection to queue with cooldown check"""
        current_time = time.time() - self.start_time

        # Don't push if we're in blink cooldown
        if self.in_blink_cooldown:
            return False

        # Horizontal signal-type cooldown
        if (current_time - self.last_h_movement_time) < config.GLOBAL_COOLDOWN:
            return False

        # Store the detection for later use in final selection
        self.pending_h = det

        # Update global cooldown timer
        self.last_h_movement_time = current_time
        return True

    def _push_vertical(self, det: Detection):
        """Push vertical detection to queue with cooldown check"""
        current_time = time.time() - self.start_time

        # Don't push if we're in blink cooldown
        if self.in_blink_cooldown:
            return False

        # Vertical signal-type cooldown
        if (current_time - self.last_v_movement_time) < config.GLOBAL_COOLDOWN:
            return False
        
        # Store the detection for later use in final selection
        self.pending_v = det

        # Update vertical cooldown timer
        self.last_v_movement_time = current_time
        return True
    
    def _finalize_combined_detection(self):
        """Create a final detection from pending horizontal and vertical detections"""
        current_time = time.time() - self.start_time

        # Nothing detected
        if self.pending_h is None and self.pending_v is None:
            return

        # Both axes detected → create combined detection
        if self.pending_h is not None and self.pending_v is not None:
            horiz = self.pending_h.direction
            vert = self.pending_v.direction
            combined = Detection(
                ts=current_time,
                direction=f"{vert} {horiz}",
                is_horizontal=False,
                is_blink=False,
                is_final=True,
                h_value=self.pending_h.h_value,
                v_value=self.pending_v.v_value,
                h_velocity=self.pending_h.h_velocity,
                v_velocity=self.pending_v.v_velocity
            )
            self.out_queue.append(combined)
            self.last_final_detection = combined
            # Print the combined detection
            print(f"[FINAL DETECTION - COMBINED] {combined.direction} at {current_time:.2f}s "
                f"(H: {horiz}, V: {vert}, H_val: {combined.h_value:.3f}, V_val: {combined.v_value:.3f})")

        # Only horizontal
        elif self.pending_h is not None:
            det = self.pending_h
            det.is_final = True
            det.ts = current_time  # Update timestamp to current time
            self.out_queue.append(det)
            self.last_final_detection = det
            # Print the horizontal detection
            print(f"[FINAL DETECTION - HORIZONTAL] {det.direction} at {current_time:.2f}s "
                f"(H_val: {det.h_value:.3f}, V_val: {det.v_value:.3f})")

        # Only vertical
        elif self.pending_v is not None:
            det = self.pending_v
            det.is_final = True
            det.ts = current_time  # Update timestamp to current time
            self.out_queue.append(det)
            self.last_final_detection = det
            # Print the vertical detection
            print(f"[FINAL DETECTION - VERTICAL] {det.direction} at {current_time:.2f}s "
                f"(H_val: {det.h_value:.3f}, V_val: {det.v_value:.3f})")

        # Keep queue bounded
        while len(self.out_queue) > self.max_queue:
            self.out_queue.popleft()

        # Clear pending buffers for next cycle
        self.pending_h = None
        self.pending_v = None

    def _push_blink(self, det: Detection):
        """Push blink detection to queue and set blink cooldown"""
        current_time = time.time() - self.start_time
        # Check global cooldown
        if (current_time - self.last_blink_time) >= config.GLOBAL_COOLDOWN:
            self.out_queue.append(det)
            while len(self.out_queue) > self.max_queue:
                self.out_queue.popleft()
            self.last_blink_time = current_time
            self.in_blink_cooldown = True  # Set blink cooldown flag
            self.last_final_detection = det
            # Print the blink detection
            print(f"[FINAL DETECTION - BLINK] at {current_time:.2f}s "
                f"(Duration: {det.blink_duration:.3f}s, V_val: {det.v_value:.3f})")

            # Start a timer to clear the blink cooldown after BLINK_COOLDOWN period
            def clear_blink_cooldown():
                time.sleep(config.BLINK_COOLDOWN)
                self.in_blink_cooldown = False

            cooldown_thread = threading.Thread(target=clear_blink_cooldown)
            cooldown_thread.daemon = True
            cooldown_thread.start()
            return True
        return False

    def process_detection_window(self):
        """Process the detection window and check for saccades and blinks"""
        try:
            # Use the detection window buffers
            times = np.array(self.detect_time_buffer)
            ch1 = np.array(self.detect_channel_buffers[0])
            ch2 = np.array(self.detect_channel_buffers[1])
            ch3 = np.array(self.detect_channel_buffers[2])
            ch8 = np.array(self.detect_channel_buffers[4])

            # Ensure all arrays have the same length
            min_length = min(len(times), len(ch1), len(ch2), len(ch3), len(ch8))
            if min_length == 0:
                return

            times = times[-min_length:]
            ch1 = ch1[-min_length:]
            ch2 = ch2[-min_length:]
            ch3 = ch3[-min_length:]
            ch8 = ch8[-min_length:]

            # Process signals with alpha compensation
            H_corrected, V, V_compensated, blink_events = process_eog_signals_with_blinks(
                ch1, ch2, ch3, ch8, self.calibration_params
            )

            # Calculate velocity (derivative)
            dt = 1.0 / config.FS
            H_velocity = np.gradient(H_corrected, dt)
            V_velocity = np.gradient(V_compensated, dt)

            if len(H_velocity) < len(H_corrected):
                H_velocity = np.pad(H_velocity, (0, len(H_corrected) - len(H_velocity)), mode='edge')
            if len(V_velocity) < len(V_compensated):
                V_velocity = np.pad(V_velocity, (0, len(V_compensated) - len(V_velocity)), mode='edge')

            #Ensure the processed signals have the same length as times
            min_processed_length = min(len(times), len(H_corrected), len(V), len(V_compensated), len(H_velocity), len(V_velocity))
            if min_processed_length < min_length:
                print(f"Warning: Processed signals shorter than expected ({min_processed_length} < {min_length})")
                # Trim all arrays to the shortest length
                times = times[-min_processed_length:]
                H_corrected = H_corrected[:min_processed_length]
                V = V[:min_processed_length]
                V_compensated = V_compensated[:min_processed_length]
                H_velocity = H_velocity[:min_processed_length]
                V_velocity = V_velocity[:min_processed_length]

                # Filter blink events to only those within our valid range
                blink_events = [b for b in blink_events if b['peak_index'] < min_processed_length]
            else:
                blink_events = blink_events

            # Update latest signals for detection
            self.latest_H = H_corrected
            self.latest_V = V_compensated
            self.latest_times = times

            # Get current time for cooldown checks
            current_time = time.time() - self.start_time
            detected_directions = set()  # Prevent multiple detections of the same direction per window

            # --- First detect blinks (they have priority) ---
            for blink in blink_events:
                # Check cooldown
                if (current_time - self.last_blink_time) < config.BLINK_COOLDOWN:
                    continue

                # Check if the blink peak is above the blink threshold
                if abs(V[blink['peak_index']]) < self.calibration_params['blink_threshold']:
                    continue  # Ignore small peaks

                # Create a blink detection
                det = Detection(
                    ts=times[blink['peak_index']],
                    direction='blink',
                    is_horizontal=False,
                    is_blink=True,
                    blink_duration=blink['duration'],
                    h_value=H_corrected[blink['peak_index']],
                    v_value=V_compensated[blink['peak_index']],
                    h_velocity=H_velocity[blink['peak_index']],
                    v_velocity=V_velocity[blink['peak_index']]
                )

                if self._push_blink(det):
                    print(f"Pushed blink detection to queue at {times[blink['peak_index']]:.2f}s")
                    self.last_blink_time = current_time
                    detected_directions.add("blink")
                    return  # Skip other detections in this window when blink is detected

            # --- Then check for other movements if not in blink cooldown ---
            if self.in_blink_cooldown:
                return
            
            # Define a helper function to find the next peak in the opposite direction
            def find_opposite_peak(signal, start_idx, threshold, is_positive_direction):
                """
                Find the next peak in the opposite direction after a threshold crossing.

                Args:
                    signal: The signal array to search
                    start_idx: Index to start searching from
                    threshold: The threshold value
                    is_positive_direction: True if the initial direction was positive, False if negative

                Returns:
                    A tuple of (peak_idx, peak_value) or (None, None) if no suitable peak is found
                """
                search_end = min(start_idx + int(0.5 * config.FS), len(signal))  # Search up to 500ms ahead

                # Determine the direction we're looking for (opposite of initial)
                if is_positive_direction:
                    # Looking for a negative peak after a positive crossing
                    min_val = float('inf')
                    min_idx = None

                    for i in range(start_idx, search_end):
                        if signal[i] < min_val:
                            min_val = signal[i]
                            min_idx = i

                    # Check if the minimum is significant enough
                    if min_idx is not None and abs(min_val) > 0.75 * threshold:
                        return min_idx, min_val
                else:
                    # Looking for a positive peak after a negative crossing
                    max_val = -float('inf')
                    max_idx = None

                    for i in range(start_idx, search_end):
                        if signal[i] > max_val:
                            max_val = signal[i]
                            max_idx = i

                    # Check if the maximum is significant enough
                    if max_idx is not None and max_val > 0.75 * threshold:
                        return max_idx, max_val

                return None, None
            
            left_crossings = np.where(H_corrected < -self.calibration_params["thresholds"]["left"])[0]
            right_crossings = np.where(H_corrected > self.calibration_params["thresholds"]["right"])[0]

            # Process left crossings
            for crossing_idx in left_crossings:

                if crossing_idx >= len(H_velocity):
                    continue  # Safety check

                if crossing_idx >= len(times):
                    continue  # Safety check

                # Check if this crossing has sufficient velocity
                if abs(H_velocity[crossing_idx]) < H_VELOCITY_THRESHOLD:
                    continue

                # Look for a right deflection after this left crossing Commented out, might be useful later
                #opposite_peak_idx, opposite_peak_val = find_opposite_peak(
                    H_corrected, crossing_idx, self.calibration_params["thresholds"]["right"], True
                #)

                #if opposite_peak_idx is not None:
                    # We found a significant opposite peak - detect this as a right movement
                    det = Detection(
                        ts=times[opposite_peak_idx],
                        direction='right',
                        is_horizontal=True,
                        h_value=opposite_peak_val,
                        v_value=V_compensated[opposite_peak_idx],
                        h_velocity=H_velocity[opposite_peak_idx],
                        v_velocity=V_velocity[opposite_peak_idx]
                    )
                    pushed = self._push_horizontal(det)

                    if pushed:
                        detected_directions.add("right")
                #else:
                    # No significant opposite peak - detect as left movement
                det = Detection(
                        ts=times[crossing_idx],
                        direction='left',
                        is_horizontal=True,
                        h_value=H_corrected[crossing_idx],
                        v_value=V_compensated[crossing_idx],
                        h_velocity=H_velocity[crossing_idx],
                        v_velocity=V_velocity[crossing_idx]
                    )
                pushed = self._push_horizontal(det)

                if pushed:
                    detected_directions.add("left")

            # Process right crossings
            for crossing_idx in right_crossings:

                if crossing_idx >= len(H_velocity):
                    continue  # Safety check

                if crossing_idx >= len(times):
                    continue  # Safety check

                # Check if this crossing has sufficient velocity
                if abs(H_velocity[crossing_idx]) < H_VELOCITY_THRESHOLD:
                    continue

                # Look for a left deflection after this right crossing Commented out, might be useful later
                #opposite_peak_idx, opposite_peak_val = find_opposite_peak(
                    H_corrected, crossing_idx, self.calibration_params["thresholds"]["left"], False
                #)

                #if opposite_peak_idx is not None:
                    # We found a significant opposite peak - detect this as a left movement
                    det = Detection(
                        ts=times[opposite_peak_idx],
                        direction='left',
                        is_horizontal=True,
                        h_value=opposite_peak_val,
                        v_value=V_compensated[opposite_peak_idx],
                        h_velocity=H_velocity[opposite_peak_idx],
                        v_velocity=V_velocity[opposite_peak_idx]
                    )
                    pushed = self._push_horizontal(det)

                    if pushed:
                        detected_directions.add("left")
                #else:
                    # No significant opposite peak - detect as right movement
                det = Detection(
                        ts=times[crossing_idx],
                        direction='right',
                        is_horizontal=True,
                        h_value=H_corrected[crossing_idx],
                        v_value=V_compensated[crossing_idx],
                        h_velocity=H_velocity[crossing_idx],
                        v_velocity=V_velocity[crossing_idx]
                    )
                pushed = self._push_horizontal(det)

                if pushed:
                    detected_directions.add("right")

            # --- Vertical movements with improved detection logic ---
            # Find all threshold crossings for up and down
            up_crossings = np.where(V_compensated < -self.calibration_params["thresholds"]["up"])[0]
            down_crossings = np.where(V_compensated > self.calibration_params["thresholds"]["down"])[0]

            # Process up crossings
            for crossing_idx in up_crossings:

                if crossing_idx >= len(V_velocity):
                    continue  # Safety check

                if crossing_idx >= len(times):
                    continue  # Safety check

                # Check if this crossing has sufficient velocity
                if abs(V_velocity[crossing_idx]) < V_VELOCITY_THRESHOLD:
                    continue

                # Check if this is actually a blink (above blink threshold)
                if V_compensated[crossing_idx] > self.calibration_params['blink_threshold']:
                    continue

                # Look for a down deflection after this up crossing Commented out, might be useful later
                #opposite_peak_idx, opposite_peak_val = find_opposite_peak(
                    V_compensated, crossing_idx, self.calibration_params["thresholds"]["down"], True
                #)

                #if opposite_peak_idx is not None:
                    # We found a significant opposite peak - detect this as a down movement
                    det = Detection(
                        ts=times[opposite_peak_idx],
                        direction='down',
                        is_horizontal=False,
                        h_value=H_corrected[opposite_peak_idx],
                        v_value=opposite_peak_val,
                        h_velocity=H_velocity[opposite_peak_idx],
                        v_velocity=V_velocity[opposite_peak_idx]
                    )
                    pushed = self._push_vertical(det)

                    if pushed:
                        detected_directions.add("down")
                #else:
                    # No significant opposite peak - detect as up movement
                det = Detection(
                        ts=times[crossing_idx],
                        direction='up',
                        is_horizontal=False,
                        h_value=H_corrected[crossing_idx],
                        v_value=V_compensated[crossing_idx],
                        h_velocity=H_velocity[crossing_idx],
                        v_velocity=V_velocity[crossing_idx]
                    )
                pushed = self._push_vertical(det)

                if pushed:
                    detected_directions.add("up")

            # Process down crossings
            for crossing_idx in down_crossings:

                if crossing_idx >= len(V_velocity):
                    continue  # Safety check

                if crossing_idx >= len(times):
                    continue  # Safety check

                # Check if this crossing has sufficient velocity
                if abs(V_velocity[crossing_idx]) < V_VELOCITY_THRESHOLD:
                    continue

                # Check if this is actually a blink (below negative blink threshold)
                if V_compensated[crossing_idx] < -self.calibration_params['blink_threshold']:
                    continue

                # Look for an up deflection after this down crossing Commented out, might be useful later
                #opposite_peak_idx, opposite_peak_val = find_opposite_peak(
                    V_compensated, crossing_idx, self.calibration_params["thresholds"]["up"], False
                #)

                #if opposite_peak_idx is not None:
                    # We found a significant opposite peak - detect this as an up movement
                    det = Detection(
                        ts=times[opposite_peak_idx],
                        direction='up',
                        is_horizontal=False,
                        h_value=H_corrected[opposite_peak_idx],
                        v_value=opposite_peak_val,
                        h_velocity=H_velocity[opposite_peak_idx],
                        v_velocity=V_velocity[opposite_peak_idx]
                    )
                    pushed = self._push_vertical(det)

                    if pushed:
                        detected_directions.add("up")
                #else:
                    # No significant opposite peak - detect as down movement
                det = Detection(
                        ts=times[crossing_idx],
                        direction='down',
                        is_horizontal=False,
                        h_value=H_corrected[crossing_idx],
                        v_value=V_compensated[crossing_idx],
                        h_velocity=H_velocity[crossing_idx],
                        v_velocity=V_velocity[crossing_idx]
                    )
                pushed = self._push_vertical(det)

                if pushed:
                    detected_directions.add("down")

            if not self.in_blink_cooldown:
                # After processing all detections, finalize combined detection if applicable
                self._finalize_combined_detection()

        except Exception as e:
            print(f"Error in detection processing: {str(e)}")
            import traceback
            traceback.print_exc()

    def run(self):
        """Main thread loop for reading and processing EOG data"""
        last_detection_check = time.time()

        while self.running:
            sample, timestamp = self.inlet.pull_sample(timeout=0.05)
            if sample is None:
                continue

            now = time.time() - self.start_time

                # Save raw data if recording is enabled
            if self.record_raw:
                self.raw_log.append([now] + list(sample))

            # Add to plot buffers (5 seconds)
            self.time_buffer.append(now)
            for i in range(config.TOTAL_CHANNELS):
                self.channel_buffers[i].append(sample[i])

            # Add to detection buffers (1 second)
            self.detect_time_buffer.append(now)
            for i in range(config.TOTAL_CHANNELS):
                self.detect_channel_buffers[i].append(sample[i])

            # Process full buffers for plotting periodically (every 0.2 seconds)
            current_time = time.time()
            if (current_time - self.last_plot_update) >= 0.005 and len(self.time_buffer) > 0:
                self.last_plot_update = current_time

                try:
                    # Process all channels for plotting
                    ch1 = np.array(self.channel_buffers[0])
                    ch2 = np.array(self.channel_buffers[1])
                    ch3 = np.array(self.channel_buffers[2])
                    ch8 = np.array(self.channel_buffers[4])

                    # Ensure all arrays have the same length
                    min_length = min(len(self.time_buffer), len(ch1), len(ch2), len(ch3), len(ch8))
                    if min_length > 0:
                        times = np.array(self.time_buffer)[-min_length:]
                        ch1 = ch1[-min_length:]
                        ch2 = ch2[-min_length:]
                        ch3 = ch3[-min_length:]
                        ch8 = ch8[-min_length:]

                        # Process signals with alpha compensation
                        H_corrected, V_corrected, V_compensated, _ = process_eog_signals_with_blinks(
                            ch1, ch2, ch3, ch8, self.calibration_params
                        )

                        # Check if we got valid results
                        if all(isinstance(arr, np.ndarray) and len(arr) > 0 for arr in [H_corrected, V_corrected, V_compensated]):
                            # Update full buffers for plotting
                            self.full_H = H_corrected
                            self.full_V = V_compensated
                            self.full_times = times
                except Exception as e:
                    print(f"Error in plotting processing: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
            # Run detection every DETECT_PERIOD seconds if we have enough samples
            # Ensure we have at least 10 samples more than needed to avoid edge cases
            required_samples = min(config.DETECT_MAX_SAMPLES, len(self.detect_time_buffer) - 10)
            if required_samples > 10 and (current_time - last_detection_check) >= DETECT_PERIOD:
                last_detection_check = current_time
                self.process_detection_window()

    def stop(self):
        """Stop the thread"""
        self.running = False
        self.join()
        print("EOGReader stopped.")
