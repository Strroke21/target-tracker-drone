
import cv2
import numpy as np
from pymavlink import mavutil
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
import time
import math

# -------------------------------------
# Mouse callback
# -------------------------------------
target_point = None
new_target = False

script_mode = 2 #1 LOITER , 2 ALT_HOLD
counter = 0

# CAMERA PARAMETERS ----------------------------------------------
W = 640
H = 480
HFOV = np.deg2rad(114.6)
VFOV = np.deg2rad(92)

fx = W / (2 * np.tan(HFOV / 2))
fy = H / (2 * np.tan(VFOV / 2))

cx_img = W / 2
cy_img = H / 2

max_pitch = np.deg2rad(45)
max_roll  = np.deg2rad(45)
target_alt = 20
cam_orientation = 1 #0 downward, 1 45deg forward
theta = np.deg2rad(45.0) # camera tilt angle

if script_mode == 1:
    throttle_descent = 1400
    throttle_approach = 1000
    pwm_center = 1500
    Kp_x = max_roll*0.85
    Kp_y = max_pitch*0.85
    throttle_takeoff = 1750
    mode = "LOITER"
    terminal_altitude = 5
    px_error_threshold = 0.1

elif script_mode == 2:
    Kp_x = max_roll*1
    Kp_y = max_pitch*1.6
    Kp_z = -2
    mode = "ALT_HOLD"
    terminal_altitude = 10
    px_error_threshold = 0.15
    angle_threshold = 0.15

elif script_mode == 3:
    pwm_center = 1500
    Kp_x = max_roll*0.3
    Kp_y = max_pitch*0.3
    Kp_z = -0.8
    mode = "ALT_HOLD"
    terminal_altitude = 5
    px_error_threshold = 0.1

# Global frame (no queue)
latest_frame = None

# ================================================================
# ROS2 IMAGE SUBSCRIBER – Best Effort QoS
# ================================================================
class ImageSubscriber(Node):
    def __init__(self):
        super().__init__("image_subscriber")
        self.bridge = CvBridge()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=1
        )

        self.subscription = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            qos
        )

    def image_callback(self, msg):
        global latest_frame
        latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")


# ================================================================
# MAVLINK FUNCTIONS (unchanged)
# ================================================================
def connect(connection_string):
    vehicle = mavutil.mavlink_connection(connection_string)
    vehicle.wait_heartbeat()
    return vehicle

def enable_data_stream(vehicle, stream_rate):
    vehicle.mav.request_data_stream_send(
        vehicle.target_system,
        vehicle.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL,
        stream_rate, 1)

def get_rangefinder_data(vehicle):
    global rng_alt
    msg = vehicle.recv_match(type='DISTANCE_SENSOR', blocking=False)
    if msg and msg.current_distance is not None:
        rng_alt = msg.current_distance / 100.0
    return rng_alt

def get_local_position(vehicle):
    msg = vehicle.recv_match(type='LOCAL_POSITION_NED', blocking=True)
    if msg:
        return msg.x, msg.y, msg.z

def compute_pose(cx, cy, alt):
    # Pixel → normalized coordinates
    x_norm = (cx - cx_img) / fx
    y_norm = (cy - cy_img) / fy

    # Ground projection with Z = alt 
    X = x_norm * alt
    Y = y_norm * alt
    Z = -alt   # by your definition: drone is alt meters above ground

    # Yaw angle (bearing on ground plane)
    yaw = np.arctan2(Y, X)

    # Ray angles relative to camera frame
    pitch = np.arctan2(np.sqrt(x_norm*x_norm + y_norm*y_norm), 1.0)
    roll = np.arctan2(-y_norm, 1.0)

    return (X, Y, Z), (roll, pitch, yaw)

def mouse_click(event, x, y, flags, param):
    global target_point, new_target
    if event == cv2.EVENT_LBUTTONDOWN:
        target_point = np.array([[x, y]], dtype=np.float32)
        new_target = True
        print(f"Target selected at: {x}, {y}")

def rc_override(vehicle, roll_pwm, pitch_pwm, throttle_pwm):
    vehicle.mav.rc_channels_override_send(
        vehicle.target_system,
        vehicle.target_component,
        roll_pwm,
        pitch_pwm,
        throttle_pwm,
        1500,
        0,0,0,0
    )

def cmd_to_rc(cmd):
    cmd = np.clip(cmd, -1, 1)
    return int(1500 + 400 * cmd)   # 1100..1900

def current_alt(vehicle):
    while True:
        msg = vehicle.recv_match(type='TERRAIN_REPORT', blocking=True)
        if msg:
            curr_alt = msg.current_height #in meters
            return curr_alt
        
def give_throttle(vehicle, throttle_pwm):
    while True:
        alt = current_alt(vehicle)
        print(f"Current Altitude: {alt} m")
        if alt<=target_alt:
            rc_override(vehicle, 1500, 1500, throttle_pwm)
        else:
            break

def arm(vehicle):
    #arm the drone
    vehicle.mav.command_long_send(vehicle.target_system, vehicle.target_component,mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
    
def VehicleMode(vehicle, mode):
    modes = ["STABILIZE","ACRO","ALT_HOLD","AUTO","GUIDED","LOITER","RTL","CIRCLE","","LAND"]
    mode_id = modes.index(mode) if mode in modes else 12
    vehicle.mav.set_mode_send(vehicle.target_system,
                              mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                              mode_id)
    
def force_disarm(master):
    master.mav.command_long_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0,          # confirmation
    0,          # param1: 0 = disarm
    21196,      # param2: force disarm magic number
    0, 0, 0, 0, 0
)

# ================================================================
# MAIN TRACKER LOOP (CSRT)
# ================================================================
def main():
    vehicle = connect('tcp:127.0.0.1:5773')
    enable_data_stream(vehicle, 100)

    # ---- ROS2 SETUP ----
    rclpy.init()
    node = ImageSubscriber()

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    global latest_frame
    print("Waiting for /camera/image_raw ...")
    while latest_frame is None:
        pass

    cv2.namedWindow("Tracker")
    cv2.setMouseCallback("Tracker", mouse_click)

    global target_point, new_target

    tracker = None
    tracking = False

    ROI_SIZE = 80       # Initial tracking window (pixels)
    # initial_roi_area = ROI_SIZE * ROI_SIZE

    while rclpy.ok():

        if latest_frame is None:
            continue

        frame = latest_frame.copy()

        alt = current_alt(vehicle)

        # ---------------------------------------------------------
        # Initialize tracker after mouse click
        # ---------------------------------------------------------
        if new_target:
            cx, cy = target_point.astype(int).ravel()

            x = max(0, cx - ROI_SIZE // 2)
            y = max(0, cy - ROI_SIZE // 2)

            w = min(ROI_SIZE, frame.shape[1] - x)
            h = min(ROI_SIZE, frame.shape[0] - y)

            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, (x, y, w, h))

            tracking = True
            new_target = False

        # ---------------------------------------------------------
        # Update tracker
        # ---------------------------------------------------------
        if tracking:

            success, bbox = tracker.update(frame)

            if success:

                x, y, w, h = [int(v) for v in bbox]
                current_roi_area = w * h
                roi_scale = current_roi_area / (W * H)
                print(f"ROI Scale: {roi_scale:.2f}")
                cx = x + w // 2
                cy = y + h // 2

                # Draw bounding box
                cv2.rectangle(frame,
                              (x, y),
                              (x + w, y + h),
                              (0, 255, 0), 2)

                # Crosshair
                size_x = 200
                size_y = 150

                cv2.line(frame,
                         (cx - size_x, cy),
                         (cx + size_x, cy),
                         (255, 0, 255), 2)

                cv2.line(frame,
                         (cx, cy - size_y),
                         (cx, cy + size_y),
                         (255, 0, 255), 2)

                # Pose, Orientation = compute_pose(cx, cy, alt)
                # print(f"Pose: {Pose}, Orientation: {Orientation}")
                # Pixel angles
                # -------------------------------------------------
                # Down-facing camera
                # -------------------------------------------------
                if cam_orientation == 0:

                    error_x = (cx - cx_img) / cx_img
                    error_y = (cy - cy_img) / cy_img

                    pitch_cmd = Kp_y * error_y
                    roll_cmd = Kp_x * error_x

                    rc_roll = cmd_to_rc(roll_cmd)
                    rc_pitch = cmd_to_rc(pitch_cmd)
                    rc_throttle = 1500

                    print(f"RC Roll={rc_roll}, Pitch={rc_pitch}")
                                        # VehicleMode(vehicle, mode)
                    print(f"vehicle mode: {vehicle.flightmode}")
                    aligned = (abs(error_x) < px_error_threshold and abs(error_y) < px_error_threshold)
                    if not aligned:
                        rc_override(vehicle,rc_roll,rc_pitch,rc_throttle)
                    if aligned:
                        if alt > terminal_altitude:
                            rc_override(vehicle,rc_roll,rc_pitch,1000)
                        elif alt <= terminal_altitude:
                            force_disarm(vehicle)
                            print("Target aligned and disarmed!")

                # -------------------------------------------------
                # Forward 45 degree camera
                # -------------------------------------------------
                if cam_orientation == 1:

                    # VehicleMode(vehicle, mode)
                    print(f"vehicle mode: {vehicle.flightmode}")

                    error_x = (cx - cx_img) / cx_img
                    error_y = (cy - cy_img) / cy_img

                    x_angle = math.atan2((cx - cx_img), fx)
                    y_angle = math.atan2((cy - cy_img), fy)
                    #effective vertical error
                    theta2 = theta + y_angle
                    slant_dist = (alt* math.sqrt(1.0 + math.tan(x_angle)**2)/ math.sin(theta2))
                    horizontal_dist = slant_dist * math.cos(theta2)
                    view_angle = math.atan2(horizontal_dist, alt)

                    # pitch_cmd = Kp_y * error_y
                    # roll_cmd = Kp_x * error_x
                    # error = math.sqrt(error_x**2 + error_y**2)
                    # throttle_cmd = Kp_z * error
                    roll_cmd = Kp_x * x_angle
                    pitch_cmd = Kp_y * y_angle
                    throttle_cmd = Kp_z * (horizontal_dist -alt)

                    rc_pitch = cmd_to_rc(pitch_cmd)
                    rc_roll = cmd_to_rc(roll_cmd)
                    rc_throttle = cmd_to_rc(throttle_cmd)

                    xy_error = math.sqrt(error_x**2 + error_y**2)

                    xy_aligned = xy_error < px_error_threshold
                    angle_error = abs(horizontal_dist - alt)
                    aligned_angle = angle_error < angle_threshold

                    # Approximate line-of-sight distance

                    print(f"Normal Distance: {slant_dist:.2f} m Horizontal Distance: {horizontal_dist:.2f} m View Angle: {np.degrees(view_angle):.2f} deg")
                    aligned = (abs(x_angle) < math.radians(2.0) and abs(theta2 - theta) < math.radians(1.0))
                    if not xy_aligned:
                        rc_override(vehicle,rc_roll,rc_pitch,rc_throttle)
                        print(f"RC Roll={rc_roll}, Pitch={rc_pitch}, Throttle={rc_throttle}, Altitude={alt:.2f} m")
                    elif not aligned_angle:
                        rc_override(vehicle,rc_roll,rc_pitch,rc_throttle)
                        print(f"RC Roll={rc_roll}, Pitch={rc_pitch}, Throttle={rc_throttle}, Altitude={alt:.2f} m")

                    if horizontal_dist < 40:
                        VehicleMode(vehicle, "STABILIZE")
                        rc_override(vehicle, rc_roll,rc_pitch,1200)
                        print("Target aligned and landing!")
                        
            else:
                tracking = False
                tracker = None
                print("Tracking lost")

        cv2.imshow("Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()