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
HFOV = np.deg2rad(87)
VFOV = np.deg2rad(58)

fx = W / (2 * np.tan(HFOV / 2))
fy = H / (2 * np.tan(VFOV / 2))

cx_img = W / 2
cy_img = H / 2

max_pitch = np.deg2rad(45)
max_roll  = np.deg2rad(45)
target_alt = 10
cam_orientation = 0 #0 downward, 1 45deg forward

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
    throttle_descent = 1400
    throttle_approach = 1000
    pwm_center = 1500
    Kp_x = max_roll*0.4
    Kp_y = max_pitch*0.4
    throttle_takeoff = 1750
    mode = "ALT_HOLD"
    terminal_altitude = 5
    px_error_threshold = 0.2

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
# ================================================================
# MAIN TRACKER LOOP
# ================================================================
def main():
    vehicle = connect('tcp:127.0.0.1:5773')
    enable_data_stream(vehicle, 100)
    VehicleMode(vehicle, mode)
    arm(vehicle)
    time.sleep(2)
    give_throttle(vehicle, throttle_takeoff)

    # ---- ROS2 SETUP ----
    rclpy.init()
    node = ImageSubscriber()

    # Spin ROS in background thread
    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    # ---- Wait for first frame ----
    global latest_frame
    print("Waiting for /camera/image_raw ...")
    while latest_frame is None:
        pass

    frame = latest_frame
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow("Tracker")
    cv2.setMouseCallback("Tracker", mouse_click)

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    global target_point, new_target
    tracking = False
    point_prev = None

    # =========================================================
    # ---------------------- MAIN LOOP -------------------------
    # =========================================================
    while rclpy.ok():

        # If no new frame, continue loop
        if latest_frame is None:
            continue
        frame = latest_frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        alt = current_alt(vehicle)
        # print(f"Current Altitude: {alt} m")
        rc_override(vehicle, pwm_center, pwm_center, pwm_center)

        # New target
        if new_target:
            point_prev = target_point.reshape(-1, 1, 2)
            tracking = True
            new_target = False

        # Optical flow
        if tracking and point_prev is not None:

            point_next, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray,
                point_prev,
                None,
                **lk_params
            )

            if status[0] == 1:
                x, y = point_next[0, 0]
                cx = int(x)
                cy = int(y)

                # Crosshair
                size_x = 200
                size_y = 150
                color = (255, 0, 255) 
                thickness = 2
                cv2.line(frame, (cx - size_x, cy), (cx + size_x, cy), color, thickness)
                cv2.line(frame, (cx, cy - size_y), (cx, cy + size_y), color, thickness)

                point_prev = point_next

                Pose, Orientation = compute_pose(cx, cy, alt)
                print(f"Pose: {Pose}, Orientation: {Orientation}")

                if cam_orientation == 0:
                    error_x = (cx - cx_img)/cx_img
                    error_y = (cy - cy_img)/cy_img

                    pitch_cmd = (Kp_y * error_y)
                    roll_cmd = (Kp_x * error_x)

                    rc_roll = cmd_to_rc(roll_cmd)
                    rc_pitch = cmd_to_rc(pitch_cmd)

                print(f"RC Roll={rc_roll}, Pitch={rc_pitch}")
                aligned = (abs(error_x) < px_error_threshold) and (abs(error_y) < px_error_threshold)  

                if (alt > terminal_altitude):
                    if not aligned:
                        rc_override(vehicle, rc_roll, rc_pitch,pwm_center)
                        print("aligning to target...")
                    elif aligned:
                        rc_override(vehicle, 0, 0, 1000)
                        print("target aligned!")

                else:
                    rc_override(vehicle, pwm_center, pwm_center, throttle_approach)
                    print("target final approach...")

            else:
                tracking = False

        cv2.imshow("Tracker",frame)
        prev_gray = gray.copy()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
