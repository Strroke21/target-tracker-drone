import cv2
import numpy as np
from pymavlink import mavutil

# -------------------------------------
# Mouse callback: click to select target
# -------------------------------------
target_point = None
new_target = False

# CAMERA PARAMETERS ----------------------------------------------
W = 720
H = 480
HFOV = np.deg2rad(87)   # horizontal FOV in radians
VFOV = np.deg2rad(58)   # vertical FOV               # height above ground (meters)

# Focal lengths in pixels
fx = W / (2 * np.tan(HFOV / 2))
fy = H / (2 * np.tan(VFOV / 2))

cx_img = W / 2
cy_img = H / 2
max_pitch = np.deg2rad(45) # max tilt angle of drone (radians)
max_roll = np.deg2rad(45) # max roll angle of drone (radians)
throttle_hover = 1500    # hover throttle
throttle_approach = 1300
pwm_center = 1500
px_error_threshold = 0.1

Kp_x = max_roll*0.9
Kp_y = max_pitch*0.9
terminal_altitude = 2

def connect(connection_string):
    vehicle = mavutil.mavlink_connection(connection_string)
    vehicle.wait_heartbeat()
    return vehicle

def enable_data_stream(vehicle, stream_rate):
    vehicle.mav.request_data_stream_send(vehicle.target_system,
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
    # Pixel → camera plane normalization
    x_norm = (cx - cx_img) / fx
    y_norm = (cy - cy_img) / fy

    # Project onto ground plane Z = –h (camera pointing down)
    X = x_norm * alt
    Y = y_norm * alt
    Z = -alt   # ground plane

    # Orientation of LOS vector
    yaw = np.arctan2(Y, X)
    pitch = np.arctan(np.sqrt(X*X + Y*Y) / alt)
    roll = 0.0

    return (X, Y, Z), (roll, pitch, yaw)

def mouse_click(event, x, y, flags, param):
    global target_point, new_target
    if event == cv2.EVENT_LBUTTONDOWN:
        target_point = np.array([[x, y]], dtype=np.float32)
        new_target = True
        print(f"Target selected at: {x}, {y}")

def cmd_to_rc(cmd):
    cmd = np.clip(cmd, -1, 1)
    return int(1500 + 400 * cmd)   # 1100..1900

def rc_override(vehicle, roll_pwm, pitch_pwm, throttle_pwm):
    vehicle.mav.rc_channels_override_send(
        vehicle.target_system,
        vehicle.target_component,
        roll_pwm,    # RC1: Roll
        pitch_pwm,   # RC2: Pitch
        throttle_pwm,# RC3: Throttle
        1500,        # RC4: Yaw
        0, 0, 0, 0   # RC5-RC8: not used
    )

def VehicleMode(vehicle, mode):
    modes = ["STABILIZE","ACRO","ALT_HOLD","AUTO","GUIDED","LOITER","RTL","CIRCLE","","LAND"]
    mode_id = modes.index(mode) if mode in modes else 12
    vehicle.mav.set_mode_send(vehicle.target_system,
                              mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                              mode_id)
    
def give_throttle(vehicle, throttle_pwm):
    while True:
        alt = get_rangefinder_data(vehicle)
        print(f"Current Altitude: {alt} m")
        if alt<=10:
            rc_override(vehicle, 1500, 1500, throttle_pwm)
        else:
            break

def arm(vehicle):
    #arm the drone
    vehicle.mav.command_long_send(vehicle.target_system, vehicle.target_component,mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
    

# -------------------------------------
# Main Tracker
# -------------------------------------
def main():

    vehicle = connect('tcp:127.0.0.1:5773')
    enable_data_stream(vehicle, 100)

    cap = cv2.VideoCapture(4)#"/home/deathstroke/Downloads/bikers.mp4")
    if not cap.isOpened():
        print("Error opening video")
        return

    cv2.namedWindow("Tracker")
    cv2.setMouseCallback("Tracker", mouse_click)

    # LK optical flow params
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    ret, prev_frame = cap.read()
    prev_frame = cv2.resize(prev_frame, (720, 480))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    global target_point, new_target
    tracking = False
    point_prev = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (720, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        alt = get_local_position(vehicle)[2]

        # -------------------------------------
        # New target clicked
        # -------------------------------------
        if new_target:
            point_prev = target_point.reshape(-1, 1, 2)
            tracking = True
            new_target = False

        # -------------------------------------
        # Track selected point
        # -------------------------------------
        if tracking and point_prev is not None:
            point_next, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray,
                point_prev,
                None,
                **lk_params
            )

            if status[0] == 1:  # good tracking
                x, y = point_next[0, 0]
                cx, cy = int(x), int(y)

                # Draw crosshair
                size_x = 720
                size_y = 480
                color = (0, 0, 255)
                thickness = 2
                cv2.line(frame, (cx - size_x, cy), (cx + size_x, cy), color, thickness)
                cv2.line(frame, (cx, cy - size_y), (cx, cy + size_y), color, thickness)

                point_prev = point_next

                Pose, Orientation = compute_pose(cx, cy, alt)
                print(f"Pose: {Pose}, Orientation: {Orientation}")

                error_x = (cx - cx_img)/cx_img
                error_y = (cy - cy_img)/cy_img

                pitch_cmd = -Kp_x * error_x
                roll_cmd = Kp_y * error_y

                rc_roll = cmd_to_rc(roll_cmd)
                rc_pitch = cmd_to_rc(pitch_cmd)
                print(f"RC Roll={rc_roll}, Pitch={rc_pitch}")
                print(f"RC Commands - Roll: {rc_roll}, Pitch: {rc_pitch}")
                aligned = (abs(error_x) < px_error_threshold) and (abs(error_y) < px_error_threshold)
                if (alt > terminal_altitude):
                    if not aligned:
                        rc_override(vehicle, rc_roll, rc_pitch,pwm_center)
                        print("aligning to target...")
                    elif aligned:
                        rc_override(vehicle, rc_roll, rc_pitch, throttle_approach)
                        print("target aligned!")

                else:
                    rc_override(vehicle, pwm_center, pwm_center, throttle_approach)
                    print("target final approach...")

            else:
                tracking = False  # lost target

        cv2.imshow("Tracker", frame)
        prev_gray = gray.copy()

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
