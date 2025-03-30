import cv2
import numpy as np
import time
import board
import busio
from adafruit_pca9685 import PCA9685
from simple_pid import PID

# Constants
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CONTOUR_AREA_THRESHOLD = 5000
SERVO_YAW_CHANNEL = 1
SERVO_PITCH_CHANNEL = 0
MIN_ANGLE = 0
MAX_ANGLE = 180
CENTER_YAW = 90
CENTER_PITCH = 90
IDLE_TIME = 10
RETURN_STEP = 2
RETURN_DELAY = 0.05
ALPHA = 0.1
PWM_FREQUENCY = 50
SERVO_MIN_PULSE = 500
SERVO_MAX_PULSE = 2500

# Initialize camera
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("❌ Camera not detected!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Initialize PCA9685 PWM Driver
i2c = busio.I2C(board.SCL, board.SDA)
pwm = PCA9685(i2c)
pwm.frequency = PWM_FREQUENCY

# PID controllers
yaw_pid = PID(0.1, 0.01, 0.05, setpoint=CENTER_YAW)
pitch_pid = PID(0.1, 0.01, 0.05, setpoint=CENTER_PITCH)
yaw_pid.output_limits = (MIN_ANGLE, MAX_ANGLE)
pitch_pid.output_limits = (MIN_ANGLE, MAX_ANGLE)

# Variables
last_movement_time = time.time()
hand_detected = False
yaw_angle = CENTER_YAW
pitch_angle = CENTER_PITCH
camera_error_count = 0

def smooth_value(prev, new):
    return ALPHA * new + (1 - ALPHA) * prev

def angle_to_pulse(angle):
    return int(SERVO_MIN_PULSE + (angle / 180) * (SERVO_MAX_PULSE - SERVO_MIN_PULSE))

def set_servo_angle(channel, angle):
    angle = max(MIN_ANGLE, min(MAX_ANGLE, angle))
    pulse = angle_to_pulse(angle)
    duty_cycle = int((pulse / 20000) * 65535)
    pwm.channels[channel].duty_cycle = duty_cycle

def return_to_center():
    global yaw_angle, pitch_angle, last_movement_time, hand_detected
    if time.time() - last_movement_time > IDLE_TIME and not hand_detected:
        while abs(yaw_angle - CENTER_YAW) > RETURN_STEP or abs(pitch_angle - CENTER_PITCH) > RETURN_STEP:
            yaw_angle = smooth_value(yaw_angle, CENTER_YAW)
            pitch_angle = smooth_value(pitch_angle, CENTER_PITCH)
            set_servo_angle(SERVO_YAW_CHANNEL, yaw_angle)
            set_servo_angle(SERVO_PITCH_CHANNEL, pitch_angle)
            time.sleep(RETURN_DELAY)
        set_servo_angle(SERVO_YAW_CHANNEL, CENTER_YAW)
        set_servo_angle(SERVO_PITCH_CHANNEL, CENTER_PITCH)
        time.sleep(0.5)
        pwm.channels[SERVO_YAW_CHANNEL].duty_cycle = 0
        pwm.channels[SERVO_PITCH_CHANNEL].duty_cycle = 0

# Center servos at startup
set_servo_angle(SERVO_YAW_CHANNEL, CENTER_YAW)
set_servo_angle(SERVO_PITCH_CHANNEL, CENTER_PITCH)
time.sleep(1)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Camera error!")
            camera_error_count += 1
            if camera_error_count > 10:
                return_to_center()
                break
            continue
        camera_error_count = 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        edges = cv2.Canny(blurred, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_contour) > CONTOUR_AREA_THRESHOLD:
                hand_detected = True
                last_movement_time = time.time()
                x, y, w, h = cv2.boundingRect(max_contour)
                cx, cy = x + w // 2, y + h // 2

                yaw_pid.setpoint = np.interp(cx, [0, CAMERA_WIDTH], [MAX_ANGLE, MIN_ANGLE])
                pitch_pid.setpoint = np.interp(cy, [0, CAMERA_HEIGHT], [MAX_ANGLE, MIN_ANGLE])

                yaw_angle = yaw_pid(yaw_angle)
                pitch_angle = pitch_pid(pitch_angle)

                set_servo_angle(SERVO_YAW_CHANNEL, yaw_angle)
                set_servo_angle(SERVO_PITCH_CHANNEL, pitch_angle)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            else:
                hand_detected = False
        else:
            hand_detected = False

        return_to_center()

        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Shutting down...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    pwm.deinit()
    print("Resources released.")



