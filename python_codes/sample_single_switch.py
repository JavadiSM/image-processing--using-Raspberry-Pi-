import RPi.GPIO as GPIO
import time
import threading

GPIO.setmode(GPIO.BCM)
switch_pin = 18 
GPIO.setup(switch_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

def check_switch():
    while True:
        while GPIO.input(switch_pin) == GPIO.LOW:
            time.sleep(0.01)
        press_time = time.time()
        while GPIO.input(switch_pin) == GPIO.HIGH:
            time.sleep(0.01)
        hold_time = time.time() - press_time
        if hold_time >= 1:
            print("mode 2")
        else:
            print("mode 1")
        time.sleep(0.1)

thread = threading.Thread(target=check_switch)
thread.daemon = True
thread.start()

try:
    while True:
        print("Main program is running...")
        time.sleep(1)
except KeyboardInterrupt:
    GPIO.cleanup()
    print("Exiting program.")
