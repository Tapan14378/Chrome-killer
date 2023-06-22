import cv2
import mediapipe as mp
import numpy as np
import subprocess
import time

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands.Hands()
        self.cap = cv2.VideoCapture(0)
        self.gesture_state = 0
        self.last_trigger_time = 0
        self.cooldown_period = 2  # Cooldown period in seconds

    def detect_hands(self):
        success, image = self.cap.read()
        if success:
            # Convert the image to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect the hands in the image.
            hand_results = self.mp_hands.process(image)

            # If hands are detected, return the coordinates of the hand points.
            if hand_results.multi_hand_landmarks:
                return hand_results.multi_hand_landmarks
            else:
                return None

    def recognize_gesture(self, hand_landmarks):
        # Check if the hand landmarks are None.
        if hand_landmarks is None:
            return 0

        # Get the coordinates of the thumb tip and the index, middle, ring, and pinky MCP (first joint).
        thumb_tip = hand_landmarks[0].landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
        index_mcp = hand_landmarks[0].landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
        middle_mcp = hand_landmarks[0].landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_mcp = hand_landmarks[0].landmark[mp.solutions.hands.HandLandmark.RING_FINGER_MCP]
        pinky_mcp = hand_landmarks[0].landmark[mp.solutions.hands.HandLandmark.PINKY_MCP]

        # Check if the thumb is extended and the other fingers are closed.
        if thumb_tip.x < index_mcp.x and thumb_tip.x < middle_mcp.x and thumb_tip.x < ring_mcp.x and thumb_tip.x < pinky_mcp.x:
            return 1
        else:
            return 0

    def assign_tasks(self, gesture):
        current_time = time.time()

        if gesture == 1 and self.gesture_state == 0:
            # Gesture detected and not in gesture state
            if current_time - self.last_trigger_time >= self.cooldown_period:
                print("Closing Chrome browser...")
                subprocess.call(["taskkill", "/IM", "chrome.exe"])
                self.last_trigger_time = current_time
                self.gesture_state = 1
        elif gesture == 0 and self.gesture_state == 1:
            # Gesture not detected and in gesture state
            self.gesture_state = 0

def main():
    hand_detector = HandDetector()

    while True:
        hand_landmarks = hand_detector.detect_hands()
        gesture = hand_detector.recognize_gesture(hand_landmarks)
        hand_detector.assign_tasks(gesture)

if __name__ == "__main__":
    main()
