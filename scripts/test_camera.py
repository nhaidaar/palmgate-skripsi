#!/usr/bin/env python3
import cv2
import sys

device = sys.argv[1] if len(sys.argv) > 1 else "/dev/video0"
print(f"Testing {device}...")

cap = cv2.VideoCapture(device)
if not cap.isOpened():
    print(f"✗ Failed to open {device}")
    sys.exit(1)

ret, frame = cap.read()
cap.release()

if ret and frame is not None:
    h, w = frame.shape[:2]
    print(f"✓ {device} works — {w}x{h}")
else:
    print(f"✗ {device} opened but no frame")
    sys.exit(1)
