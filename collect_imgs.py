import os
import cv2

# Data folder
DATA_DIR = './data'

# Labels A to Z + SPACE
labels = [chr(i) for i in range(ord('A'), ord('Z')+1)] + ['SPACE']

# Create folders if not exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

for label in labels:
    path = os.path.join(DATA_DIR, label)
    if not os.path.exists(path):
        os.makedirs(path)

# Use phone cam (change index if needed)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

current_label = None
capture = False
counter = 0

print("\nControls:")
print("Press A–Z to select letter class")
print("Press 0 (zero) to select SPACE class")
print("Press SPACEBAR to start/stop capture")
print("Press ESC to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)

    # Show info overlay
    status_color = (0, 255, 100) if capture else (100, 100, 255)
    status_text = "● CAPTURING" if capture else "○ PAUSED"

    if current_label:
        cv2.putText(frame, f'Label: {current_label}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Images: {counter}', (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)
        cv2.putText(frame, status_text, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    else:
        cv2.putText(frame, "Select a class (A-Z or 0 for SPACE)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Capture images
    if capture and current_label:
        img_path = os.path.join(DATA_DIR, current_label, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        counter += 1

    # Instructions on screen
    cv2.putText(frame, "A-Z / 0: Select | SPACE: Start/Stop | ESC: Quit",
                (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Data Collection', frame)

    key = cv2.waitKey(1)

    # ESC to quit
    if key == 27:
        print("Exiting...")
        break

    # SPACEBAR to toggle capture
    elif key == 32:
        if current_label is None:
            print("Please select a label first (A-Z or 0 for SPACE)")
        else:
            capture = not capture
            print(f"Capture {'STARTED' if capture else 'STOPPED'} for class '{current_label}' — {counter} images so far")

    # 0 key for SPACE label
    elif key == ord('0'):
        current_label = 'SPACE'
        folder = os.path.join(DATA_DIR, 'SPACE')
        counter = len(os.listdir(folder))
        capture = False
        print(f"Switched to class SPACE, current images: {counter}")

    # A–Z keys
    else:
        for label in labels[:-1]:  # exclude SPACE from this loop
            if key == ord(label.lower()):
                current_label = label
                folder = os.path.join(DATA_DIR, label)
                counter = len(os.listdir(folder))
                capture = False
                print(f"Switched to class {label}, current images: {counter}")

cap.release()
cv2.destroyAllWindows()
