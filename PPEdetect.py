from ultralytics import YOLO
import cv2

# Load PPE model
model = YOLO("ppe.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, verbose=False)

    helmet_missing = False
    vest_missing = False

    for r in results:
        boxes = r.boxes

        for box in boxes:
            class_id = int(box.cls)
            label = model.names[class_id]

            # Ignore mask completely
            if label == "NO-Mask":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Check helmet
            if label == "NO-Hardhat":
                helmet_missing = True
                color = (0,0,255)

            # Check vest
            elif label == "NO-Safety Vest":
                vest_missing = True
                color = (0,0,255)

            else:
                color = (0,255,0)

            # Draw bounding box
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,color,2)

    # PPE decision
    if helmet_missing or vest_missing:
        text = "PPE VIOLATION"
        status_color = (0,0,255)
    else:
        text = "SAFE"
        status_color = (0,255,0)

    # Show status
    cv2.putText(frame,text,(40,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,status_color,3)

    cv2.imshow("Trinetra PPE Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()