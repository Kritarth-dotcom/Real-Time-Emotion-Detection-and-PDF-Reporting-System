import os
import cv2
from fer import FER
import time
from fpdf import FPDF
from collections import Counter

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize the FER detector
emotion_detector = FER(mtcnn=True)

# Set the desired frame width and height (smaller values for better performance)
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Frame skip settings to reduce lag (process every 2nd frame)
frame_skip = 2
frame_count = 0

# Duration of scanning (in seconds)
scan_duration = 60  # 1 minute
start_time = time.time()

# Data collection
emotion_data = []

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip this frame

    # Resize the frame for faster processing
    small_frame = cv2.resize(frame, (frame_width, frame_height))

    # Detect emotions in the resized frame
    emotions = emotion_detector.detect_emotions(small_frame)

    # Loop through detected faces and their corresponding emotions
    for face in emotions:
        # Get the bounding box for the face
        (x, y, w, h) = face['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Get the predicted emotion and its confidence score
        dominant_emotion, score = emotion_detector.top_emotion(small_frame)

        # Only display if a valid emotion and score were found
        if dominant_emotion and score is not None:
            # Display the emotion on the frame
            cv2.putText(frame, f"{dominant_emotion} ({score*100:.2f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            # Collect data for the report
            emotion_data.append({
                'time': time.strftime("%H:%M:%S", time.localtime()),
                'emotion': dominant_emotion,
                'confidence': f"{score*100:.2f}%"
            })
        else:
            cv2.putText(frame, "No Emotion Detected", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    # Display the frame with emotion predictions
    cv2.imshow("Emotion Detection", frame)

    # Break the loop after 1 minute
    if time.time() - start_time > scan_duration:
        break

    # Break the loop if 'Esc' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

# Generate the PDF report
pdf = FPDF()

# Adding a page
pdf.add_page()

# Set title
pdf.set_font("Arial", 'B', 16)
pdf.cell(200, 10, "Emotion Detection Report", ln=True, align='C')

# Add summary details
pdf.ln(10)
pdf.set_font("Arial", '', 12)
pdf.multi_cell(0, 10, '''
The following report documents the real-time emotion detection performed over the last minute.
The system detects the following emotions and their corresponding confidence scores:
''')

# Add table of detected emotions
pdf.ln(5)
for entry in emotion_data:
    pdf.cell(0, 10, f"Time: {entry['time']}, Emotion: {entry['emotion']}, Confidence: {entry['confidence']}", ln=True)

# Analyze the emotion data for the summary and suggestions
if emotion_data:
    emotion_counts = Counter([entry['emotion'] for entry in emotion_data])
    most_common_emotion, most_common_count = emotion_counts.most_common(1)[0]
    avg_confidence = sum(float(entry['confidence'][:-1]) for entry in emotion_data) / len(emotion_data)
else:
    most_common_emotion = "None"
    most_common_count = 0
    avg_confidence = 0

# Add summary at the end of the report
pdf.ln(10)
pdf.set_font("Arial", 'B', 14)
pdf.cell(200, 10, "Summary", ln=True, align='C')

pdf.ln(5)
pdf.set_font("Arial", '', 12)
pdf.multi_cell(0, 10, f'''
- Most frequently detected emotion: {most_common_emotion}
- Number of times this emotion was detected: {most_common_count}
- Average confidence score of detected emotions: {avg_confidence:.2f}% 
''')

# Add suggestions for improvement based on emotion detection data
pdf.ln(10)
pdf.set_font("Arial", 'B', 14)
pdf.cell(200, 10, "Suggestions for Improvement", ln=True, align='C')

pdf.ln(5)
pdf.set_font("Arial", '', 12)

if most_common_emotion in ["angry", "sad", "disgust", "fear"]:
    pdf.multi_cell(0, 10, f'''
It was observed that the dominant emotion detected was {most_common_emotion}, which could indicate underlying stress or frustration.
To improve emotional balance and well-being, it is recommended to:
- Practice mindfulness techniques like deep breathing or meditation during breaks.
- Focus on work-life balance and ensure that adequate rest is taken.
- Consider seeking support or feedback from colleagues or supervisors when tasks become challenging.
    ''')
elif most_common_emotion == "happy":
    pdf.multi_cell(0, 10, '''
The predominant emotion detected was happiness, which is a great indicator of a positive work environment. To maintain this level of emotional well-being:
- Continue fostering a positive attitude and maintain good communication with colleagues.
- Share any successful strategies you use for managing stress with your team.
- Keep up the good work and strive to balance workload while sustaining this level of emotional health.
    ''')
else:
    pdf.multi_cell(0, 10, '''
No strong emotional trends were detected. It may be helpful to:
- Identify any external factors that could be impacting emotional stability.
- Ensure that you take time to reflect on your emotions throughout the workday.
- Practice techniques such as mindfulness to stay emotionally centered.
    ''')

# Save the PDF report with suggestions
report_file = "emotion_detection_report_with_suggestions.pdf"
pdf.output(report_file)

print(f"Report saved as {report_file}")

# Open the PDF report automatically after saving
os.system(f"start {report_file}")  # For Windows
# Use "open" for MacOS, or "xdg-open" for Linux systems.
