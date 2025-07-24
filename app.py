from flask import Flask, render_template, Response
import cv2
import os
from deepface import DeepFace

app = Flask(__name__)

def generate_frames():
    cap = cv2.VideoCapture(0)  # Open the webcam
    if not cap.isOpened():
        print("Error: Unable to access the camera")
        return

    while True:
        success, frame = cap.read()  # Read a frame from the webcam
        if not success:
            break
        
        try:
            # Analyze the emotions of the frame
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            
            # Add emotion text overlay to the frame
            cv2.putText(frame, f'Emotion: {emotion}', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except Exception as e:
            print("Error with emotion detection:", e)
        
        # Convert the frame to JPEG format for the web stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as part of the MJPEG stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
