from flask import Flask, jsonify, render_template, request, Response, redirect, url_for
import torch
import os
import threading
import pyttsx3
from PIL import Image
import cv2
import requests
import re
import time
import speech_recognition as sr
import json
from model_loader import ensure_models_loaded

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize models - will be loaded on first request
yolo = None
processor = None
model = None

last_spoken = ""
speak_lock = threading.Lock()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyAOsNr4W61u6eAs-szc0d2_urLle9AFedc")

# Global flag to control object detection
detection_paused = False

# Lazy loading function for models
def get_models():
    global yolo, processor, model
    if yolo is None or processor is None or model is None:
        yolo, processor, model = ensure_models_loaded()
    return yolo, processor, model

def speak_text(text):
    global last_spoken
    with speak_lock:
        if text != last_spoken:
            last_spoken = text
            def _speak():
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            threading.Thread(target=_speak).start()

def speak_navigation_steps(steps):
    for step in steps:
        speak_text(step)
        time.sleep(20)  

def listen_for_command():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    try:
        with mic as source:
            speak_text("Please say your command clearly after the beep.")
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
        
        try:
            command = recognizer.recognize_google(audio).lower()
            speak_text(f"I heard: {command}")
            return command
        except sr.UnknownValueError:
            speak_text("Sorry, I could not understand. Please try again.")
            return None
        except sr.RequestError:
            speak_text("Could not request results from Google Speech Recognition.")
            return None
    except Exception as e:
        speak_text("An error occurred while listening. Please try again.")
        print(f"Listening error: {e}")
        return None

def generate_caption(image_path):
    """Generate detailed descriptions for images by combining captioning model with object detection."""
    # Load models if not already loaded
    yolo, processor, model = get_models()
    
    # Load and process the image
    image = Image.open(image_path).convert('RGB')
    
    # Get the base caption from BLIP
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs, max_length=100)  # Increased max length for more detail
    base_caption = processor.decode(out[0], skip_special_tokens=True)
    
    # Run object detection to get detailed object information
    image_cv = cv2.imread(image_path)
    results = yolo(image_cv)
    detections = results.pandas().xyxy[0]
    
    # Extract object counts and positions
    objects = {}
    for _, row in detections.iterrows():
        label = row['name']
        if label in objects:
            objects[label] += 1
        else:
            objects[label] = 1

    # Build a detailed description
    detailed_desc = base_caption.capitalize()
    
    # Add object detection info if available
    if objects:
        detailed_desc += ". I can detect the following objects: "
        object_descriptions = []
        
        for obj, count in objects.items():
            if count == 1:
                object_descriptions.append(f"one {obj}")
            else:
                object_descriptions.append(f"{count} {obj}s")
        
        detailed_desc += ", ".join(object_descriptions)
    
    # Add positioning context when there are multiple objects
    if len(objects) > 1:
        # Get image dimensions for relative positioning
        height, width = image_cv.shape[:2]
        
        # Add spatial relationships for up to 3 prominent objects
        top_objects = sorted([(label, row) for _, row in detections.iterrows()], 
                            key=lambda x: (x[1]['xmax'] - x[1]['xmin']) * (x[1]['ymax'] - x[1]['ymin']), 
                            reverse=True)[:3]
        
        if top_objects:
            detailed_desc += ". For location context: "
            location_desc = []
            
            for label, row in top_objects:
                x_center = (row['xmin'] + row['xmax']) / 2
                y_center = (row['ymin'] + row['ymax']) / 2
                
                # Determine horizontal position
                if x_center < width / 3:
                    h_pos = "on the left side"
                elif x_center < 2 * width / 3:
                    h_pos = "in the center"
                else:
                    h_pos = "on the right side"
                
                # Determine vertical position
                if y_center < height / 3:
                    v_pos = "at the top"
                elif y_center < 2 * height / 3:
                    v_pos = "in the middle"
                else:
                    v_pos = "at the bottom"
                
                location_desc.append(f"the {label} is {h_pos} {v_pos}")
            
            detailed_desc += ", ".join(location_desc)
    
    # Add a general scene context based on detected objects
    scene_contexts = {
        "person": ["There appear to be people in this scene", "This looks like a scene with people"],
        "car": ["This might be taken outdoors, possibly on a street or parking area"],
        "chair": ["This might be an indoor setting, possibly a room or office"],
        "dining table": ["This appears to be in a dining area or kitchen"],
        "tv": ["This seems to be in a living room or entertainment area"],
        "book": ["This might be in a study, library, or reading area"]
    }
    
    for key_object in scene_contexts:
        if key_object in objects and "might be" not in detailed_desc:
            detailed_desc += f". {scene_contexts[key_object][0]}"
            break
    
    return detailed_desc

def gen_frames():
    global detection_paused
    # Load models if not already loaded
    yolo, _, _ = get_models()
    
    cap = cv2.VideoCapture(0)
    previous_labels = set()
    frame_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        
        # Only perform detection if not paused and on every 10th frame
        if not detection_paused and frame_count % 10 == 0:
            results = yolo(frame)
            detections = results.pandas().xyxy[0]
            current_labels = set()
            objects_with_positions = []
            
            height, width = frame.shape[:2]

            for _, row in detections.iterrows():
                label = row['name']
                current_labels.add(label)
                x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Determine position in frame
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                
                # Simple position description
                h_pos = "left" if x_center < width/2 else "right"
                v_pos = "top" if y_center < height/2 else "bottom"
                
                position = f"{v_pos} {h_pos}"
                objects_with_positions.append(f"{label} in the {position}")
            
            # If there are new objects detected
            if current_labels != previous_labels:
                if current_labels:
                    # Create a descriptive message about what's visible
                    if len(current_labels) <= 3:
                        # For fewer objects, include position information
                        message = f"I can see: {', '.join(objects_with_positions)}"
                    else:
                        # For many objects, just list them
                        object_list = ", ".join(current_labels)
                        message = f"Multiple objects detected: {object_list}"
                    
                    # Add caution prefix for certain objects
                    caution_objects = ["person", "car", "truck", "bicycle", "motorcycle"]
                    if any(obj in current_labels for obj in caution_objects):
                        message = "Caution! " + message
                        
                    speak_text(message)
                previous_labels = current_labels
        elif not detection_paused:
            # Just draw the previous detections without reprocessing
            for label in previous_labels:
                cv2.putText(frame, f"Detected: {label}", (50, 50 + list(previous_labels).index(label)*30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            # Add a visual indicator that detection is paused
            cv2.putText(frame, "Detection Paused", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/', methods=['GET', 'POST'])
def index():
    caption = ""
    image_path = ""
    if request.method == 'POST':
        if 'image' in request.files:
            img = request.files['image']
            if img.filename != '':
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
                img.save(image_path)
                caption = generate_caption(image_path)
                threading.Thread(target=speak_text, args=(caption,)).start()
        elif request.form.get('capture') == 'Capture from Webcam':
            cap = cv2.VideoCapture(0)
            time.sleep(1)
            ret, frame = cap.read()
            cap.release()
            if ret:
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'captured.jpg')
                cv2.imwrite(image_path, frame)
                caption = generate_caption(image_path)
                threading.Thread(target=speak_text, args=(caption,)).start()
    return render_template('index.html', caption=caption, image_path=image_path)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live')
def live():
    return render_template('live.html')

@app.route('/navigate', methods=['POST'])
def navigate():
    source = request.form['source']
    destination = request.form['destination']

    directions_url = f"https://maps.googleapis.com/maps/api/directions/json?origin={source}&destination={destination}&mode=walking&key={GOOGLE_API_KEY}"
    response = requests.get(directions_url)
    steps = []
    navigation_instructions = []
    
    if response.status_code == 200:
        data = response.json()
        if data['routes']:
            for leg in data['routes'][0]['legs']:
                for step in leg['steps']:
                    html_instruction = step['html_instructions']
                    text_instruction = re.sub('<[^<]+?>', '', html_instruction)
                    steps.append(text_instruction)
                    # Add distance and duration if available
                    if 'distance' in step and 'text' in step['distance']:
                        distance_text = step['distance']['text']
                        text_instruction += f" ({distance_text})"
                    navigation_instructions.append({
                        'instruction': text_instruction,
                        'distance': step.get('distance', {}).get('text', ''),
                        'duration': step.get('duration', {}).get('text', '')
                    })
        else:
            steps.append("No navigation routes found.")
            navigation_instructions.append({
                'instruction': "No navigation routes found.",
                'distance': '',
                'duration': ''
            })
    else:
        steps.append("Unable to fetch navigation directions.")
        navigation_instructions.append({
            'instruction': "Unable to fetch navigation directions.",
            'distance': '',
            'duration': ''
        })

    threading.Thread(target=speak_navigation_steps, args=(steps,)).start()
    map_url = f"https://www.google.com/maps/embed/v1/directions?key={GOOGLE_API_KEY}&origin={source}&destination={destination}&mode=walking"
    speak_text("Navigation started. Follow the route instructions. You will be alerted about obstacles ahead.")
    return render_template('navigation_result.html', map_url=map_url, navigation_instructions=navigation_instructions)

@app.route('/smart_navigation')
def smart_navigation_page():
    return render_template("smart_navigation.html")

@app.route('/smart_navigate', methods=['POST'])
def smart_navigate():
    destination = request.form.get('destination')
    user_lat = request.form.get('latitude')
    user_lng = request.form.get('longitude')

    if not destination or not user_lat or not user_lng:
        error = "Missing destination or location coordinates. Please allow location access and try again."
        return render_template("smart_navigation.html", map_url=None, error=error)

    source = f"{user_lat},{user_lng}"
    directions_url = f"https://maps.googleapis.com/maps/api/directions/json?origin={source}&destination={destination}&mode=walking&key={GOOGLE_API_KEY}"
    response = requests.get(directions_url)

    steps = []
    navigation_instructions = []
    
    if response.status_code == 200:
        data = response.json()
        if data['routes']:
            for leg in data['routes'][0]['legs']:
                for step in leg['steps']:
                    html_instruction = step['html_instructions']
                    text_instruction = re.sub('<[^<]+?>', '', html_instruction)
                    steps.append(text_instruction)
                    # Add distance and duration if available
                    if 'distance' in step and 'text' in step['distance']:
                        distance_text = step['distance']['text']
                        text_instruction += f" ({distance_text})"
                    navigation_instructions.append({
                        'instruction': text_instruction,
                        'distance': step.get('distance', {}).get('text', ''),
                        'duration': step.get('duration', {}).get('text', '')
                    })
        else:
            steps.append("No navigation routes found.")
            navigation_instructions.append({
                'instruction': "No navigation routes found.",
                'distance': '',
                'duration': ''
            })
    else:
        steps.append("Unable to fetch navigation directions.")
        navigation_instructions.append({
            'instruction': "Unable to fetch navigation directions.",
            'distance': '',
            'duration': ''
        })

    threading.Thread(target=speak_navigation_steps, args=(steps,)).start()
    speak_text("Smart navigation started from your current location. Please follow the voice guidance.")

    map_url = f"https://www.google.com/maps/embed/v1/directions?key={GOOGLE_API_KEY}&origin={source}&destination={destination}&mode=walking"
    return render_template("smart_navigation.html", map_url=map_url, error=None, navigation_instructions=navigation_instructions)

@app.route('/voice_command', methods=['POST'])
def voice_command():
    command = listen_for_command()
    if command:
        # Image processing commands
        if "upload image" in command or "upload photo" in command:
            speak_text("Opening image upload.")
            return redirect(url_for('index'))
        elif "capture from webcam" in command or "take picture" in command or "take photo" in command:
            speak_text("Capturing from webcam.")
            return redirect(url_for('index', capture='Capture from Webcam'))
        
        # Object detection commands
        elif "start live object detection" in command or "detect objects" in command or "live detection" in command:
            speak_text("Starting live object detection.")
            return redirect(url_for('live'))
        
        # Navigation commands
        elif "start navigation" in command or "navigate" in command:
            speak_text("Opening navigation. Please specify source and destination.")
            return redirect(url_for('index', _anchor='navigation'))
        elif "smart navigation" in command or "current location" in command:
            speak_text("Opening smart navigation using your current location.")
            return redirect(url_for('smart_navigation_page'))
        
        # General commands
        elif "go home" in command or "main menu" in command:
            speak_text("Going to home page.")
            return redirect(url_for('index'))
        elif "help" in command or "what can i say" in command:
            help_text = "Available commands include: upload image, capture from webcam, detect objects, start navigation, smart navigation, go home, and help."
            speak_text(help_text)
            return help_text
        else:
            speak_text("Command not recognized. Say help to hear available commands.")
            return "Command not recognized. Try saying 'help' for available commands."
    else:
        speak_text("I didn't hear a command. Please try again.")
        return "Voice command failed. Please try again."

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback_data = {
        "name": request.form['name'],
        "email": request.form['email'],
        "feedback": request.form['feedback']
    }

    if not os.path.exists('feedback.json'):
        with open('feedback.json', 'w') as f:
            json.dump([], f)

    with open('feedback.json', 'r+') as f:
        data = json.load(f)
        data.append(feedback_data)
        f.seek(0)
        json.dump(data, f, indent=4)

    return "Thank you for your feedback!"

# Add a new route for continuous voice control
@app.route('/continuous_listening', methods=['GET'])
def continuous_listening():
    return render_template('continuous_listening.html')

# Add a route to handle specific voice actions
@app.route('/voice_action/<action>', methods=['POST'])
def voice_action(action):
    if action == 'capture_image':
        # Logic to capture image from webcam
        cap = cv2.VideoCapture(0)
        time.sleep(1)
        ret, frame = cap.read()
        cap.release()
        if ret:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'captured.jpg')
            cv2.imwrite(image_path, frame)
            caption = generate_caption(image_path)
            speak_text(f"Image captured. {caption}")
            return jsonify({"success": True, "caption": caption, "image_path": image_path})
    elif action == 'get_current_location':
        # Would need to be implemented with client-side GPS
        speak_text("Getting your current location.")
        return jsonify({"success": True, "message": "Location request initiated"})
    
    return jsonify({"success": False, "message": "Action not recognized"})

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global detection_paused
    try:
        data = request.json
        detection_paused = data.get('paused', False)
        return jsonify({"success": True, "paused": detection_paused})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# Comment out the direct run for Vercel deployment
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)

# For local development, uncomment the above block
# For Vercel deployment, we need to export the app variable