"""
Flask server that bridges face recognition with the web interface.
Run this file to start the server, then open http://localhost:5001 in your browser.
"""

from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import face_recognition
from PIL import Image
import numpy as np
import base64
import threading
import time
import pickle
import os
import subprocess

app = Flask(__name__, static_folder='.', static_url_path='')
app.config['SECRET_KEY'] = 'spectra_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# -------------------------------
# PEOPLE DATABASE (info for each person)
# -------------------------------
PEOPLE_DATABASE = {
    "tom holland": {
        "age": 29,
        "jobs": "Actor, Producer, Dancer",
        "fun_fact": "Started out as a ballet dancer",
        "neighborhood": "Richmond, Southwest London",
        "family": "Parents: Dominic & Nikki Holland | Siblings: Harry, Sam, Paddy",
        "relationship": "Girlfriend: Zendaya",
        "possessions": "Luxury watches, Spider-Man props"
    },
    "the rock": {
        "age": 53,
        "jobs": "Actor, Producer, Former Wrestler",
        "fun_fact": "National champion college football player",
        "neighborhood": "Beverly Hills, California",
        "family": "Parents: Rocky & Ata Johnson",
        "relationship": "Wife: Lauren Hashian",
        "possessions": "Real estate, luxury cars, private jets"
    },
    "taylor swift": {
        "age": 36,
        "jobs": "Singer-songwriter, Producer, Businesswoman",
        "fun_fact": "Named after James Taylor",
        "neighborhood": "Watch Hill, Rhode Island",
        "family": "Parents: Scott & Andrea | Brother: Austin",
        "relationship": "Boyfriend: Travis Kelce",
        "possessions": "Real estate, jets, music catalog, 3 cats"
    },
    "omar farooq": {
        "age": 31,
        "jobs": "Content Creator",
        "fun_fact": "Learned professional camera work at 15",
    },
    "jennifer lopez": {
        "age": 56,
        "jobs": "Singer, Actress, Dancer, Producer, Businesswoman",
        "fun_fact": "Her dress inspired Google Images",
        "neighborhood": "Beverly Hills, California",
        "family": "Parents: Guadalupe & David Lopez",
        "possessions": "Fashion collection, real estate, luxury cars"
    },
    "flipperachi": {
        "age": 37,
        "jobs": "Rapper / Artist",
        "fun_fact": "Stage name inspired by Flip Mode Squad",
        "family": "Spouse: Najwa | Child: Young son"
    },
    "Ali Waheed Alshaikh": {
        "age": 12,
        "Occupation": "BIBF Student",
        "Hobbies": "Reading",
        "fun_fact": "Loves posting ML on linkedin and dynamite boneless chicken, eats 3x the normal human",
        "random": "got IBM AI developer certificate, Deep Learning on NVIDIA certificate, AWS's machine learning etc..."
    },
    "Dr. Mohammed bin Mubarak Juma": {
        "age": "unknown",
        "jobs": "Minister of Education",
        "fun_fact": "Long-serving government official"
    },
    "Zahi wehbe": {
        "age": "43",
        "birthday": "31st jan 1983",
        "jobs": "head of innovation hub",
        "fun_fact": "wants to go to dubai to work soon, likes shawarma with lots of sauce",
        "nationality": "Lebanese"
    },
    "Husam": {
        "age": "23",
        "jobs": "bapco, 4th year @ polytechnic, NCST graduate",
        "fun_fact": "black belt in taekwondo, likes chocolate cake",
        "random": "hands get tired within 10 minutes of climbing",
        "weakness": "yellow foothold at gravity climbing"
    },
    "Kabas": {
        "age": "25",
        "jobs": "Doo Space, AI success engineer",
        "fun_fact": "prompt engineering enthusiast",
        "random": "Lost 40 pounds from his carnivore diet"
    },
    "Leila the goat": {
        "age": "20",
        "jobs": "Mech Eng intern at dentist office","Arts and Crafts major at MIT"
        "nationality": "Morrocan", 
        "weakness": "tiktok",
        "fun_fact": "App with highest screentime is Talabat",
        "random": "shorter than Medha, THRIVE scholar, Bio research at Stanford and Tufts"
    },
    "Hannah Chung": {
        "age": "20",
        "jobs": "TrustAI CEO",
        "nationality": "Korean",
        "fun_fact": "has fallen off a ski lift twice",
        "random": "Watched the Coldplay concert 7 times, likes Topoyaki",
        "gpa": "4.00 gpa until 6.1220"
    },
    "Atharv Mehrotra": {
        "age": "17",
        "jobs": "Student at St. Christopher's School, Bahrain",
        "fun_fact": "flooribear youtuber",
        "nationality": "Indian",
    }, 
    "Khalid": {
        "age": "19",
        "random": "wants to make spy glasses too"
    },
    "Raj Mehta": {
        "age": "38",
        "jobs": "Joining family business",
        "fun_fact": "does bhangra dance, likes visiting temples",
        "weakness": "ER addict",
        "nationality": "Indian",
        "random": "Masters student at MIT"
    },
    "Rashed Almansoori": {
        "age": "22",
        "birthday": "12/01/2004",
        "jobs": "student @ polytechnic university, RPQ intern, ",
        "fun_fact": "House #7115 in Sar neighborhood",
        "random": "Fidgets with yellow beads, valued at 30 BHD"
    },
    "Medha Venkatapathy": {
        "age": "21",
        "birthday": "9/19/2004",
        "nationality": "Indian",
        "jobs": "TrustAI CTO, Defense intern, 3rd year @ MIT",
        "fun_fact": "went cliff jumping, likes turtles"
    },
    "Ethan Dougherty": {
        "age": "17",
        "jobs": "Junior at Bahrain school",
        "fun_fact": "Once slept for 21 hours straight",
        "hobby": "Boxing"
    },
    "Alejandro Jose Saa": {
        "age": "14",
        "occupation": "High school student at Bahrain school",
        "gpa": "4.0",
        "nationality": "Columbian",
        "hobbies": "football, collect legos and funko pops",
        "fun_fact": "makes video games, goes to US for summer break"
    },
    "Ali Ayyad": {
        "age": "17",
        "occupation": "Student at NCST",
        "GPA": "3.8",
        "fun_fact": "he doesn't know",
        "hobbies": "grey hat hacking"
    },
    "Maryam Aysha": {
        "age": "20",
        "occupation": "Student at University of Technology Bahrain",
        "Hobbies": "painting, reading"
    },
    "Vassilios Mingos": {
        "age": "37",
        "jobs": "Math teacher at RVIS",
        "Hobbies": "Snow boarding, tide surfing, football",
        "fun_fact": "has been to more than 50 countries"
    },
    "John Artista": {
        "age": "33",
        "jobs": "RVIS",
        "Hobbies": "Programming",
        "fun_fact": "is a gamer, plays fighting games, fps games, league of legends"
    },
    "Eom": {
        "age": "14",
        "occupation": "Student at Capital school",
        "GPA": "4.0",
        "fun_fact": "Doesnt play any games",
        "hobbies": "Studying"
    },
    "Hamzeh Luay": {
        "age": "16",
        "occupation": "Student at the International school of Choueifat",
        "GPA": "3.7",
        "Hobbies": "Video games",
        "fun_fact": "Played in an international basketball tournament"
    },
    "Mohammed Daggag": {
        "age": "18",
        "occupation": "AUBH Student",
        "GPA": "4.0",
        "Hobbies": "Home Labbing",
        "fun_fact": "Born 7 weeks early"
    },
    "Elyas Rahimi": {
        "age": "17",
        "occupation": "Student at NCST",
        "GPA": "4.0",
        "Hobbies": "Programming",
        "fun_fact": "can speak 5 languages"
    }
}

# -------------------------------
# SETTINGS
# -------------------------------
RESIZE_SCALE = 0.4
TOLERANCE = 0.48

# -------------------------------
# LIST OF KNOWN FACES
# -------------------------------
people = {
    "tom holland": ["faces/tom1.jpg", "faces/tom2.jpg","faces/tom3.jpg","faces/tom4.jpg"],
    "taylor swift": ["faces/taylor1.jpg", "faces/taylor2.jpg", "faces/taylor3.jpg", "faces/taylor4.jpg", "faces/taylor5.jpg"],
    "the rock": ["faces/rock1.jpg", "faces/rock2.jpg", "faces/rock3.jpg", "faces/rock4.jpg"],
    "omar farooq": ["faces/omar1.jpg", "faces/omar2.jpg", "faces/omar3.jpg", "faces/omar4.jpg", "faces/omar5.jpg"],
    "jennifer lopez": ["faces/jennifer1.jpg", "faces/jennifer2.jpg", "faces/jennifer3.jpg", "faces/jennifer4.jpg"],
    "flipperachi": ["faces/flipperachi1.jpg", "faces/flipperachi2.jpg", "faces/flipperachi3.jpg", "faces/flipperachi4.jpg"],
    "Dr. Mohammed bin Mubarak Juma": ["faces/minister1.jpg", "faces/minister2.jpg", "faces/minister3.jpg"],
    "Zahi wehbe": ["faces/zahi1.jpg", "faces/zahi2.jpg", "faces/zahi3.jpg", "faces/zahi4.jpg", "faces/zahi5.jpg"],
    "Maryam Aysha": ["faces/maryam1.jpg", "faces/maryam2.jpg", "faces/maryam3.jpg"],
    "Ali Ayyad": ["faces/alia1.jpg", "faces/alia2.jpg", "faces/alia3.jpg"],
    "Ali waheed": ["faces/ali1.jpg"],
    "Alejandro Jose Saa": ["faces/alejandro1.jpg", "faces/alejandro2.jpg"],
    "Atharv Mehrotra": ["faces/atharv1.jpg", "faces/atharv2.jpg"],
    "Mohammed Daggag": ["faces/daggag1.jpg", "faces/daggag2.jpg"],
    "Elyas Rahimi": ["faces/elyas1.jpg", "faces/elyas2.jpg", "faces/elyas3.jpg"],
    "Hannah Chung": ["faces/hana1.jpg", "faces/hana2.jpg"],
    "Khalid": ["faces/khalid1.jpg", "faces/khalid2.jpg", "faces/khalid3.jpg"],
    "Leila": ["faces/layla1.jpg", "faces/layla2.jpg", "faces/layla3.jpg"],
    "Sara": ["faces/sara1.jpg", "faces/sara2.jpg", "faces/sara3.jpg"],
    "Raj Mehta": ["faces/raj1.jpg", "faces/raj2.jpg", "faces/raj3.jpg"],
    "Husam": ["faces/husam1.jpg", "faces/husam2.jpg", "faces/husam3.jpg"],
    "Ethan Dougherty": ["faces/ethan1.jpg", "faces/ethan2.jpg"],
    # "Alia": ["faces/alya1.jpg", "faces/alya2.jpg"],
    "Vassilios Mingos": ["faces/vass1.jpg", "faces/vass2.jpg", "faces/vass3.jpg"],
    "John Artista": ["faces/john1.jpg", "faces/john2.jpg", "faces/john3.jpg"],
    "Kabas": ["faces/kabas1.jpg", "faces/kabas2.jpg", "faces/kabas3.jpg"],
    "Medha Venkatapathy": ["faces/medha1.jpg", "faces/medha2.jpg"],
    "Rashed Almansoori": ["faces/rashed1.jpg"],
    "Eom": ["faces/eom1.jpg", "faces/eom2.jpg", "faces/eom3.jpg"],
    "Hamzeh Luay": ["faces/hamzeh1.jpg", "faces/hamzeh2.jpg", "faces/hamzeh3.jpg"],
}

# Global variables
known_face_encodings = []
known_face_names = []
currently_visible = set()
video_capture = None
is_streaming = False
search_target = None  # Name to search for in locate page
last_spoken_person = None  # Track last person announced via TTS
tts_lock = threading.Lock()  # Lock to ensure only one TTS at a time
tts_speaking = False  # Flag to track if TTS is currently speaking

def speak_text(text):
    """Speak text aloud using macOS say command (runs in background thread)."""
    global tts_speaking

    # Check if already speaking - if so, skip this request
    if tts_speaking:
        print(f"  [TTS] Skipping (already speaking): {text}")
        return

    def _speak():
        global tts_speaking
        with tts_lock:
            tts_speaking = True
            try:
                # Use macOS 'say' command - will output to default audio (Bluetooth if connected)
                subprocess.run(['say', '-r', '180', text], check=True)
            except Exception as e:
                print(f"TTS Error: {e}")
            finally:
                tts_speaking = False

    # Run in background thread so it doesn't block video processing
    threading.Thread(target=_speak, daemon=True).start()

# Path to save/load face encodings
ENCODINGS_FILE = "face_encodings.pkl"

def save_encodings():
    """Save face encodings to a pickle file."""
    data = {
        "encodings": known_face_encodings,
        "names": known_face_names
    }
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {len(known_face_encodings)} encodings to {ENCODINGS_FILE}")

def load_encodings_from_file():
    """Load face encodings from pickle file if it exists."""
    global known_face_encodings, known_face_names

    if os.path.exists(ENCODINGS_FILE):
        print(f"Loading encodings from {ENCODINGS_FILE}...")
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        known_face_encodings = data["encodings"]
        known_face_names = data["names"]
        print(f"Loaded {len(known_face_encodings)} encodings from cache.")
        return True
    return False

def load_faces(force_reload=False):
    """Load all known face encodings. Uses cached file if available."""
    global known_face_encodings, known_face_names

    # Try to load from cache first (unless force_reload is True)
    if not force_reload and load_encodings_from_file():
        return

    print("Processing face images (this may take a moment)...")
    known_face_encodings = []
    known_face_names = []

    for name, image_paths in people.items():
        for path in image_paths:
            try:
                img = Image.open(path)
                img = img.convert("RGB")
                img_np = np.array(img).astype(np.uint8)

                encodings = face_recognition.face_encodings(img_np)
                if encodings:
                    known_face_encodings.append(encodings[0])
                    known_face_names.append(name)
                    print(f"Loaded {name} from {path}")
                else:
                    print(f"No face detected in {path}")
            except Exception as e:
                print(f"Failed {path} | {e}")

    print(f"\nLoaded {len(known_face_encodings)} face encodings.")

    # Save to cache for next time
    save_encodings()

def get_person_info(key):
    """Get person info from database, handling case-insensitive lookup."""
    # Try exact match first
    if key in PEOPLE_DATABASE:
        return PEOPLE_DATABASE[key]

    # Try case-insensitive match
    key_lower = key.lower()
    for db_key, value in PEOPLE_DATABASE.items():
        if db_key.lower() == key_lower:
            return value

    return None

def determine_designation(name):
    """Determine if someone is a celebrity, ministry official, or citizen."""
    celebrities = ["tom holland", "taylor swift", "the rock", "omar farooq", "jennifer lopez", "flipperachi"]
    ministry = ["dr. mohammed bin mubarak juma", "zahi wehbe"]

    name_lower = name.lower()
    if name_lower in celebrities:
        return "celebrity"
    elif name_lower in ministry:
        return "ministry"
    else:
        return "citizen"

def generate_frames():
    """Generate frames from camera with face recognition."""
    global currently_visible, video_capture, is_streaming, last_spoken_person

    video_capture = cv2.VideoCapture(0)
    is_streaming = True

    while is_streaming:
        ret, frame = video_capture.read()
        if not ret:
            continue

        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB).astype(np.uint8)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        detected_now = set()

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=TOLERANCE)
            name_to_display = "Unknown"
            color = (0, 0, 255)  # Red for unknown

            if True in matches:
                index = matches.index(True)
                key = known_face_names[index]
                name_to_display = key.title()
                color = (0, 255, 0)  # Green for known
                detected_now.add(key)

                # Send person info to frontend when newly detected
                if key not in currently_visible:
                    person_info = get_person_info(key)
                    designation = determine_designation(key)

                    # Build the data to send
                    data = {
                        "name": name_to_display,
                        "designation": designation,
                        "recognized": True
                    }

                    if person_info:
                        # Get age - try different key variations
                        age = person_info.get("age", person_info.get("Age", "--"))
                        data["age"] = str(age)

                        # Get occupation - try different key variations
                        occupation = (person_info.get("jobs") or
                                    person_info.get("occupation") or
                                    person_info.get("Occupation") or
                                    "Unknown")
                        data["occupation"] = occupation

                        # Get nationality
                        data["nationality"] = person_info.get("nationality", "Unknown")

                        # Get fun fact
                        data["fun_fact"] = person_info.get("fun_fact", "")

                        # Add all other info
                        data["all_info"] = person_info
                    else:
                        data["age"] = "--"
                        data["occupation"] = "Unknown"
                        data["nationality"] = "Unknown"
                        data["all_info"] = {}

                    # Emit to all connected clients
                    socketio.emit('person_identified', data)
                    print(f"\nPERSON IDENTIFIED: {name_to_display}")
                    if person_info:
                        for k, v in person_info.items():
                            print(f"  {k}: {v}")

                    # Speak the fun fact aloud only if this is a different person
                    if key != last_spoken_person:
                        last_spoken_person = key
                        fun_fact = person_info.get("fun_fact", "") if person_info else ""
                        if fun_fact:
                            speech_text = f"{name_to_display}. {fun_fact}"
                            speak_text(speech_text)
                            print(f"  [TTS] Speaking: {speech_text}")

            # Scale rectangle back to original frame size
            top = int(top / RESIZE_SCALE)
            right = int(right / RESIZE_SCALE)
            bottom = int(bottom / RESIZE_SCALE)
            left = int(left / RESIZE_SCALE)

            # Draw rectangle and label
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name_to_display, (left, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        currently_visible = detected_now

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()

        # Yield as multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Serve the main page."""
    return app.send_static_file('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('Client connected')
    emit('status', {'message': 'Connected to SPECTRA server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')

@socketio.on('start_camera')
def handle_start_camera():
    """Start the camera stream."""
    global is_streaming
    if not is_streaming:
        threading.Thread(target=generate_frames, daemon=True).start()
    emit('camera_status', {'status': 'started'})

@socketio.on('stop_camera')
def handle_stop_camera():
    """Stop the camera stream."""
    global is_streaming, video_capture
    is_streaming = False
    if video_capture:
        video_capture.release()
        video_capture = None
    emit('camera_status', {'status': 'stopped'})

@socketio.on('search_person')
def handle_search_person(data):
    """Handle person search request from locate page."""
    global search_target
    name = data.get('name', '').strip()
    if name:
        search_target = name.lower()
        print(f"Searching for: {search_target}")
        emit('search_result', {'name': name, 'searching': True, 'found': False})
    else:
        search_target = None

@socketio.on('chat_message')
def handle_chat_message(data):
    """Handle chat/LLM queries from the terminal."""
    query = data.get('message', '').strip()
    if not query:
        emit('chat_response', {'response': 'No query provided.'})
        return

    print(f"Chat query: {query}")

    # Generate response using simple logic or API
    response = generate_llm_response(query)
    emit('chat_response', {'response': response})

def generate_llm_response(query):
    """Generate a response to user queries. Uses Groq API if available, otherwise falls back to local responses."""
    import os
    import json

    query_lower = query.lower()

    # Try to use Groq API (free tier available)
    groq_api_key = os.environ.get('GROQ_API_KEY')

    if groq_api_key:
        try:
            import urllib.request
            import urllib.error

            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            }

            system_prompt = """You are SPECTRA AI, an advanced intelligence system assistant.
You provide concise, professional responses in a military/intelligence style.
Keep responses brief (1-3 sentences max). Use technical language when appropriate.
You help with facial recognition queries, person lookups, and general intelligence questions."""

            payload = json.dumps({
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                "max_tokens": 150,
                "temperature": 0.7
            })

            req = urllib.request.Request(url, data=payload.encode('utf-8'), headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"Groq API error: {e}")
            # Fall through to local responses

    # Local response system (fallback)
    # Check for person queries
    for name in PEOPLE_DATABASE.keys():
        if name.lower() in query_lower:
            person = PEOPLE_DATABASE[name]
            info_parts = [f"{k}: {v}" for k, v in list(person.items())[:3]]
            return f"SUBJECT FOUND: {name.title()}. {'; '.join(info_parts)}."

    # Check for specific keywords
    if any(word in query_lower for word in ['hello', 'hi', 'hey']):
        return "SPECTRA AI online. How may I assist with your intelligence operation?"

    if any(word in query_lower for word in ['help', 'commands', 'what can']):
        return "Available commands: Ask about any person in database, request system status, or query threat levels. Type 'list subjects' for known individuals."

    if 'status' in query_lower or 'system' in query_lower:
        return f"All systems nominal. {len(PEOPLE_DATABASE)} subjects in database. Face recognition: ACTIVE. Threat monitoring: ACTIVE."

    if 'list' in query_lower and ('subject' in query_lower or 'people' in query_lower or 'person' in query_lower):
        names = list(PEOPLE_DATABASE.keys())[:10]
        return f"Known subjects: {', '.join(n.title() for n in names)}{'...' if len(PEOPLE_DATABASE) > 10 else ''}"

    if any(word in query_lower for word in ['threat', 'danger', 'criminal']):
        return "5 active threats in database. Access DANGER tab for full threat assessment and criminal profiles."

    if any(word in query_lower for word in ['location', 'where', 'gps']):
        return "GPS tracking active. Current position displayed on LOCATE tab. Accuracy within operational parameters."

    if any(word in query_lower for word in ['who', 'identify', 'recognize']):
        return "Point camera at subject for facial recognition. System will auto-identify known individuals from database."

    if any(word in query_lower for word in ['time', 'date']):
        from datetime import datetime
        now = datetime.now()
        return f"Current timestamp: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC. System uptime: nominal."

    if any(word in query_lower for word in ['thank', 'thanks']):
        return "Acknowledged. SPECTRA AI standing by for further queries."

    # General knowledge responses
    # Unit conversions
    if 'feet' in query_lower and 'yard' in query_lower:
        return "3 feet in a yard. Standard imperial measurement."
    if 'inch' in query_lower and 'feet' in query_lower:
        return "12 inches in a foot. Standard imperial measurement."
    if 'meter' in query_lower and ('feet' in query_lower or 'foot' in query_lower):
        return "1 meter equals approximately 3.28 feet."
    if 'mile' in query_lower and ('km' in query_lower or 'kilometer' in query_lower):
        return "1 mile equals approximately 1.609 kilometers."
    if 'pound' in query_lower and ('kg' in query_lower or 'kilogram' in query_lower):
        return "1 pound equals approximately 0.454 kilograms."

    # Math queries
    if any(op in query_lower for op in ['plus', 'minus', 'times', 'divided', '+', '-', '*', '/', 'add', 'subtract', 'multiply', 'divide']):
        try:
            # Try to evaluate simple math expressions
            import re
            # Extract numbers and operators
            expr = query_lower.replace('plus', '+').replace('minus', '-').replace('times', '*').replace('divided by', '/').replace('add', '+').replace('subtract', '-').replace('multiply', '*').replace('divide', '/')
            # Find mathematical expression
            math_match = re.search(r'[\d\.\s\+\-\*\/\(\)]+', expr)
            if math_match:
                result = eval(math_match.group().strip())
                return f"Calculation result: {result}"
        except:
            pass

    # Weather (mock response)
    if 'weather' in query_lower:
        return "Weather data unavailable. SPECTRA operates on intelligence data, not meteorological systems."

    # General questions
    if query_lower.startswith('what is') or query_lower.startswith('what\'s'):
        return "Query acknowledged. For detailed information, please specify if this relates to a known subject or intelligence operation."

    if query_lower.startswith('how') and ('many' in query_lower or 'much' in query_lower):
        return "Quantitative query detected. Please provide more context for accurate intelligence assessment."

    if query_lower.startswith('who is') or query_lower.startswith('who\'s'):
        return "Subject query detected. If target is in database, face recognition will provide full profile. Otherwise, information unavailable."

    if query_lower.startswith('where'):
        return "Location query detected. Use LOCATE tab for GPS tracking or specify subject name for last known position."

    if query_lower.startswith('why') or query_lower.startswith('when'):
        return "Query logged. SPECTRA specializes in identification and tracking. For detailed analysis, consult mission briefing."

    # Default response
    return "Query processed. For optimal results, ask about subjects in database, system status, or use specific commands. Type 'help' for available commands."

def generate_frames_search():
    """Generate frames with search highlighting for locate page."""
    global search_target, video_capture, is_streaming

    # Use existing video capture or create new one
    if video_capture is None or not video_capture.isOpened():
        video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue

        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB).astype(np.uint8)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        target_found = False

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=TOLERANCE)
            name_to_display = "Unknown"
            color = (128, 128, 128)  # Gray for unknown

            if True in matches:
                index = matches.index(True)
                key = known_face_names[index]
                name_to_display = key.title()

                # Check if this is the search target
                if search_target and search_target in key.lower():
                    color = (0, 255, 0)  # Bright green for found target
                    target_found = True
                    # Emit found result
                    socketio.emit('search_result', {
                        'name': key.title(),
                        'searching': False,
                        'found': True
                    })
                else:
                    color = (100, 100, 100)  # Dim gray for non-target known faces
            else:
                # Check if searching for unknown and this is unknown
                if search_target:
                    color = (50, 50, 50)  # Very dim for unknown when searching

            # Scale rectangle back to original frame size
            top = int(top / RESIZE_SCALE)
            right = int(right / RESIZE_SCALE)
            bottom = int(bottom / RESIZE_SCALE)
            left = int(left / RESIZE_SCALE)

            # Draw rectangle - thicker for target
            thickness = 3 if (search_target and search_target in name_to_display.lower()) else 2
            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)

            # Draw label
            label_color = color
            cv2.putText(frame, name_to_display, (left, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2)

            # If target found, draw additional highlighting
            if search_target and search_target in name_to_display.lower():
                # Draw corner brackets
                bracket_len = 20
                cv2.line(frame, (left - 5, top - 5), (left - 5 + bracket_len, top - 5), (0, 255, 0), 3)
                cv2.line(frame, (left - 5, top - 5), (left - 5, top - 5 + bracket_len), (0, 255, 0), 3)
                cv2.line(frame, (right + 5, top - 5), (right + 5 - bracket_len, top - 5), (0, 255, 0), 3)
                cv2.line(frame, (right + 5, top - 5), (right + 5, top - 5 + bracket_len), (0, 255, 0), 3)
                cv2.line(frame, (left - 5, bottom + 5), (left - 5 + bracket_len, bottom + 5), (0, 255, 0), 3)
                cv2.line(frame, (left - 5, bottom + 5), (left - 5, bottom + 5 - bracket_len), (0, 255, 0), 3)
                cv2.line(frame, (right + 5, bottom + 5), (right + 5 - bracket_len, bottom + 5), (0, 255, 0), 3)
                cv2.line(frame, (right + 5, bottom + 5), (right + 5, bottom + 5 - bracket_len), (0, 255, 0), 3)

                # Draw "LOCATED" text
                cv2.putText(frame, "LOCATED", (left, bottom + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # If searching but not found
        if search_target and not target_found and len(face_locations) > 0:
            socketio.emit('search_result', {
                'name': search_target,
                'searching': False,
                'found': False
            })

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed_search')
def video_feed_search():
    """Video streaming route for locate page with search highlighting."""
    return Response(generate_frames_search(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    import sys

    print("=" * 50)
    print("SPECTRA Intelligence System - Server")
    print("=" * 50)

    # Check for --reload flag to force reprocessing images
    force_reload = "--reload" in sys.argv

    if force_reload:
        print("\n[--reload flag detected] Reprocessing all face images...")

    # Load face encodings (uses cache unless --reload is passed)
    load_faces(force_reload=force_reload)

    print("\nStarting server...")
    print("Open http://localhost:5001 in your browser")
    print("Press Ctrl+C to stop")
    print("\nTip: Run with --reload to reprocess face images\n")

    socketio.run(app, host='0.0.0.0', port=5001, debug=False)
