import cv2
import face_recognition
from PIL import Image
import numpy as np

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
        "nationality": "lebanese"
    },
    "Husam": {
        "age": "23",
        "jobs": "bapco, 4th year @ polytechnic, NCST graduate",
        "fun_fact": "black belt in taekwondo, likes chocolate cake", 
        "random" : "hands get tired within 10 minutes of climbing"
    },
    "Kabas": {
        "age": "25", 
        "jobs": "Doo Space, AI success engineer",
        "fun_fact": "prompt engineering enthusiast",
        "random": "Lost 40 pounds from his carnivore diet"
    },
    "Leila": {
        "age": "20", 
        "jobs": "Mech Eng intern at dentist office",
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
    "Khalid": {
        "age": "19", 
        "random": "wants to make spy glasses too"
    },
    "Raj Mehta": {
        "age": "38",
        "jobs": "Joining family business",
        "fun_fact": "does bhangra dance, likes visiting temples",
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
        "hobbies": "football",
        "fun_fact": "makes video games"
    },
    "Ali Ayyad": {
        "age": "17",
        "occupation": "Student at NCST",
        "GPA": "3.8",
        "fun_fact": "Not Known",
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
RESIZE_SCALE = 0.4   # scale down for faster processing
TOLERANCE = 0.48     # face recognition tolerance

# -------------------------------
# LIST OF KNOWN FACES
# Each person can have multiple images
# -------------------------------
people = {
    # celebrity faces
    "tom holland": ["faces/tom1.jpg", "faces/tom2.jpg","faces/tom3.jpg","faces/tom4.jpg"],
    "taylor swift": ["faces/taylor1.jpg", "faces/taylor2.jpg", "faces/taylor3.jpg", "faces/taylor4.jpg", "faces/taylor5.jpg"],
    "the rock": ["faces/rock1.jpg", "faces/rock2.jpg", "faces/rock3.jpg", "faces/rock4.jpg"],
    "omar farooq": ["faces/omar1.jpg", "faces/omar2.jpg", "faces/omar3.jpg", "faces/omar4.jpg", "faces/omar5.jpg"],
    "jennifer lopez": ["faces/jennifer1.jpg", "faces/jennifer2.jpg", "faces/jennifer3.jpg", "faces/jennifer4.jpg"],
    "flipperachi": ["faces/flipperachi1.jpg", "faces/flipperachi2.jpg", "faces/flipperachi3.jpg", "faces/flipperachi4.jpg"],
    # ministry faces
    "Dr. Mohammed bin Mubarak Juma": ["faces/minister1.jpg", "faces/minister2.jpg", "faces/minister3.jpg"],
    "Zahi wehbe": ["faces/zahi1.jpg", "faces/zahi2.jpg", "faces/zahi3.jpg", "faces/zahi4.jpg", "faces/zahi5.jpg"],
    # normal people faces
    "Maryam Aysha": ["faces/maryam1.jpg", "faces/maryam2.jpg", "faces/maryam3.jpg"],
    "Ali Ayyad": ["faces/alia1.jpg", "faces/alia2.jpg", "faces/alia3.jpg"],
    "Ali waheed": ["faces/ali1.jpg"],
    "Alejandro Jose Saa ": ["faces/alejandro1.jpg", "faces/alejandro2.jpg"],
    "Mohammed Daggag": ["faces/daggag1.jpg", "faces/daggag2.jpg"],
    "Elyas Rahimi": ["faces/elyas1.jpg", "faces/elyas2.jpg", "faces/elyas3.jpg"],
    "Hannah Chung": ["faces/hana1.jpg", "faces/hana2.jpg"],
    "Khalid": ["faces/khalid1.jpg", "faces/khalid2.jpg", "faces/khalid3.jpg"],
    "Leila": ["faces/layla1.jpg", "faces/layla2.jpg", "faces/layla3.jpg"],
    "Sara": ["faces/sara1.jpg", "faces/sara2.jpg", "faces/sara3.jpg"],
    "Raj Mehta": ["faces/raj1.jpg", "faces/raj2.jpg", "faces/raj3.jpg"],
    "Husam": ["faces/husam1.jpg", "faces/husam2.jpg", "faces/husam3.jpg"],
    "Alia": ["faces/alya1.jpg", "faces/alya2.jpg"],
    "Vassilios Mingos": ["faces/vass1.jpg", "faces/vass2.jpg", "faces/vass3.jpg"],
    "John Artista": ["faces/john1.jpg", "faces/john2.jpg", "faces/john3.jpg"],
    "Kabas": ["faces/kabas1.jpg", "faces/kabas2.jpg", "faces/kabas3.jpg"],
    "Medha Venkatapathy": ["faces/medha1.jpg", "faces/medha2.jpg"],
    "Rashed Almansoori": ["faces/rashed1.jpg"],
    "Eom": ["faces/eom1.jpg", "faces/eom2.jpg", "faces/eom3.jpg"],
    "Hamzeh Luay": ["faces/hamzeh1.jpg", "faces/hamzeh2.jpg", "faces/hamzeh3.jpg"],
}

# -------------------------------
# LOAD ALL FACE ENCODINGS
# -------------------------------
known_face_encodings = []
known_face_names = []

print("🔍 Loading known faces...\n")

for name, image_paths in people.items():
    for path in image_paths:
        try:
            # Open image with PIL
            img = Image.open(path)

            # Force RGB (3 channels)
            img = img.convert("RGB")

            # Convert to numpy array
            img_np = np.array(img).astype(np.uint8)  # <-- force 8-bit uint8

            # Get face encoding
            encodings = face_recognition.face_encodings(img_np)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f"✅ Loaded {name} from {path}")
            else:
                print(f"❌ No face detected in {path}")

        except Exception as e:
            print(f"❌ Failed {path} | {e}")

print("\n✅ All known faces loaded.\n")

# -------------------------------
# START CAMERA
# -------------------------------
video = cv2.VideoCapture(0)
cv2.namedWindow("AI Smart Spy Glasses", cv2.WINDOW_NORMAL)

currently_visible = set()

while True:
    ret, frame = video.read()
    if not ret:
        continue

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB).astype(np.uint8)  # force uint8

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

            if key not in currently_visible:
                print(f"\n🕵️ PERSON IDENTIFIED: {name_to_display}")
                if key in PEOPLE_DATABASE:
                    for k, v in PEOPLE_DATABASE[key].items():
                        print(f"{k.title()}: {v}")
                else:
                    print("No additional info available.")

        # Scale rectangle back to original frame size
        top = int(top / RESIZE_SCALE)
        right = int(right / RESIZE_SCALE)
        bottom = int(bottom / RESIZE_SCALE)
        left = int(left / RESIZE_SCALE)

        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name_to_display, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    currently_visible = detected_now
    cv2.imshow("AI Smart Spy Glasses", frame)

    if cv2.waitKey(1) == 27:  # ESC key to exit
        break

video.release()
cv2.destroyAllWindows()