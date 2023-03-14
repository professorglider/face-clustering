import cv2
import face_recognition
import psycopg2
import datetime

# Connect to the PostgreSQL database
conn = psycopg2.connect(
    host="postgres",
    database="faces_db",
    user="postgres",
    password="postgres"
)

# Create a cursor
cur = conn.cursor()

# Create the faces table if it does not exist
cur.execute("""
    CREATE TABLE IF NOT EXISTS faces (
        face_id SERIAL PRIMARY KEY,
        face_embedding TEXT,
        pitch FLOAT,
        roll FLOAT,
        yaw FLOAT,
        date DATE,
        time TIME
    )
""")
conn.commit()

# Load the face recognition model
face_recognition_model = face_recognition.api.load_model('/app/models')

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Iterate over the detected faces
    for face_location, face_encoding in zip(face_locations, face_encodings):
        # Extract the face pitch, roll, and yaw angles
        pitch, roll, yaw = face_recognition.api.face_rotation(face_encoding)

        # Convert the face embedding to a string for storage in the database
        face_embedding_str = ','.join(map(str, face_encoding))

        # Insert the face data into the database
        cur.execute("""
            INSERT INTO faces (face_embedding, pitch, roll, yaw, date, time)
            VALUES (%s, %s, %s, %s, %s, %s)
    """, (face_embedding_str, pitch, roll, yaw, datetime.date.today(), datetime.datetime.now().time()))
    conn.commit()

# Release the camera
cap.release()

# Close the database connection
conn.close()
