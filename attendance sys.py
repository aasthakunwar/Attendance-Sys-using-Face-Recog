import cv2
import face_recognition
import os
import pandas as pd
from datetime import datetime

# path to folders
students_folder = r"C:\Users\aasth\OneDrive\Desktop\project\photos"
attendance_file = r"C:\Users\aasth\OneDrive\Desktop\project\attendance.csv"

# Webcam
cap = cv2.VideoCapture(0)

# Loading and encoding faces
known_face_encodings = []
known_face_names = []

def load_known_faces():
    for filename in os.listdir(students_folder):                #constructing image paths 
        if filename.lower().endswith(('.jpg', '.png')):
            image_path = os.path.join(students_folder, filename)    

            try:                                                
                student_image = face_recognition.load_image_file(image_path)      #image is loaded
                face_encodings = face_recognition.face_encodings(student_image)   #face is encoded

                if face_encodings:   
                    known_face_encodings.append(face_encodings[0])        #populates the empty list initialized above
                    known_face_names.append(filename.split('.')[0])

            except Exception as e:
                print(f"Error loading image {filename}: {e}")

 
def mark_attendance(name):
    now = datetime.now()                        
    current_time = now.strftime("%H:%M:%S")  
    today_date = now.strftime("%d-%m-%Y")    

    # Check if the CSV file exists
    if not os.path.isfile(attendance_file):
        # Create a new file 
        df = pd.DataFrame(columns=["Date", "Name", "Time"])
    else:
        # Loading the existing file
        df = pd.read_csv(attendance_file)

    # Ensure the "Date" column exists (runtime error)
    if "Date" not in df.columns:
        df["Date"] = ""

    # Check if attendance has already been marked for this name and date
    if not df[(df["Name"] == name) & (df["Date"] == today_date)].empty:
        print(f"Attendance for {name} has already been marked today.")
        return 

    # Add a new entry if attendance is not already marked today
    new_entry = pd.DataFrame({"Date": [today_date], "Name": [name], "Time": [current_time]})
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(attendance_file, index=False)  # Save the updated attendance list
    print(f"Attendance for {name} has been marked at {current_time}.")



load_known_faces()
print("Attendance System Initialized")

while True:
    ret, frame = cap.read()         #captures video frame
    if not ret: 
        print("Failed to grab frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)         #converts BGR to RGB    
    face_locations = face_recognition.face_locations(rgb_frame)      # detects all faces
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)  #computes face encodings 

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):   
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)    #matches encodings
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)       #returns index of 1st true value
            name = known_face_names[first_match_index]    #returns name of 1st encoding
            mark_attendance(name)
        else:
            print("No exact match found, but you may try showing the student's face image.") 


            # Capture image to check if it is an image of the student
            captured_image = rgb_frame[top:bottom, left:right]   
            face_encoding_from_image = face_recognition.face_encodings(captured_image)
            if face_encoding_from_image:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding_from_image[0])
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                    mark_attendance(name)

        #draws a rectangle and adds name
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2) 
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow("Attendance System", frame)

    # Close the program when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
