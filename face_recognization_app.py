import face_recognition
import cv2
import numpy as np
import os

# Function to load images and their encodings for a given person
def load_images_and_encodings(folder_path):
    face_encodings = []
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = face_recognition.load_image_file(image_path)
        # Check if any faces are detected in the image
        face_encodings_in_image = face_recognition.face_encodings(image)
        if face_encodings_in_image:
            encoding = face_encodings_in_image[0]  # Assuming only one face per image
            face_encodings.append(encoding)
    return face_encodings

# Load images and encodings for each person
GangaRaju_encodings = load_images_and_encodings(r"C:\Users\venkey\OneDrive\Desktop\gan\face_reco\face_reco\GangaRaju/")
Gayathri_encodings = load_images_and_encodings(r"C:\Users\venkey\OneDrive\Desktop\gan\face_reco\face_reco\Gayathri/")
Venkatesh_encodings = load_images_and_encodings(r"C:\Users\venkey\OneDrive\Desktop\gan\face_reco\face_reco\Venkatesh/")
Harika_encodings = load_images_and_encodings(r"C:\Users\venkey\OneDrive\Desktop\gan\face_reco\face_reco\Harika/")
badrinath_encodings = load_images_and_encodings(r"C:\Users\venkey\OneDrive\Desktop\gan\face_reco\face_reco\Badrinath/")

# Create arrays of known face encodings and their names
known_face_encodings = [
    GangaRaju_encodings,
    Gayathri_encodings,
    Venkatesh_encodings,
    Harika_encodings,
    badrinath_encodings,
    ]
# known_face_names = [
#     "GangaRaju",
#     "Gayathri",
#     'Venkatesh',
#     'Harika'
#     'Badrinath' 
# ]


known_face_encodings = GangaRaju_encodings + Gayathri_encodings + Venkatesh_encodings + Harika_encodings+badrinath_encodings

known_face_names = ["GangaRaju"] * len(GangaRaju_encodings) + ["Gayathri"] * len(Gayathri_encodings) + ["Venkatesh"] * len(Venkatesh_encodings) + ["Harika"] * len(Harika_encodings) +["badrinath"] *len(badrinath_encodings)# Assign names accordingly

###########TESTING###############TESTING############TESTING########TESTING################TESTING############TESTING###################

for i in os.listdir(r"C:\Users\venkey\OneDrive\Desktop\gan\blur/"):
    # image_path = r"C:/Users/venkey/OneDrive/Desktop/gan/face_reco/face_reco/GangaRaju/WhatsApp Image 2024-03-12 at 12.47.51 PM (4).jpeg"
    image_path = os.path.join(r"C:\Users\venkey\OneDrive\Desktop\gan\blur/" + i)
    image = face_recognition.load_image_file(image_path)
    
    # Find all the faces and face encodings in the image
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    # Initialize an empty list to store the names of recognized faces
    face_names = []
    
    for face_encoding in face_encodings:
        # See if the face is a match for the known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
    
        # Find the best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
    
        # If a match is found, assign the name of the known face to the recognized face
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
    
        face_names.append(name)
    
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
    
    # Display the resulting image
    # cv2.imshow('Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt.imshow(image, cmap = 'gray')
    plt.show()
    image_name = image_path.split("\\")[-1].split('/')[-1]
    cv2.imwrite(r'C:\Users\venkey\Downloads\test_output/' + image_name, image)
    
    


##########################################################################################################################
# TESTING ON SINGLE IMAGE
import matplotlib.pyplot as plt
%matplotlib inline
image_path = r"C:\Users\venkey\OneDrive\Desktop\Screenshot 2024-05-08 201054.png"
image = face_recognition.load_image_file(image_path)

# Find all the faces and face encodings in the image
face_locations = face_recognition.face_locations(image)
face_encodings = face_recognition.face_encodings(image, face_locations)

# Initialize an empty list to store the names of recognized faces
face_names = []

for face_encoding in face_encodings:
    # See if the face is a match for the known faces
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    # Find the best match
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)

    # If a match is found, assign the name of the known face to the recognized face
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    face_names.append(name)

# Display the results
for (top, right, bottom, left), name in zip(face_locations, face_names):
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

# Display the resulting image
# cv2.imshow('Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
plt.imshow(image, cmap = 'gray')
plt.show()
image_name = image_path.split("\\")[-1].split('/')[-1]
cv2.imwrite(r'C:\Users\venkey\OneDrive\Desktop\gan\face_rec_output/' + image_name, image)


















