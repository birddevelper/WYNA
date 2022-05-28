import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import dlib
from scipy.spatial.distance import cosine
import speech_recognition as sr
import pyttsx3 as tts
import random
import time
import threading

text_to_speech_engine = tts.init()
speech_recognizer = sr.Recognizer()
cap = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()
facenet_model = load_model("facenet_keras.h5",compile=False)
embedding_db = {}

mic = sr.Microphone()
last_speech = ""

speech_history=[]
    

stop_listening = None

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# initialize video recorder
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
last_ask = 0
last_mic_reset = time.time()
closed_uknown_person =  None

class faceAttr:
    face = None
    x1 = None
    x2 = None
    y1 = None
    y2 = None
    def __init__(self, face, x1, x2, y1, y2):
            self.face = face
            self.x1 = x1
            self.x2 = x2
            self.y1 = y1
            self.y2 = y2
    # this function calculates the face_area/video_screen_area
    def get_area_ratio(self, imageWidth, imageHeight):
        facearea = ( self.x2- self.x1) * ( self.y2- self.y1)
        imageArea = imageWidth * imageHeight
        ratio = (facearea / imageArea)
        #print("face area" + str(facearea) + " image area :" + str(imageArea))
        #print ("ration :" + str(round(ratio,3)))
        return ratio


def put_speech_history_on_image(img):
    height = img.shape[0] - 10
    for speech_text in speech_history :
        cv2.putText(img, str(speech_text), (5, height), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (10,30,255), 1)
        height-=15



def put_image_text(img, face, text, upper = True , color = (0,0,255)) :
    cv2.rectangle(img, (face.x1, face.y1), (face.x2, face.y2),color, 2)
    if(upper) :
        cv2.putText(img, text, (face.x1, face.y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else :
        cv2.putText(img, text, (face.x1, face.y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return

def extract_face_and_preprocessing(image):
    face_list =[]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_detector(gray_image,1)
    
    for detected_face in faces :
        x1,y1 = detected_face.left(), detected_face.top()
        x2,y2 = detected_face.right(), detected_face.bottom()
        try :
            face = image[y1:y2, x1:x2]
            face = cv2.resize(face, (160, 160))
            face = np.expand_dims(face, axis = 0)
            face = (face - face.mean())/face.std()
            face_list.append(faceAttr(face,x1,x2,y1,y2))
        except :
            pass
    return face_list



def face_identifier(face_embedding) :
    
    name = "Unkown"
    distance = float("inf")
    for db_name, db_embedding in embedding_db.items():
        dist = cosine(face_embedding, db_embedding)
        if dist < 0.5 and dist < distance:
            name = db_name
            distance = dist
    return name



def text_to_speech(text) :
    text_to_speech_engine.say(text)
    text_to_speech_engine.runAndWait()


def mark_as_uknown(face, face_embedding,img) :
        put_image_text(img,face,"Unkown")
        cv2.imshow("WYNA", img)
        return


def ask_what_is_your_name(face,img,voice) :
    
    put_image_text(img,face,"What is your Name ?",False,(255,0,0))
    cv2.imshow("WYNA", img)
    if(voice) :
        text_to_speech("What is your name")
        last_ask =  time.time()
   

    return 



def put_name_on_face(face,img,name) :
    
    put_image_text(img,face,name,True,(0,255,0))
    return

def callback(recognizer, audio):
    # received audio data, now we'll recognize it using Google Speech Recognition
    global last_speech 
    global stop_listening
    try:
       
       print("Processing the voice...")
       speech = speech_recognizer.recognize_google(audio) #recognize_sphinx
       #embedding_db[name] = face_embedding
       #greetings= ['Hello ', 'What\'s up ', 'Nice to meet you ', 'How are you ', 'Nice Name ']
       print ("Listened :" + speech )
       speech_history.append(speech)
       if(True) : #closed_uknown_person != None
            if(speech.lower().startswith("my name is") or speech.lower().startswith("i am") ) :
                 last_speech = speech.lower().replace("my name is ","").replace("i am ","")
                 print("Name :" + last_speech)
       
    
    except sr.UnknownValueError:
        print("Sorry, did not understand")
    except sr.RequestError as e:
        text_to_speech("Sorry, I got a problem")
        print("Sorry, I got a problem")
    
   

    return 

def recorder() :
    while True :
        try :
            speech_recognizer.pause_threshold = 3
            print("Mic Started...")
            with mic as source:
                speech_recognizer.energy_threshold = 300
                #speech_recognizer.adjust_for_ambient_noise(source)
                audio = speech_recognizer.listen(source,5,10)
            callback(speech_recognizer,audio)
        except :
            print("Mic Problem Handled")



# thread.start_new_thread( recorder, ("Thread-1", 2, ) )
x = threading.Thread(target=recorder)
x.start()

while True :
    

    _ , img = cap.read()

    if img is None :
        print("Capture Problem")
        continue
    



    
    faces = extract_face_and_preprocessing(img)   
    if( len(faces) > 0 ) :
        
        closed_uknown_person =  None
        closest_uknown_person_ratio = 0.0
        closest_face_attr = None
        for face_attr in faces :

            face_embedding = facenet_model.predict(face_attr.face)
            face_name = face_identifier(face_embedding)
            
            # If the face is new, try to learn it
            if(face_name == "Unkown") :
                mark_as_uknown(face_attr, face_embedding ,img)
                faceRatio = face_attr.get_area_ratio(frame_width,frame_height)
                if(faceRatio > closest_uknown_person_ratio ) :
                    if(faceRatio>= 0.05) :
                        closed_uknown_person = face_embedding
                        closest_uknown_person_ratio = faceRatio
                        closest_face_attr = face_attr
            else :
                put_name_on_face(face_attr,img,face_name)

        if(closed_uknown_person is not None) :
            
            
            ask_what_is_your_name(closest_face_attr,img,last_ask - time.time() > 20)

            if(len(last_speech)>1) :
                print("Want to assign Name " + last_speech)
                embedding_db[last_speech] = closed_uknown_person
                greetings= ['Hello ', 'What\'s up ', 'Nice to meet you ', 'How are you ', 'Nice Name ']
                text_to_speech(greetings[random.randint(0, 4)] + last_speech )
                last_speech = ""
    put_speech_history_on_image(img)
    cv2.imshow("WYNA", img)
    out.write(img)
    if(cv2.waitKey(30)==ord("q")) :
        break

cv2.destroyAllWindows()
