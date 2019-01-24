import cv2 
  
# capture frames from a video 
cap = cv2.VideoCapture(0)##"F:/PYTHON/Cars Detection/video2.avi") 
  
# Trained XML classifiers describes some features of some object we want to detect 
car_cascade = cv2.CascadeClassifier('cars.xml')
faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0, 255, 0)  
# loop runs if capturing has been initialized. 
while True: 
    # reads frames from a video 
    ret, frames = cap.read() 
    count_list = [] 
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY) 
      
  
    # Detects cars of different sizes in the input image 
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    people = faceDetect.detectMultiScale(gray,1.1,1)
    count = 0
    # To draw a rectangle in each cars 
    for (x,y,w,h) in cars: 
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(frames,"Car",(x,y+h),fontFace,fontScale,fontColor)
       
    for (x,y,w,h) in people: 
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(frames,"People",(x,y+h),fontFace,fontScale,fontColor)
        
  
   # Display frames in a window  
    cv2.imshow('video2', frames) 
      
    # Wait for Esc key to stop 
    if cv2.waitKey(33) == 27: 
        break
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()
print(count_list)
