import cv2
import sys, re

## 해도리도리도리 

# #입력파일지정하기
if len(sys.argv) <= 1:
    print("no input file")
    quit()

image_file =sys.argv[1]
# image_file = "51.png"
#출력파일 이름
output_file = re.sub(r'\,jpg|jpeg|PNG$', 'output2.jpg',image_file)

#캐스캐이드 파일 경로 지정하기
cascade_file = "haarcascade_frontalface_alt.xml"

# 이미지 읽어들이기
image = cv2.imread(image_file)
# 그레이스케일로 변환하기
image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 얼굴 인식 특징 파일 읽어들이기
cascade = cv2.CascadeClassifier(cascade_file)
# 얼굴인식 실행하기
face_list = cascade.detectMultiScale(image_gs,
                                     scaleFactor=1.1,
                                     minNeighbors=1,
                                     minSize=(10, 10))

if len(face_list) == 0:
    print("no face")
    quit()

#확인한 부분에 모자이크 걸기
print(face_list)
color = (0, 0, 255)
mosaic_rate = 30
for (x, y, w, h) in face_list:
    #얼굴부분자르기
    face_img = image[y:y+h, x:x+w]
    #자른 이미지를 지정한 배율로 확대/축소하기
    face_img = cv2.resize(face_img, (w//mosaic_rate, h//mosaic_rate))
    #확대/축소한 그림을 원래 크기로 돌리기
    face_img = cv2.resize(face_img, (w, h), interpolation=cv2.INTER_LINEAR)
    #원래이미지에붙이기
    image[y:y+h, x:x+w] = face_img
#렌더링 결과를 파일에 출력
cv2.imwrite('output2.PNG', image)
