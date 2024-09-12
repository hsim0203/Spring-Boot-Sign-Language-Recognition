from flask import Flask
from flask import request
import base64
import json

from tensorflow import keras
import mediapipe as mp
import numpy.linalg as LA
import numpy as np
import cv2
import matplotlib.pyplot as plt

#LSTM 모델 경로
model = keras.models.load_model('C://ai_project01/hand_lstm_train_result')
#LSTM모델 출력
model.summary()

#손동작 5번째마다 예측
seq_length = 5

gesture = {
    0:'A', 1:"B", 2:"C", 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P',
    16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z',
}

mp_hands = mp.solutions.hands

app = Flask(__name__)


@app.route("/lstm_detect", methods=["POST"])
def lstm_detect01():
    #탐지 결과를 저장하는 변수
    lstm_result = []
    #손동작을 저장하는 리스트
    seq = []

    #화면에서 손과 손가락 위치 탐지
    with mp_hands.Hands() as hands:
        json_image = request.get_json()
        print("=" * 100)
        print("json_image=", json_image)
        print("=" * 100)

        encoded_data_arr = json_image.get("data")
        print("=" * 100)
        print("encoded_data_arr=", encoded_data_arr)
        print("=" * 100)

        for index, encoded_data in enumerate(encoded_data_arr):
            print("=" * 100)
            print("index=", index)
            print("=" * 100)
            print("encoded_data=", encoded_data)
            print("=" * 100)

            encoded_data = encoded_data.replace("image/jpeg;base64,", "")
            decoded_data = base64.b64decode(encoded_data)

            #decoded_data -> 1차원 배열 변환
            nparr = np.fromstring(decoded_data, np.uint8)
            print("=" * 100)
            print("nparr=",nparr)
            print("=" * 100)

            #nparr -> BGR 3차원 배열 변환
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            print("=" * 100)
            print("image=", image)
            print("=" * 100)

            #BGR -> RGB로 변환 후 손, 손가락 관절위치 탐지 후 리턴
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # results.multi_hand_landmarks - 탐지된 손의 keypoint 값들이 저장
            if results.multi_hand_landmarks != None:

                #hand_landmarks에 탐지된 keypoint값을 순서대로 1개씩 저장,
                for hand_landmarks in results.multi_hand_landmarks:

                    joint = np.zeros((21, 3))

                    #hand_landmarks.landmark - 손의 keypoint 좌표 리턴
                    #j - keypoint의 index
                    #lm - keypoint의 좌표
                    for j, lm in enumerate(hand_landmarks.landmark):
                        print("j=", j)
                        print("lm=", lm)
                        #keypoint의 x,y,z좌표
                        print("lm.x", lm.x)
                        print("lm.y", lm.y)
                        print("lm.z", lm.z)
                        #각도를 구하기 위해 x,y,z 좌표 배열에 대입
                        joint[j] = [lm.x, lm.y, lm.z]
                        print("=" * 100)

                    
                    print("=" * 100)
                    print("joint=", joint)
                    print("=" * 100)

                    
                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0,  9, 10, 11,  0, 13, 14, 15,  0, 17, 18, 19], :]
                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
                    
                    #v1에서 v2를 빼서 거리를 계산
                    #v는 2차원 배열
                    v = v2 - v1
                    print("=" * 100)
                    print("v=", v)
                    print("=" * 100)

                    #v를 1차원배열로 정규화
                    v_normal = LA.norm(v, axis=1)
                    print("=" * 100)
                    print("v_normal=", v_normal)
                    print("=" * 100)

                    #v와 연산을 위해 2차원배열로 변환
                    v_normal2 = v_normal[:, np.newaxis]
                    print("=" * 100)
                    print("v_normal2=", v_normal2)
                    print("=" * 100)

                    #v/v_normal2 로 나눠서 거리를 정규화
                    v2 = v / v_normal2
                    print("=" * 100)
                    print("v2=", v2)
                    print("=" * 100)


                    a = v2[[0, 1, 2, 4, 5, 6, 8,  9, 10, 12, 13, 14, 16, 17, 18], :]
                    b = v2[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]

                    #ein - 행렬곱
                    #a와 b 배열의 곱 계산
                    ein = np.einsum('ij,ij->i', a, b)
                    print("=" * 100)
                    print("ein=", ein)
                    print("=" * 100)

                    #radian - 코사인값(1차원 배열)
                    radian = np.arccos(ein)
                    print("=" * 100)
                    print("radian=", radian)
                    print("=" * 100)

                    #radian값을 각도로 변환
                    angle = np.degrees(radian)
                    print("=" * 100)
                    print("angle=", angle)
                    print("=" * 100)

                    #joint.flatten() - 관절 좌표를 1차원 배열로 변환
                    # data에 관절 좌표, 각도 저장
                    data = np.concatenate([joint.flatten(), angle])
                    print("=" * 100)
                    print("data=", data)
                    print("=" * 100)

                    seq.append(data)

                    if len(seq) < 5:
                        continue

                    #last_seq <- 마지막 손동작 5개 행 대입
                    last_seq = seq[-5:]
                    #last_seq -> 배열로 변환
                    input_arr = np.array(last_seq, dtype=np.float32)
                    print("input_arr=", input_arr.shape)

                    #input_arr을 3차원 배열로 변환
                    input_lstm_arr = input_arr.reshape(1, 5, 78)
                    print("=" * 100)
                    print("input_lstm_arr=", input_lstm_arr)
                    print("=" * 100)
                    print("=" * 100)
                    print("input_lstm_arr.shape=", input_lstm_arr.shape)
                    print("=" * 100)

                    #y_pred <- lstm 모델을 통해 수어 예측 후 대입
                    y_pred = model.predict(input_lstm_arr)
                    print("=" * 100)
                    print("y_pred=", y_pred)
                    print("=" * 100)

                    #idx <- y_pred에서 가장 예측 확률 값이 가장 높은 인덱스 대입
                    idx = int(np.argmax(y_pred))
                    print("=" * 100)
                    print("idx=", idx)
                    print("=" * 100)


                    letter = gesture[idx]
                    print("=" * 100)
                    print("letter=", letter)
                    print("=" * 100)

                    #conf <- idx번째의 확률 대입
                    conf = y_pred[0, idx]
                    print("=" * 100)
                    print("conf=", conf)
                    print("=" * 100)

                    #탐지 결과 lstm_result에 추가
                    lstm_result.append({
                        "text":f"{letter} { round(conf * 100, 2) } percent!!"
                        #손위치의 x,y 좌표
                        ,"x": int(hand_landmarks.landmark[0].x * image.shape[1])
                        ,"y": int(hand_landmarks.landmark[0].y * image.shape[0])
                    })

                    print("=" * 100)
                    print("lstm_result=",lstm_result)
                    print("=" * 100)

    #결과를 json으로 변환 후 return
    return json.dumps(lstm_result)

#JSON으로 변환된 이미지 수신 테스트
#문구 반환 및 json 객체 출력
@app.route("/image_test01", methods=["POST"])
def image_send_test01():
    json_image = request.get_json
    print("json_image=", json_image)

    return "스프링 컨트롤러가 보낸 이미지 잘 받았습니다"


#전송받은 json 이미지 인코딩후 저장하는 함수 테스트
@app.route("/image_test02", methods=["POST"])
def image_send_test02():
    #image 저장
    json_image = request.get_json()
    print("json_image=", json_image)
    print("=" * 100)

    #image에서 data의 값 저장
    encoded_data_arr = json_image.get("data")
    print("=" * 100)
    print("encoded_data_arr=", encoded_data_arr)
    print("=" * 100)

    #배열에 저장된 data를 이미지 순서대로 1개씩 저장
    for index, encoded_data in enumerate(encoded_data_arr):
        print("=" * 100)
        print("index=",index)
        print("=" * 100)
        print("encoded_data=",encoded_data)
        print("=" * 100)
        #인코딩된 데이터에서 image/jpeg;base64, 제거
        encoded_data = encoded_data.replace("image/jpeg;base64,","")
        decoded_data = base64.b64decode(encoded_data)

        #index번호.jpg로 파일 저장
        with open(f"image{index}.jpg","wb") as f:
            f.write(decoded_data)

    return "스프링 컨트롤러가 보낸 이미지 잘 받았습니다"

if __name__ == '__main__':
    app.run()