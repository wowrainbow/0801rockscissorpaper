# AIFFEL Data Scientist Campus Code Peer Review Templete

코더 : 남태욱

리뷰어 : 배성우

---

🔑 **PRT(Peer Review Template)**

[ o]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
- 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
	- (문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 퀘스트 문제 요구조건 등을 지칭)
- 해당 조건을 만족하는 코드를 캡쳐해 근거로 첨부
2. 오버피팅을 극복하기 위한 적절한 시도가 있었는가?	오버피팅 극복을 위하여 데이터셋의 다양성, 정규화 등을 2가지 이상 시도해보았음
![딥러닝](https://github.com/user-attachments/assets/c38f144b-fc8c-4469-9331-d95f968fa228)

![딥러닝2](https://github.com/user-attachments/assets/21c0dae9-06b4-4cbc-94e5-f9d8fa81b1c9)


정규화 과정과 데이터 셋을 추가했다.

3. 분류모델의 test accuracy가 기준 이상 높게 나왔는가?	85% 이상 도달하였음
![딥러닝 3](https://github.com/user-attachments/assets/63032903-4ac7-44d6-a3ed-1371301e86f5)


[o ]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
	주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
- 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
- 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
- 주석을 보고 코드 이해가 잘 되었는지 확인
	- 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
 n_channel_1=8
n_channel_2=16
n_dense=32
n_train_epoch=3 #하이퍼파라미터 조정

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
model.fit(x_train, y_train, epochs=n_train_epoch)
좋은 데이터셋을 넣었더니 n_train_epoch=3 -> 으로 해도 결과가 잘 나왔더니 좋았다. 
이 부분에 대해서 학습 데이터를 풍부하게 넣지 않은 나는 많은 시도(못해도 20번)은 넘게 이 부분을 돌린 거 같은데
딥러닝 CV는 요리와 같다는 느낌을 지울 수 없었다.
명확하고 깨끗한 사진을 많이 넣어서 학습을 시키면 학습이 잘 되고 정확도가 잘 나오는데
사진이 많지 않고, 깨끗하지 않거나 주변 배경이 많이 복잡하면 안에서 배치 정규화, dropout, 데이터 불균형등의 코드등을 아무리 넣고 검증 데이터로 분리해서 검증을 해봐도 결과물이 좋지 않다.

요리는 재료가 70퍼 조리가 30퍼라는 말이 있듯이, 좋은 재료를 최대한 많이 구하는 것이 우선인 것을 깨달았다. 
[o ]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록"을 남겼거나 "새로운 시도 
또는 추가 실험"을 수행해봤나요?**
- 문제 원인 및 해결 과정을 잘 기록하였는지 확인 또는
- 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 실험이 기록되어 있는지 확인
	- 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
 다양한 시도를 했지만, 결국 학습데이터가 학습을 제대로 못하는 경우가 주요 원인이라
다양한 데이터를 집어넣어서 정확도를 올리는 수행을 한 것이 고무적이다.
  image_dir_path = os.getenv('HOME') + '/aiffel/rock_scissor_paper'
(x_train, y_train)=load_data(image_dir_path)
x_train_norm = x_train/255.0

print('x_train shape {}'.format(x_train.shape))
print('y_train shape {}'.format(y_train.shape))
학습데이터의 이미지 개수는 2394 입니다.
x_train shape (3000, 28, 28, 3)
y_train shape (3000,)      
[o ]  **4. 회고를 잘 작성했나요?**
- 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해 배운점과 아쉬운점, 느낀점 등이 상세히 기록되어 있는지 확인
    - 딥러닝 모델의 경우, 인풋이 들어가 최종적으로 아웃풋이 나오기까지의 전체 흐름을 도식화하여 모델 아키텍쳐에 대한 이해를 돕고 있는지 확인

문제가 발생한 원인이 뭔지 바로 찾고 , 학습 데이터 셋을 많이 넣은 것이 인상적이다. 결과적인 부분에 있어서 효과적으로 작성을 하신 것 같다. 

[o ]  **5. 코드가 간결하고 효율적인가요?**
- 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
- 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 모듈화(함수화) 했는지
	- 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.
#일단 학습데이터 늘려보기, 캐글에서 받은 png파일 변환하기

def resize_pngimages(img_path):
    images = glob.glob(img_path + '/*.png')
    
    print(len(images), 'images to be resized')
    
    target_size=(28,28)
    for img in images:
        old_img=Image.open(img)
        new_img=old_img.resize(target_size,Image.ANTIALIAS)
        new_img.save(img, 'JPEG')
        
    print(len(images), 'images resized.')
    
image_dir_path = os.getenv('HOME') + '/aiffel/rock_scissor_paper/scissor'
resize_pngimages(image_dir_path)
print('가위 이미지 resize 완료')

image_dir_path = os.getenv('HOME') + '/aiffel/rock_scissor_paper/rock'
resize_pngimages(image_dir_path)
print('바위 이미지 resize 완료')

image_dir_path = os.getenv('HOME') + '/aiffel/rock_scissor_paper/paper'
resize_pngimages(image_dir_path)
print('보 이미지 resize 완료')

캐글에서 가져온 이미지도 변환할 수 있도록 함수를 잘 적용했다. 

---
### 참고 문헌
