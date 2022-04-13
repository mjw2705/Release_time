# 양궁 행동분석

## Description
양궁의 릴리즈 타임 측정 기술

1. [활 검출](https://github.com/mjw2705/Release_time/tree/master/release_bow)

    얼굴 roi 내에서 Canny 엣지를 통해 활을 검출 하여 활의 움직임을 보고 릴리즈 타임 측정

2. [바디 스켈레톤 검출](https://github.com/mjw2705/Release_time/tree/master/release_angle)

    mediapipe를 이용한 팔 각도 측정으로 팔 각도 및 팔의 움직임을 보고 릴리즈 타임 측정


## Demo
### 활 검출 
![bow_demo](bow_demo.gif)

### 바디 스켈레톤 검출
![body_demo](body_demo.gif)