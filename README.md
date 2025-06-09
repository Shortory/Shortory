# 🎬 숏토리(Shortory) - 시청자 반응 기반 숏폼 자동 생성 서비스  
찰나를 포착하다, Shortory(숏토리) 

<p align="center">
  <img src="https://github.com/user-attachments/assets/fb1a31f0-7e21-4b52-897d-69b454030164" width="300" />
</p>

시청자의 감정 😊과 댓글의 타임스탬프(1:42)를 분석해 영상 속 하이라이트를 자동으로 추출하는 숏폼 제작 웹 서비스

---

## 📌 프로젝트 개요
*숏토리(shortory)*는 유튜브 댓글의 타임스탬프와 시청자의 시선·표정을 종합 분석하여, 숏폼(Short-form) 하이라이트 영상을 자동 생성하는 웹 서비스입니다.  
웹캠을 사용하여 사용자의 감정 데이터를 분석하고, 분석 결과를 바탕으로 감정이 두드러지게 나타나는 장면을 자동으로 선정하여 숏폼 영상을 생성합니다. 

---

## 🌱 감정 인식 모델 생성 과정
MobileNetV2 기반 전이 학습(Transfer Learning)과 파인튜닝(Fine-tuning)을 활용하여 감정 인식 모델을 구축하였습니다.

### 1. 데이터 구성
- **데이터셋**: FER2013
- **클래스**: angry, happy, neutral, sad, surprise (총 5개 감정)

### 2. 모델 구조
- **기반 모델**: MobileNetV2 (ImageNet 사전학습)

### 3. 학습 전략
- **1단계 (전이 학습)** →  **2단계 (파인튜닝)** → **클래스 불균형 보정** → **콜백**

#### 📉 Phase 1 Train VS Validation
<img width="600" alt="image" src="https://github.com/user-attachments/assets/c5567d31-d10f-4719-9613-0a7caa605ceb" />

#### 📉 Phase 2 Fine-tuning
<img width="600" alt="image" src="https://github.com/user-attachments/assets/3e408dd9-1cd0-4607-967f-db3828e4e2aa" />
