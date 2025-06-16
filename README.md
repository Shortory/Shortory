# 🎬 숏토리(Shortory) - 시청자 반응 기반 숏폼 자동 생성 서비스  


<h3 align="center">찰나를 포착하다, <strong>Shortory(숏토리)</strong></h3>






<p align="center">
  <img src="https://github.com/user-attachments/assets/255f9526-a164-4130-b27d-3e6b5ef13110" width="300" />
</p>

 
<h4 align="center">시청자의 감정과 댓글의 타임스탬프를 분석해 영상 속 하이라이트를 자동으로 추출하는 숏폼 제작 웹 서비스</strong></h4>

<br />

## 프로젝트 개요
**숏토리(shortory)** 는 유튜브 댓글의 타임스탬프와 시청자의 표정을 분석하여, 숏폼(Short-form) 하이라이트 영상을 자동 생성하는 웹 서비스입니다.  
웹캠을 사용하여 사용자의 감정 데이터를 분석하고, 분석 결과를 바탕으로 감정이 두드러지게 나타나는 장면을 자동으로 선정하여 숏폼 영상을 생성합니다. 

<br />

## 핵심 아이디어

> 댓글과 표정, 모두가 말해주는 진짜 하이라이트!
> 
- `타임스탬프` 가 포함된 댓글 자동 수집&분석
- 웹캠 기반 **표정 분석으로 감정 + 집중도** 추적
- **몰입도 높은 순간**만 골라 감정 기반 하이라이트 완성!

<br />

## ⭐️ 주요 기능 
### 1. 감정 기반 숏츠 생성

시청자의 **웹캠 영상을 프레임 단위로 분석** 하여, 얼굴에서 추출한 감정과 집중도를 바탕으로 5가지 감정(`happy`, `sad`, `angry`, `neutral`, `surprise`) 중 가장 높은 확률의 감정을 기반으로 숏폼 영상을 자동 생성합니다.

### 2. 타임스탬프 기준 숏츠 생성

**유튜브 댓글 속 타임스탬프** 를 자동으로 추출하고, **가장 많이 언급된 상위 5개 구간을 기준으로** 하이라이트 클립 영상을 자동 생성합니다.

| 기능 | 설명 |
| --- | --- |
| 🔍 댓글 분석 | 유튜브 댓글에서 타임스탬프 수집 -> 그룹화 → 빈도 계산 → 상위 5개 → 클립 생성 |
| 🧠 감정/집중 분석 | 표정 분석과 집중도 분석을 통해 감정 몰입도 높은 장면 감지 |
| ✂️ 클립 생성 | ±10초 범위 영상 추출 |
| 📂 숏폼 분류 | 감정별 숏폼 정렬 및 다운로드 |


<br />

## ⚙️ Architecture 구조도

<p align="center">
  <img src="https://github.com/user-attachments/assets/9354f1c0-da93-4daf-b28c-4f297fa61d68" alt="아키텍처" width="70%">
</p>


<br />

## 타임스탬프(timestamp) 정의

**타임스탬프**는 영상 속 특정 시점을 나타내는 시간 정보로, 일반적으로 `"00:45"`, `"3:15"`, `"12:34"`와 같은 **`분:초` 또는 `시:분:초` 형식**으로 표현됩니다.

유튜브 댓글에서는 시청자들이 인상 깊었던 장면에 대해 “**10:43 아이유 레전드**”처럼 타임스탬프를 남기며 명장면을 직접 표시합니다.

> ✅ 숏토리에서는 이 댓글 속 타임스탬프를 자동 감지하여, 시청자가 선택한 하이라이트 구간을 숏폼 영상으로 자동 생성합니다.
>

<img src="https://github.com/user-attachments/assets/fb7c50c5-995d-4a43-9227-73cd851575a2" alt="타임스탬프 예시" width="60%">

출처: 뿅뿅지구오락실3 tvN D ENT 유튜브 영상 댓글 화면 캡처
<br>
https://www.youtube.com/watch?v=aLF3YHUvm7E 


<br />

## 감정 인식 모델 생성 과정
MobileNetV2 기반 전이 학습(Transfer Learning)과 파인튜닝(Fine-tuning)을 활용하여 감정 인식 모델을 구축하였습니다.
emotion_tl2_model.h5 모델을 다운받아 실행 가능합니다.

아래 버튼을 클릭하면 Colab에서 직접 확인할 수 있습니다:

👉 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1x7WFKhHi4zHMAH6r4oPw4L2PdZCwyqqW?usp=sharing)

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


<br />


## 결과 화면
<h4 align="center">[메인 화면]</strong></h4>
<p align="center">
  <img src="https://github.com/user-attachments/assets/ecc7faff-036a-49ef-ae78-73f645144bff" width="700"/>
</p>

<br>
<h4 align="center">[감정 분석 기반 숏폼 생성]</strong></h4>

<p align="center">
  <img src="https://github.com/user-attachments/assets/c1d13bfb-fb07-4805-a288-fe94f46e25ae" width="700"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/cd7020b7-423a-4f74-96ee-617b13228042" width="700"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/e3357126-5745-46cc-949c-35716c54babf" width="700"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/1d266558-a883-4040-b43c-c1818b0b33c2" width="700"/>

<br>
<h4 align="center">[타임스탬프 기반 숏폼 생성]</strong></h4>

<p align="center">
  <img src="https://github.com/user-attachments/assets/8a930645-317e-440b-983e-13675fbcb409" width="700" />
  
<p align="center">
  <img src="https://github.com/user-attachments/assets/612ce1b7-e56e-4361-b134-efd1e6195990" width="700" />

<p align="center">
  <img src="https://github.com/user-attachments/assets/3b49a1d3-f3ec-43e4-9740-119852824c05" width="700" />

</p>

## 실행 방법
**1. 가상환경 및 패키지 설치**
<br>
pip install -r requirements.txt

**2. 감정 인식 모델 다운로드**
<br>
🔗 https://drive.google.com/file/d/18ryNf-Tt2eEFnr6hsnPOJA6nmwyaEuwA/view?usp=share_link

다운로드 후 프로젝트 내의 models 폴더에 저장

**3. Flask 서버 실행**
<br>
python app.py



<br />   

## 프로젝트 폴더 구조

```
shortoty_web/
│
├── backend/                          # 백엔드 로직 처리 (Flask 서버 및 분석 스크립트)
│   ├── app.py                        # 메인 Flask 서버 및 라우팅
│   ├── run_analysis.py               # 실시간 감정 분석 처리 로직
│   └── create_shorts.py              # 댓글 기반 타임스탬프 분석 및 숏폼 생성
│
├── models/                           # 감정 분석을 위한 학습 모델
│   └── emotion_tl2_model.h5          # 감정 분류 모델 (Keras 기반)
│
├── timestamp_uploads/                # 댓글 기반 분석을 위한 원본 영상 저장 폴더
│
├── emotion_uploads/                  # 감정 분석용 원본 영상 저장 폴더
│
├── static/                           # 정적 파일 및 분석 결과 저장소
│   ├── shorts_output/                # 감정 분석 기반 숏폼 영상 결과
│   │   └── categories/               # 감정별 클립 분류 (Angry, Happy, etc.)
│   └── timestamp_output/             # 댓글 기반 숏폼 결과 저장
│
├── frontend/                         # 프론트엔드 화면 구성 (Flask HTML 템플릿)
│   └── templates/
│       ├── home.html                # 메인 진입 페이지
│       ├── emotion_form.html        # 감정 분석용 URL 입력 폼
│       ├── timestamp_form.html      # 댓글 기반 분석 URL 입력 폼
│       ├── loading.html             # 댓글 분석 대기 화면
│       ├── analyzing.html           # 감정 분석 진행 중 화면
│       ├── result.html              # 감정 분석 결과 페이지
│       ├── categories.html          # 감정별 클립 보관함
│       └── shorts_comment_result.html # 댓글 기반 숏폼 결과 및 다운로드
│
├── requirements.txt                  # 의존성 라이브러리 목록
└── README.md                         # 프로젝트 개요 및 설명 문서

```

## 깃허브 폴더 설명

- **backend/** : Flask 서버와 분석 스크립트 (app.py, run_analysis.py, create_shorts.py)
- **models/** : 감정 분석 모델 파일
- **frontend/** : 사용자 인터페이스 HTML 템플릿
- **static/** : 생성된 숏폼 영상 및 감정별 클립 저장 폴더
- **emotion_uploads/** : 감정 분석용 업로드 원본 영상 저장
- **timestamp_uploads/** : 댓글 기반 분석용 업로드 원본 영상 저장

<br />

<p align="center"><strong>
  
## 💚 팀원 소개

<table align="center">
  <thead>
    <tr align="center">
      <th>FULLSTACK</th>
      <th>BACKEND</th>
      <th>AI</th>
      <th>AI & DESIGN</th>
    </tr>
  </thead>
  <tbody>
    <tr align="center">
      <td>황채원</td>
      <td>문성원</td>
      <td>박시현</td>
      <td>한송미</td>
    </tr>
    <tr align="center">
      <td><img src="https://github.com/user-attachments/assets/0a99a63f-3aee-4581-b3c6-f8d7a3017a27" width="100"/></td>
      <td><img src="https://github.com/user-attachments/assets/bb1d2e9a-f1e6-4a12-ba8c-19bcfcab2492" width="100"/></td>
      <td><img src="https://github.com/user-attachments/assets/ce52b7d9-39cc-4004-ac1b-9caf91b4b8e6" width="100"/></td>
      <td><img src="https://github.com/user-attachments/assets/07d9bece-9517-4d7f-9bf3-b553fbb877cb" width="100"/></td>
    </tr>
    <tr align="center">
      <td><a href="https://github.com/ChaewonHwang-01">ChaewonHwang-01</a></td>
      <td><a href="https://github.com/m-seongwon">m-seongwon</a></td>
      <td><a href="https://github.com/Sihyun32">Sihyun32</a></td>
      <td><a href="https://github.com/0weny">0weny</a></td>
    </tr>
  </tbody>
</table>
