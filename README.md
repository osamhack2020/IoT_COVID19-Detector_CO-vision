# Initial page

> [2020 군장병 공개SW 온라인 해커톤](https://osam.kr/main/page.jsp?pid=offline.offline19)

![CO-vision\_LOGO](https://user-images.githubusercontent.com/41141851/97099932-c0968d00-16d1-11eb-96b0-1afd2c0c455f.PNG)

### 프로젝트 소개

#### Co-Vision

코로나를 물리치자 \[추가 예정2\]

### 팀소개

#### 팀 구성원

* 조동현 Donghyeon Cho \(hyeon9698@naver.com\), Github Id: [hyeon9698](https://github.com/hyeon9698)
* 허동준 

  **프로젝트 설명**

  **프로젝트 배경**

  **기능 설계**

  * 발사믹, 카카오 오븐 등 본인이 편한 목업 프레임워크를 이용하여 제작 후 링크 
  * 수기로 작성시 찍어서 올려주세요

### 컴퓨터 구성 / 필수 조건 안내 \(Prerequisites\)

* ECMAScript 6 지원 브라우저 사용
* 권장: Google Chrome 버젼 77 이상

### 기술 스택 \(Technique Used\) \(예시\)

#### Server\(back-end\)

* Keras/Tensorflow
* python
* opencv
* [MobileNet V2](https://arxiv.org/abs/1801.04381)
* Google-vision API
* telepot \(tellegram\)
* nodejs, php, java 등 서버 언어 버전 
* express, laravel, sptring boot 등 사용한 프레임워크 
* DB 등 사용한 다른 프로그램 

  **embedded devices**

* Raspberry Pi or NVIDIA Jetson Nano

  **front-end**

* react.js, vue.js 등 사용한 front-end 프레임워크 
* UI framework
* 기타 사용한 라이브러리

### 설치 안내 \(Installation Process\)

```bash
$ git clone git주소
$ pip install opencv-python
$ pip install google-cloud-vision
$ pip install telepot
```

### 프로젝트 사용법 \(Getting Started\)

**마크다운 문법을 이용하여 자유롭게 기재**

#### Mask Training

**구조**

```bash
├── dataset
│   ├── with_mask [690 entries]
│   └── without_mask [686 entries]
├── examples
│   ├── example_01.png
│   ├── example_02.png
│   └── example_03.png
├── face_detector
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── detect_mask_image.py
├── detect_mask_video.py
├── mask_detector.model
├── plot.png
└── train_mask_detector.py
```

잘 모를 경우 구글 검색 - 마크다운 문법 [https://post.naver.com/viewer/postView.nhn?volumeNo=24627214&memberNo=42458017](https://post.naver.com/viewer/postView.nhn?volumeNo=24627214&memberNo=42458017)

편한 마크다운 에디터를 찾아서 사용 샘플 에디터 [https://stackedit.io/app\#](https://stackedit.io/app#)

### 팀 정보 \(Team Information\)

* Donghyeon Cho \(hyeon9698@naver.com\), Github Id: hyeon9698
* kim su ji \(suji999@gmail.com\), Github Id: suji999

### 저작권 및 사용권 정보 \(Copyleft / End User License\)

* [MIT](https://github.com/osam2020-WEB/Sample-ProjectName-TeamName/blob/master/license.md)

  **😷**

