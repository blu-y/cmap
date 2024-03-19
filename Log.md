
#### 2024-02-21
- 연구실 청소
- 책상 배치 변경 및 배선정리
- 라이센스 서버 컴퓨터 설정
- Win 데스크탑 조립 및 세팅

#### 2024-02-22
- Windows에서 ROS2 설치 [-](/log/01_ros2win.md)
- ROS2 Talker, Listener test

#### 2024-02-23
- Windows에서 ROS2 환경 설정 [-](log/02_ros2win_setting.md)
- Turtlebot Topic subscribe
- teleop_twist_keyboard 설치 [-](log/02_ros2win_setting.md)
- Simple LiDAR Subscriber 작성 [🔗](log/03_lidar_viz.md)

#### 2024-02-26
- Ubuntu에서 Windows 프로그램 설치 [-](log/02_ros2win_setting.md#ubuntu에서-windows-프로그램-설치)
- Environment setting [🔗](log/04_exp_setting.md#environment-setting)
- Dataset searching, setting [🔗](log/04_exp_setting.md#robothome2-dataset)
- CLIP model setting [🔗](log/04_exp_setting.md#clip-model-setting)
- Semi-experiment: CLIP inference test [🔗](log/05_semi-exp.md#semi-experiment-clip-inference-test)

#### 2024-02-27
- Semi-experiment with other CLIP model [🔗](log/05_semi-exp.md#semi-experiment-clip-inference-test)
    - Result: 
    5 fps (200 ms) / i7-11370H / GeForce MX450 / ViT-B-16-SigLIP(203.16M params, 46.44B FLOPs)  
    10 fps (100 ms) / i7-11370H / GeForce MX450 / ViT-B-32-256(151.29M params, 17.46B FLOPs)  
    10 fps (100 ms) / i7-11370H / GeForce MX450 / ViT-B-32(151.28M params, 14.78B FLOPs)  
    - Laptop으로 실험하여 원하는 처리속도 (<30ms) 달성 X, 데스크탑용 GPU로 재실험 예정

#### 2024-02-28
- Dataset API 정리 [🔗](log/06_robotathome.md#robothome2-dataset)

#### 2024-03-04
- Dataset에서 좌표 추출 후 visualization [🔗](log/05_semi-exp.md#semi-experiment-visualization)
- Dimension reduction을 위한 general data 추출
- 기존 작업물 함수화

#### 2024-03-05
- PCA dimension reduction for visualization [🔗](log/07_dim-reduct.md#pca)
    - Result: 잘 나오긴 하였으나, 좌표계가 room 마다 새로시작되는 것 같음. data의 좌표가 home session 안에서 이어져 있는지 살펴볼 필요가 있음 

#### 2024-03-07
- Semi-exp with desktop
    - Result: 
    8.05 fps (124 ms) / i5-13400F / GeForce RTX 3090 / ViT-B-16-SigLIP(203.16M params, 46.44B FLOPs)  
    19.57 fps (51 ms) / i5-13400F / GeForce RTX 3090 / ViT-B-32-256(151.29M params, 17.46B FLOPs)  
    23.09 fps (43 ms) / i5-13400F / GeForce RTX 3090 / ViT-B-32(151.28M params, 14.78B FLOPs)

#### 2024-03-08
- PCA explained variance ratio 계산 [🔗](log/07_dim-reduct.md#pca)
    - Explained variance ratio: [0.080, 0.064, 0.052] >> 20% variance explained with 3 dimenion

#### 2024-03-11~13
- Preparation for data collecting [🔗](log/08_datagen.md)

#### 2024-03-14~15
- Jetson TX2 environment setting **(suspended)**
    - Install OS (Jetpack 4.6)
    - Install pytorch, torchvision
    - ~~Install Miniconda for python 3.7 or later~~ Aborted 
        - OpenCLIP not supported in py3.6
        - CUDA 10.2 not supported in arm conda environmnet
    - ~~Using default python with `open-clip-torch-any-py3` for py3.6~~ Aborted
        - No matching dependency for python 3.6 for other packages
    - Install python 3.7
        - Run with CPU (no cuda due to low drive space remained)
        - ~~Run with CUDA~~ *later with SSD equiped*
- Semi-exp with Jetson TX2 **(suspended)**
    - Result: **(Done with CPU)**
     5.62 s / ARM A57 / CPU Only / ViT-B-32(151.28M params, 14.78B FLOPs)
- *separate_session s_x_v >> s_x_c*

#### 2024-03-18~19
- Preparation for data collecting [🔗](log/08_datagen.md)
  - **slam toolbox에서 pose estimation할 때 odometry를 사용하는지?**
- Turtlebot4 패키지 코드 분석
  - turtlebot4_description (URDF)
  - turtlebot4_viz
- 'GLIBCXX_3.4.30' not found 오류 [🔗](log/00_debug.md#glibcxx_3430-not-found)


#### 2024-03-20
- SLAM Toolbox, Turtlebot4 패키지 코드 분석
  - turtlebot4_navigation
  - slam_toolbox
 
#### TODO
- 오검출 교차검증 필요, 알고리즘 보완해야함
- CLIP vector가 이미 normalized된 건지?
- data diet
- filtering points with multi camera view
- how to choose keyframe (현재 사용하는 방식 조사, 보완)
- saving lesser dimension with dimension reduction techniques
- Other dimension reduction techniques (NMF, SVD, ICA)
- GUI, search, sort

- github classroom / autograding
