
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
    5.62 s / Cortex A57 / CPU Only / ViT-B-32(151.28M params, 14.78B FLOPs)
- *separate_session s_x_v >> s_x_c*

#### 2024-03-18~19
- Preparation for data collecting [🔗](log/08_datagen.md)
- Turtlebot4 패키지 코드 분석
  - turtlebot4_description (URDF)
  - turtlebot4_viz
- 'GLIBCXX_3.4.30' not found 오류 [🔗](log/00_debug.md#glibcxx_3430-not-found)
- ImportError: /lib/libgdal.so.30 오류  [🔗](log/00_debug.md#importerror-liblibgdalso30)
  
#### 2024-03-21
- Bunker
  - Rosbag으로 explore하기 debug
    - pub_tf arguments in bunker_control_base.launch was declared 'string' not 'bool'
    - LeGO-LoAM simulation parameter to true

#### 2024-03-22
- Jetson Orin Nano Jetpack 6 DP 설치 [🔗](log/10_jetson.md)

#### 2024-03-23
- Semi-exp with Jetson Orin Nano
    - Result:  
    0.94 fps (1067 ms) / Cortex A78AE / CPU Only / ViT-B-16-SigLIP(203.16M params, 46.44B FLOPs)  
    1.76 fps (567 ms) / Cortex A78AE / CPU Only / ViT-B-32-256(151.29M params, 17.46B FLOPs)  
    2.51 fps (398 ms)  / Cortex A78AE / CPU Only / ViT-B-32(151.28M params, 14.78B FLOPs)  
    - CUDA unavailable 문제로 CPU Only로 실험, 문제 해결 중.
- Bunker
  - tf 문제 해결 후 재실행하여 제어 문제 해결되는지 확인

#### 2024-03-24
- Jetson Orin Nano Pytorch CUDA 문제 해결 중 (torch는 해결하였으나 torchvision 설치 불가)
- Bunker
  - Outdoor exp, Indoor exp 진행
  - dynamic object가 obstacle로 등록됨(octomap)
  - segment point pure에서 수신되는 point가 너무 적음(octomap)
  - 뚫려있는 곳을 바라보면 그 곳은 free space로 등록하지 않아 unexplored 상태임(octomap)
  - 벽에 가까이 있으면 local path가 소극적이며 이상하게 동작(m-explore)

#### 2024-03-25
- Jetson Orin Nano CUDA Solved, but performance is same with CPU only
- Retry with clean installed ubuntu (with no conda) [🔗](log/10_jetson.md#installing-python-packages)
- Semi-exp with Jetson Orin Nano (CUDA)
    - Result:  
    0.66 fps (1504 ms) / Cortex A78AE / 1024-core Ampere / ViT-B-16-SigLIP(203.16M params, 46.44B FLOPs)  
    2.07 fps (482 ms) / Cortex A78AE / 1024-core Ampere / ViT-B-32-256(151.29M params, 17.46B FLOPs)  
    2.56 fps (391 ms)  / Cortex A78AE / 1024-core Ampere / ViT-B-32(151.28M params, 14.78B FLOPs)  
    - torch.cuda.is_available() = True로 나왔으나 성능이 CPU Only와 똑같음. --> 추후 docker를 사용하여 재시도

#### 2024-03-27~28
- DRL Term Project 1

#### 2024-03-30
- SLAM Toolbox, Turtlebot4 패키지 코드 분석
  - turtlebot4_navigation
    - `ros2 launch turtlebot4_navigation slam.launch.py`
  - slam_toolbox

#### 2024-03-31~01
- Bunker
  - 지난 실험 replay를 통해 ground 정보가 map에 반영되지 않음을 확인  
  - Octomap에 들어가는 cloud는 이미 LeGO-LOAM에서 바닥이 제거된 cloud.
  - Octomap에서는 따로 바닥 제거 알고리즘이 존재하며, 여기서 분류된 바닥은 free space를 등록하는 데 사용되나 바닥이 없는 cloud가 들어가면서 바닥이 free space로 등록되지 않음.
  - 따라서 octomap에서 LeGO-LOAM의 ground_cloud를 subscribe하여 이를 free space로 등록하도록 수정하였음.
  - IMU를 사용하여 실험해 보았지만, 오히려 더 오차가 커졌음.
  - Indoor exp[🔗](https://www.youtube.com/watch?v=zUZfFqObhPU), Outdoor exp[🔗](https://www.youtube.com/watch?v=9FNuZD3T66I) 진행
    - 바닥이 free space로 등록되어 map이 더욱 잘 생성되는 것을 확인. path 또한 잘 생성됨.
    - LeGO-LOAM에서 Localization이 부정확하여(rolling 발생, z축으로 가라앉거나 떠오름) 바닥의 z값이 달라져, free space로 등록되지 않는 경우때문에 map 제작이 불완전함.


#### 2024-04-02
- CMAP 패키지 작성 [🔗](log/11_cmap_node.md)
  - cmap node 작성
    - embed previous works (CLIP, VIZ, PCA)
  - 작성된 부분 실험
  
#### 2024-04-03~04
- CMAP 패키지 작성 [🔗](log/11_cmap_node.md)
  - Goal point publisher node 작성
  - cmap node goal point calculation 부분 작성
  - 작성된 부분 실험
  
#### 2024-04-05~06
- CMAP 패키지 작성 [🔗](log/11_cmap_node.md)
  - Keyframe selection
  - 패키지화


#### TODO (장기)
- Jetson Orin Nano experiment using docker
- slam toolbox에서 pose estimation할 때 odometry를 사용하는지?
- 오검출 교차검증 필요, 알고리즘 보완해야함
- AI Powered search
- CLIP vector가 이미 normalized된 건지?
- data diet
- filtering points with multi camera view
- how to choose keyframe (현재 사용하는 방식 조사, 보완, 좋은 화질의 frame 필요)
- saving lesser dimension with dimension reduction techniques
- GUI, search, sort
- Other dimension reduction techniques (NMF, SVD, ICA)
