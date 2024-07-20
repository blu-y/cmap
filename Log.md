
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
- ~~Jetson Orin Nano CUDA Solved, but performance is same with CPU only~~ (Solved)
- Retry with clean installed ubuntu (with no conda) [🔗](log/10_jetson.md#installing-python-packages)
- ~~Semi-exp with Jetson Orin Nano (CUDA)~~
    - ~~Result:  ~~
    ~~0.66 fps (1504 ms) / Cortex A78AE / 1024-core Ampere / ViT-B-16-SigLIP(203.16M params, 46.44B FLOPs)~~
    ~~2.07 fps (482 ms) / Cortex A78AE / 1024-core Ampere / ViT-B-32-256(151.29M params, 17.46B FLOPs)~~
    ~~2.56 fps (391 ms)  / Cortex A78AE / 1024-core Ampere / ViT-B-32(151.28M params, 14.78B FLOPs)~~
    - ~~torch.cuda.is_available() = True로 나왔으나 성능이 CPU Only와 똑같음. --> 추후 docker를 사용하여 재시도~~ (Solved)

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

#### 2024-04-05~06
- CMAP 패키지 실험 [🔗](log/11_cmap_node.md)
    - Result:  
    - 초기에 찍은 사진은 잘 되는 경향이 있으나, 점점 frame이 쌓여갈수록 정확도가 떨어지는 듯함  
    - 다시 본 frame을 업데이트 하는 알고리즘이 필요함  
    - SLAM의 loop closure 같이 정확도를 향상시키는 알고리즘이 필요함
    - Good working case [🔗](https://www.youtube.com/watch?v=xkdtDuR6BVM)
    - Bad working case [🔗](https://www.youtube.com/watch?v=ydCWAEa_9SU)
      - Bad working case에서 pose부분이 뒤늦게 따라오는 것을 확인
      - 과거의 pose를 optimization하는 과정에서 과거가 아닌 현재시간으로 db를 제작하는 듯
      - pose토픽이 아니라 tf를 이용하여 실시간 처리로 개선할 필요 있음
  
#### 2024-04-09~10
- Bunker
  - Explore 패키지 수정 (max distance, heading angle)
  - 연석 부분은 라이다 위치 때문에, 거리에 따라 인식 불가
    - 너무 가까우면 로봇 몸체에 막혀서 인식 X
    - 너무 멀면 채널 사이 공간에 연석이 들어가 인식 X

#### 2024-04-11~13
- Bunker
  - Explore 패키지 수정 (exploration percentage 측정 알고리즘)
    - costmap 업데이트 과정에서 오류가 있음, copy 또는 mutex를 사용해야 할 듯
- CMAP 패키지 작성 [🔗](log/11_cmap_node.md)
- CMAP 실험
  - place, object, ambiguous로 나누어 실험.
    - place: 
    - object: 
    - ambiguous: 
- CMAP 논문 작성

#### 2024-04-14
- CMAP 논문 작성
  
#### 2024-04-15
- Bunker
  - map topic을 내부에 저장하여 exploration rate 계산
  - 시뮬레이션 결과 오류가 발생하지 않음. rate가 조금씩 꾸준히 증가.
- CMAP 논문 작성, 제출

#### 2024-04-18
- Bunker
  - Scan position 수정
    - 이전 scan과의 distance 계산, >20m 일 때 다른 모든 scan과 >20m이면 scan.
- CMAP 논문 수정, 제출

#### 2024-04-19
- Bunker
  - Scan position, explore node와 상호작용
  - Scan 중 explore 일시정지

#### 2024-04-21~22
- Bunker
  - 최적 scan position으로 이동하도록 frontier의 cost에 추가
    - scan position과의 거리가 20m일 때 cost가 가장 낮도록.
    - min( abs(400-distance^2), 1200 )

#### 2024-04-23
- Bunker
  - Frontier의 cost에 heading angle 항목 추가
    - 이전에는 heading angle이 우선 -> cost로 추가해서 비교
      - 방법 1: heading angle 내부에서 -1500 주는 방법
      - 방법 2: angle differenct(deg) * 100 하는 방법

#### 2024-04-24~26
- CMAP 논문 작성

#### 2024-05-02~03
- Jetson Orin Nano (Restart)
  - Docker 사용 환경설정

#### 2024-05-06
- Jetson Orin Nano
  - Docker 환경에서 cmap 세팅 후 commit (cmap:0.1)

#### 2024-05-07
- Jetson Orin Nano
  - Docker 사용하여 GPU 사용 성공
    - 원인을 찾아본 결과 코드 내에서 GPU 사용 오류
      - docker을 사용하지 않아도 사용이 가능할 듯
      - 하지만 docker 사용하는 이점도 있어 계속 사용
    - Text encoder(Tokenizer)에서 오류가 나서 수정 필요
  - Result
    - ~~20 fps (50 ms) / Cortex A78AE / 1024-core Ampere / ViT-B-16-SigLIP(203.16M params, 46.44B FLOPs)~~
  - 새로운 HW에서의 카메라 해상도(현재는 250x250)를 고려하여 모델 선택 가능할 듯.

#### 2024-05-08
- Text encoder에서 오류가 나서 수정 필요
  - Result(Jetson)
    - 14.69 fps (68 ms) / Cortex A78AE / 1024-core Ampere / ViT-B-32(151.28M params, 14.78B FLOPs)  
    - 14.10 fps (71 ms) / Cortex A78AE / 1024-core Ampere / ViT-B-32-256(151.29M params, 17.46B FLOPs)  
    - 13.85 fps (72 ms) / Cortex A78AE / 1024-core Ampere / ViT-B-16-SigLIP(203.16M params, 46.44B FLOPs)
    - 6.35 fps (158 ms) / Cortex A78AE / 1024-core Ampere / ViT-L-14-quickgelu(427.62M params, 175.33B FLOPs)
    - ~~-- fps (-- ms) / Cortex A78AE / 1024-core Ampere / ViT-H-14-quickgelu(986.11M params, 381.68B FLOPs)~~(failed)

#### 2024-05-21~22
  - Result(Laptop)
    - 15.75 fps (64 ms) / i7-11370H / GeForce MX450 / ViT-B-32(151.28M params, 14.78B FLOPs)  
    - 9.05 fps (111 ms) / i7-11370H / GeForce MX450 / ViT-B-32-256(151.29M params, 17.46B FLOPs)  
    - 5.53 fps (181 ms) / i7-11370H / GeForce MX450 / ViT-B-16-SigLIP(203.16M params, 46.44B FLOPs)
    - ~~-- fps (-- ms) / i7-11370H / GeForce MX450 / ViT-L-14-quickgelu(427.62M params, 175.33B FLOPs)~~(failed)
    - ~~-- fps (-- ms) / i7-11370H / GeForce MX450 / ViT-H-14-quickgelu(986.11M params, 381.68B FLOPs)~~(failed)

#### 2024-05-23~28
  - CMAP Upgrade 방안
    - Divide frame into 3 sub-frame and localize
      - Map building 시 localize vs. Inference 시 localize
      - Inference 시 localize하는 방법이 적절하다고 판단

#### 2024-05-29~30
  - Jetson Orin Nano Setup 파일 작성

#### 2024-06-03~04
  - Sub-frame semi experiment
    - 3분할로 process하니 속도가 1/3, 이 부분은 병렬로 처리가 가능할지 의문
  - Tracer Mini Setting [~~🔗~~](log/12_tracer.md)
    - base setting
    - LiDAR setting
    - SLAM setting
      - base_link의 x축이 반대로 되어있어 제대로 mapping되지 않음

#### 2024-06-06
  - Kinova Gen3 Lite Manipulator demo [🔗](log/13_kinova.md)
    - 최신 Driver는 x86으로 작성되어서 Jetson에서 구동 불가
    - 설정해준 Driver는 Jetson호환이지만 Gen3 Lite 세팅파일이 없음
    - 두 파일을 합쳐서 일단 Jetson에서 Gen3 Lite 사용 가능하게 설정함

#### 2024-06-07
  - Tracer Mini SLAM 정상 구동 확인
    - base 노드가 출력하는 base_link frame을 base_link_reversed로 변경, static tf publisher로 base_link frame을 reverse 시켜 publish
  - 3분할 처리 프로세스
    - Mapping:
      - 현재 scan을 map 좌표계에서 voxelize (0.25 x 0.25 m)
      - 3분할 CLIP vector을 각 화각의 voxel에 방향과 거리와 함께 저장
      - 겹치는 부분은 거리, 방향에 따라 update (한 voxel에 근거리 8방향, 원거리 4방향으로 12방향으로 제한)(작은 공간에서도 용량이 커지는 문제 있음)
    - Inference:
      - cosine similarity 순서로 정렬, 음수값 제외
      - log probability, linear probability 사용하여 localize

#### 2024-06-08~11
  - 3분할 처리
    - filtering scan points in camera angle(L,M,R)
    - voxelizing scan points
    - minimize camera delay using threading (75%)
  
#### 2024-06-12
  - 3분할 처리
    - enhanced performance in CLIP with batch (>1.5x)
    - getting local images
    - tf extrapolation solved
  
#### 2024-06-13~14
  - 3분할 처리
    - 결과로 나온 features을 이용하여 입력 text에 해당하는 point 찾기
    - ~~softmax similarity로 정렬 후 0.01로 cut한 후 해당 frame에서 각 point에 누적된 similarity/n_point로 찾기~~(failed)

#### 2024-06-15
  - 3분할 처리
    - 결과로 나온 features을 이용하여 입력 text에 해당하는 point 찾기
    - softmax similarity로 정렬 후 0.01로 cut한 후 해당 frame에서 각 point에 누적된 similarity로 찾기
    - softmax similarity로 정렬 후 0.01로 cut한 후 해당 frame에서 각 point에 누적된 횟수로 찾기
      - ~~두 실험 모두 비슷하긴 하지만 완전 정확한 곳으로 이동하지 못하는 것을 확인(같이 찍힌 다른 point들이 문제)~~(해결)
        - 사용하는 range의 거리를 0.5m~5.0m로 제한
    - tf data가 밀려서 제대로 voxel로 저장이 안되는 것을 확인
      - multithread로 실행한 결과 tf는 밀리지 않았지만 clip 연산이 느려졌다(5~6fps -> 4fps)
      - multithread를 2, 4, 6으로 모두 실행하여도 3fps으로 동일함.

#### 2024-06-19
  - 3분할 처리
    - 정확도 문제를 해결하기 위해 확인해 본 결과
    - 카메라 frame은 cw로 indexing 한 반면, lidar는 ccw로 indexing한 것을 확인
    - 해당 오류 해결하니 제대로 작동함.

#### 2024-06-20,22
  - 긴 시간의 sequence를 실험하기 위한 데이터 수집
    - exp0620, exp0622
  - space도 넓게 복도까지 포함(복도에서 SLAM이 삐뚤어지는 것을 확인)

#### 2024-06-23
  - PPT작성

#### 2024-06-24
  - 3분할 처리
    - exp0622 파일 테스트
      - ~~프레임이 1fps정도 나오는 것을 확인~~(해결, split하는 과정에서 오류가 있었음)

#### 2024-06-25~30
  - Mapsaver
  - Probability map generation
  - Experiment with 4 ViT, n=1,2,3,4,5
  - 발표자료 만들기

#### 2024-07-01~10
  - 학회 발표 및 논문 작성 마무리

#### 2024-07-12
  - 방학 중 계획
      - 패키지 완성
      - LLM finetuning / Jetson embedding
      - Manipulator demo

TODO: 
  - keyframe selection 적용
  - voxel update (이미지 품질을 확인하는 과정이 필요)

      
#### 2024-04~
- CMAP 패키지화 [🔗](log/11_cmap_node.md)

#### TODO
- **Short Term**
  - Ablation study (KFS, IQE)
  - ~~Keyframe selection  (로봇이 천천히 움직이면 중복되는 프레임이 너무 많아져서 데이터가 너무 커진다 / viewpoint에 따라 keyframe인지 확인해야함)~~(Done, weighted keyframe selection)
  - Point에 image embedding mapping
  - Feature update
  - Exploration
  - ~~HW 보완 Camera 높이 올리기, more cameras, PC에 직접 연결, 화각~~(Done)
  - Backend Operation (Queue)
  - Lifelong mapping (맵 저장 및 로드, feature 저장 및 로드)
  - TensorRT
  - ~~Jetson Orin Nano experiment using docker~~(Done)
- **Long Term**
  - 오검출 교차검증 필요, 알고리즘 보완해야함
  - AI Powered search
  - data diet
  - filtering points with multi camera view
  - saving lesser dimension with dimension reduction techniques
  - GUI, search, sort
  - Other dimension reduction techniques (NMF, SVD, ICA)
