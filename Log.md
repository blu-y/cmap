
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
- Dataset searching, setting [🔗](log/04_exp_setting.md#robothome2-dataset)
- CLIP model setting [🔗](log/04_exp_setting.md#clip-model-setting)
- Semi-experiment: CLIP inference test [🔗](log/05_semi-exp#semi-experiment-clip-inference-test)

#### 2024-02-27
- Semi-experiment with other CLIP model [🔗](log/05_semi-exp#semi-experiment-clip-inference-test)
    - Result: 
    5 fps (200ms) / i7-11370H / GeForce MX450 / ViT-B-16-SigLIP(203.16M params, 46.44B FLOPs)
    10 fps (100ms) / i7-11370H / GeForce MX450 / ViT-B-32-256(151.29M params, 17.46B FLOPs)
    10.85 fps (92ms) / i7-11370H / GeForce MX450 / ViT-B-32(151.28M params, 14.78B FLOPs)
    - Laptop으로 실험하여 원하는 처리속도 (<30ms) 달성 X, 데스크탑용 GPU로 재실험 예정

#### 2024-02-28
- Dataset API [🔗](log/240226.md#robothome2-dataset)
