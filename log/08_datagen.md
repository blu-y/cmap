## Dataset Generation

##### 데이터셋 제작 시 사용되는 주요 방법

1. 광학 모션 캡처 시스템 (OMC): 고정된 카메라 시스템을 사용. 높은 정확도, 높은 비용.
2. 관성 측정 장치 (IMU): 가속도계와 자이로스코프를 사용. OMC에 비해 저렴하지만, 오차 누적, 정밀도 낮음.
3. 마커 기반 시스템: 대상에 마커를 부착하고, 카메라로 마커의 위치 트래킹. 설치 간편, 마커의 가시성에 영향을 받음, 실외 사용 어려움.
4. LiDAR 스캐너: 레이저를 사용하여 주변 환경의 3D 구조를 측정합니다. 높은 정확도, 높은 비용, 실내 환경에서 효과적.
5. SLAM (Simultaneous Localization and Mapping): 카메라나 LiDAR 센서를 사용하여 위치 추정과 지도 생성. 낮은 비용, 실시간 처리, 낮은 정확도.

##### TUM 데이터셋에서 ground truth 좌표를 얻는 데 사용된 방법

IMU 데이터: Xsens MTi-G 센서. 3축 가속도계, 3축 자이로스코프, 3축 마그네토미터.
GNSS 데이터: u-blox LEA-6T 센서.
모션 캡처 시스템: Vicon Motion Capture System. 12개의 카메라.

#### Dataset Generation with TB4

터틀봇4를 사용하여 SLAM으로 얻은 좌표를 ground truth로 하는 dataset 생성

##### TF to save
```
# dynamic
map  
└── odom  
    ├── base_footprint  
    └── base_link  
# static
base_link  
├── imu_link  
└── shell_link  
    ├── rplidar_link  
    └── oakd_link  
        └── oakd_rgb_camera_frame  
            └── oakd_rgb_camera_optical_frame 
``` 

##### How to see TF Tree

```bash
ros2 run tf2_tools view_frames # to save tf tree
ros2 run tf2_ros tf2_echo [source_frame] [target_frame] # to see transform between to frames
```

##### Topics to save

|Topic|Type|Count|
|---|---|---|
|/odom | nav_msgs/msg/Odometry|1765|
|/scan | sensor_msgs/msg/LaserScan|819|
|/imu | sensor_msgs/msg/Imu|3612|
|/pose | geometry_msgs/msg/PoseWithCovarianceStamped|253|
|/map | nav_msgs/msg/OccupancyGrid|225|
|/oakd/rgb/preview/image_raw/compressed | sensor_msgs/msg/CompressedImage|2928|
|/tf_static | tf2_msgs/msg/TFMessage|4|
|/tf | tf2_msgs/msg/TFMessage|6460|
* Count is example for 112s rosbag data

### Usage
```bash
python3 ./ex/datagen.py
```
```bash
ros2 bag play rosbag2_2024_03_15-20_17_07
```
#### Datastructure
```
cmap  
└── dataset  
    └── tb4  
        ├── image  
        │   └── image_files.png  
        ├── pcd 
        │   └── pcd_files.pcd  
        ├── db.csv 
        ├── imu.csv
        ├── odometry.csv
        ├── pose.csv
        ├── tf_static.csv
        └── tf.csv
``` 

##### Use generated data
```bash
mkdir -p dataset/tb4
cd dataset
tar -xvf tb4.tar.xz
```