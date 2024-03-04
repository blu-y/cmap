## Robot@Home2 Dataset
[Dataset Page](https://zenodo.org/records/7811795)

##### Set Dataset Path
```python
from robotathome import RobotAtHome, logger
my_rh_path = '~/cmap/dataset'
my_rgbd_path = os.path.join(my_rh_path, 'files/rgbd')
my_scene_path = os.path.join(my_rh_path, 'files/scene')
my_wspc_path = '~/cmap/dataset/result'
```
##### Dataset Loading
```python
try: 
      rh = RobotAtHome(my_rh_path, my_rgbd_path, my_scene_path, my_wspc_path)
except:
      logger.error("Something was wrong")
```
##### Get RGBD LASER data
```python
df = rh.get_sensor_observations('rgbdlsr')
print(df.head())
```
##### Sort with timestamp
```python
df = df.sort_values("timestamp")
print(df.head())
```
Laser, RGBD 1,2,3,4 확인
##### Extract RGBD Data
RGBD_1 데이터만 추출
```python
df = df.loc[df.sensor_name=="RGBD_1"]
df = df.reset_index(drop=True)
print(df.head())
```
##### ID List Generation
id를 통하여 데이터의 좌표, 이미지를 추출할 수 있기 때문에 id list를 생성
```python
ids = df.id.tolist()
```
##### Get RGBD Frame by ID
get_RGBD_files로 id를 통해 RGB와 depth frame을 추출할 수 있다.
```python
from PIL import Image
for id in ids:
    [rgb_frame, d_frame] = rh.get_RGBD_files(id)
    image = Image.open(rgb_frame)
    plt.imshow(image)
    plt.show()
```