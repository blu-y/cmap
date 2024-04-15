## Cmap Node
#### from Turtlebot
```
ros2 launch ~/cmap/cmap/launch/run.launch.py
```
```
python ./cmap/src/cmap_node.py 
```
```
python ./cmap/src/cmap_goal.py
```

#### from bag
```
ros2 bag play <bag_file> --clock
```
```
ros2 launch ~/cmap/cmap/launch/run.launch.py
```
```
python ./cmap/src/cmap_node.py 
```
```
python ./cmap/src/cmap_goal.py
```

#### from bag + extra image
```
python ./cmap/src/image_publisher.py 
```
```
ros2 launch turtlebot4_viz view_robot_cmap.launch.py use_sim_time:=true
```
```
python ./cmap/src/cmap_node.py 
```
```
python ./cmap/src/cmap_goal.py
```
```
ros2 bag play <bag_file> --clock
```


#### Result
![result1](../image/cmap_node_1.png) 
![result2](../image/cmap_node_2.png) 