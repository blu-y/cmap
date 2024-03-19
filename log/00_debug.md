## Debug Note

## 'GLIBCXX_3.4.30' not found

conda environment에서 rclpy를 import할 때 발생함.

```bash
conda install -c conda-forge gcc=12.1.0
```

위에서 해결되지 않는다면

```bash
rm /home/$USER/anaconda3/bin/../lib/libstdc++.so.6
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 /home/$USER/anaconda3/bin/../lib # 6.0.30 will be the latest version
ln -s /home/$USER/anaconda3/bin/../lib/libstdc++.so.6.0.30 /home/$USER/anaconda3/bin/../lib/libstdc++.so.6 # 6.0.30 will be the latest version
```

## ImportError: /lib/libgdal.so.30

```
    from cv_bridge.boost.cv_bridge_boost import cvtColor2
ImportError: /lib/libgdal.so.30: undefined symbol: TIFFReadRGBATileExt, version LIBTIFF_4.0
```

cv2_bridge import 할 때 발생

```bash
conda install -c conda-forge libtiff
conda install -c conda-forge gdal
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/' >> ~/.bashrc
source ~/.bashrc
```
