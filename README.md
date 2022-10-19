## ROS package for ConvPoint Python

This repo contains the ROS package for the Python implementation of ConvPoint model.

Before running the code, install the KNN module by compiling it:

```bash
$ cd convpoint_ws/src/convpoint/knn
$ python setup.py install --home="."
```

To run the code, clone the repo into the home directory and follow the commands.

```bash
$ cd convpoint_ws
$ chmod +x run.sh
$ ./run.sh
```
---
**In the run_segmentation.launch file,**

to change the log file location,

```markdown
<!-- Path to log file  -->
<param name="log_file" value="<whole path to the log file>" />
```

to change the model weights location,
```markdown
<!-- Path to model ckpt dir. -->
<param name="model_dir" value="<path to the pretrained weights dir>" />
```

to change the hyperparameters,
```markdown
<param name="batch_size" value="8" type="int" />
<param name="npoints" value="5000" type="int" />
```

to rename the topic,
```markdown
<remap from="segmentation/velodyne_points"        to="/segmentation/map_local_seg" />
```
---

**In the run_rosconvpoint file,**

to change the segmented color map, lines 233 - 237,

```python
self.map_label_to_color = {
    0: [128, 0, 0],  # maroon -> unlandable
    1: [128, 128, 0]  # greenyellow -> landable
    }
```
to rename the predictions folder, line 245,
```py
Path(os.path.join(self.logdir, "predictions")).mkdir(parents=True, exist_ok=True)
```

to change the time file name of the runtime log, line 246,
```py
self.time_file = os.path.join(self.logdir, "infer_time_ros.csv")
```

to change the subscriber topics, lines 258, 259,
```py
self.seg_pub = rospy.Publisher("segmentation/colored_map", PointCloud2, queue_size=1)
self.vel_sub = rospy.Subscriber("segmented/map", PointCloud2, self.infer_callback, queue_size=1)
```

to change the predictions file name, line 304,
```py
save_fname = os.path.join(self.logdir, "predictions", str(self.i)+"_lbls.txt")
```
