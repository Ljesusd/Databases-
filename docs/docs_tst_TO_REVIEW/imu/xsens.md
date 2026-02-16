***
> :bulb:**Note:**
> 
> This example might be outdated. Please refer to the specific project folder using that equipment for the latest versions.
> This folder is kept for reference purposes only and to illustrate the generic conversion characterize_element.
***

# Data pipeline XSens:

An XLS file is often the output of the IMU software. 
Only a subset of sheets is often of interest:

- **Sensor Orientation – Euler,** that holds information about gyroscope.
- **Sensor Magnetic Field,** that holds information about magnetograph.
- **Sensor Free Acceleration,** that holds information about accelerometer.
- **General Information,** that holds information about the recording conditions. This is a special sheet that contains information about the recording characterize_element. The content of the file is as follows in the next example.


> | MVN version | 2022.0.0 |
> | --- | --- |
> | Original File Name | C:\Users\Usuario\Documents\NEUROMARK\SOFT\subject_04_cond_03_run_-002.mvn |
> | Suit Label | Subject_04 |
> | Recorded Date UTC | 17/oct/2022 10:23:25 |
> | Frame Rate | 60 |
> | Processing Quality | Live |

* * *
## **INPUT:** 

A XLS with pages according to the XSENS plain text export format is the input source. The pages contain the following information:

1. General Information
2. Segment Orientation - Quat
3. **Segment Orientation - Euler**
4. Segment Position
5. Segment Velocity
6. **Segment Acceleration**
7. Segment Angular Velocity
8. Segment Angular Acceleration
9. Joint Angles ZXY
10. Joint Angles XZY
11. Ergonomic Joint Angles ZXY
12. Ergonomic Joint Angles XZY
13. Center of Mass
14. **Sensor Free Acceleration**
15. Sensor Magnetic Field
16. Sensor Orientation - Quat
17. Sensor Orientation – Euler

Only a subset of sheets is of interest to the project:

- **Sensor Orientation – Euler,** that holds information about gyroscope.
- **Sensor Magnetic Field,** that holds information about magnetograph.
- **Sensor Free Acceleration,** that holds information about accelerometer.

### **Example:**

> | Frame | Pelvis x | Pelvis y | Pelvis z | T8 x | T8 y | T8 z | Head x | Head y | Head z |
> | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
> | 0 | 0,0528 | -0,718 | -0,168 | -0,427 | 0,546 | 0,006 | 0,037 | -0,553 | 0,677 |
> | 1 | 0,0432 | -0,719 | -0,163 | -0,433 | 0,541 | 0,017 | 0,028 | -0,551 | 0,685 |
> | 2 | 0,0476 | -0,725 | -0,168 | -0,428 | 0,537 | 0,007 | 0,033 | -0,557 | 0,682 |
> | 3 | 0,0429 | -0,719 | -0,166 | -0,433 | 0,543 | 0,011 | 0,030 | -0,552 | 0,688 |
> | 4 | 0,0539 | -0,718 | -0,170 | -0,424 | 0,542 | 0,004 | 0,041 | -0,552 | 0,678 |
> | 5 | 0,0519 | -0,721 | -0,169 | -0,424 | 0,542 | 0,009 | 0,038 | -0,557 | 0,678 |

* * *
## **OUTPUT:** 
csv with columns following the EUROBENCH format:

### **Example:**

> | time  | `lumbar_x` | `lumbar_y` | `lumbar_z` | `r_shoulder_x` | `r_shoulder_y` | `r_shoulder_z` | ... |
> |-------|------------|------------|------------|----------------|----------------|----------------|-----|
> | 0.003 | 0.014      | 0.004      | 0.016      | 0.005          | 0.006          | -0.039         | ... |
> | 0.016 | 0.01       | 0.001      | 0.012      | 0.001          | -0.003         | -0.039         | ... |
> | 0.033 | 0.04       | 0.008      | -0.02      | 0.085          | -0.002         | -0.047         | ... |
> | 0.055 | -0.013     | -0.007     | -0.013     | 0.025          | -0.011         | -0.068         | ... |
> | 0.066 | -0.006     | -0.009     | -0.024     | 0.043          | -0.007         | -0.056         | ... |

- For the magnetometer file the suffix _magnetometer is included in the filename
- For the accelerometer file the suffix _accelerometer is included in the filename
- For the gyroscope file the suffix _gyroscope is included in the filename