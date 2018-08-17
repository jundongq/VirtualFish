# VirtualFish
Virtual fish developed with deep learning technique

1. Training data of virtual fish is stored in the folder 'Data'
2. Trained virtual fish is the folder

## Virtual fish concept, definition and performance are illustrated as below:
### 1. Virtual Fish Concept
The virtual fish model contains ANN and CNN modules, which takes temporal information and spatial information flow respectively.

<img src="https://github.com/jundongq/VirtualFish/blob/master/TrainedModel/VirtualFish_1.jpg" width="600">


### 2. Virtual Fish Model Architecture
Virutal fish takes multiple inputs, hydrodynamic cues such as velocity (u, v), vorticity, turbulence kinetic energy, swirl, and strain rate

<img src="https://github.com/jundongq/VirtualFish/blob/master/TrainedModel/VirtualFish_2.jpg" width="600">


### 3. Virtual Fish Action Definition
There are four discrete actions for virtual fish

<img src="https://github.com/jundongq/VirtualFish/blob/master/Virtual%20Fish%20Action%20Definition.jpg" width="600">

### 4. Run Virtual Fish in Virtual Environment
Put virtual fish in a virtual Eulerian environment, it makes decisions based on its perception on surrounding hydrodynamics. The decision takes it to next location.

<img src="https://github.com/jundongq/VirtualFish/blob/master/EL_VirtualFish.jpg" width="600">

### 5. Virtual Fish Trend Compared to Observed Fish Trend
Running 364 virtual fish in 52 virtual environment, the averaged trend of virtual fish trajectory with 1 standard devision is shown in this plot. The averaged trend of virtual fish is in a reasonable agreement with observed fish trajectory trend.

<img src="https://github.com/jundongq/VirtualFish/blob/master/VirtualFishTrajectories_RealFishTrajectories_1sd.png" width="1200">



