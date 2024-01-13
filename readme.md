# Optimization of Downlink Beamforming with Reconfigurable Intelligent Surface using Channel Covariances

### System Model

See the folder [sytemModel](./systemModel/fig1.png) to understand the custom Downlink Beamforming with Reconfigurable Intelligent Surface environment.

The architecture of the proposed neural network model is shown in [systemModel](/systemModel/modelArchitecture.png) or as belows.

For more details, please view the paper [Insert URL]().

### Results
Figures of the sum rates in the paper are found in the folder [sumRates](./sumRates/) or as belows.  The hyperparameters follow all figures presented in the paper. 

Please modify `N`, `Nt`, `totalUsers`, `Lm`, `Lk` in [NNUtils.py](./NNUtils.py) and respective `python` `plot` files to reproduce all figures in the paper.


### Run
**0.Requirements**
```bash
matplotlib==3.7.1
numpy==1.24.3
tensorflow==2.15.0
keras==2.15.0
```

**1.Installing**
* Clone this Repository:
    ```bash 
    git clone https://github.com/khinthandarkyaw98/BF-RIS-Channel-Covariance-DeepLearning.git
    cd BF-RIS-Channel-Covariance-DeepLearning
    ```
* Install Python requriements:
    ```bash
    pip install -r requirements.txt
    ```

**2.Implementation**
* Generate the dataset:
  ```bash 
  python covariance.py
  ```

* Calculate the sum rate of ZF beams w/ water-filling pwr:
  ```bash 
  python waterFilling.py
  ```

* Train the model: 
  ```bash
  python TrainSuper.py
  ```

* Test the model:
  ```bash
  python TestSuper.py
  ```

* Check the elapsed time:
  ```bash 
  python timerCalculation.py
  ```
  
* Plotting the graph:
  ```bash
  python plot.py
  ```
  
Eplased time info, Loss curves and sum rate plots can also be viewed in `timer`, `train` and  `Plotting` folders which will be automatically created after running the abovementioned files.

### Using the code
Please cite this repository if you utilize our code:
```
```

### System Model
<div align="center">
  <img src="https://github.com/khinthandarkyaw98/BF-RIS-Channel-Covariance-DeepLearning/blob/main/systemModel/fig1.png" style="width:550px; height:380px">
</div>

### Numerical Results
<div align="center">
  <img src="https://github.com/khinthandarkyaw98/BF-RIS-Channel-Covariance-DeepLearning/blob/main/sumRates/fig2.png" style="width:250px; height:200px">
  <img src="https://github.com/khinthandarkyaw98/BF-RIS-Channel-Covariance-DeepLearning/blob/main/sumRates/fig3.png" style="width:250px; height:200px">
  <img src="https://github.com/khinthandarkyaw98/BF-RIS-Channel-Covariance-DeepLearning/blob/main/sumRates/fig4.png" style="width:250px; height:200px">
</div>

