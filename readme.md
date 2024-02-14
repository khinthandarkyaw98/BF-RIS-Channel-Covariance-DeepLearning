# Optimization of Transmit Beamforming With Channel Covariances for MISO Downlink Assisted by Reconfigurable Intelligent Surfaces

<div align="justify">Our proposed BNN utilizes only channel covariances of UEs, which do not change often, and hence the transmit beams do not need frequent updates. The BNN outperforms the ZF scheme when the UE channels are sparse with <b>rank one</b> covariance. The sum-rate gain over ZF is pronounced in heavily loaded systems in which the number of UEs is closer to that of the BS antennas. The complexity of the BNN is shown to be much lower than that of the ZF. Future work includes improving the BNN for channel covariances whose rank is greater than one and joint optimization of the transmit beams with RIS elements.</div>

### System Model

The implementation of the neural network model is adapted from [TianLin0509/BF-design-with-DL](https://github.com/TianLin0509/BF-design-with-DL) to meet our system requriements.

> [!IMPORTANT]
> For details on the custom Downlink Beamforming with Reconfigurable Intelligent Surface environment, please refer to the paper: [](Insert URL).
<div align="center">
  <img src="https://github.com/khinthandarkyaw98/BF-RIS-Channel-Covariance-DeepLearning/blob/main/systemModel/fig1.png">
</div>

### Numerical Results
Figures of the sum rates and computaion time in the paper are found in the folder [sumRates](./sumRates/) and [elapsedTime](./elapsedTime/Bar_time.png) respectively or as belows.  The hyperparameters follow all figures presented in the paper. 

<div align="center">
  <img src="https://github.com/khinthandarkyaw98/BF-RIS-Channel-Covariance-DeepLearning/blob/main/sumRates/fig2.png" style="width:400px; height:350px">
  <img src="https://github.com/khinthandarkyaw98/BF-RIS-Channel-Covariance-DeepLearning/blob/main/sumRates/fig3.png" style="width:400px; height:350px">
  <img src="https://github.com/khinthandarkyaw98/BF-RIS-Channel-Covariance-DeepLearning/blob/main/sumRates/fig4.png" style="width:400px; height:350px">
  <img src="https://github.com/khinthandarkyaw98/BF-RIS-Channel-Covariance-DeepLearning/blob/main/elapsedTime/Bar_time.png" style="width:400px; height:350px">
</div>

Please modify `N`, `Nt`, `totalUsers`, `Lm`, `Lk` in [NNUtils.py](./NNUtils.py) and respective `python` `plot` files to reproduce all figures in the paper.

### Run
**0.Requirements**
```bash
python==3.10.10
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
  python plot_corresponding_number_.py
  ```
  
Eplased time info, Loss curves and sum rate plots can also be viewed in `timer`, `train` and  `Plotting` folders which will be automatically created after running the abovementioned files.








