# Optimization of Transmit Beamforming With Channel Covariances for MISO Downlink Assisted by Reconfigurable Intelligent Surfaces 

>[!NOTE]
>[Manuscript Accepted]

<div align="justify">We propose an <b>unsupervised</b> beamforming neural network (BNN) to optimize transmit beamforming in downlink multiple input single output (MISO) channels. Our proposed BNN utilizes only channel covariances of UEs, which do not change often, and hence the transmit beams do not need frequent updates. The BNN outperforms the ZF scheme when the UE channels are sparse with <b>rank one</b> covariance. The sum-rate gain over ZF is pronounced in heavily loaded systems in which the number of UEs is closer to that of the BS antennas. The complexity of the BNN is shown to be much lower than that of the ZF. Future work includes improving the BNN for channel covariances whose rank is greater than one and joint optimization of the transmit beams with RIS elements.</div>

### System Model
***
The implementation of the neural network model is adapted from [TianLin0509/BF-design-with-DL](https://github.com/TianLin0509/BF-design-with-DL) to meet our system requriements.

> [!IMPORTANT]
> For details on the custom Downlink Beamforming with Reconfigurable Intelligent Surface environment, please refer to the paper: [](Will be published on IEEE Xplore in May, 2024).
<div align="center">
  <img src="https://github.com/khinthandarkyaw98/BF-RIS-Channel-Covariance-DeepLearning/blob/main/systemModel/fig1.png">
</div>

### Simulation Parameters
|  Parameter |  Current Value | 
|---|---|
|  Number of UEs|  Default: 8 <br/> Otherwise: 6, or 10|  
|  Number of BS transmit antenna ($N_t$)  | Default: 16 <br/> Otherwise: 10 |  
|  Number of RIS elements ($N$) |  Default: 30 <br/> Otherwise: 60 | 
|  Downlink bandwidth | Assume mmWave Frequencies > 30 GHz |
|  Channel bandwidth |  Rayleigh Fading Model |
|  Antenna configuration|  MISO |
|  Frequency reuse scheme | Large Frequency Reuse Factor |
|  Mobility model | Stationary |
|  Learning type| Unsupervised |


### Implementation Details of the proposed BNN
| Layer Name | Output Dimension | Activation Function |
|---|---|---|
| Input layer 1 | [M+K, 2, $N_t$, $N_t$] |<center>-</center> |
| Input layer 2 | [1] |<center>-</center> |
| Input layer 3 | [M+K, 2, $N_t$, 1] |<center>-</center> |
| Concatenate layer | [2$N_t$(M+K)($N_t$+1)+1, 1] |<center>-</center> |
| Dense layer 1 | [256, 1] |<center>softplus</center> |
| Dense layer 2 | [128, 1] |<center>softplus</center> |
| Dense layer 3 | [64, 1] |<center>softplus</center> |
| Lambda layer 1 | [32, 1] |<center>-</center> |
| Lambda layer 2 | [32, 1] |<center>-</center> |
| Dense layer 4 | [M+K, 1] |<center>softplus</center> |
| Dense layer 5 | [M+K, 1] |<center>softplus</center> |
| Lambda layer 3 | [M+K, 1] |<center>-</center> |
| Lambda layer 4 | [M+K, 1] |<center>-</center> |
| Lambda layer 5 | [M+K, $N_t$, 1] |<center>-</center> |
| Lambda layer 6 | [1] |<center>-</center> |

### Training Hyperparameters of BNN
| Hyperparameters | Value |
|---|---|
| Number of episodes | Maximum episodes = $500$ |
| Mini-batch size | $32$ samples |
| Network weight initializations | Keras' default wegihts |
| Optimizer | Adam |
| Learning rate | Maximium value = $1e-5$ <br/> Minimum value = $1e-7$ |

### Numerical Results
***
Figures of the sum rates and computaion time in the paper are found in the folder [sumRates](./sumRates/) and [elapsedTime](./elapsedTime/Bar_time.png) respectively or as belows.  The hyperparameters follow all figures presented in the paper. 

<div align="center">
  <img src="https://github.com/khinthandarkyaw98/BF-RIS-Channel-Covariance-DeepLearning/blob/main/sumRates/fig2.png" style="width:400px; height:350px">
  <img src="https://github.com/khinthandarkyaw98/BF-RIS-Channel-Covariance-DeepLearning/blob/main/sumRates/fig3.png" style="width:400px; height:350px">
  <img src="https://github.com/khinthandarkyaw98/BF-RIS-Channel-Covariance-DeepLearning/blob/main/sumRates/fig4.png" style="width:400px; height:350px">
  <img src="https://github.com/khinthandarkyaw98/BF-RIS-Channel-Covariance-DeepLearning/blob/main/elapsedTime/Bar_time.png" style="width:400px; height:350px">
</div>

Please modify `N`, `Nt`, `totalUsers`, `Lm`, `Lk` in [NNUtils.py](./NNUtils.py) and respective `python` `plot` files to reproduce all figures in the paper.

### How to use
***
**0.Requirements**
```bash
python==3.10.10
matplotlib==3.7.1s
numpy==1.24.3
tensorflow==2.15.0
keras==2.15.0
```

**1.Implementation**
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








