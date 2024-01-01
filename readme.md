# Optimization of Downlink Beamforming with Reconfigurable Intelligent Surface using Channel Covariances

### System Model

See the folder [sytemModel](./systemModel/fig1.png) to understand the custom Downlink Beamforming with Reconfigurable Intelligent Surface environment.

The architecture of the proposed neural network model is shown in [systemModel](/systemModel/modelArchitecture.png).

For more details, please view the paper [Insert URL]().

### Results
Figures of the sum rates in the paper are found in the folder [sumRates](./sumRates/). Please look at folder [lossCurves](./lossCurves/Fig3/) to know the loss rates of `Fig. 3` in the paper. The hyperparameters follow all figures presented in the paper. 

Please modify `N` in [NNUtils.py](./NNUtils.py) and respective `python` `plot` files to reproduce all figures in the paper.


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
    git clone https://github.com/khinthandarkyaw98/BF-RIS-Channel-Covariance-DeepLearning
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

* Train the model: 
  ```bash
  python TrainSuper.py
  ```

* Test the model:
  ```bash
  python TestSuper.py
  ```
  
* Plotting the graph:
  ```bash
  python plot.py
  ```
  
Loss curves and sum rate plots can also be viewed in `train` and  `Plotting` folders which will be automatically created after running the abovementioned files.

### Using the code
Please cite this repository if you utilize our code:
```
```

