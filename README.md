# DSAI-ElectrictyBidding

The goal of this project is to design an agent which helps with electricity bidding to minimize the electricty bill of house.

## Data

The dataset used in this project would be accumulation of 50 household dataset```NASDAQ:IBM```.  
Data contains 3 columns: ```Hourly Power Consumption(kWh), Hourly Solar Power Generation(kWh), Hourly Bidding records```  
The dataset contains 8 months worth of dataset
<!-- ![Table sample](/img/table.png) -->

The chart below shows the graph of the training data (*close* data):  
<!-- ![Chart sample](/img/chart.png)   -->

The dataset is then splited into two parts :
- Power consumption (kWh)
- Solar Power generation (kWh
our goal is create two dataset which allows us to train two different LSTM model to each predict :
- Consumption rate
- Generation rate
Which then be used to create the perfect agent for electricity bidding.

## Training with Long Short-Term Memory (LSTM)
Since we are mainly focusing on the time series problem. 
- LSTM contains feedback connections which make them different to traditional feedforward neural networks.
- which allows useful information about preious data in sequence to help with the processing of new data points.

### How does LSTM works?
3 main dependencies:
- cell state (current long-term memory of network)
- previous hidden state (output of previous data)
- input data 

LSTM uses series of gates to control information flow
which acts as filters to generate information for training:
- forget gate
- input gate
- output gates  

For more detailed explanation, please refer here [LSTM](https://towardsdatascience.com/lstm-networks-a-detailed-explanation-8fae6aefc7f9)

### Model Architecture
Below is the summary of the model architecture used:  
![Model summary](/img/model.png)

## Testing
To test the data
not yet written
We used 2 different methods of testing:  
1. Split training data to test data [First 800 data are training, the rest are test].
![Prediction chart: Split](/img/predict.png)  
2. Use all training data to train, test data use ```testing.csv```  
![Prediction chart: Separated](/img/test.png)  

The model should be fed with actual test data, otherwise the result will be terrible.  
![Bad test](/img/badtest.png)  

## Trading Algorithm
Our approach to minimize electricity bill:
- Check for the electricity usage per hour
- If the **generated** > **consumption**, then use it for the day and sell the rest for **generated amount + 0.2**
- If the **generated** < **consumption**, then buy the needed electricity with the price of **1 unit for 2.3**
- If the **generated** = **consumption**, do no action