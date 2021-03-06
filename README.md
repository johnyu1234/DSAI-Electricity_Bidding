# DSAI-ElectrictyBidding

The goal of this project is to design an agent which helps with electricity bidding to minimize the electricty bill of house.

## Data

The dataset used in this project would be accumulation of ```50 household dataset```.  
Data contains 3 columns: ```Hourly Power Consumption(kWh), Hourly Solar Power Generation(kWh), Hourly Bidding records```  
>

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
Two different trained model, same architecture
![Model summary](/img/model_1.png)
![Model summary](/img/model_2.png)
## Testing
To test the data
Since there are two values that we are focusing on:
1. prediction value of generated electricty

![Prediction chart: Split](/img/generation.png)  

2. prediction value of consumption electricty

![Prediction chart: Separated](/img/consumption.png)  

This is an example of the prediction of each model

## Trading Algorithm
Our approach to minimize electricity bill:
- Check for the electricity usage per hour
- If the **generated** > **consumption**, then use it for the day and sell the remaining with the price of **1 unit for 2.2**
- If the **generated** < **consumption**, then buy the needed electricity with the price of **1 unit for 2.3**
- If the **generated** = **consumption**, do no action