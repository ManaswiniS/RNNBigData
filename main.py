import pyspark.pandas as ps
import numpy as np
from pyspark.sql.types import ArrayType, DoubleType, FloatType
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql.functions import col,sum,to_date,lag,collect_list,lead
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.linalg import Vectors
from pyspark.sql.window import Window


spark_data = spark.read.csv('/FileStore/tables/train.csv', header=True, inferSchema=True)
spark_data.count()

spark_data = spark_data.withColumn("sales", col("sales").cast("float"))
spark_data = spark_data.withColumn("date", to_date(col("date")))
local_data=spark_data.toPandas()
ts_plot=local_data[["sales"]]
plt.plot(ts_plot)


col_to_normalize = "sales"


min_max_values = spark_data.selectExpr(f"min({col_to_normalize}) as min_value", f"max({col_to_normalize}) as max_value").collect()[0]

# Extract min and max values
min_value = min_max_values["min_value"]
max_value = min_max_values["max_value"]

# Perform min-max normalization for the selected column
normalized_col_name = col_to_normalize + "_normalized"
spark_data = spark_data.withColumn(normalized_col_name, (col(col_to_normalize) - min_value) / (max_value - min_value))

# Show the DataFrame with normalized column
display(spark_data)


spark_dataw=spark_data.alias("spark_dataw")

def create_timeseq_pair(data):
    X, y = [], []
    sequence_length = len(data)
    d = 0
    while d < sequence_length - 3:
        sequence_x , sequence_y = data[d:d+3] , data[d+3]
        X.append(sequence_x)
        y.append(sequence_y)
        d += 1
    return np.array(X), np.array(y)


data = ps.read_csv('/FileStore/tables/train.csv')
X, y = create_timeseq_pair(data[["sales"]].values.astype('float32'))

input_seq_length = 3  # Length of input sequences

# Define a window to create sequences
w = Window.partitionBy('store', 'item').orderBy('date')

# Create a new column 'input_sequence' containing arrays of input sequences
spark_dataw = spark_dataw.withColumn(
    'input_sequence',
    collect_list('sales_normalized').over(w.rowsBetween(0,input_seq_length-1))
)

# Create a new column 'target' containing the target sequences
spark_dataw = spark_dataw.withColumn('target', lag('sales_normalized', -input_seq_length).over(w))

# Drop rows where 'target' is null (due to lag function)
spark_dataw = spark_dataw.dropna(subset=["target"])
spark_dataw = spark_dataw.drop("date")

spark_dataw = spark_dataw.drop("sales")
spark_dataw = spark_dataw.drop("sales_normalized")

# Show the DataFrame with input sequences and target
display(spark_dataw)

filtered_df = spark_dataw.filter(spark_dataw['store']==1).filter(spark_dataw['item']==1)
filtered_df = filtered_df.drop("store")
filtered_df = filtered_df.drop("item")
# Show the filtered DataFrame
filtered_df.show(truncate=False)
train_data = filtered_df


class CustomLSTMModel:
    def __init__(self, in_size, h_size, out_size, lr):
        self.in_size, self.h_size, self.out_size  = in_size, h_size, out_size
        self.lr , self.i = lr , 0.01
        # Initializing the weights with randon values with respective input size and hidden layer size
        # For input layer
        self.W_xh = np.random.randn(h_size, in_size) * self.i
        # For hidden layer
        self.W_hh = np.random.randn(h_size, h_size) * self.i
        # For output layer
        self.W_yh = np.random.randn(out_size, h_size) * self.i
        # Initializing the bias with zeroes, for hidden layer
        self.b_h = np.zeros((h_size, 1))
        # Initializing the bias with zeroes, for the output layer.
        self.b_y = np.zeros((out_size, 1))
    
    # tanh as the activation function 
    def tanh_function(self, val):
        calculation =  np.tanh(val)
        return calculation
    
    # sigmoid as the activation function
    def sigmoid_function(self, val):
        calculation = 1 / (1 + np.exp(-val))
        return calculation
 
    
    def forward_pass(self, value_input):
        # Initializing to the zeroes
        hidden_layer = np.zeros((self.h_size, 1))
        self.previous_hidden , self.value_input = hidden_layer , value_input
        self.h_s , self.y_s = dict() , dict()

        l = 0
        while l < len(self.value_input):

            x_t = np.array(value_input[l])
            hidden_layer = self.tanh_function(np.dot(self.W_xh, x_t) + 
                                              np.dot(self.W_hh, hidden_layer) + 
                                              self.b_h)
            output = np.dot(self.W_yh, hidden_layer) 
            output +=  self.b_y
 
            self.h_s[l] = hidden_layer
            self.y_s[l] = output

            l += 1
 
        return output
 
    def backward_pass(self, d_y):
        k = len(self.value_input)
        d_W_yh , d_b_y = np.dot(d_y, self.h_s[k - 1].T) , d_y
        d_h = np.dot(self.W_yh.T, d_y)
        d_W_xh , d_W_hh = np.zeros(self.W_xh.shape) , np.zeros(self.W_hh.shape)
        d_b_h = np.zeros(self.b_h.shape)

        l = len(self.value_input) - 1

        while l >= 0 :

            d_t = np.array(d_h) * (1 - self.h_s[l] * self.h_s[l])
            d_b_h = d_b_h + d_t
            # For the input layer.
            d_W_xh , d_W_hh =  d_W_xh + np.dot(d_t, self.value_input[l]) , d_W_hh + np.dot(d_t, self.previous_hidden.T)
            # for the hidden layer.
            d_h = np.dot(self.W_hh.T, d_t)

            self.previous_hidden = self.h_s[l]

            l -= 1
        return d_W_xh, d_W_hh, d_W_yh, d_b_h, d_b_y
 
    def update_parameters(self, d_W_xh, d_W_hh, d_W_yh, d_b_h, d_b_y):

        # To perform the optimization we need to update the parameters like weights and biases of hidden and output layers.
        # Weights for input layer
        self.W_xh -= self.lr * d_W_xh
        # For output and hidden layer
        self.W_hh , self.W_yh = self.W_hh - self.lr * d_W_hh , self.W_yh - self.lr * d_W_yh
        # For bias hidden and output.
        self.b_h , self.b_y = self.b_h - self.lr * d_b_h , self.b_y - self.lr * d_b_y

    def train(self, train_features, train_targets, epochs):
        for epoch in range(epochs):
            for features, target in zip(train_features, train_targets):
                predicted_value = self.forward_pass(features)
                loss = np.square(target - predicted_value)
                dy = 2 * (predicted_value - target)
                dWxh, dWhh, dWhy, dbh, dby = self.backward_pass(dy)
                self.update_parameters(dWxh, dWhh, dWhy, dbh, dby)

    def predict(self, test_features):
        predictions = []
        for features in test_features:
            predicted_value = self.forward_pass(features)
            predictions.append(predicted_value[0][0])  # Assuming a single output value
        return np.array(predictions)
        

inp_custom = {
    "in_size" : 1,
    "h_size"  : 4,
    "out_size" : 1,
    "lr" : 0.1
}
epochs_count = 10
lstm_model = CustomLSTMModel(inp_custom["in_size"], inp_custom["h_size"], inp_custom["out_size"], inp_custom["lr"])


# Calculation loss and rmse metrics for the model 
loss_per_epoch , rmse_per_epoch = list() , list()


#function to calculate root mean square error
def calculateRootMeanSquareError(loss):
    return np.sqrt(loss/training_data_length)



#modification
currentEpoch = 1
loss = 0
training_data = train_data.rdd.collect()
training_data_length = len(training_data)

while currentEpoch <= 10:
    
    for data in training_data:
        features = data["input_sequence"]
        actual_value = data["target"]
        predicted_value = lstm_model.forward_pass(features)
        loss += np.square(actual_value - predicted_value)
        dy = 2 * (predicted_value - actual_value)
        dWxh, dWhh, dWhy, dbh, dby = lstm_model.backward_pass(dy)
        lstm_model.update_parameters(dWxh, dWhh, dWhy, dbh, dby)

    loss_per_epoch.append(loss.squeeze())
    rmse_per_epoch.append(calculateRootMeanSquareError(loss).squeeze())

    print(f"Current Epoch: {currentEpoch}, Loss: {loss}")

    #reset loss
    loss = 0
    currentEpoch += 1




#function to rescale list back to original
def rescale(input_list):
    return np.array(input_list)* (max_value - min_value) + min_value


future_predictions = []
no_of_predictions = 5
next_input = X[-1]

while no_of_predictions > 0:
    next_prediction = lstm_model.forward_pass(next_input)
    future_predictions.append(next_prediction[0][0])
    next_input = np.vstack((next_input[1:], next_prediction[0][0]))
    no_of_predictions -= 1

future_predictions = rescale(future_predictions)
print("Future predicted Values:", future_predictions)




# Identify the most recent date in the Spark DataFrame
last_date = spark_data.select("Date").orderBy(col("Date").desc()).first()["Date"]

forecasted_dates = pd.date_range(start=last_date, periods=len(future_predictions)+1, closed='right')
forecasted_series = pd.Series(future_predictions, index=forecasted_dates)

# Extract relevant data from the Spark DataFrame
selected_data = spark_data.select("Date", "sales").filter(spark_data['store']==1).filter(spark_data['item']==1).toPandas()

# Plot the actual time series and the predicted values
plt.figure(figsize=(8, 6))
plt.plot(selected_data["Date"], selected_data["sales"], label='Actual Time Series')
plt.plot(forecasted_series.index, forecasted_series, label='Predicted Values', color='red', linestyle='dashdot')
plt.scatter(forecasted_series.index, forecasted_series, color='red')  
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('LSTM Predictions')
plt.legend()
plt.show()




# Plot Loss and RMSE on separate axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

epochs = np.arange(1, len(loss_per_epoch) + 1)

ax1.plot(epochs, loss_per_epoch, label='Loss', marker='o')
ax1.set_ylabel('Loss')
ax1.set_title('Loss and RMSE over Epochs')

ax2.plot(epochs, rmse_per_epoch, label='RMSE', marker='o', color='orange')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('RMSE')

plt.tight_layout()
plt.show()




import logging
import datetime
from sklearn.metrics import accuracy_score, mean_squared_error


def run_experiment(train_data, input_size, hidden_size, output_size, learning_rate, epochs, validation_size=0.2):

    model = CustomLSTMModel(input_size, hidden_size, output_size, learning_rate)
    
    # Split the data into training and testing sets
    train_size = int(train_data.count() * (1 - validation_size))

    training_data = train_data.limit(train_size)
    test_data = train_data.subtract(training_data)

    training_features_rdd = training_data.select("input_sequence").rdd.map(lambda row: row[0])
    training_targets_rdd = training_data.select("target").rdd.map(lambda row: row[0])

    testing_features_rdd = test_data.select("input_sequence").rdd.map(lambda row: row[0])
    testing_targets_rdd = test_data.select("target").rdd.map(lambda row: row[0])

  

    for epoch in range(epochs):
        for features, target in zip(training_features_rdd.collect(), training_targets_rdd.collect()):
            predicted_value = model.forward_pass(features)
            loss = np.square(target - predicted_value)
            dy = 2 * (predicted_value - target)
            dWxh, dWhh, dWhy, dbh, dby = model.backward_pass(dy)
            model.update_parameters(dWxh, dWhh, dWhy, dbh, dby)
    
    # loss_per_epoch.append(loss.squeeze())
    # rmse_per_epoch.append(calculateRootMeanSquareError(loss).squeeze())

    training_predictions_rdd = training_features_rdd.map(lambda features: model.forward_pass(features))
    training_predictions = np.array(training_predictions_rdd.collect()).flatten() 
    training_targets = np.array(training_targets_rdd.collect()).flatten() 

    # Convert predictions to binary (0 or 1) for classification
    # binary_predictions = np.round(training_predictions)

    # training_accuracy = accuracy_score(training_targets, training_predictions)
    train_accuracy = 1 - (losses[-1] / len(train_data.rdd.collect()))
    print(training_accuracy)
    training_rmse = mean_squared_error(training_targets, training_predictions, squared=False)

    # Evaluate the model on test data
    testing_predictions_rdd = testing_features_rdd.map(lambda features: model.forward_pass(features))

    testing_predictions = np.array(testing_predictions_rdd.collect()).flatten() 
    testing_targets = np.array(testing_targets_rdd.collect()).flatten() 

    #test_accuracy = accuracy_score(testing_targets_rdd, np.round(test_predictions))
    test_rmse = mean_squared_error(testing_targets, testing_predictions, squared=False)

    return {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "validation_size": validation_size,
        # "training_accuracy": training_accuracy,
        # "test_accuracy": test_accuracy,
        "training_rmse": training_rmse,
        "test_rmse": test_rmse,
        "model": model
    }



# Specify the hyperparameter values you want to try
input_sizes = [1]
hidden_sizes = [4, 8, 16]
output_sizes = [1]
learning_rates = [0.01, 0.1, 0.2]
epochs_values = [10, 20, 30]

# Create an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['input_size', 'hidden_size', 'output_size', 'lr', 'epochs', 'training_rmse', 'test_rmse'])

# Iterate over hyperparameter combinations
for input_size in input_sizes:
    for hidden_size in hidden_sizes:
        for output_size in output_sizes:
            for learning_rate in learning_rates:
                for epochs in epochs_values:
                    # Run experiment with current hyperparameters
                    result = run_experiment(train_data, input_size, hidden_size, output_size, learning_rate, epochs, validation_size=0.2)
                    
                    # Append results to the DataFrame
                    results_df = results_df.append({
                        'input_size': input_size,
                        'hidden_size': hidden_size,
                        'output_size': output_size,
                        'lr': learning_rate,
                        'epochs': epochs,
                        'training_rmse': result['training_rmse'],
                        'test_rmse': result['test_rmse']
                    }, ignore_index=True)

# Display the results table
print(results_df)