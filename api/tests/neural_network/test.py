import pandas as pd
from api.static.preprocessing import standard_features, train_test_split
from models.ml.backwards_propagation_neural_network import NeuralNetwork, Layer


# Read Spotify Dataset
df = pd.read_excel('output_shuffled.xlsx').drop_duplicates()
x_data = df.iloc[:, 0:9].to_numpy()
y_data = df.iloc[:, 13].to_numpy().reshape(-1, 1)

# Pre processing
x_data = standard_features(x=x_data)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
n_inputs = x_train.shape[1]
nn = NeuralNetwork()
nn.add_layer(Layer(n_inputs, 18, 'sigmoid', name='Hidden Layer 1'))
nn.add_layer(Layer(18, 9, 'sigmoid', name='Output Layer'))
nn.add_layer(Layer(9, 2, 'sigmoid', name='Output Layer'))
errors = nn.train(x_train, y_train, x_test, y_test, learning_rate=0.02, max_epochs=300, plot_errors=True, plot_accuracies=True)
y_predictions = nn.predict(x_test)
train_accuracy = nn.accuracy(x_train, y_train)
test_accuracy = nn.accuracy(x_test, y_test)
nn.roc_curve(x_test=x_test, y_test=y_test)
print(train_accuracy, test_accuracy)
nn.visualize()
print(1)

def test_hidden_layers_1():
    epoch = 1
    results = []
    for i in range(51):
        nn = NeuralNetwork()
        nn.add_layer(Layer(n_inputs, 9, 'relu', name='Hidden Layer 1'))
        nn.add_layer(Layer(9, 9, 'sigmoid', name='Output Layer'))
        nn.add_layer(Layer(9, 8, 'sigmoid', name='Output Layer'))
        nn.add_layer(Layer(8, 7, 'sigmoid', name='Output Layer'))
        nn.add_layer(Layer(7, 6, 'sigmoid', name='Output Layer'))
        nn.add_layer(Layer(6, 5, 'sigmoid', name='Output Layer'))
        nn.add_layer(Layer(5, 2, 'sigmoid', name='Output Layer'))
        errors = nn.train(x_train, y_train, x_test, y_test, learning_rate=0.01, max_epochs=epoch, plot_errors=False)
        train_accuracy = nn.accuracy(x_train, y_train)
        test_accuracy = nn.accuracy(x_test, y_test)
        results.append((epoch, train_accuracy, test_accuracy, min(errors)))
        epoch = epoch + 10
        print(i)
    COLUMNS = ['size', 'Accuracy Train', 'Accuracy Test', 'Minimum Loss']
    name = r'h1-34-a01-bp-epoch-relu'
    pd.DataFrame(list(results), columns=COLUMNS).to_excel(name + '.xlsx', sheet_name=name)

#test_hidden_layers_1()