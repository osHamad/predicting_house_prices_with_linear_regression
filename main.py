import numpy as np
from csv import reader
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import mean_absolute_error as mae

def train_test_split(data, x_pos, y_pos, train_size, index=0):
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    train_size *= len(data)
    for i in data:
        if index < train_size:
            train_x.append(float(i[x_pos]))
            train_y.append(float(i[y_pos]))
        else:
            test_x.append(float(i[x_pos]))
            test_y.append(float(i[y_pos]))
        index += 1

    return train_x, train_y, test_x, test_y


def read_csv(file):
    with open(file, newline='') as data:
        return list(reader(data))

def model_eval(y_data, x_data, model):
    y = []
    yhat = []
    for i in range(len(y_data)):
        y.append(y_data[i])
        yhat.append(float(model.predict([[x_data[i]]])))
    return mae(y, yhat)

def make_predictions(model):
    while True:
        house_area = input('enter a house area in sqft to make a prediction of price: ')
        if house_area == 'exit':
            return
        else:
            try:
                price = model.predict(np.array([[int(house_area)]]))
                print(f'price: {round(price[0], 2)}')
                print('\n')
            except ValueError:
                print('Invalid number\n')



def main():
    data = read_csv('house_dataset.csv')
    data.pop(0)
    train_x, train_y, test_x, test_y = train_test_split(data, 5, 2, 0.70)
    x = np.array(train_x)
    x = x.reshape((x.size, 1))
    y = np.array(train_y)
    reg = lr().fit(x, y)
    coef = reg.intercept_
    intercept = reg.coef_
    score = reg.score(x, y)
    error = model_eval(train_y, train_x, reg)
    print('model fit')
    print(f'coeffecient: {coef}')
    print(f'intercept: {intercept}')
    print(f'model has a Score of {score} and a Mean Absolue Error of {error}')
    print('\n\n')
    make_predictions(reg)

main()
