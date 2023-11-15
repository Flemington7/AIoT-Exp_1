import matplotlib.pyplot as plt

from load import *
from server import *

# train
features_train = torch.tensor(all_features_train, dtype = torch.float)
features_train = features_train.to(device)
fitting_output = net(features_train)
fitting = fitting_output.detach().cpu().numpy()

# calculate the train accuracy
train_count = 0
train_accuracy_array = np.abs((fitting - all_labels_train.reshape(-1, 1)) / all_labels_train.reshape(-1, 1))
for i in range(len(train_accuracy_array)):
    if train_accuracy_array[i][0] < 0.1:
        train_count += 1
train_accuracy = train_count / len(train_accuracy_array)

past_date_day = np.char.add(np.char.add(np.array(train_data['year']).astype(str),
                                        np.array(train_data['month']).astype(str)),
                            np.array(train_data['day']).astype(str))
past_year = np.array(train_data['year']).astype(str)[::365]

plt.scatter(past_date_day, all_labels_train, c = 'b', marker = '+', label = 'actual')
plt.scatter(past_date_day, fitting, c = 'r', marker = '+', label = 'fitting')

plt.xlabel('date')
plt.xticks(np.arange(0, 3000, 365), past_year, rotation = 45)
plt.ylabel('temperature')
plt.text(1500, 103, 'train accuracy:{:.2f} '.format(train_accuracy), ha = 'center', va = 'center', fontsize = 10)
plt.legend(loc = 'upper right')

plt.title('past time actual temperature & fitting temperature')
plt.savefig('past time actual temperature & fitting temperature.png', dpi = 300)
plt.grid()
plt.show()

# test
test_csv = "test.csv"
test_data = pd.read_csv(test_csv)

all_labels_test = np.array(test_data['actual'])  # pd -> np
features_test = test_data.drop('actual', axis = 1)  # axis=1 means drop column
features_test = pd.get_dummies(features_test)  # one-hot
all_features_test = preprocessing.StandardScaler().fit_transform(features_test)  # standardization, pd -> np

features_test = torch.tensor(all_features_test, dtype = torch.float)
features_test = features_test.to(device)
predict_output = net(features_test)
predict = predict_output.detach().cpu().numpy()

'''with open('future predict.csv', 'w') as f:
    f.write('predict,actual\n')
    for i in range(len(predict)):
        f.write(str(predict[i][0]) + ',' + str(all_labels_test[i]) + '\n')'''

# calculate the test accuracy
test_count = 0
test_accuracy_array = np.abs(
    (predict - all_labels_test.reshape(-1, 1)) / all_labels_test.reshape(-1, 1))
for i in range(len(test_accuracy_array)):
    if test_accuracy_array[i][0] < 0.1:
        test_count += 1
test_accuracy = test_count / len(test_accuracy_array)

future_date_month = np.char.add(np.array(test_data['year']).astype(str), np.array(test_data['month']).astype(str))
future_date_day = np.char.add(future_date_month, np.array(test_data['day']).astype(str))
future_month = future_date_month[::30]

plt.scatter(future_date_day, predict, c = 'r', marker = '+', label = 'predict')
plt.scatter(future_date_day, all_labels_test, c = 'b', marker = '+', label = 'actual')

plt.xlabel('date')
plt.xticks(np.arange(0, 480, 30), future_month, rotation = 45)
plt.ylabel('temperature')
plt.legend(loc = 'upper right')
plt.text(240, 103, 'test accuracy:{:.2f} '.format(test_accuracy), ha = 'center', va = 'center', fontsize = 10)

plt.title('future time actual temperature & predict temperature')
plt.savefig('future time actual temperature & predict temperature.png', dpi = 300)
plt.grid()
plt.show()
