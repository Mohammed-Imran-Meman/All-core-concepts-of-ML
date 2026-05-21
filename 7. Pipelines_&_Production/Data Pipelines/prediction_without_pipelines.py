import pickle
import numpy as np

ohe_sex = pickle.load(open('models/ohe_sex.pkl','rb'))
ohe_embarked = pickle.load(open('models/ohe_embarked.pkl','rb'))
clf = pickle.load(open('models/clf.pkl','rb'))

test_input = np.array([2, 'male', 31.0, 0, 0, 10.5, 'S'],dtype=object).reshape(1,7)

test_input_sex = ohe_sex.transform(test_input[:,1].reshape(1,1))

test_input_embarked = ohe_embarked.transform(test_input[:,-1].reshape(1,1))

test_input_age = test_input[:,2].reshape(1,1)

test_input_transformed = np.concatenate((test_input[:,[0,3,4,5]],test_input_age,test_input_sex,test_input_embarked),axis=1)

clf.predict(test_input_transformed)