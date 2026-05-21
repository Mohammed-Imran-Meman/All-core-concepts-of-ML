import pickle
import numpy as np

pipe = pickle.load(open('models/pipe.pkl','rb'))

test_input2 = np.array([2, 'male', 31.0, 0, 0, 10.5, 'S'],dtype=object).reshape(1,7)
pipe.predict(test_input2)