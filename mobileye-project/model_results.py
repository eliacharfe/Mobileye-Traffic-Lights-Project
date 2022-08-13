import pickle
import matplotlib.pyplot as plt

with open(r'C:\leftImg8bit\logs\my_model_final_2\model_0059.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)