import pickle

# Load the file
with open('fraud_model.pkl', 'rb') as file:
    data = pickle.load(file)

# Now you can inspect the data
print(type(data))
print(data)
