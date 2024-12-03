import numpy as np
import pickle

# Load the cosine similarity array from the file
file_path = 'cosine_sim/cosine_sim_resnet_cifar10.npy'
cosine_sim_array = np.load(file_path, allow_pickle=True)

# with open('cosine_sim_vgg_flower.pkl', 'rb') as f:
#     cosine_sim_array = pickle.load(f)
# print("Cosine similarity array loaded successfully.")

total = len(cosine_sim_array)
print(total)
thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


out = []
for threshold in thresholds:
    correct = 0
    for cosine_sim, label in cosine_sim_array:
        prediction = 0
        if cosine_sim > threshold:
            prediction = 1
        if prediction == label:
            correct+=1
    accuracy = correct / total
    out.append(accuracy)
    print(f"threshold: {threshold},  Accuracy: {accuracy * 100:.2f}%")
print(out)