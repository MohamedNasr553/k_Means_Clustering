import random
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

class K_Means:
    def __init__(self, k = 3, max_number_of_iterations = 1000):
        self.max_iter = max_number_of_iterations
        self.k = k

    def start(filename, percentage, k):
        try:
            data = CSVReader.read_data(filename, percentage)

            # Detect and remove outliers
            points, outliers = Outlier.iqr(data['IMDB Rating'])

            # Convert relevant columns to numpy array for clustering
            features = points.values.reshape(-1, 1)

            k_means = K_Means(k = k)

            # Fit the data (without outliers) to the model
            k_means.fit(features)

            # Output cluster contents
            results = {}
            for i in range(k):
                cluster_results = [jj for j, jj in enumerate(
                    points.values) if k_means.predict(features[j]) == i]

                results[f"Cluster {i+1}"] = cluster_results

            return results, outliers

        except Exception as e:
            print("Something went wrong :(")
            print(e)
            messagebox.showerror("Error", "Something went wrong :(")
            return None, None

    def fit(self, data):
        # Initialize centroids
        self.centroids = {}

        # Choose random data points as initial centroids
        for i in range(self.k):
            self.centroids[i] = self.__random_item(data)
            print(self.centroids[i])
        # Optimize centroids by iterating through data points
        for _ in range(self.max_iter):
            # Initialize dict clusters NO. => Data 
            self.clusters = {}

            for i in range(self.k):
                self.clusters[i] = []

            # Put data points into clusters
            for point in data:
                distances = [self.__euclidean_distance(point, centroid) for centroid in self.centroids]

                smallest_distance = min(distances)

                cluster = distances.index(smallest_distance)

                self.clusters[cluster].append(point)

            # Copy old centroids
            old_centroids = {i: self.centroids[i] for i in self.centroids}

            # Update centroids (average of all points in the cluster)
            for cluster in self.clusters:
                # Check if cluster is not empty
                if self.clusters[cluster]:
                    self.centroids[cluster] = self.__calculate_average(cluster)

            # Check for convergence
            optimum = True
            for centroid in self.centroids:
                old_centroid = old_centroids[centroid]
                new_centroid = self.centroids[centroid]

                if self.__converged(old_centroid, new_centroid):
                    optimum = False

            # Break if centroids have converged
            if optimum:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid])
                    for centroid in self.centroids]
        cluster = distances.index(min(distances))
        return cluster

    def __converged(self, old_centroid, new_centroid):
        return np.sum((new_centroid-old_centroid)/old_centroid*100.0) > 0.001

    def __calculate_average(self, cluster):
        return sum(
            self.clusters[cluster]) / len(self.clusters[cluster])

    def __random_item(self, data):
        return data[random.randint(0, len(data)-1)]

    def __euclidean_distance(self, features, centroid):
        return np.linalg.norm(
            features - self.centroids[centroid])

class Outlier:
    def z_score(data):
        # Calculate Z-score
        z = np.abs((data - data.mean()) / data.std())

        # Calculate outliers using Z-score
        outliers = data[z > 3]

        # Remove outliers from data
        data_cleaned = data[z <= 3].dropna()

        return data_cleaned, outliers

    def iqr(data):
        # Calculate IQR
        quantile25 = data.quantile(0.25)
        quantile75 = data.quantile(0.75)
        iqr = quantile75 - quantile25

        # Identify outliers using IQR
        lower_bound = quantile25 - 1.5 * iqr
        upper_bound = quantile75 + 1.5 * iqr

        # data points falling below the lower bound or above the upper bound are considered outliers
        outliers = data[(data < lower_bound) | (data > upper_bound)]

        # Remove outliers from data
        data_cleaned = data[(data >= lower_bound) & (data <= upper_bound)]

        return data_cleaned, outliers

class CSVReader:
    def read_data(filename, percentage):
        data = pd.read_csv(filename)
        # Percentage of data to read
        data = data.sample(frac=percentage, random_state=1)
        return data

def browse_file():
    filename = filedialog.askopenfilename(filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    file_entry.delete(0, tk.END)
    file_entry.insert(0, filename)

def main():
    filename = file_entry.get()
    percentage = float(percentage_spinbox.get()) / 100
    k = int(k_entry.get())

    clustering_result, outliers = K_Means.start(filename, percentage, k)

    result_text.config(state=tk.NORMAL)
    result_text.delete('1.0', tk.END)
    result_text.insert(tk.END, "Clusters:\n\n")
    for cluster, contents in clustering_result.items():
        result_text.insert(tk.END, f"{cluster}: {contents}\n\n")

    result_text.insert(tk.END, "\nOutliers:\n")
    result_text.insert(tk.END, outliers)

    result_text.config(state=tk.DISABLED)

root = tk.Tk()
root.title("K-Means Clustering")

frame = tk.Frame(root)
frame.pack(pady=20)

file_label = tk.Label(frame, text="Select CSV File:")
file_label.grid(row=0, column=0, padx=10, pady=5)

file_entry = tk.Entry(frame, width=50)
file_entry.grid(row=0, column=1, padx=10, pady=5)

browse_button = tk.Button(frame, text="Browse", command=browse_file, bg="lightblue", width= 10)
browse_button.grid(row=0, column=2, padx=10, pady=5)

percentage_label = tk.Label(frame, text="Percentage of data to read (%):")
percentage_label.grid(row=1, column=0, padx=10, pady=5)

percentage_spinbox = ttk.Spinbox(frame, from_=0, to=100, width=18)
percentage_spinbox.grid(row=1, column=1, padx=10, pady=5)

k_label = tk.Label(frame, text="Number of Clusters (k):")
k_label.grid(row=2, column=0, padx=10, pady=5)

k_entry = tk.Entry(frame)
k_entry.grid(row=2, column=1, padx=10, pady=5)

start_button = tk.Button(root, text="Start Clustering", command=main, bg="lightblue", width=12)
start_button.pack(pady=10)

result_text = tk.Text(root, height=20, width=70, fg="darkblue")
result_text.pack()

root.mainloop()
