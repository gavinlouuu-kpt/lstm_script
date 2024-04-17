import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def peak_detection(data: pd.DataFrame, threshold: float, window_size: int) -> pd.DataFrame:
    """
    Detects peaks in the data based on a threshold and window size.
    """
    # Initialize a list to hold the peak indices
    peak_indices = []
    # Iterate over the data
    for i in range(len(data) - window_size):
        # Get the window of data
        window = data[i:i + window_size]
        # Check if the maximum value in the window exceeds the threshold
        if max(window) > threshold:
            # Find the index of the peak value
            peak_index = i + np.argmax(window)
            # Add the peak index to the list
            peak_indices.append(peak_index)
    # Create a new DataFrame with the peak indices
    peaks = pd.DataFrame({'peak_index': peak_indices})
    return peaks
import pandas as pd
import matplotlib.pyplot as plt

def main():
    file_path = "../data/aligned_df.pq"
    data_set = pd.read_parquet(file_path)
    threshold = 1000
    window_size = 500
    # detect peaks within each group by exp_no
    groupby_exp = data_set.groupby('exp_no')
    for name, group in groupby_exp:
        peaks = peak_detection(group['A1_Sensor'], threshold, window_size)
        print(f"Peaks for experiment {name}:")
        print(peaks)

    # Plotting peak detection results over the data of a specific exp_no
    exp_no = 1
    group = groupby_exp.get_group(exp_no)
    peaks = peak_detection(group['A1_Sensor'], threshold, window_size)
    
    # Plot raw data
    plt.plot(group['A1_Sensor'], label='Sensor')

    # Overlay peaks on the plot
    # Ensure 'peak_index' from your 'peak_detection' function is the array of indices where peaks occur
    plt.scatter(peaks['peak_index'], group['A1_Sensor'].iloc[peaks['peak_index']], color='red', label='Peaks')
    
    plt.title(f"Peak Detection for Experiment {exp_no}")
    plt.xlabel("Time")
    plt.ylabel("Sensor")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

