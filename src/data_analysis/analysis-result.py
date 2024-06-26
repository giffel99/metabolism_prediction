import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain

colors = ['#b7ded2', '#f6a6b2', '#f7c297', '#90d2d8', '#ffecb8', '#30afba']

def plot_graph(x_values, y_values, x_label, y_label, file_name): 
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_values, y_values)
    plt.savefig(f"../admin/results/{file_name}.png")

def remove_non_beamsize(results, non_beam_size): 
    results = results[["Beam_size", non_beam_size]]
    results = results[results["Beam_size"].notna()]
    return results

def visualize_precision_beam_size(results):
    results = remove_non_beamsize(results, "Precision")
    beam_size = results["Beam_size"]
    precision = results["Precision"]
    plot_graph(beam_size, precision, "Beam Size", "Precision", "beam_precision") 

def visualize_precision_false_positives(results):
    results = remove_non_beamsize(results, "False_positives")
    beam_size = results["Beam_size"]
    fp = results["False_positives"]
    plot_graph(beam_size, fp, "Beam Size", "False positives", "beam_fp") 

def visualize_fingerprint_beam_size(results): 
    results = remove_non_beamsize(results, "Total_avg_sim")
    beam_size = results["Beam_size"]
    precision = results["Total_avg_sim"]
    plot_graph(beam_size, precision, "Beam Size", "Average Fingerprint similiarity", "beam_fp_sim") 

def main():
    results = pd.read_csv(f'data/evaluation/evaluation_runs.csv')  
    visualize_precision_beam_size(results)
    visualize_fingerprint_beam_size(results)
    visualize_precision_false_positives(results)
    #combine_results_and_save(results)
   
if __name__ == "__main__":    
   main()
