import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

class SolarFlareHMM:
    """
    Hidden Markov Model for solar flare detection and prediction.
    This class implements a HMM from scratch for solar flare detection
    and prediction based on x-ray flux time series data.
    """

    # Solar flare classifications and their threshold values (from https://www.sws.bom.gov.au/Educational/2/1/3)
    FLARE_CLASSES = {
        'A': 1e-7,
        'B': 1e-6,
        'C': 1e-5,
        'M': 1e-4,
        'X': float('inf')
    }

    def __init__(self, n_states=5, n_emissions=20, flare_classes_to_detect=None):
        """
        Initialize the hmm for solar flare detection

        Args:
            n_states (int): # of hidden states in the HMM
            n_emissions (int): # of possible emission values
            flare_classes_to_detect (list): List of flare classes to detect, ex: 'C, 'M', 'X'
        """
        self.n_states = n_states
        self.n_emissions = n_emissions

        # Initialize model params
        self.A = np.ones((n_states, n_states)) / n_states  # matrix for transitions
        self.B = np.ones((n_states, n_emissions)) / n_emissions  # matrix for emissions
        self.pi = np.ones(n_states) / n_states  # init state probs

      
        self.scaler = MinMaxScaler()

        # flare classes to detect (default all)
        if flare_classes_to_detect is None:
            self.flare_classes_to_detect = list(self.FLARE_CLASSES.keys())
        else:
            self.flare_classes_to_detect = flare_classes_to_detect

        self.convergence_threshold = 1e-6

        # stores for predicted results
        self.predictions = None
        self.future_dates = None
        self.trained = False

    def preprocess_data(self, df):
        """
        To preprocess the input data for HMM training

        Args:
            df (pd.DataFrame): Input dataframe with 'time' and 'xray_flux' columns

        Returns:
            tuple: (processed_df, observations, time_indices)
        """
        # Ensure it's in datetime
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])

        # Make a copy to not alter og data
        df_processed = df.copy()

        # missing value at ~2015 (and probably more)
        df_processed['xray_flux'] = df_processed['xray_flux'].interpolate()

        # Sort by time
        df_processed = df_processed.sort_values('time')

        df_processed = df_processed.reset_index(drop=True)

        # scale xray_flux values
        scaled_values = self.scaler.fit_transform(df_processed[['xray_flux']])

        # Discretize the scaled values into the indice of the emissions
        emissions = np.floor(scaled_values * self.n_emissions).astype(int)
        emissions[emissions == self.n_emissions] = self.n_emissions - 1

        return df_processed, emissions.flatten(), df_processed.index.values

    def forward_algorithm(self, observations):
        """
        This here implements the forward algorithm for HMM

        Args:
            observations (np.array): Sequence of emission indices

        Returns:
            tuple: (alpha, scale_factors)
        """
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        scale_factors = np.zeros(T)

        alpha[0] = self.pi * self.B[:, observations[0]]

        scale_factors[0] = np.sum(alpha[0])
        if scale_factors[0] != 0:
            alpha[0] /= scale_factors[0]
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, observations[t]]

            # now we scale alpha
            scale_factors[t] = np.sum(alpha[t])
            if scale_factors[t] != 0:
                alpha[t] /= scale_factors[t]

        return alpha, scale_factors

    def backward_algorithm(self, observations, scale_factors):
        """
        this here implements the backward algorithm for HMM

        Args:
            observations (np.array): Sequence of emission indices
            scale_factors (np.array): Scale factors from forward algorithm

        Returns:
            np.array: Beta values
        """
        T = len(observations)
        beta = np.zeros((T, self.n_states))

        # init beta @ t=T-1
        beta[T-1] = 1.0
        if scale_factors[T-1] != 0:
            beta[T-1] /= scale_factors[T-1]

        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, observations[t+1]] * beta[t+1, :])

            # now we scale beta
            if scale_factors[t] != 0:
                beta[t] /= scale_factors[t]

        return beta

    def compute_xi_gamma(self, alpha, beta, observations):
        """
        find xi and gamma for the baum-welch algorithm

        Args:
            alpha (np.array): Forward probabilities
            beta (np.array): Backward probabilities
            observations (np.array): Sequence of emission indices

        Returns:
            tuple: (xi, gamma)
        """
        T = len(observations)
        xi = np.zeros((T-1, self.n_states, self.n_states))
        gamma = np.zeros((T, self.n_states))

        # get gamma
        gamma = alpha * beta
        row_sums = np.sum(gamma, axis=1, keepdims=True)
        gamma = np.divide(gamma, row_sums, out=np.zeros_like(gamma), where=row_sums!=0)

        # get xi
        for t in range(T-1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    numerator = alpha[t, i] * self.A[i, j] * self.B[j, observations[t+1]] * beta[t+1, j]
                    denominator = np.sum(alpha[t, :] * self.A[:, :] * self.B[:, observations[t+1]].reshape(1, -1) * beta[t+1, :].reshape(1, -1))
                    if denominator != 0:
                        xi[t, i, j] = numerator / denominator

        return xi, gamma

    def baum_welch_step(self, observations):
        """
        gets a step from the Baum-Welch algorithm for param estimation

        Args:
            observations (np.array): Sequence of emission indices

        Returns:
            float: Log likelihood of the model
        """
        alpha, scale_factors = self.forward_algorithm(observations)
        beta = self.backward_algorithm(observations, scale_factors)

        # find relevat vars
        xi, gamma = self.compute_xi_gamma(alpha, beta, observations)

        # Update parameters & initial store params
        self.pi = gamma[0]

        for i in range(self.n_states):
            denominator = np.sum(gamma[:-1, i])
            if denominator != 0:
                self.A[i] = np.sum(xi[:, i, :], axis=0) / denominator

        for j in range(self.n_states):
            for k in range(self.n_emissions):
                numerator = np.sum(gamma[observations == k, j])
                denominator = np.sum(gamma[:, j])
                if denominator != 0:
                    self.B[j, k] = numerator / denominator

        # get log-likelihood
        log_likelihood = -np.sum(np.log(scale_factors))

        return log_likelihood

    def train(self, df, epochs=50, verbose=True):
        """
        Train the HMM using the Baum-Welch algorithm

        Args:
            df (pd.DataFrame): Training data with 'time' and 'xray_flux' columns
            epochs (int): Maximum number of training epochs
            verbose (bool): Whether to print training progress

        Returns:
            list: Log likelihoods during training
        """
        df_processed, observations, _ = self.preprocess_data(df)

        # Initialize model params randomly
        self.A = np.random.rand(self.n_states, self.n_states)
        self.A = self.A / np.sum(self.A, axis=1, keepdims=True)

        self.B = np.random.rand(self.n_states, self.n_emissions)
        self.B = self.B / np.sum(self.B, axis=1, keepdims=True)

        self.pi = np.random.rand(self.n_states)
        self.pi = self.pi / np.sum(self.pi)

        log_likelihoods = []
        prev_log_likelihood = float('-inf')

        iterator = range(epochs)
        if verbose:
            iterator = tqdm(iterator, desc="Training HMM")

        for i in iterator:
            log_likelihood = self.baum_welch_step(observations)
            log_likelihoods.append(log_likelihood)

            # Checks convergence
            if i > 0 and abs(log_likelihood - prev_log_likelihood) < self.convergence_threshold:
                if verbose:
                    print(f"Converged after {i+1} iterations")
                break

            prev_log_likelihood = log_likelihood

            if verbose and (i+1) % 10 == 0:
                print(f"Iteration {i+1}/{epochs}, Log Likelihood: {log_likelihood}")

        self.trained = True
        return log_likelihoods

    def viterbi_algorithm(self, observations):
        """
        This implements Viterbi algorithm to find the most likely state sequence

        Args:
            observations (np.array): Sequence of emission indices

        Returns:
            np.array: Most likely hidden state sequence
        """
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        # Initialize
        delta[0] = self.pi * self.B[:, observations[0]]
        psi[0] = 0

        # Recursive here
        for t in range(1, T):
            for j in range(self.n_states):
                delta[t, j] = np.max(delta[t-1] * self.A[:, j]) * self.B[j, observations[t]]
                psi[t, j] = np.argmax(delta[t-1] * self.A[:, j])

        q_star = np.zeros(T, dtype=int)
        q_star[T-1] = np.argmax(delta[T-1])

        # backtracking
        for t in range(T-2, -1, -1):
            q_star[t] = psi[t+1, q_star[t+1]]

        return q_star

    def detect_flares(self, df):
        """
        Detect solar flares in the given data

        Args:
            df (pd.DataFrame): Data with 'time' and 'xray_flux' columns

        Returns:
            pd.DataFrame: DataFrame with flare detection information
        """
        if not self.trained:
            raise ValueError("Model needs to be trained before detection")

        # Processing
        df_processed, observations, _ = self.preprocess_data(df)

        # Find the most likely hidden state sequence
        states = self.viterbi_algorithm(observations)

        # copy dataframe
        result_df = df_processed.copy()
        result_df['hidden_state'] = states

        # get high activity states
        state_means = np.zeros(self.n_states)
        for s in range(self.n_states):
            if np.sum(states == s) > 0:
                state_means[s] = np.mean(df_processed.loc[states == s, 'xray_flux'])

        # Sorts states by mean activity
        sorted_states = np.argsort(state_means)

        flare_state = sorted_states[-1]

        # classify flares based on thresholds
        result_df['flare_detected'] = False
        result_df['flare_class'] = None

        # Determine flare class and mark detected flares
        for idx, row in result_df.iterrows():
            flux = row['xray_flux']
            flare_class = None

            # Determine flare class
            if flux < self.FLARE_CLASSES['A']:
                flare_class = 'Below A'
            elif flux < self.FLARE_CLASSES['B']:
                flare_class = 'A'
            elif flux < self.FLARE_CLASSES['C']:
                flare_class = 'B'
            elif flux < self.FLARE_CLASSES['M']:
                flare_class = 'C'
            elif flux < self.FLARE_CLASSES['X']:
                flare_class = 'M'
            else:
                flare_class = 'X'

            result_df.at[idx, 'flare_class'] = flare_class

            # mark it as detected if in the specified classes and in a high activity state
            if (flare_class in self.flare_classes_to_detect and
                row['hidden_state'] == flare_state):
                result_df.at[idx, 'flare_detected'] = True

        return result_df

    def predict_future(self, df, days_ahead=7, num_simulations=100):
        """
        Predict future solar flare activity

        Args:
            df (pd.DataFrame): Historical data with 'time' and 'xray_flux' columns
            days_ahead (int): # of days to predict ahead
            num_simulations (int): # of simulations to run for prediction

        Returns:
            pd.DataFrame: dataframe w/ predictions
        """
        if not self.trained:
            raise ValueError("Model needs to be trained before prediction")

        # Process input
        df_processed, observations, _ = self.preprocess_data(df)

        # Get last observation and find most likely state
        last_emission = observations[-1]
        last_state_probs = np.zeros(self.n_states)

        for s in range(self.n_states):
            last_state_probs[s] = self.pi[s] * self.B[s, last_emission]

        current_state = np.argmax(last_state_probs)

        # Determine prediction timeframe
        last_time = df_processed['time'].max()
        minutes_per_day = 24 * 60
        prediction_horizon = days_ahead * minutes_per_day

        # Create future timestamps (1/min)
        future_dates = [last_time + timedelta(minutes=i+1) for i in range(prediction_horizon)]

        # Run multiple simulations
        all_predictions = np.zeros((num_simulations, prediction_horizon))
        all_states = np.zeros((num_simulations, prediction_horizon), dtype=int)

        for sim in range(num_simulations):
            state_sequence = np.zeros(prediction_horizon, dtype=int)
            emission_sequence = np.zeros(prediction_horizon, dtype=int)

            # Start with current state
            state = current_state

            # Generate future states and emissions
            for t in range(prediction_horizon):
                # Transition to next state based on transition matrix
                state = np.random.choice(self.n_states, p=self.A[state])
                state_sequence[t] = state

                # Generate emission based on emission matrix
                emission = np.random.choice(self.n_emissions, p=self.B[state])
                emission_sequence[t] = emission

            # Convert emissions back to flux values
            predicted_values = emission_sequence / self.n_emissions
            predicted_values = self.scaler.inverse_transform(predicted_values.reshape(-1, 1)).flatten()

            all_predictions[sim] = predicted_values
            all_states[sim] = state_sequence

        # Compute mean & confidence intervals
        mean_predictions = np.mean(all_predictions, axis=0)
        lower_ci = np.percentile(all_predictions, 5, axis=0)
        upper_ci = np.percentile(all_predictions, 95, axis=0)

        pred_df = pd.DataFrame({
            'time': future_dates,
            'predicted_xray_flux': mean_predictions,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci
        })

        pred_df['predicted_flare_class'] = 'Below A'

        for idx, flux in enumerate(mean_predictions):
            if flux >= self.FLARE_CLASSES['X']:
                pred_df.loc[idx, 'predicted_flare_class'] = 'X'
            elif flux >= self.FLARE_CLASSES['M']:
                pred_df.loc[idx, 'predicted_flare_class'] = 'M'
            elif flux >= self.FLARE_CLASSES['C']:
                pred_df.loc[idx, 'predicted_flare_class'] = 'C'
            elif flux >= self.FLARE_CLASSES['B']:
                pred_df.loc[idx, 'predicted_flare_class'] = 'B'
            elif flux >= self.FLARE_CLASSES['A']:
                pred_df.loc[idx, 'predicted_flare_class'] = 'A'

        # store the predictions for later
        self.predictions = pred_df
        self.future_dates = future_dates

        return pred_df

    def plot_training_data(self, df, detected_flares=None):
        """
        Actually plotsthe training data with flare detection results

        Args:
            df (pd.DataFrame): Original data with 'time' and 'xray_flux' columns
            detected_flares (pd.DataFrame): dataframe with flare detection results

        Returns:
            matplotlib.figure.Figure: Generated plot
        """
        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot OG  data
        ax.plot(df['time'], df['xray_flux'], color='blue', alpha=0.7, label='X-ray Flux')

        # Add logarithmic scale otherwise it's way out of propertion
        ax.set_yscale('log')

        # Adds some horizontal lines for flare classifications
        for flare_class, threshold in self.FLARE_CLASSES.items():
            if flare_class != 'X':  # x class has no upper threshold
                ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5,
                          label=f"Class {flare_class} Threshold")

        # Plot detected flares (if provided)
        if detected_flares is not None:
            flare_points = detected_flares[detected_flares['flare_detected']]
            if not flare_points.empty:
                ax.scatter(flare_points['time'], flare_points['xray_flux'], color='red',
                          s=50, label='Detected Flares')

                for idx, row in flare_points.iterrows():
                    ax.annotate(row['flare_class'],
                               (row['time'], row['xray_flux']),
                               xytext=(5, 5),
                               textcoords='offset points',
                               fontsize=8)

        #x-axis
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('X-ray Flux (W/m²)')
        ax.set_title('Solar X-ray Flux with Detected Flares')
        plt.tight_layout()
        return fig

    def plot_predictions(self, historical_df=None):
        """
        Plots predicted solar flare activity

        Args:
            historical_df (pd.DataFrame): Historical data to show before predictions

        Returns:
            matplotlib.figure.Figure: Generated plot
        """
        if self.predictions is None:
            raise ValueError("No predictions available. Run predict_future() first.")

        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot historical data (if provided)
        if historical_df is not None:
            # gets the last week of historical data for context
            start_date = self.predictions['time'].min() - timedelta(days=7)
            hist_subset = historical_df[historical_df['time'] >= start_date].copy()

            if not hist_subset.empty:
                ax.plot(hist_subset['time'], hist_subset['xray_flux'],
                       color='blue', label='Historical Data')

                # Add vertical line to show historical vs predictions
                last_hist_date = hist_subset['time'].max()
                ax.axvline(x=last_hist_date, color='black', linestyle='--',
                          label='Prediction Start')

        # Plot predictions w/ confidence interval
        ax.plot(self.predictions['time'], self.predictions['predicted_xray_flux'],
               color='green', label='Predicted X-ray Flux')

        ax.fill_between(self.predictions['time'],
                        self.predictions['lower_ci'],
                        self.predictions['upper_ci'],
                        color='green', alpha=0.2, label='95% Confidence Interval')

        # Add more horizontal lines for the flare classifications
        for flare_class, threshold in self.FLARE_CLASSES.items():
            if flare_class != 'X':  # X class has no upper threshold
                ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5,
                          label=f"Class {flare_class} Threshold")

        # log scale for ui
        ax.set_yscale('log')

        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        ax.set_xlabel('Time')
        ax.set_ylabel('X-ray Flux (W/m²)')
        ax.set_title('Solar Flare Prediction with Confidence Interval')
        plt.tight_layout()
        return fig

    def plot_flare_probability(self):
        """
        Plot the probability of different flare classes over prediction period

        Returns:
            matplotlib.figure.Figure: Generated plot
        """
        if self.predictions is None:
            raise ValueError("No predictions available. Run predict_future() first please.")

        # Groups predictions by day and calculate flare probabilities
        self.predictions['date'] = self.predictions['time'].dt.date
        daily_groups = self.predictions.groupby('date')

        days = sorted(self.predictions['date'].unique())
        flare_classes = ['A', 'B', 'C', 'M', 'X']

        # Calculate daily probability for each type of flare 
        probs = {fc: [] for fc in flare_classes}

        for day in days:
            day_data = daily_groups.get_group(day)
            total = len(day_data)

            for fc in flare_classes:
                count = np.sum(day_data['predicted_flare_class'] == fc)
                probs[fc].append(count / total * 100)  # Convert to%

      #Plot stuff
        fig, ax = plt.subplots(figsize=(15, 8))
        width = 0.15
        x = np.arange(len(days))
        for i, fc in enumerate(flare_classes):
            offset = (i - len(flare_classes)/2 + 0.5) * width
            ax.bar(x + offset, probs[fc], width, label=f'Class {fc}')
        ax.set_xticks(x)
        ax.set_xticklabels([d.strftime('%Y-%m-%d') for d in days], rotation=45)
        ax.set_xlabel('Date')
        ax.set_ylabel('Probability (%)')
        ax.set_title('Daily Probability of Solar Flare Classes')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        plt.tight_layout()
        return fig

    def plot_training_convergence(self, log_likelihoods):
        """
        Plot the convergence of the HMM training process

        Args:
            log_likelihoods (list): Log likelihoods from training

        Returns:
            matplotlib.figure.Figure: Generated plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(range(1, len(log_likelihoods) + 1), log_likelihoods,
               marker='o', linestyle='-', markersize=4)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Log Likelihood')
        ax.set_title('HMM Training Convergence')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def evaluate_model(self, test_df):
        """
        Evaluate the model on test data by measuring flare detection accuracy.

        Args:
            test_df (pd.DataFrame): Test data with 'time' and 'xray_flux' columns

        Returns:
            dict: Dictionary with evaluation metrics
        """
        if not self.trained:
            raise ValueError("Model needs to be trained before evaluation")

        # Detect flares in test data
        detected_flares = self.detect_flares(test_df)
        class_counts = detected_flares['flare_class'].value_counts().to_dict()
        detected_counts = detected_flares[detected_flares['flare_detected']]['flare_class'].value_counts().to_dict()
        detection_rates = {}
        for fc in self.flare_classes_to_detect:
            if fc in class_counts:
                detected = detected_counts.get(fc, 0)
                total = class_counts[fc]
                detection_rates[fc] = detected / total
            else:
                detection_rates[fc] = 0

        # Calc overall detection rate
        total_detected = sum(detected_counts.values())
        total_points = len(detected_flares)
        overall_rate = total_detected / total_points if total_points > 0 else 0

        return {
            'detection_rates': detection_rates,
            'overall_detection_rate': overall_rate,
            'class_counts': class_counts,
            'detected_counts': detected_counts
        }

    def plot_evaluation(self, eval_metrics):
        """
        Plot the evaluation stuff (like detection raes, etc)

        Args:
            eval_metrics (dict): Evaluation metrics from evaluate_model

        Returns:
            matplotlib.figure.Figure: Generated plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        detection_rates = eval_metrics['detection_rates']
        classes = list(detection_rates.keys())
        rates = list(detection_rates.values())
        ax1.bar(classes, rates, color='skyblue')
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Detection Rate')
        ax1.set_title('Flare Detection Rate by Class')
        for i, v in enumerate(rates):
            ax1.text(i, v + 0.05, f'{v:.2f}', ha='center')
        class_counts = eval_metrics['class_counts']
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        ax2.bar(classes, counts, color='lightgreen')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Flare Classes')
        for i, v in enumerate(counts):
            ax2.text(i, v + 0.5, str(v), ha='center')

        plt.tight_layout()
        return fig


# Usage examples

def load_sample_data(file_path):
    """
    Load and prepare sample solar flare data from CSV file.

    Args:
        file_path (str): Path to CSV file with solar data

    Returns:
        pd.DataFrame: Prepared DataFrame with 'time' and 'xray_flux' columns
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        return None

    # Try converting 'time' column to datetime with error handling for mixed formats
    df['time'] = pd.to_datetime(df['time'], errors='coerce', format='mixed')

    # Drop rows where 'time' could not be parsed
    df.dropna(subset=['time'], inplace=True)

    # Ensure the dataframe is sorted by time
    df = df.sort_values('time')

    return df


def demo_solar_flare_detection(data_path):
    """
    Demonstration of the SolarFlareHMM class functionality

    Args:
        data_path (str): Path to CSV file with solar X-ray flux data

    Returns:
        SolarFlareHMM: Trained model instance
    """
    # Load data
    print("Loading solar flux data...")
    df = load_sample_data(data_path)
    print(f"Loaded {len(df)} data points")

    # Split data into training and test sets (80/20 split)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Initialize the HMM model
    hmm = SolarFlareHMM(n_states=5, n_emissions=20, flare_classes_to_detect=['C', 'M', 'X'])

    # Train the model
    print("\nTraining HMM model...")
    log_likelihoods = hmm.train(train_df, epochs=30, verbose=True)

    # Plot training conv.
    print("Plotting training convergence...")
    convergence_fig = hmm.plot_training_convergence(log_likelihoods)
    convergence_fig.savefig('training_convergence.png')

    # Detect flares in test data
    print("\nDetecting flares in test data...")
    detected_flares = hmm.detect_flares(test_df)
    print(f"Detected {detected_flares['flare_detected'].sum()} flares")

    # Plot test data with detected flares
    print("Plotting detected flares...")
    detection_fig = hmm.plot_training_data(test_df, detected_flares)
    detection_fig.savefig('flare_detection.png')
    print("\nEvaluating model performance...")
    eval_metrics = hmm.evaluate_model(test_df)
    print("Detection rates by class:")
    for flare_class, rate in eval_metrics['detection_rates'].items():
        print(f"  Class {flare_class}: {rate:.2f}")
    print(f"Overall detection rate: {eval_metrics['overall_detection_rate']:.2f}")

    # Eval stuff
    eval_fig = hmm.plot_evaluation(eval_metrics)
    eval_fig.savefig('model_evaluation.png')

    # Prediction of future flare activity
    print("\nPredicting future solar flare activity...")
    prediction_df = hmm.predict_future(df, days_ahead=7, num_simulations=100)

    # Plot stuff
    print("Plotting predictions...")
    prediction_fig = hmm.plot_predictions(df)
    prediction_fig.savefig('flare_predictions.png')
    prob_fig = hmm.plot_flare_probability()
    prob_fig.savefig('flare_probability.png')

    return hmm


# Example usage
if __name__ == "__main__":
    # Replace with your actual path to data
    data_path = "combined_data.csv"

    try:
        df = pd.read_csv(data_path)

        # Calculate and extract just 10% of the data
        num_rows = len(df)
        rows_to_use = int(num_rows * 0.1)
        subset_data_path = "temp_subset_data.csv"

        # Save just 10% of the data to a temporary file
        df.iloc[:rows_to_use].to_csv(subset_data_path, index=False)

        print(f"Using {rows_to_use} rows (10% of the original dataset)")

        # Use the subset for modeling
        hmm_model = demo_solar_flare_detection(subset_data_path)
        print("\nSuccessfully completed solar flare modeling and prediction!")
        os.remove(subset_data_path)

    except Exception as e:
        print(f"Error during demonstration: {str(e)}")


# Example of 'advanced' usage of SolarFlareHMM'
def advanced_usage():
    """
    Examples of more advanced usage patterns for the SolarFlareHMM class
    """
    # placeholder - replace with your actual path
    data_path = "combined_data.csv"

    # Load data
    df = load_sample_data(data_path)

    # Example 1: model with more states for better discrimination
    print("\nExample 1: model with more states")
    hmm_fine = SolarFlareHMM(n_states=8, n_emissions=30,
                             flare_classes_to_detect=['M', 'X'])  # here it shows how you can focus on stronger flares
    hmm_fine.train(df, epochs=50)

    # Example 2: long-term prediction 
    print("\nExample 2: Long-term prediction (30 days)")
    hmm_longterm = SolarFlareHMM(n_states=5, n_emissions=20)
    hmm_longterm.train(df, epochs=30)
    long_predictions = hmm_longterm.predict_future(df, days_ahead=30, num_simulations=200)

    # Example 3: Real-time monitoring setup
    print("\nExample 3: Real-time monitoring setup")
    hmm_monitor = SolarFlareHMM(n_states=5, n_emissions=20,
                               flare_classes_to_detect=['C', 'M', 'X'])
    hmm_monitor.train(df, epochs=30)

    # Simulate real-time data updates (last 2 days of data)
    last_date = df['time'].max()
    two_days_ago = last_date - timedelta(days=2)
    recent_data = df[df['time'] >= two_days_ago]

    # Detect recent flares
    recent_flares = hmm_monitor.detect_flares(recent_data)
    detected_today = recent_flares[recent_flares['time'].dt.date == last_date.date()]
    print(f"Detected {detected_today['flare_detected'].sum()} flares today")

    # Make a short-term prediction (for the next day in this case)
    tomorrow_prediction = hmm_monitor.predict_future(df, days_ahead=1, num_simulations=100)

class SolarFlarePredictor:
    """

    This here is just as add-on for simpler usage of the 'advanced interface'.
    It essentially consolidates the advanced param options in SolarFlareHMM, and could be good for
    more common use cases.

    """

    def __init__(self, sensitivity='medium'):
        """
        Initialize the predictor with the specified sensitivity.

        Args:
            sensitivity (str): Sensitivity level - 'low', 'medium', or 'high'
        """
        # set model params based on sensitivity
        if sensitivity == 'low':
            self.n_states = 3
            self.flare_classes = ['X']  # Only detect X flares
        elif sensitivity == 'medium':
            self.n_states = 5
            self.flare_classes = ['M', 'X']  # M & X flares
        elif sensitivity == 'high':
            self.n_states = 7
            self.flare_classes = ['C', 'M', 'X']  # for C, M, and X flares
        else:
            raise ValueError("Sensitivity must be 'low', 'medium', or 'high'")

        # Create the HMM model
        self.model = SolarFlareHMM(
            n_states=self.n_states,
            n_emissions=20,
            flare_classes_to_detect=self.flare_classes
        )

        self.is_trained = False

    def train_on_data(self, data_path, train_epochs=30):
        """
        Train the model on the provided data

        Args:
            data_path (str): Path to CSV file with solar data
            train_epochs (int): Number of training epochs

        Returns:
            bool: True if training was successful
        """
        try:
            # Load data
            df = load_sample_data(data_path)

            # Train model
            self.model.train(df, epochs=train_epochs, verbose=True)
            self.is_trained = True
            self.data = df

            return True
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return False

    def detect_flares(self, data_path=None):
        """
        Detect flares in the provided data or in the training data.

        Args:
            data_path (str, optional): Path to CSV file with new data

        Returns:
            pd.DataFrame: dataframe with flare detection results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before detection")

        # Use new data if provided, otherwise use training data
        if data_path:
            df = load_sample_data(data_path)
        else:
            df = self.data

        # Detect flares
        return self.model.detect_flares(df)

    def predict_future_activity(self, days=7):
        """
        Predict future solar flare activity

        Args:
            days (int): Number of days to predict ahead

        Returns:
            pd.DataFrame: dataframe with predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        return self.model.predict_future(self.data, days_ahead=days)

    def generate_report(self, output_dir="."):
        """
        creates report on the solar flare activity

        Args:
            output_dir (str): Directory to save report files

        Returns:
            dict: Dict with report info
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before creating report")
        os.makedirs(output_dir, exist_ok=True)

        report = {}

        # get flares in the training data (if any)
        detected_flares = self.model.detect_flares(self.data)
        report['detected_flares_count'] = detected_flares['flare_detected'].sum()

        # Plots detected flares
        detection_fig = self.model.plot_training_data(self.data, detected_flares)
        detection_path = os.path.join(output_dir, "detected_flares.png")
        detection_fig.savefig(detection_path)
        report['detection_plot'] = detection_path

        # Makes predictions
        prediction_df = self.model.predict_future(self.data, days_ahead=7)
        report['prediction_dataframe'] = prediction_df

        # Plots predictions
        prediction_fig = self.model.plot_predictions(self.data)
        prediction_path = os.path.join(output_dir, "predictions.png")
        prediction_fig.savefig(prediction_path)
        report['prediction_plot'] = prediction_path

        # Plots flare prob.
        prob_fig = self.model.plot_flare_probability()
        prob_path = os.path.join(output_dir, "flare_probability.png")
        prob_fig.savefig(prob_path)
        report['probability_plot'] = prob_path

        # full text summary
        summary = f"""
        Solar Flare Activity Report
        --------------------------
        Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Sensitivity Level: {self.get_sensitivity()}
        Flare Classes Monitored: {', '.join(self.flare_classes)}

        Historical Analysis:
        - Data points analyzed: {len(self.data)}
        - Date range: {self.data['time'].min().strftime('%Y-%m-%d')} to {self.data['time'].max().strftime('%Y-%m-%d')}
        - Detected flares: {report['detected_flares_count']}

        Prediction Summary:
        - Prediction period: {prediction_df['time'].min().strftime('%Y-%m-%d')} to {prediction_df['time'].max().strftime('%Y-%m-%d')}
        - Highest predicted flux: {prediction_df['predicted_xray_flux'].max():.2e} W/m²
        - Highest predicted flare class: {prediction_df['predicted_flare_class'].max()}
        """

        summary_path = os.path.join(output_dir, "report_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(summary)
        report['summary'] = summary_path

        return report

    def get_sensitivity(self):
        """
        Get the current sensitivity level based on model params

        Returns:
            str: Sensitivity level
        """
        if self.n_states <= 3:
            return 'low'
        elif self.n_states <= 5:
            return 'medium'
        else:
            return 'high'

    def set_sensitivity(self, sensitivity):
        """
        Change the sensitivity level of the model. For this, you must retrain the model

        Args:
            sensitivity (str): Sensitivity level - 'low', 'medium', or 'high'

        Returns:
            bool: True if successful
        """
        # Create a new instance with the user's chosen sensitivity
        new_predictor = SolarFlarePredictor(sensitivity)

        # Transfer the data if available
        if hasattr(self, 'data'):
            new_predictor.model.train(self.data)
            new_predictor.is_trained = True
            new_predictor.data = self.date
        self.n_states = new_predictor.n_states
        self.flare_classes = new_predictor.flare_classes
        self.model = new_predictor.model
        self.is_trained = new_predictor.is_trained

        return True
