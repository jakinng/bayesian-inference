import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_state_election_forecast_from_notebook_data(
    state_abbr, 
    fit_model, 
    loaded_state_mapping, 
    polls_df, 
    model_dates_array, 
    current_election_day
    ):
    """
    Generates an election forecast plot for a given state using data
    processed within the notebook.

    Args:
        state_abbr (str): The abbreviation of the state (e.g., 'FL').
        fit_model: The CmdStanMCMC fit object from the notebook.
        loaded_state_mapping (dict): Dictionary mapping state abbreviation to numerical index.
        polls_df (pd.DataFrame): DataFrame with observed poll data (e.g., all_polls_df_S).
                                 Must contain 't' (datetime), 'pct_obama', 'index_s'.
        model_dates_array (list or pd.DatetimeIndex): Dates corresponding to the daily estimates in 'pi'.
                                                    (e.g., day_objects_linzner).
        current_election_day (pd.Timestamp): Election day.
    """
    if state_abbr not in loaded_state_mapping:
        print(f"State {state_abbr} not found in loaded_state_mapping.")
        return

    state_idx = loaded_state_mapping[state_abbr] - 1 # Stan is 1-indexed

    # 1. Extract data from Stan fit for the specific state
    try:
        pi_all_states_samples = fit_model.stan_variable("pi")
        if pi_all_states_samples is None:
            print("Failed to extract 'pi' variable from Stan fit object. Is the model run?")
            return
    except Exception as e:
        print(f"Error extracting 'pi' from fit_model: {e}")
        print("Please ensure 'fit' is a valid, completed CmdStanMCMC object.")
        return
        
    model_dates_idx = pd.to_datetime(model_dates_array)

    if state_idx >= pi_all_states_samples.shape[1]:
        print(f"State index {state_idx} for {state_abbr} is out of bounds for 'pi' variable samples.")
        return
    
    # pi_state_samples has shape (draws, days) for the selected state
    pi_state_samples = pi_all_states_samples[:, state_idx, :]  

    # Ensure model_dates_idx aligns with the days dimension of pi_state_samples
    if len(model_dates_idx) != pi_state_samples.shape[1]:
        print(f"Warning: Mismatch between number of model dates ({len(model_dates_idx)}) "
              f"and days in pi samples ({pi_state_samples.shape[1]}) for state {state_abbr}.")
        # If there's a mismatch, we might need to reconsider how 'J' or model_dates_idx are defined
        # For now, we'll proceed but the x-axis might be misaligned if J is incorrect.
        # It's crucial that J (number of days in stan model) matches len(model_dates_idx)
        # And that model_dates_idx covers the period for which pi is estimated.
        
    pi_daily_mean = np.mean(pi_state_samples, axis=0)
    pi_daily_lower = np.percentile(pi_state_samples, 5, axis=0)
    pi_daily_upper = np.percentile(pi_state_samples, 95, axis=0)
    
    # Election Day forecast (last day of modeled pi)
    # The Stan model `pi` variable is indexed by day `j` from 1 to J.
    # So, the last column pi_state_samples[:, -1] corresponds to the forecast for the J-th day.
    # We assume model_dates_idx[-1] is the date for this J-th day.
    election_day_plot_date = model_dates_idx[-1] # Date for the final forecast point
                                                 # This should ideally be current_election_day if J is set up to end on election day

    pi_election_day_samples = pi_state_samples[:, -1] # Samples for the J-th day
    pi_election_day_mean = np.mean(pi_election_day_samples)
    pi_election_day_lower = np.percentile(pi_election_day_samples, 5)
    pi_election_day_upper = np.percentile(pi_election_day_samples, 95)

    # 2. Get observed poll data for the state
    state_polls = polls_df[polls_df['index_s'] == state_idx+1].copy() # Use .copy() to avoid SettingWithCopyWarning
    state_polls['t'] = pd.to_datetime(state_polls['t'])

    # 3. Late Time-for-Change Forecast (National, as per user)
    try:
        # Ensure state_results_2004_df is indexed by 'state' if it's not already
        # Or filter directly as you suggested.
        late_tfc_series = state_results_2004.loc[state_results_2004['state'] == state_abbr, 'h_prior']
        if not late_tfc_series.empty:
            late_tfc_forecast_state_specific = late_tfc_series.iloc[0]
        else:
            print(f"Warning: h_prior not found for {state_abbr} in state_results_2004_df. Using NaN.")
            late_tfc_forecast_state_specific = np.nan
    except KeyError:
        print(f"Warning: 'state' or 'h_prior' column not found in state_results_2004_df. Using NaN for TFC forecast.")
        late_tfc_forecast_state_specific = np.nan
    except Exception as e:
        print(f"Error accessing h_prior for {state_abbr}: {e}. Using NaN for TFC forecast.")
        late_tfc_forecast_state_specific = np.nan


    # 4. Get Actual election outcome from data/2008.csv
    try:
        results_df_2008_actual = pd.read_csv('data/2008.csv')
        state_actual_data = results_df_2008_actual[results_df_2008_actual['state'] == state_abbr]
        if state_actual_data.empty:
            print(f"Actual 2008 results not found for {state_abbr} in data/2008.csv")
            actual_outcome_state = np.nan # Or handle as error
        else:
            obama_votes = state_actual_data['obama_count'].iloc[0]
            mccain_votes = state_actual_data['mccain_count'].iloc[0]
            if (obama_votes + mccain_votes) > 0:
                actual_outcome_state = obama_votes / (obama_votes + mccain_votes)
            else:
                actual_outcome_state = np.nan
                print(f"Zero total votes for Obama and McCain in {state_abbr}, cannot calculate share.")
    except FileNotFoundError:
        print("Error: 'data/2008.csv' not found. Please ensure the file exists.")
        return
    except Exception as e:
        print(f"Error processing 'data/2008.csv': {e}")
        actual_outcome_state = np.nan


    # 5. Plotting
    plt.figure(figsize=(13, 7)) # Adjusted for better legend display
    ax = plt.gca()

    # Plot observed polls
    pct_obama = state_polls['obama'] / state_polls['two_party_sum']
    ax.plot(state_polls['t'], pct_obama, 'o', color='skyblue', alpha=0.6, label='Observed Polls', markersize=5, zorder=2)
    
    # Plot Late Time-for-Change forecast
    ax.axhline(y=late_tfc_forecast_state_specific, color='dodgerblue', linestyle='--', linewidth=2, 
               label=f'Late Time-for-Change Forecast (National: {late_tfc_forecast_state_specific:.3f})', zorder=3)

    # Plot Actual election outcome
    if not np.isnan(actual_outcome_state):
        ax.axhline(y=actual_outcome_state, color='red', linestyle='-', linewidth=2, 
                   label=f'Actual Election Outcome ({actual_outcome_state:.3f})', zorder=3)

    # Plot daily estimate trend
    ax.plot(model_dates_idx, pi_daily_mean, color='dimgray', linewidth=2, label='Daily Preference Estimate (π_ij)', zorder=4)

    # Plot 90% CI for daily estimates
    ax.fill_between(model_dates_idx, pi_daily_lower, pi_daily_upper, color='silver', alpha=0.5, label='90% CI Daily Estimate', zorder=1)

    # Plot Election Day forecast point and CI bar
    ax.plot(election_day_plot_date, pi_election_day_mean, 'ko', markersize=8, label=f'Model Election Day Forecast ({pi_election_day_mean:.3f})', zorder=5)
    ax.vlines(election_day_plot_date, pi_election_day_lower, pi_election_day_upper, color='black', linewidth=3.5, label='90% CI Model Election Day', zorder=5)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Fraction Supporting Obama", fontsize=12)
    ax.set_title(f"Obama Support Forecast - {state_abbr}", fontsize=14)
    
    # Improve legend placement
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    
    plt.ylim(0.4, 0.6) # Adjusted for typical vote shares
    plt.xticks(rotation=30, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.82, 1]) # Adjust layout to make space for legend outside
    
    # Save the plot
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    plot_filename = f'figures/{state_abbr}_election_forecast_plot.png'
    try:
        plt.savefig(plot_filename)
        print(f"Plot for {state_abbr} saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot for {state_abbr}: {e}")
    plt.show() # Show plot inline as well
    plt.close() # Close the plot to free up memory


# 2. Modified Plotting Function
def plot_parameter_trends(
    parameter_samples,
    dates_array,
    dates_dict,
    title,
    y_label="Logit-scale Value",
    param_name="parameter",
    overlay_param_samples=None, # New: For overlaying another parameter's mean
    overlay_dates_array=None,   # New: Dates for the overlay parameter
    overlay_param_label='Mean Overlay Trend' # New: Label for the overlay
):
    """
    Plots the trend of a model parameter with a 90% CI and important dates.
    Optionally overlays the mean trend of a second parameter.
    """
    if parameter_samples is None or parameter_samples.size == 0:
        print(f"No samples provided for {title}. Skipping plot.")
        return
    if len(dates_array) != parameter_samples.shape[1]:
        print(f"Mismatch in length of dates_array ({len(dates_array)}) and parameter_samples time points ({parameter_samples.shape[1]}) for {title}.")
        if abs(len(dates_array) - parameter_samples.shape[1]) == 1 and len(dates_array) > parameter_samples.shape[1]:
             dates_array = dates_array[:parameter_samples.shape[1]]
             print(f"Adjusted primary dates_array to length {len(dates_array)}")
        else:
            print("Cannot automatically adjust dates for primary samples. Please check inputs.")
            return

    mean_trend = np.mean(parameter_samples, axis=0)
    lower_ci = np.percentile(parameter_samples, 5, axis=0)
    upper_ci = np.percentile(parameter_samples, 95, axis=0)

    plt.figure(figsize=(14, 7))
    ax = plt.gca()

    # Plot primary parameter trend and CI
    ax.plot(dates_array, mean_trend, color='black', linewidth=1.5, label=f'Mean {param_name.replace("_", " ").title()}')
    ax.fill_between(dates_array, lower_ci, upper_ci, color='grey', alpha=0.3, label='90% CI Primary Parameter')

    # Plot overlay parameter trend if provided
    if overlay_param_samples is not None and overlay_dates_array is not None:
        if overlay_param_samples.size > 0 and len(overlay_dates_array) == overlay_param_samples.shape[1]:
            mean_overlay_trend = np.mean(overlay_param_samples, axis=0)
            ax.plot(overlay_dates_array, mean_overlay_trend, color='blue', linestyle='--', linewidth=1.5, label=overlay_param_label, alpha=0.7)
        elif overlay_param_samples.size > 0:
            print(f"Warning: Mismatch in length of overlay_dates_array ({len(overlay_dates_array)}) and overlay_param_samples time points ({overlay_param_samples.shape[1]}) for overlay on {title}.")
            if abs(len(overlay_dates_array) - overlay_param_samples.shape[1]) == 1 and len(overlay_dates_array) > overlay_param_samples.shape[1]:
                temp_overlay_dates = overlay_dates_array[:overlay_param_samples.shape[1]]
                mean_overlay_trend = np.mean(overlay_param_samples, axis=0)
                ax.plot(temp_overlay_dates, mean_overlay_trend, color='blue', linestyle='--', linewidth=1.5, label=overlay_param_label + " (dates adjusted)", alpha=0.7)
                print(f"Adjusted overlay dates_array to length {len(temp_overlay_dates)} for plotting.")
            else:
                print("Skipping overlay plot due to date mismatch.")


    # Add vertical lines for important dates
    min_date_plot = dates_array.min()
    if overlay_dates_array is not None:
        min_date_plot = min(min_date_plot, overlay_dates_array.min())
    
    max_date_plot = dates_array.max()
    if overlay_dates_array is not None:
        max_date_plot = max(max_date_plot, overlay_dates_array.max())

    for label, date_val in dates_dict.items():
        if min_date_plot <= date_val <= max_date_plot:
            ax.axvline(date_val, color='crimson', linestyle='--', linewidth=1, alpha=0.8)
            ax.text(date_val + pd.Timedelta(days=1), ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    label, rotation=90, verticalalignment='bottom', color='crimson', fontsize=9, alpha=0.9)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left')
    plt.xticks(rotation=30, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()

    plot_filename = f'{param_name}_trend_plot.png'
    try:
        plt.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot {plot_filename}: {e}")
    plt.show()
    plt.close()