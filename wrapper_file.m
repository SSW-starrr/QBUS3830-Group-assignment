clear; clc; close all;
Exploratory_Data_Analysis;
data_file = "ASX_2000_2025.csv";
data_file_cleaned = "ASX200_Cleaned_Last10Years.csv";
current_data_file = data_file;
while true
    exit = input("Do you want to exit the wrapper? Y/N [N]: ", 's');
    if upper(exit) == 'Y'
        break;
    end
    x = input("Do you want to use the cleaned data file? Y/N [Y]: ", 's');
    if isempty(x) || upper(x) == 'Y'
        current_data_file = data_file_cleaned;
    end
    model = input("Select model - 1: Automatic tuning, 2: Manual tuning, 3: FFVB, 4: RWM [1]")
    if isempty(model) || model == 1
        current_data_file = data_file;
        fprintf("Using full sample data file: %s\n", current_data_file);
        % Start timer, call the function, then report elapsed time
        tic;
        % Replace the function name below with the target function you want to time
        results = automatic_metro_in_gibbs(current_data_file, 12345);
        elapsed_time = toc;
        fprintf('Function automatic_metro_in_gibbs took %.2f seconds.\n', elapsed_time);
    elseif model == 2
        current_data_file = data_file_cleaned;
        fprintf("Using last 10 years data file: %s\n", current_data_file);
        % Start timer, call the function, then report elapsed time
        tic;
        % Replace the function name below with the target function you want to time
        results = Metropolis_within_Gibbs(current_data_file, 12345);
        elapsed_time = toc;
        fprintf('Function Metropolis_within_Gibbs took %.2f seconds.\n', elapsed_time);
    elseif model == 3
        fprintf("Using FFVB model with data file: %s\n", current_data_file);
        % Start timer, call the function, then report elapsed time
        tic;
        % Replace the function name below with the target function you want to time
        results = FFVB_LogGARCH_Single(current_data_file, 12345);
        elapsed_time = toc;
        fprintf('Function FFVB_LogGARCH_Single took %.2f seconds.\n', elapsed_time);
    elseif model == 4
        fprintf("Using RWM model with data file: %s\n", current_data_file);
        % Start timer, call the function, then report elapsed time
        tic;
        % Replace the function name below with the target function you want to time
        results = rwm_loggarch_baseline(current_data_file, 5000, 30000, 12345);
        elapsed_time = toc;
        fprintf('Function RWM took %.2f seconds.\n', elapsed_time);
    end
end
