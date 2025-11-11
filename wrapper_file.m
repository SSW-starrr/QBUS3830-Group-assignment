clear; clc; close all;
Exploratory_Data_Analysis;
data_file = "ASX_2000_2025.csv";
data_file_cleaned = "ASX200_Cleaned_Last10Years.csv";
current_data_file = data_file;
while true
    exit = input("Do you want to exit the wrapper? Y/N [N]: ", 's');
    if upper(exit) == 'Y'
        quit(0);
    end
    x = input("Do you want to use the cleaned data file? Y/N [Y]: ", 's');
    if isempty(x) || upper(x) == 'Y'
        current_data_file = data_file_cleaned;
    else
        current_data_file = data_file;
    end
    model = input("Select model - 1: Automatic tuning, 2: Manual tuning, 3: FFVB, 4: RWM [1]: ");
    if isempty(model) || model == 1
        fprintf("Using full sample data file: %s\n", current_data_file);
        % Start timer, call the function, then report elapsed time
        tic;        
        results = automatic_metro_in_gibbs(current_data_file, 12345);
        elapsed_time = toc;
        fprintf('Function automatic_metro_in_gibbs took %.2f seconds.\n', elapsed_time);
    elseif model == 2
        fprintf("Using last 10 years data file: %s\n", current_data_file);
        % Start timer, call the function, then report elapsed time
        tic;        
        results = Metropolis_within_Gibbs(current_data_file, 12345);
        elapsed_time = toc;
        fprintf('Function Metropolis_within_Gibbs took %.2f seconds.\n', elapsed_time);
    elseif model == 3
        fprintf("Using FFVB model with data file: %s\n", current_data_file);
        % Start timer, call the function, then report elapsed time
        tic;        
        results = FFVB_LogGARCH_Single(current_data_file, 12345);
        elapsed_time = toc;
        fprintf('Function FFVB_LogGARCH_Single took %.2f seconds.\n', elapsed_time);
    elseif model == 4
        fprintf("Using RWM model with data file: %s\n", current_data_file);
        % Start timer, call the function, then report elapsed time
        tic;        
        results = rwm_loggarch_baseline(current_data_file, 10000, 30000, 12345);
        elapsed_time = toc;
        fprintf('Function RWM took %.2f seconds.\n', elapsed_time);
    elseif model == 5
        fprintf("Running all with data file with data file: %s\n", current_data_file);
        tic;        
        results = automatic_metro_in_gibbs(current_data_file, 12345);
        elapsed_time = toc;
        fprintf('Function automatic_metro_in_gibbs took %.2f seconds.\n', elapsed_time);
        % Start timer, call the function, then report elapsed time
        tic;        
        results = Metropolis_within_Gibbs(current_data_file, 12345);
        elapsed_time = toc;
        fprintf('Function Metropolis_within_Gibbs took %.2f seconds.\n', elapsed_time);
        tic;        
        results = Metropolis_within_Gibbs(current_data_file, 12345);
        elapsed_time = toc;
        fprintf('Function Metropolis_within_Gibbs took %.2f seconds.\n', elapsed_time);
        tic;        
        results = rwm_loggarch_baseline(current_data_file, 10000, 30000, 12345);
        elapsed_time = toc;
        fprintf('Function RWM took %.2f seconds.\n', elapsed_time);
        tic;        
        results = rwm_loggarch_baseline(current_data_file, 10000, 30000, 12345);
        elapsed_time = toc;
        fprintf('Function RWM took %.2f seconds.\n', elapsed_time);
    end
end
