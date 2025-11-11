clear; clc; close all;
% Exploratory_Data_Analysis;
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
        % Run all models and record elapsed times, then show a table and plot
        fprintf("Running all models with data file: %s\n", current_data_file);

        tic;
        results_auto = automatic_metro_in_gibbs(current_data_file, 12345);
        elapsed_auto = toc;
        fprintf('automatic_metro_in_gibbs took %.2f seconds.\n', elapsed_auto);

        tic;
        results_mwg = Metropolis_within_Gibbs(current_data_file, 12345);
        elapsed_mwg = toc;
        fprintf('Metropolis_within_Gibbs took %.2f seconds.\n', elapsed_mwg);

        tic;
        results_ffvb = FFVB_LogGARCH_Single(current_data_file, 12345);
        elapsed_ffvb = toc;
        fprintf('FFVB_LogGARCH_Single took %.2f seconds.\n', elapsed_ffvb);

        tic;
        results_rwm = rwm_test(current_data_file, 6500, 30000, 12345);
        elapsed_rwm = toc;
        fprintf('rwm_test took %.2f seconds.\n', elapsed_rwm);

        % Collate timings
        model_names = {'automatic_metro_in_gibbs', 'Metropolis_within_Gibbs', 'FFVB_LogGARCH_Single', 'rwm_test'}';
        elapsed_seconds = [elapsed_auto; elapsed_mwg; elapsed_ffvb; elapsed_rwm];

        % Display a table in the command window
        timing_table = table(model_names, elapsed_seconds, 'VariableNames', {'Model','ElapsedSeconds'});
        disp(timing_table);

        % Create a bar plot of timings and a GUI table for inspection
        fig = figure('Name','Model run times','NumberTitle','off','Position',[100 100 900 400]);
        subplot(1,2,1);
        bar(elapsed_seconds, 0.6, 'FaceColor',[0.2 0.6 0.8]);
        set(gca, 'XTickLabel', model_names, 'XTick', 1:numel(model_names));
        xtickangle(45);
        ylabel('Seconds');
        title('Model run times');

        subplot(1,2,2);
        % Create a uitable showing the timings. Use table2cell so mixed types are handled.
        try
            uit = uitable('Parent', fig, 'Data', table2cell(timing_table), 'ColumnName', timing_table.Properties.VariableNames, ...
                'Units','normalized','Position',[0.53 0.08 0.44 0.84], 'ColumnFormat', {'char','numeric'});
        catch
            % fallback for older MATLAB versions
            uit = uitable('Parent', fig, 'Data', table2cell(timing_table), 'ColumnName', timing_table.Properties.VariableNames, ...
                'Units','normalized','Position',[0.53 0.08 0.44 0.84]);
        end

        % Save figure
        try
            saveas(fig, 'all_models_timing.png');
        catch
            % ignore save errors in headless environments
        end

        % keep last results in `results` for compatibility
        results = struct('automatic',results_auto,'mwg',results_mwg,'ffvb',results_ffvb,'rwm',results_rwm,'timing',timing_table);

    end
end
