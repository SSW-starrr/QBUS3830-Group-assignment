clear; clc; close all;
data_file = "ASX_2000_2025.csv";
printf('Running automatic_metro_in_gibbs with data file: %s\n', data_file);
results = automatic_metro_in_gibbs(data_file, 3781);