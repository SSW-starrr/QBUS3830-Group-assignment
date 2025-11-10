%  QBUS3830 Assignment 2 — EDA + Data Cleaning
%  ASX200 Daily Returns (4 Jan 2000 – 14 Oct 2025)


clc; clear; close all;

%% 1. Load data 
data = readtable('ASX_2000_2025.csv');
data.Properties.VariableNames = {'Index','ret_asx'};

n = height(data);

% Rebuild true daily trading timeline (6513 evenly spaced points)
startDate = datetime(2000,1,4);
endDate   = datetime(2025,10,14);
data.Date = dateshift(linspace(startDate, endDate, n)', 'start', 'day');

fprintf('Full dataset: %s → %s (%d obs)\n', ...
    string(min(data.Date)), string(max(data.Date)), n);

%% 2. Define rolling 10-year estimation window
winEnd   = max(data.Date);           % 14-Oct-2025
winStart = winEnd - years(10);       % 15-Oct-2015
mask = data.Date >= winStart & data.Date <= winEnd;

% Cleaned subset for all later modelling
data_clean = data(mask, :);
y      = data_clean.ret_asx;
dates  = data_clean.Date;

fprintf('Estimation window: %s → %s  |  %d obs (≈ %.1f years)\n\n', ...
    string(min(dates)), string(max(dates)), height(data_clean), height(data_clean)/252);

%% 3. Custom visual style
set(groot,'DefaultFigureColor',[1 1 1], ...
           'DefaultAxesFontName','Arial', ...
           'DefaultAxesFontSize',11, ...
           'DefaultAxesGridLineStyle',':', ...
           'DefaultAxesXColor',[0 0 0], ...
           'DefaultAxesYColor',[0 0 0], ...
           'DefaultAxesXGrid','on', ...
           'DefaultAxesYGrid','on');

blue   = [0.05 0.35 0.75];
red    = [0.85 0.2 0.2];
orange = [0.95 0.55 0.15];

% 4. Full-period plot (highlight last-10-year window)

figure('Name','Full Period Returns','Color','w'); hold on;
yl = [min(data.ret_asx) max(data.ret_asx)];

fill([winStart winEnd winEnd winStart],[yl(1) yl(1) yl(2) yl(2)], ...
     [1 0.9 0.75],'EdgeColor','none','FaceAlpha',0.35);
plot(data.Date, data.ret_asx,'Color',blue,'LineWidth',0.7);
yline(0,'--','Color',[0.5 0.5 0.5]);

xlabel('Date (Daily Frequency)');
ylabel('Daily Return (%)');
title('ASX200 Daily Returns (4 Jan 2000 – 14 Oct 2025) with Last-10-Year Forecast Window', ...
      'FontWeight','bold','FontSize',13);
xlim([min(data.Date), max(data.Date)]);
legend({'Last-10-Year Forecast Window','ASX200 Daily Returns'}, ...
       'Location','southoutside','Orientation','horizontal','Box','off');
grid on; box off;
hold off;

% 5. EDA for last-10-year window

% (1) Daily returns
figure('Name','ASX200 Daily Returns (Last 10 Years)','Color','w');
plot(dates, y,'Color',blue,'LineWidth',0.8);
xlabel('Date'); ylabel('Daily Return (%)');
title('ASX200 Daily Returns (Last 10 Years)','FontWeight','bold','FontSize',13);
yline(0,'--','Color',[0.5 0.5 0.5]); xlim([dates(1) dates(end)]); grid on;

% (2) Squared returns (volatility clustering) 
figure('Name','Squared Returns','Color','w');
plot(dates, y.^2,'Color',red,'LineWidth',0.8);
xlabel('Date'); ylabel('Squared Daily Return');
title('Volatility Clustering in Squared Returns (Last 10 Years)','FontWeight','bold','FontSize',13);
xlim([dates(1) dates(end)]); grid on;

% (3) Distribution of returns 
figure('Name','Return Distribution','Color','w');
h = histogram(y,40,'FaceColor',blue,'FaceAlpha',0.75,'EdgeColor','none');
hold on;
x = linspace(min(y),max(y),200);
pd = fitdist(y,'Normal');
y_pdf = pdf(pd,x);
y_pdf_scaled = y_pdf * max(h.Values)/max(y_pdf);
plot(x,y_pdf_scaled,'LineWidth',2,'Color',orange);
hold off;
xlabel('Daily Return (%)'); ylabel('Frequency');
title('Distribution of ASX200 Daily Returns (Last 10 Years)','FontWeight','bold','FontSize',13);
grid on; box off;

% (4) ACF of returns
figure('Name','ACF of Returns','Color','w');
autocorr(y,'NumLags',40);
title('Autocorrelation of Daily Returns (Last 10 Years)','FontWeight','bold','FontSize',13);

% (5) ACF of squared returns
figure('Name','ACF of Squared Returns','Color','w');
autocorr(y.^2,'NumLags',40);
title('Autocorrelation of Squared Daily Returns (Last 10 Years)','FontWeight','bold','FontSize',13);

%% 6. Export results
outFolder = fullfile(pwd,'EDA_Plots');
if ~exist(outFolder,'dir'), mkdir(outFolder); end
figs = findall(groot,'Type','figure');
for i = 1:length(figs)
    exportgraphics(figs(i),fullfile(outFolder,sprintf('EDA_Fig%d.png',i)),'Resolution',300);
end
disp('All EDA plots saved to "EDA_Plots" (300 DPI).');

% Save cleaned dataset for later use
cleanFile = fullfile(pwd,'ASX200_Cleaned_Last10Years.csv');
writetable(data_clean, cleanFile);
disp(['Cleaned dataset saved as: ', cleanFile]);
