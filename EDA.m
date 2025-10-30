    %% ======== EDA Visualisations (Light Mode): ASX200 Returns (2015–2025) ======== %%
    clc; clear; close all;
    
    % === Load data ===
    data = readtable('ASX_2000_2025.csv');
    data.Properties.VariableNames = {'Index','Return'};
    
    % === Create synthetic dates for plotting ===
    startDate = datetime(2000,1,4);
    data.Date = startDate + caldays(data.Index - 1);
    
    % === Subset to 2015–2025 ===
    data = data(year(data.Date) >= 2015, :);
    y = data.Return;
    
    %% ===== Custom Visual Style =====
    set(groot, 'DefaultFigureColor', [1 1 1], ...
               'DefaultAxesColor', [1 1 1], ...
               'DefaultAxesFontName', 'Arial', ...
               'DefaultAxesFontSize', 11, ...
               'DefaultAxesGridLineStyle', ':', ...
               'DefaultAxesXColor', [0 0 0], ...
               'DefaultAxesYColor', [0 0 0], ...
               'DefaultAxesXGrid', 'on', ...
               'DefaultAxesYGrid', 'on');
    
    % Define color palette
    blue = [0.05 0.35 0.75];
    red  = [0.85 0.2 0.2];
    orange = [0.95 0.55 0.15];
    gray = [0.3 0.3 0.3];
    
    %% ---- Plot 1: Daily returns ----
    figure('Name','ASX200 Daily Returns','Color','w');
    plot(data.Date, y, 'Color',blue, 'LineWidth',0.8);
    xlabel('Date'); ylabel('Return (%)');
    title('ASX200 Daily Returns (2015–2025)','FontWeight','bold','FontSize',13,'Color','k');
    yline(0,'--','Color',[0.5 0.5 0.5]);
    xlim([data.Date(1), data.Date(end)]);
    grid on; box off;
    
    %% ---- Plot 2: Squared returns ----
    figure('Name','Squared Returns','Color','w');
    plot(data.Date, y.^2, 'Color',red, 'LineWidth',0.8);
    xlabel('Date'); ylabel('Squared Return');
    title('Volatility Clustering in Squared Returns','FontWeight','bold','FontSize',13,'Color','k');
    xlim([data.Date(1), data.Date(end)]);
    grid on; box off;
    
    %% ---- Plot 3: Histogram + normal fit ----
    figure('Name','Return Distribution','Color','w');
    h = histogram(y,40,'FaceColor',blue,'FaceAlpha',0.75,'EdgeColor','none');
    hold on;
    x = linspace(min(y),max(y),200);
    pd = fitdist(y,'Normal');
    y_pdf = pdf(pd,x);
    y_pdf_scaled = y_pdf * max(h.Values)/max(y_pdf);
    plot(x,y_pdf_scaled,'LineWidth',2,'Color',orange);
    hold off;
    xlabel('Return (%)'); ylabel('Frequency');
    title('Distribution of ASX200 Daily Returns','FontWeight','bold','FontSize',13,'Color','k');
    grid on; box off;
    
    %% ---- Plot 4: ACF of Returns ----
    figure('Name','ACF of Returns','Color','w');
    autocorr(y,'NumLags',40);
    title('Autocorrelation of Returns','FontWeight','bold','FontSize',13,'Color','k');
    set(gca,'LineWidth',1.2,'FontSize',11,'XColor','k','YColor','k');
    
    %% ---- Plot 5: ACF of Squared Returns ----
    figure('Name','ACF of Squared Returns','Color','w');
    autocorr(y.^2,'NumLags',40);
    title('Autocorrelation of Squared Returns','FontWeight','bold','FontSize',13,'Color','k');
    set(gca,'LineWidth',1.2,'FontSize',11,'XColor','k','YColor','k');
    
    %% ---- Optional: Save all figures automatically ----
    saveFolder = fullfile(pwd,'EDA_Plots');
    if ~exist(saveFolder,'dir')
        mkdir(saveFolder);
    end
    figHandles = findall(groot, 'Type', 'figure');
    for i = 1:length(figHandles)
        f = figHandles(i);
        exportgraphics(f, fullfile(saveFolder, sprintf('EDA_Fig%d.png', i)), 'Resolution',300);
    end
    disp('✅ All EDA plots saved in "EDA_Plots" folder at 300 DPI (light mode visuals).');
