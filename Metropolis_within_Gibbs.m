clear; clc; close all;
rng(12345); 
fprintf('Loading data...\n');
data = readtable('ASX_2000_2025.csv');
y = data.ret_asx;  % T x 1 vector
T = length(y);
mu0 = mean(y);
omega0 = log(std(y));
alpha0 = 0.1;
beta0 = 0.8;
beta_tilde0 = atanh(beta0);  % transform to unconstrained space

theta_tilde = [mu0, omega0, alpha0, beta_tilde0]';  % [mu, omega, alpha, beta_tilde]

%% Step 3: MCMC settings
n_iter = 30000;
burn_in = 5000;
sigma_proposal = [0.05, 0.1, 0.05, 0.2];  % tune for 20%-40% acceptance

%% Step 4: Pre-allocate
chain = zeros(n_iter, 4);
accept = zeros(1, 4);

%% Step 5: Initialize
current_theta = theta_tilde;
current_loglik = loglik_logGARCH(current_theta, y);
current_logprior = logprior_logGARCH(current_theta);
chain(1, :) = current_theta';

%% Step 6: MCMC loop
fprintf('Running Metropolis-within-Gibbs...\n');
for iter = 2:n_iter
    for j = 1:4
        % Propose new value
        proposal = current_theta;
        proposal(j) = current_theta(j) + randn * sigma_proposal(j);
        
        % Compute log-likelihood and log-prior for proposal
        prop_loglik = loglik_logGARCH(proposal, y);
        prop_logprior = logprior_logGARCH(proposal);
        
        % Handle invalid proposals
        if isinf(prop_logprior) || isnan(prop_loglik)
            prop_loglik = -Inf;
            prop_logprior = -Inf;
        end
        
        % Acceptance probability (symmetric proposal => q ratio = 1)
        log_alpha = (prop_loglik + prop_logprior) - (current_loglik + current_logprior);
        alpha = min(1, exp(log_alpha));
        
        % Accept or reject
        if rand < alpha
            current_theta = proposal;
            current_loglik = prop_loglik;
            current_logprior = prop_logprior;
            accept(j) = accept(j) + 1;
        end
    end
    chain(iter, :) = current_theta';
end

%% Step 7: Post-processing
accept_rate = accept / (n_iter - 1);
fprintf('Acceptance rates: mu=%.1f%%, omega=%.1f%%, alpha=%.1f%%, beta_tilde=%.1f%%\n', ...
    accept_rate * 100);

% Discard burn-in
post_burn = chain(burn_in+1:end, :);

% Transform beta_tilde back to beta
mu_samp = post_burn(:,1);
omega_samp = post_burn(:,2);
alpha_samp = post_burn(:,3);
beta_tilde_samp = post_burn(:,4);
beta_samp = tanh(beta_tilde_samp);

theta_samp = [mu_samp, omega_samp, alpha_samp, beta_samp];
param_names = {'mu', 'omega', 'alpha', 'beta'};

% Compute posterior statistics
post_mean = mean(theta_samp);
post_std = std(theta_samp);
post_CI = quantile(theta_samp, [0.025, 0.975]);

% Round to 4 decimal places
post_mean = round(post_mean, 4);
post_std = round(post_std, 4);
post_CI = round(post_CI, 4);

% Display results
fprintf('\nPosterior Estimates (4 decimal places):\n');
for i = 1:4
    fprintf('%s: mean=%.4f, std=%.4f, 95%% CI=[%.4f, %.4f]\n', ...
        param_names{i}, post_mean(i), post_std(i), post_CI(1,i), post_CI(2,i));
end

%% Step 8: Trace plots
figure('Position', [100, 100, 800, 600]);
for i = 1:4
    subplot(2, 2, i);
    plot(post_burn(:,i));  % plot on transformed scale for convergence check
    title(['Trace plot: ', param_names{i}]);
    xlabel('Iteration');
    ylabel('Value');
end
sgtitle('Trace Plots (after burn-in)');
saveas(gcf, 'trace_plots.png');

%% Step 9: Volatility Forecast for 15-Oct-2025 and 16-Oct-2025
fprintf('\nForecasting volatility...\n');
[vol_1step, vol_2step] = forecast_volatility(theta_samp, y);
fprintf('1-step forecast (15-Oct-2025): %.4f\n', mean(vol_1step));
fprintf('2-step forecast (16-Oct-2025): %.4f\n', mean(vol_2step));

%% Supporting Functions

function logL = loglik_logGARCH(theta_tilde, y)
% Compute log-likelihood for log-GARCH(1,1) model
% theta_tilde = [mu, omega, alpha, beta_tilde]

mu = theta_tilde(1);
omega = theta_tilde(2);
alpha = theta_tilde(3);
beta_tilde = theta_tilde(4);
beta = tanh(beta_tilde);  % transform back

T = length(y);
sigma2 = zeros(T, 1);
sigma2(1) = var(y);  % initial value

% Recursive computation of sigma_t
for t = 2:T
    epsilon_prev = (y(t-1) - mu) / sqrt(sigma2(t-1));
    log_sigma_t = omega + alpha * (abs(epsilon_prev) - sqrt(2/pi)) + beta * log(sqrt(sigma2(t-1)));
    sigma2(t) = exp(2 * log_sigma_t);
end

% Log-likelihood
residuals = y - mu;
logL = -0.5 * sum(log(2*pi) + log(sigma2) + (residuals.^2) ./ sigma2);
end

function logp = logprior_logGARCH(theta_tilde)
% Compute log-prior with Jacobian correction for beta transformation
% theta_tilde = [mu, omega, alpha, beta_tilde]

mu = theta_tilde(1);
omega = theta_tilde(2);
alpha = theta_tilde(3);
beta_tilde = theta_tilde(4);
beta = tanh(beta_tilde);

% Check uniform prior bounds
if (omega < -2 || omega > 2) || (alpha < -2 || alpha > 2)
    logp = -Inf;
    return;
end

% Jacobian: d beta / d beta_tilde = 1 - beta^2
jacobian = log(1 - beta^2);

% Flat prior for mu => log p(mu) = 0
% Constants (log(1/4), etc.) omitted as they cancel in MH ratio
logp = jacobian;
end

function [vol_1step, vol_2step] = forecast_volatility(theta_samp, y)
% Generate 1-step and 2-step ahead volatility forecasts
% theta_samp: N x 4 matrix of posterior samples [mu, omega, alpha, beta]

N = size(theta_samp, 1);
T = length(y);
vol_1step = zeros(N, 1);
vol_2step = zeros(N, 1);

for i = 1:N
    mu = theta_samp(i, 1);
    omega = theta_samp(i, 2);
    alpha = theta_samp(i, 3);
    beta = theta_samp(i, 4);
    
    % Reconstruct sigma_T using the full sample
    sigma2 = zeros(T, 1);
    sigma2(1) = var(y);
    for t = 2:T
        epsilon_prev = (y(t-1) - mu) / sqrt(sigma2(t-1));
        log_sigma_t = omega + alpha * (abs(epsilon_prev) - sqrt(2/pi)) + beta * log(sqrt(sigma2(t-1)));
        sigma2(t) = exp(2 * log_sigma_t);
    end
    
    % 1-step forecast: t = T+1
    epsilon_T = (y(T) - mu) / sqrt(sigma2(T));
    log_sigma_T1 = omega + alpha * (abs(epsilon_T) - sqrt(2/pi)) + beta * log(sqrt(sigma2(T)));
    vol_1step(i) = exp(log_sigma_T1);
    
    % 2-step forecast: t = T+2
    % E[|epsilon_{T+1}|] = sqrt(2/pi) for N(0,1)
    log_sigma_T2 = omega + alpha * (sqrt(2/pi) - sqrt(2/pi)) + beta * log_sigma_T1;
    vol_2step(i) = exp(log_sigma_T2);
end
end

%% End of main script