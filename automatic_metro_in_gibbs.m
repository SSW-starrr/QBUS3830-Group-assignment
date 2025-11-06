function results = automatic_metro_in_gibbs(dataFile, start_idx)
% automatic_metro_in_gibbs(dataFile)
% Wraped version of the original script. dataFile is optional (default
% 'ASX_2000_2025.csv'). Returns a struct `results` with posterior
% summaries and forecasts.

if nargin < 1 || isempty(dataFile)
    dataFile = 'ASX_2000_2025.csv';
end

% Keep RNG for reproducibility inside the function
rng(12345);

fprintf('Loading data...\n');
data = readtable(dataFile);
y_full = data.ret_asx;  % Full sample: 2000 to 2025

% Estimate start index for 2015-01-01
% Approx: 15 years * 252 trading days = 3780
if nargin < 2 || isempty(start_idx)
    start_idx = 3781;  % 1-based indexing
end
y = y_full(start_idx:end);  % Subsample: 2015 to 2025
T = length(y);

fprintf('Using %d observations from 2015 to 2025.\n', T);

mu0 = mean(y);
omega0 = log(std(y));
alpha0 = 0.1;
beta0 = 0.8;
beta_tilde0 = atanh(beta0);  % transform to unconstrained space
theta_tilde = [mu0, omega0, alpha0, beta_tilde0]';  % [mu, omega, alpha, beta_tilde]
%% Step 3: MCMC settings
n_iter = 5000;
burn_in = 5000;

%% Step 3.5 Automatically tune the sigma proposal
sigma_proposal = [0.05, 0.1, 0.05, 0.2];  % Initial guess
sigma_min = zeros(1, 4);                 % Lower bound for bisection
sigma_max = repmat(Inf, 1, 4);         % Upper bound for bisection
max_tuning_iter = 25;                    % Safety break for tuning
tuning_iter = 0;
target_min = 0.20;
target_max = 0.30;

end_loop = 0;
fprintf('Starting MCMC with auto-tuning...\n');
fprintf('Target acceptance rate: [%.0f%%, %.0f%%]\n\n', target_min*100, target_max*100);

while end_loop == 0
    tuning_iter = tuning_iter + 1;
    if tuning_iter > max_tuning_iter
        warning('Tuning failed to converge after %d iterations. Using last sigmas.', max_tuning_iter);
        end_loop = 1; % Force exit and use current (suboptimal) results
    end
    
    %% Step 4: Pre-allocate
    chain = zeros(n_iter, 4);
    accept = zeros(1, 4);
    
    %% Step 5: Initialize
    current_theta = theta_tilde;
    current_loglik = loglik_logGARCH(current_theta, y);
    current_logprior = logprior_logGARCH(current_theta);
    chain(1, :) = current_theta';
    
    %% Step 6: MCMC loop
    fprintf('Tuning Iter %d: Running Metropolis-within-Gibbs...\n', tuning_iter);
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
    
    %% Step 7: Post-processing and Tuning Check
    accept_rate = accept / (n_iter - 1);
    fprintf('Tuning Iter %d: Rates: mu=%.1f%%, omega=%.1f%%, alpha=%.1f%%, beta_tilde=%.1f%%\n', ...
        tuning_iter, accept_rate * 100);
    
    % Check if all rates are within the target range
    all_in_range = all(accept_rate >= target_min & accept_rate <= target_max);
    
    if all_in_range
        fprintf('Acceptance rates are in the target range. Tuning complete.\n');
        end_loop = 1; % Exit the while loop
    else
        % Adjust sigmas for the next iteration
        if end_loop == 0 % Only adjust if we haven't forced an exit
            fprintf('Adjusting proposal sigmas using bisection...\n');
            new_sigma_proposal = sigma_proposal;
            
            for j = 1:4
                if accept_rate(j) > target_max
                    % Rate is too high, increase sigma
                    sigma_min(j) = sigma_proposal(j); % Current sigma is new lower bound
                    if isinf(sigma_max(j))
                        new_sigma_proposal(j) = sigma_proposal(j) * 1.5; % Explore upwards
                    else
                        new_sigma_proposal(j) = (sigma_min(j) + sigma_max(j)) / 2; % Bisect
                    end
                elseif accept_rate(j) < target_min
                    % Rate is too low, decrease sigma
                    sigma_max(j) = sigma_proposal(j); % Current sigma is new upper bound
                    if sigma_min(j) == 0
                        new_sigma_proposal(j) = sigma_proposal(j) * 0.5; % Explore downwards
                    else
                        new_sigma_proposal(j) = (sigma_min(j) + sigma_max(j)) / 2; % Bisect
                    end
                end
                % If rate is in range, we don't adjust its sigma,
                % but the loop continues for the other parameters.
            end
            
            sigma_proposal = new_sigma_proposal;
            
            fprintf('New sigmas for next run: mu=%.4f, omega=%.4f, alpha=%.4f, beta_tilde=%.4f\n\n', ...
                sigma_proposal(1), sigma_proposal(2), sigma_proposal(3), sigma_proposal(4));
            
            % Skip plotting/forecasting for this tuning run
            continue;
        end
    end
   
    
end

n_iter = 30000;
burn_in = 5000;

%% Step 4: Pre-allocate
chain = zeros(n_iter, 4);
accept = zeros(1, 4);

%% Step 5: Initialize
current_theta = theta_tilde;
current_loglik = loglik_logGARCH(current_theta, y);
current_logprior = logprior_logGARCH(current_theta);
chain(1, :) = current_theta';

%% Step 6: MCMC loop
fprintf('Running Iter %d: Running Metropolis-within-Gibbs with n_iter=%d...\n', tuning_iter, n_iter);
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

%% Step 7: Post-processing and Tuning Check
accept_rate = accept / (n_iter - 1);
fprintf('Tuning Iter %d: Rates: mu=%.1f%%, omega=%.1f%%, alpha=%.1f%%, beta_tilde=%.1f%%\n', ...
    tuning_iter, accept_rate * 100);

% --- This section is only reached when end_loop = 1 ---
fprintf('\nProceeding to final analysis...\n');

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


% Prepare results to return
results = struct();
results.post_mean = post_mean;
results.post_std = post_std;
results.post_CI = post_CI;
results.theta_samp = theta_samp;
results.post_burn = post_burn;
results.chain = chain;
results.sigma_proposal = sigma_proposal;
results.accept_rate = accept_rate;
results.vol_1step = vol_1step;
results.vol_2step = vol_2step;

end % function automatic_metro_in_gibbs

%% Supporting Functions
% (Functions are unchanged from your original script)

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

% Handle potential numerical instability in first observation
if sigma2(1) <= 0
    sigma2(1) = 1e-6; 
end

% Recursive computation of sigma_t
for t = 2:T
    epsilon_prev = (y(t-1) - mu) / sqrt(sigma2(t-1));
    log_sigma_t = omega + alpha * (abs(epsilon_prev) - sqrt(2/pi)) + beta * log(sqrt(sigma2(t-1)));
    
    % Prevent overflow/underflow
    if log_sigma_t > 35 % exp(2*35) is huge
        sigma2(t) = exp(70);
    elseif log_sigma_t < -35 % exp(2*-35) is near zero
        sigma2(t) = exp(-70);
    else
        sigma2(t) = exp(2 * log_sigma_t);
    end
    
    % Ensure positivity
    if sigma2(t) <= 0
       sigma2(t) = 1e-6;
    end
end

% Log-likelihood
residuals = y - mu;
logL_terms = -0.5 * (log(2*pi) + log(sigma2) + (residuals.^2) ./ sigma2);

% Check for NaNs or Infs which can result from bad proposals
if any(isnan(logL_terms)) || any(isinf(logL_terms))
    logL = -Inf;
else
    logL = sum(logL_terms);
end
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

% Check beta bounds (tanh is always in [-1, 1], but let's be explicit)
% We need alpha + beta < 1 for stationarity of log-vol, but this is
% hard to enforce on transformed parameters. The prior on alpha
% and the transformation on beta often keep it stable.
% Let's just ensure beta is not exactly 1 or -1.
if abs(beta) >= 0.99999
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
    if sigma2(1) <= 0, sigma2(1) = 1e-6; end
    
    for t = 2:T
        epsilon_prev = (y(t-1) - mu) / sqrt(sigma2(t-1));
        log_sigma_t = omega + alpha * (abs(epsilon_prev) - sqrt(2/pi)) + beta * log(sqrt(sigma2(t-1)));
        
        if log_sigma_t > 35
            sigma2(t) = exp(70);
        elseif log_sigma_t < -35
            sigma2(t) = exp(-70);
        else
            sigma2(t) = exp(2 * log_sigma_t);
        end
        if sigma2(t) <= 0, sigma2(t) = 1e-6; end
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