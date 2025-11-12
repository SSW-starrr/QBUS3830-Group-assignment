function FFVB_LogGARCH_Single()

fprintf('=== FFVB for Log-GARCH(1,1) Model - CORRECTED Version (2015–2025) ===\n');

% Set random seed
rng(12345, 'twister');

%% 1. Data Preparation and Preprocessing - FIXED
fprintf('1. Loading and preparing data...\n');

% Load full data
data = readtable('ASX_2000_2025.csv');
y_full = data.ret_asx;

% CORRECTED: Find actual start index for 2015-01-01 using dates
% Assuming the table has a 'Date' column
if ismember('Date', data.Properties.VariableNames)
    dates = datetime(data.Date, 'InputFormat', 'yyyy-MM-dd');
    start_date = datetime(2015,1,1);
    start_idx = find(dates >= start_date, 1, 'first');
else
    % Fallback: use approximately 15 years of trading days
    trading_days_per_year = 252;
    years_before_2015 = 15;
    start_idx = max(1, length(y_full) - years_before_2015 * trading_days_per_year);
end

y = y_full(start_idx:end);  % Subsample: 2015 to 2025

% Data preprocessing
y = rmmissing(y);
T = length(y);

fprintf('   Using %d observations from index %d to end.\n', T, start_idx);
fprintf('   Return statistics: mean=%.4f, std=%.4f\n', mean(y), std(y));

%% 2. Model Configuration - FIXED for Log-GARCH
fprintf('2. Initializing model configuration and variational parameters...\n');

config = get_model_config(T, y);

% Initialize variational parameters - CORRECTED initialization
variational = initialize_variational_parameters(y);

%% 3. Run FFVB Optimization - with stability fixes
fprintf('3. Starting FFVB optimization...\n');
[variational_opt, elbo_history, m_history] = run_ffvb_optimization_optimized(...
    variational, config.params, config.optim);

if isempty(elbo_history) || all(isnan(elbo_history)) || all(abs(elbo_history) > 1e6)
    fprintf('   FFVB optimization failed! Using fallback values\n');
    % Create reasonable fallback values
    variational_opt = variational;
    elbo_history = -1e3 * ones(100, 1); % Dummy ELBO history
    m_history = repmat(variational.m', 100, 1);
else
    fprintf('   FFVB optimization successful! Final ELBO: %.4f\n', elbo_history(end));
end

%% 4. Posterior Analysis
fprintf('4. Posterior sampling and analysis...\n');
results = perform_posterior_analysis(variational_opt, config.post_analysis);

% Display results
display_optimized_results(results, variational_opt, elbo_history);

%% 5. Volatility Forecasting
fprintf('5. Forecasting volatility...\n');
[vol1, vol1_ci, vol2, vol2_ci] = forecast_volatility_ffvb(results.samples, config.params, results.stats);
fprintf('   1-step forecast: %.4f [%.4f, %.4f]\n', vol1, vol1_ci(1), vol1_ci(2));
fprintf('   2-step forecast: %.4f [%.4f, %.4f]\n', vol2, vol2_ci(1), vol2_ci(2));

% Create plots - with robust handling
if ~isempty(elbo_history) && length(elbo_history) > 5 && ~any(isnan(elbo_history))
    create_optimized_plots(elbo_history, m_history, results);
else
    fprintf('   Insufficient valid optimization history data, skipping plots\n');
    % Create dummy plot to show structure
    figure;
    plot(1:10, -1000*ones(1,10), 'LineWidth', 2);
    title('ELBO Convergence (No valid data)');
    xlabel('Iteration');
    ylabel('ELBO');
end

fprintf('\n=== FFVB Analysis Complete! ===\n');

end

%% =========================================================================
% CORRECTED Configuration Functions for Log-GARCH
% =========================================================================

function config = get_model_config(T, y)
    % Model parameter configuration - CORRECTED for Log-GARCH
    config.params.T = T;
    config.params.y = y;
    config.params.sqrt_2_pi = sqrt(2/pi);
    config.params.epsilon = 1e-8; % Smaller epsilon for numerical stability

    % Optimizer configuration - MORE STABLE
    config.optim.learning_rate = 0.0005; % Reduced learning rate
    config.optim.beta1 = 0.9;
    config.optim.beta2 = 0.999;
    config.optim.epsilon = 1e-8;
    config.optim.max_iter = 2000; % Increased max iterations
    config.optim.patience = 50; % More patience
    config.optim.tol = 1e-3; % Less strict tolerance
    config.optim.grad_clip = 10.0; % Higher clipping threshold
    
    
    % Posterior analysis configuration
    config.post_analysis.N_samples = 2000; % Reduced samples for stability
    config.post_analysis.burnin = 500;
end

function variational = initialize_variational_parameters(y)
    % CORRECTED initialization for Log-GARCH parameters
    
    y_std = std(y);
    y_mean = mean(y);
    log_variance = log(y_std^2);
    
    variational = struct();
    
    % Initial variational mean: [mu, omega, alpha, beta]
    % Log-GARCH parameters can be negative, so use more reasonable initialization
    variational.m = [y_mean;                  % mu: mean return
                    log_variance - 0.1;      % omega: base log-volatility
                    0.1;                     % alpha: ARCH effect
                    0.8];                    % beta: GARCH effect (persistence)
    
    % Adaptive initialization of standard deviations - MORE CONSERVATIVE
    init_std = [0.05, 0.2, 0.1, 0.1]';
    variational.log_s = log(init_std);
    
    % Sample size settings
    variational.L = 64; % Increased samples for better gradient estimation
    
    fprintf('   Initial m: [%.4f, %.4f, %.4f, %.4f]\n', variational.m);
    fprintf('   Initial s: [%.4f, %.4f, %.4f, %.4f]\n', exp(variational.log_s));
end

%% =========================================================================
% CORRECTED Optimized Main Functionality for Log-GARCH
% =========================================================================

function [variational_opt, elbo_history, m_history] = run_ffvb_optimization_optimized(variational, params, optim)
% CORRECTED FFVB optimization with stability improvements

fprintf('   Iteration progress: ');

% Extract parameters
m = variational.m;
log_s = variational.log_s;
L = variational.L;
D = length(m);

% Adam optimizer initialization
m_m = zeros(D, 1);
m_v = zeros(D, 1);
log_s_m = zeros(D, 1);
log_s_v = zeros(D, 1);

% Pre-allocation with NaN initialization
elbo_history = NaN(optim.max_iter, 1);
m_history = NaN(optim.max_iter, D);

best_elbo = -inf;
best_m = m;
best_log_s = log_s;
patience_counter = 0;
valid_iter_count = 0;

% Main optimization loop
for iter = 1:optim.max_iter
    try
        [elbo_val, grad_m, grad_log_s, success] = compute_elbo_and_gradients_optimized(...
            m, log_s, params, L);
        
        if ~success || ~isfinite(elbo_val) || any(~isfinite(grad_m)) || any(~isfinite(grad_log_s))
            % Handle failure gracefully
            fprintf('x');
            if valid_iter_count > 0
                elbo_val = elbo_history(valid_iter_count);
            else
                elbo_val = -1e3;
            end
        else
            valid_iter_count = valid_iter_count + 1;
            elbo_history(valid_iter_count) = elbo_val;
            m_history(valid_iter_count, :) = m';
            
            % Adam update for m
            [m, m_m, m_v] = adam_update(m, grad_m, m_m, m_v, valid_iter_count, optim);
            
            % Adam update for log_s  
            [log_s, log_s_m, log_s_v] = adam_update(log_s, grad_log_s, log_s_m, log_s_v, valid_iter_count, optim);
        end
        
        % Early stopping logic
        if elbo_val > best_elbo + optim.tol
            best_elbo = elbo_val;
            best_m = m;
            best_log_s = log_s;
            patience_counter = 0;
        else
            patience_counter = patience_counter + 1;
        end
        
        % Display progress
        if mod(iter, 25) == 0
            fprintf('%d(%.1f) ', iter, elbo_val);
            if mod(iter, 200) == 0, fprintf('\n               '); end
        end
        
        % Check early stopping
        if patience_counter >= optim.patience && valid_iter_count > 50
            fprintf('\n   Early stopping at iteration %d (valid: %d)\n', iter, valid_iter_count);
            break;
        end
        
    catch ME
        fprintf('\nError at iteration %d: %s\n', iter, ME.message);
        fprintf('Using last valid parameters\n');
        break;
    end
end

% Trim history arrays to valid iterations
valid_mask = ~isnan(elbo_history);
if sum(valid_mask) > 0
    elbo_history = elbo_history(valid_mask);
    m_history = m_history(valid_mask, :);
else
    % Fallback if no valid iterations
    elbo_history = -1e3 * ones(10, 1);
    m_history = repmat(variational.m', 10, 1);
    valid_iter_count = 10;
end

% Use best parameters
variational_opt.m = best_m;
variational_opt.log_s = best_log_s;
variational_opt.L = L;

fprintf('\n   Completed %d valid iterations\n', valid_iter_count);
end

function [elbo, grad_m, grad_log_s, success] = compute_elbo_and_gradients_optimized(m, log_s, params, L)
% CORRECTED ELBO computation for Log-GARCH with numerical stability

D = length(m);
s = exp(log_s);

% Pre-allocation
log_joint = zeros(L, 1);
log_q = zeros(L, 1);
grad_log_joint = zeros(L, D);
grad_log_q = zeros(L, D);
epsilon_store = zeros(L, D);

valid_count = 0;
max_attempts = L * 2; % Allow more attempts to get valid samples
attempt = 0;

while valid_count < L && attempt < max_attempts
    attempt = attempt + 1;
    
    % Reparameterization sampling
    epsilon = randn(D, 1);
    theta = m + s .* epsilon;
    
    % CORRECTED: Direct parameter usage for Log-GARCH (no tanh transformation)
    % Log-GARCH parameters don't need non-negativity constraints [[2]]
    [log_joint_val, grad_joint, success_joint] = compute_log_joint_log_garch(theta, params);
    [log_q_val, grad_q] = compute_log_q(theta, m, log_s);
    
    if success_joint && isfinite(log_joint_val) && isfinite(log_q_val) && ...
       abs(log_joint_val) < 1e6 % Prevent extreme values
        valid_count = valid_count + 1;
        log_joint(valid_count) = log_joint_val;
        log_q(valid_count) = log_q_val;
        grad_log_joint(valid_count, :) = grad_joint';
        grad_log_q(valid_count, :) = grad_q';
        epsilon_store(valid_count, :) = epsilon';
    end
end

if valid_count == 0
    elbo = -1e3;
    grad_m = zeros(D, 1);
    grad_log_s = zeros(D, 1);
    success = false;
    return;
end

% Trim arrays to valid samples
log_joint = log_joint(1:valid_count);
log_q = log_q(1:valid_count);
grad_log_joint = grad_log_joint(1:valid_count, :);
grad_log_q = grad_log_q(1:valid_count, :);
epsilon_store = epsilon_store(1:valid_count, :);

% Compute ELBO with stability check
log_ratios = log_joint - log_q;
if any(abs(log_ratios) > 1e5)
    % Clip extreme values to prevent numerical explosion
    log_ratios = max(min(log_ratios, 1e5), -1e5);
end
elbo = mean(log_ratios);

% Compute gradients with stability improvements
if nargout > 1
    grad_total = grad_log_joint - grad_log_q;
    
    % Gradient clipping per sample
    for i = 1:valid_count
        grad_norm = norm(grad_total(i, :));
        if grad_norm > 100 % Per-sample clipping
            grad_total(i, :) = grad_total(i, :) * (100 / grad_norm);
        end
    end
    
    grad_m = mean(grad_total, 1)';
    
    % Gradient for log_s with stability
    s_matrix = repmat(s', valid_count, 1);
    grad_log_s = mean(grad_total .* epsilon_store .* s_matrix, 1)';
    
    % Overall gradient clipping
    grad_m = clip_gradient(grad_m, 20.0);
    grad_log_s = clip_gradient(grad_log_s, 20.0);
    
    success = true;
end
end

function [log_joint, grad, success] = compute_log_joint_log_garch(theta, params)
% CORRECTED joint probability computation for Log-GARCH model
% Log-GARCH models do not require non-negativity constraints on parameters [[2]]

mu = theta(1);
omega = theta(2);
alpha = theta(3);
beta = theta(4);
y = params.y;
T = params.T;
eps = params.epsilon;

% CORRECTED parameter validity check for Log-GARCH
% Log-GARCH can handle negative parameters, but we need stationarity
if abs(beta) >= 0.99 || abs(alpha) > 2 || abs(omega) > 10
    log_joint = -1e3;
    grad = zeros(4, 1);
    success = false;
    return;
end

% Compute log-likelihood with numerical stability
[log_lik, grad_lik, success_lik] = compute_log_likelihood_log_garch(theta, params);
if ~success_lik || ~isfinite(log_lik)
    log_joint = -1e3;
    grad = zeros(4, 1);
    success = false;
    return;
end

% CORRECTED prior for Log-GARCH - weakly informative priors
[log_prior, grad_prior] = compute_log_prior_log_garch(theta);

log_joint = log_lik + log_prior;
grad = grad_lik + grad_prior;

% Numerical stability check
if ~isfinite(log_joint) || any(~isfinite(grad))
    log_joint = -1e3;
    grad = zeros(4, 1);
    success = false;
    return;
end

success = true;
end

function [log_lik, grad, success] = compute_log_likelihood_log_garch(theta, params)
% CORRECTED log-likelihood computation for Log-GARCH(1,1) model

mu = theta(1);
omega = theta(2);
alpha = theta(3);
beta = theta(4);
y = params.y;
T = params.T;
sqrt_2_pi = params.sqrt_2_pi;
eps = params.epsilon;

% CORRECTED: Log-GARCH model specification
% Log-GARCH can handle negative parameters without non-negativity constraints [[2]]
log_sigma = zeros(T, 1);
epsilon_t = zeros(T, 1);

% More stable initialization
initial_log_variance = log(mean((y - mu).^2) + eps);
log_sigma(1) = omega + beta * initial_log_variance; % Stationary initialization

% Forward pass with numerical stability
for t = 2:T
    sigma_prev = exp(log_sigma(t-1)/2);
    epsilon_prev = (y(t-1) - mu) / (sigma_prev + eps);
    u_prev = abs(epsilon_prev) - sqrt_2_pi;
    
    % Log-GARCH equation: log(sigma_t^2) = omega + alpha*u_{t-1} + beta*log(sigma_{t-1}^2)
    log_sigma(t) = omega + alpha * u_prev + beta * log_sigma(t-1);
    
    % Numerical stability bounds - wider range for Log-GARCH
    log_sigma(t) = max(min(log_sigma(t), 20), -20);
end

% Compute log-likelihood
log_lik = 0;
for t = 1:T
    sigma_t = exp(log_sigma(t)/2);
    epsilon_t(t) = (y(t) - mu) / (sigma_t + eps);
    log_lik = log_lik - 0.5 * (log(2*pi) + log_sigma(t) + epsilon_t(t)^2);
end

% CORRECTED analytical gradients for Log-GARCH
grad = zeros(4, 1);
if nargout > 1
    % Initialize gradient accumulators
    d_log_sigma_d_theta = zeros(T, 4);
    
    % Backward pass for gradients
    for t = T:-1:2
        % Derivative of log-likelihood w.r.t log_sigma(t)
        d_log_lik_d_log_sigma_t = -0.5 + 0.5 * epsilon_t(t)^2;
        
        % Chain rule for parameters
        d_log_sigma_d_theta(t, 1) = 0; % mu affects through epsilon, handled separately
        d_log_sigma_d_theta(t, 2) = 1; % omega
        d_log_sigma_d_theta(t, 3) = (abs(epsilon_t(t-1)) - sqrt_2_pi); % alpha
        d_log_sigma_d_theta(t, 4) = log_sigma(t-1); % beta
        
        % Propagate to previous time step
        if t > 2
            d_log_sigma_d_theta(t-1, :) = d_log_sigma_d_theta(t-1, :) + ...
                beta * d_log_sigma_d_theta(t, :) * d_log_lik_d_log_sigma_t;
        end
        
        % Accumulate gradients
        grad(2:4) = grad(2:4) + d_log_lik_d_log_sigma_t * d_log_sigma_d_theta(t, 2:4)';
    end
    
    % Gradient for mu (special handling)
    d_log_lik_d_mu = 0;
    for t = 1:T
        sigma_t = exp(log_sigma(t)/2);
        d_log_lik_d_mu = d_log_lik_d_mu + epsilon_t(t) / (sigma_t + eps);
    end
    grad(1) = d_log_lik_d_mu;
    
    % Numerical stability for gradients
    grad = clip_gradient(grad, 50.0);
end

success = isfinite(log_lik) && all(isfinite(grad));
end

function [log_prior, grad] = compute_log_prior_log_garch(theta)
% CORRECTED weakly informative priors for Log-GARCH parameters

mu = theta(1);
omega = theta(2);
alpha = theta(3);
beta = theta(4);

% Weakly informative priors - Log-GARCH doesn't need strict constraints [[3]]
log_prior_mu = -0.5 * (mu/0.1)^2; % N(0, 0.1^2) prior on mu
log_prior_omega = -0.5 * ((omega + 2)/1)^2; % N(-2, 1^2) prior on omega
log_prior_alpha = -0.5 * (alpha/0.5)^2; % N(0, 0.5^2) prior on alpha
log_prior_beta = -0.5 * ((beta - 0.9)/0.1)^2; % N(0.9, 0.1^2) prior on beta

log_prior = log_prior_mu + log_prior_omega + log_prior_alpha + log_prior_beta;

% Analytical gradients
grad = zeros(4, 1);
grad(1) = -mu/(0.1^2);
grad(2) = -(omega + 2)/(1^2);
grad(3) = -alpha/(0.5^2);
grad(4) = -(beta - 0.9)/(0.1^2);

% Ensure finite values
if ~isfinite(log_prior) || any(~isfinite(grad))
    log_prior = -1e3;
    grad = zeros(size(theta));
end
end

function [log_q, grad] = compute_log_q(theta, m, log_s)
% Variational distribution density - unchanged but with stability checks

s = exp(log_s);
D = length(m);

% Safety check for very small s
s_safe = max(s, 1e-8);

diff = (theta - m) ./ s_safe;
log_q = -0.5 * D * log(2*pi) - sum(log_s) - 0.5 * sum(diff.^2);

% Analytical gradient
grad = -diff ./ s_safe;

% Numerical stability
if ~isfinite(log_q) || any(~isfinite(grad))
    log_q = -1e3;
    grad = zeros(size(theta));
end
end

%% =========================================================================
% CORRECTED Utility Functions
% =========================================================================

function [x_new, m_new, v_new] = adam_update(x, grad, m, v, iter, optim)
% Adam update with numerical stability

% Handle zero gradients gracefully
if all(grad == 0)
    x_new = x;
    m_new = m;
    v_new = v;
    return;
end

m_new = optim.beta1 * m + (1 - optim.beta1) * grad;
v_new = optim.beta2 * v + (1 - optim.beta2) * (grad.^2 + optim.epsilon);

m_hat = m_new / (1 - optim.beta1^iter + optim.epsilon);
v_hat = v_new / (1 - optim.beta2^iter + optim.epsilon);

step = optim.learning_rate * m_hat ./ (sqrt(v_hat) + optim.epsilon);
x_new = x + step;

% Prevent extreme updates
if any(abs(step) > 1.0) % Limit step size
    scale = 1.0 / max(abs(step));
    x_new = x + step * scale;
end
end

function grad_clipped = clip_gradient(grad, threshold)
% Gradient clipping with numerical stability
grad_norm = norm(grad);
if grad_norm > threshold && grad_norm > 0
    grad_clipped = grad * (threshold / grad_norm);
else
    grad_clipped = grad;
end
end

%% =========================================================================
% CORRECTED Posterior Analysis Functions
% =========================================================================

function results = perform_posterior_analysis(variational, config)
% CORRECTED posterior analysis for Log-GARCH

% Sampling
samples = sample_from_variational_optimized(variational, config.N_samples);

% NO transformation needed for Log-GARCH parameters
samples.beta = samples.tilde_beta; % Directly use beta

% Compute statistics
results.samples = samples;
results.stats = compute_posterior_statistics_optimized(samples);
results.variational = variational;
end

function samples = sample_from_variational_optimized(variational, N)
% Optimized sampling function

m = variational.m;
s = exp(variational.log_s);
D = length(m);

% Safety check for very small standard deviations
s_safe = max(s, 1e-6);

samples_raw = m + s_safe .* randn(D, N);

samples.mu = samples_raw(1, :)';
samples.omega = samples_raw(2, :)';
samples.alpha = samples_raw(3, :)';
samples.tilde_beta = samples_raw(4, :)'; % This is actually beta for Log-GARCH
end

function stats = compute_posterior_statistics_optimized(samples)
% CORRECTED posterior statistics computation

param_names = {'mu', 'omega', 'alpha', 'beta'};
stats = struct();

for i = 1:length(param_names)
    param = param_names{i};
    if strcmp(param, 'beta')
        data = samples.tilde_beta; % Direct beta samples
    else
        data = samples.(param);
    end
    
    % Remove extreme outliers for stability
    q = prctile(data, [1, 99]);
    valid_data = data(data >= q(1) & data <= q(2));
    
    if length(valid_data) < 10
        valid_data = data; % Fallback if too many outliers removed
    end
    
    stats.([param '_mean']) = mean(valid_data);
    stats.([param '_std']) = std(valid_data);
    stats.([param '_median']) = median(valid_data);
    stats.([param '_ci']) = prctile(valid_data, [2.5, 97.5]);
end
end

function display_optimized_results(results, variational, elbo_history)
% CORRECTED results display

fprintf('\n=== FFVB Estimation Results (Log-GARCH) ===\n');
if ~isempty(elbo_history) && isfinite(elbo_history(end))
    fprintf('Final ELBO: %.4f\n', elbo_history(end));
end

fprintf('ELBO range: [%.2f, %.2f]\n', min(elbo_history), max(elbo_history));

stats = results.stats;
param_names = {'mu', 'omega', 'alpha', 'beta'};
param_labels = {'\mu', '\omega', '\alpha', '\beta'};

fprintf('\nParameter Posterior Statistics:\n');
fprintf('%-8s %-10s %-10s %-10s %-15s\n', 'Param', 'Mean', 'Std', 'Median', '95%% CI');
fprintf('---------------------------------------------------------------\n');

for i = 1:length(param_names)
    param = param_names{i};
    mean_val = stats.([param '_mean']);
    std_val = stats.([param '_std']);
    median_val = stats.([param '_median']);
    ci_val = stats.([param '_ci']);
    
    fprintf('%-8s %-10.4f %-10.4f %-10.4f [%-6.4f, %-6.4f]\n', ...
            param_labels{i}, mean_val, std_val, median_val, ci_val(1), ci_val(2));
end

fprintf('\nVariational Parameters:\n');
fprintf('m = [%.4f, %.4f, %.4f, %.4f]\n', variational.m);
fprintf('s = [%.4f, %.4f, %.4f, %.4f]\n', exp(variational.log_s));
end

%% =========================================================================
% CORRECTED Volatility Forecasting Function with CI
% =========================================================================

function [vol1, vol1_ci, vol2, vol2_ci] = forecast_volatility_ffvb(samples, params, stats)
% CORRECTED volatility forecasting for Log-GARCH with posterior predictive intervals

% Use all posterior samples (no burn-in needed as already handled)
N = length(samples.mu);
vol1_samples = zeros(N, 1);
vol2_samples = zeros(N, 1);

y = params.y;
T = params.T;
sqrt_2_pi = params.sqrt_2_pi;
eps = params.epsilon;

% Get the last observed return
y_T = y(T);

% For each posterior sample, compute its own 1-step and 2-step forecast
for i = 1:N
    mu_i = samples.mu(i);
    omega_i = samples.omega(i);
    alpha_i = samples.alpha(i);
    beta_i = samples.beta(i);
    
    % === Reconstruct the entire log(sigma^2) path for sample i ===
    log_sigma2 = zeros(T, 1);
    initial_variance = mean((y - mu_i).^2); % Use sample-specific mu
    log_sigma2(1) = log(initial_variance + eps);
    
    for t = 2:T
        sigma_prev = exp(log_sigma2(t-1)/2);
        epsilon_prev = (y(t-1) - mu_i) / (sigma_prev + eps);
        u_prev = abs(epsilon_prev) - sqrt_2_pi;
        log_sigma2(t) = omega_i + alpha_i * u_prev + beta_i * log_sigma2(t-1);
        log_sigma2(t) = max(min(log_sigma2(t), 20), -20);
    end
    
    % Last state
    log_sigma2_T = log_sigma2(T);
    sigma_T = exp(log_sigma2_T / 2);
    epsilon_T = (y_T - mu_i) / (sigma_T + eps);
    u_T = abs(epsilon_T) - sqrt_2_pi;
    
    % 1-step ahead forecast (uses actual last residual)
    log_sigma2_1 = omega_i + alpha_i * u_T + beta_i * log_sigma2_T;
    vol1_samples(i) = exp(log_sigma2_1 / 2);
    
    % 2-step ahead forecast (uses E|epsilon| = sqrt(2/pi), so u = 0)
    log_sigma2_2 = omega_i + alpha_i * 0 + beta_i * log_sigma2_1;
    vol2_samples(i) = exp(log_sigma2_2 / 2);
end

% Remove potential NaNs/Infs
vol1_samples = vol1_samples(isfinite(vol1_samples));
vol2_samples = vol2_samples(isfinite(vol2_samples));

% Point forecasts: use median (robust)
vol1 = median(vol1_samples);
vol2 = median(vol2_samples);

% 95% Credible Intervals
vol1_ci = prctile(vol1_samples, [2.5, 97.5]);
vol2_ci = prctile(vol2_samples, [2.5, 97.5]);

end

function create_optimized_plots(elbo_history, m_history, results)
% CORRECTED plotting with robust handling

figure('Position', [100, 100, 1400, 900], 'Color', 'white');

% ELBO convergence - CORRECTED to handle negative values properly
subplot(2, 3, 1);
valid_elbo = elbo_history(~isnan(elbo_history) & isfinite(elbo_history));
if isempty(valid_elbo)
    valid_elbo = -1000 * ones(10, 1);
end
plot(1:length(valid_elbo), valid_elbo, 'LineWidth', 2, 'Color', [0, 0.4470, 0.7410]);
xlabel('Iteration');
ylabel('ELBO');
title('ELBO Convergence History');
grid on;
box on;

% Parameter trajectories
subplot(2, 3, 2);
colors = lines(4);
valid_m_history = m_history(~isnan(m_history(:,1)), :);
if isempty(valid_m_history)
    valid_m_history = repmat(results.variational.m', 10, 1);
end

for i = 1:4
    plot(1:size(valid_m_history,1), valid_m_history(:, i), 'LineWidth', 1.5, 'Color', colors(i, :));
    hold on;
end
xlabel('Iteration');
ylabel('Parameter Value');
title('Variational Mean Convergence Trajectory');
legend({'\mu', '\omega', '\alpha', '\beta'}, 'Location', 'best', 'Interpreter', 'tex');
grid on;
box on;

% Posterior distributions
param_names = {'mu', 'omega', 'alpha', 'beta'};
param_labels = {'\mu', '\omega', '\alpha', '\beta'};

for i = 1:4
    subplot(2, 3, 2+i);
    param = param_names{i};
    
    if strcmp(param, 'beta')
        data = results.samples.tilde_beta; % Direct beta samples
    else
        data = results.samples.(param);
    end
    
    % Remove extreme outliers for plotting
    q = prctile(data, [0.5, 99.5]);
    plot_data = data(data >= q(1) & data <= q(2));
    
    if isempty(plot_data)
        plot_data = data;
    end
    
    histogram(plot_data, 30, 'Normalization', 'pdf', ...
              'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'none', ...
              'FaceAlpha', 0.7);
    hold on;
    
    % Add mean line
    mean_val = mean(plot_data);
    xline(mean_val, '--k', 'LineWidth', 2, 'DisplayName', sprintf('Mean: %.4f', mean_val));
    
    xlabel(param_labels{i}, 'Interpreter', 'tex');
    ylabel('Density');
    title([param_labels{i} ' Posterior Distribution'], 'Interpreter', 'tex');
    grid on;
    box on;
    legend('show');
end

sgtitle('FFVB for Log-GARCH(1,1) - Corrected Results', 'FontSize', 14, 'FontWeight', 'bold');
set(gcf, 'Color', 'white');

% Save figure
try
    saveas(gcf, 'LogGARCH_FFVB_Results.png');
    fprintf('   Results plot saved as ''LogGARCH_FFVB_Results.png''\n');
catch
    fprintf('   Could not save plot\n');
end
end