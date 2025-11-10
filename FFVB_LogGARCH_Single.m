function FFVB_LogGARCH_Single(filename, seed)

fprintf('=== FFVB for Log-GARCH(1,1) Model - Optimized Version ===\n');

% Set random seed
rng(seed, 'twister');

%% 1. Data Preparation and Preprocessing
fprintf('1. Loading and preparing data...\n');

% Load data
data = readtable(filename);
y = data.ret_asx;

% Data preprocessing
y = rmmissing(y);
T = length(y);

fprintf('   Data length: %d observations\n', T);
fprintf('   Return statistics: mean=%.4f, std=%.4f\n', mean(y), std(y));

%% 2. Model Configuration - Fix parameter passing issues
fprintf('2. Initializing model configuration and variational parameters...\n');

% Fix: Ensure y is correctly passed to params
config = get_model_config(T, y);  % Pass y parameter

% Initialize variational parameters
variational = initialize_variational_parameters(y);

%% 3. Run FFVB Optimization
fprintf('3. Starting FFVB optimization...\n');
[variational_opt, elbo_history, m_history] = run_ffvb_optimization_optimized(...
    variational, config.params, config.optim);

if isempty(elbo_history) || all(elbo_history == 0)
    fprintf('   FFVB optimization failed! Using initial values\n');
    variational_opt = variational;
else
    fprintf('   FFVB optimization successful! Final ELBO: %.4f\n', elbo_history(end));
end

%% 4. Posterior Analysis
fprintf('4. Posterior sampling and analysis...\n');
results = perform_posterior_analysis(variational_opt, config.post_analysis);

% Display results
display_optimized_results(results, variational_opt, elbo_history);

%% 5. Volatility Forecasting (NEW)
fprintf('5. Forecasting volatility...\n');
[vol1, vol2] = forecast_volatility_ffvb(results.stats, config.params);
fprintf('   1-step forecast (15-Oct-2025): %.4f\n', vol1);
fprintf('   2-step forecast (16-Oct-2025): %.4f\n', vol2);

% Create plots
if ~isempty(elbo_history) && length(elbo_history) > 10
    create_optimized_plots(elbo_history, m_history, results);
else
    fprintf('   Insufficient optimization history data, skipping plots\n');
end

fprintf('\n=== FFVB Analysis Complete! ===\n');

end

%% =========================================================================
% Configuration Functions - Fix: Add y parameter
% =========================================================================

function config = get_model_config(T, y)  % Add y parameter
    % Model parameter configuration
    config.params.T = T;
    config.params.y = y;  % Ensure y is included in params
    config.params.sqrt_2_pi = sqrt(2/pi);
    config.params.epsilon = 1e-10; % Numerical stability parameter
    
    % Optimizer configuration
    config.optim.learning_rate = 0.005;
    config.optim.beta1 = 0.9;
    config.optim.beta2 = 0.999;
    config.optim.epsilon = 1e-8;
    config.optim.max_iter = 300;
    config.optim.patience = 20;
    config.optim.tol = 1e-5;
    config.optim.grad_clip = 5.0; % Gradient clipping
    
    % Posterior analysis configuration
    config.post_analysis.N_samples = 5000;
    config.post_analysis.burnin = 1000;
end

function variational = initialize_variational_parameters(y)
    % Smart initialization of variational parameters
    y_std = std(y);
    y_mean = mean(y);
    
    variational = struct();
    % Initial variational mean: [mu, omega, alpha, tilde_beta]
    variational.m = [y_mean; log(y_std); 0.05; atanh(0.85)];  
    
    % Adaptive initialization of standard deviations
    init_std = [0.01, 0.1, 0.05, 0.1]';
    variational.log_s = log(init_std);
    
    % Sample size settings
    variational.L = 8;
    
    fprintf('   Initial m: [%.4f, %.4f, %.4f, %.4f]\n', variational.m);
    fprintf('   Initial s: [%.4f, %.4f, %.4f, %.4f]\n', exp(variational.log_s));
end

%% =========================================================================
% Optimized Main Functionality
% =========================================================================

function [variational_opt, elbo_history, m_history] = run_ffvb_optimization_optimized(variational, params, optim)
% Optimized FFVB optimization process

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

% Pre-allocation
elbo_history = zeros(optim.max_iter, 1);
m_history = zeros(optim.max_iter, D);

best_elbo = -inf;
best_m = m;
best_log_s = log_s;
patience_counter = 0;

% Main optimization loop
for iter = 1:optim.max_iter
    [elbo_val, grad_m, grad_log_s, success] = compute_elbo_and_gradients_optimized(...
        m, log_s, params, L);
    
    if ~success
        fprintf('x');
        continue;
    end
    
    % Store history
    elbo_history(iter) = elbo_val;
    m_history(iter, :) = m';
    
    % Adam update for m
    [m, m_m, m_v] = adam_update(m, grad_m, m_m, m_v, iter, optim);
    
    % Adam update for log_s  
    [log_s, log_s_m, log_s_v] = adam_update(log_s, grad_log_s, log_s_m, log_s_v, iter, optim);
    
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
    if patience_counter >= optim.patience
        fprintf('\n   Early stopping at iteration %d\n', iter);
        break;
    end
end

% Trim history arrays
valid_iter = elbo_history ~= 0;
elbo_history = elbo_history(valid_iter);
m_history = m_history(valid_iter, :);

% Use best parameters
variational_opt.m = best_m;
variational_opt.log_s = best_log_s;
variational_opt.L = L;

fprintf('\n   Completed %d valid iterations\n', sum(valid_iter));
end

function [elbo, grad_m, grad_log_s, success] = compute_elbo_and_gradients_optimized(m, log_s, params, L)
% Optimized ELBO and gradient computation

D = length(m);
s = exp(log_s);

% Pre-allocation
log_joint = zeros(L, 1);
log_q = zeros(L, 1);
grad_log_joint = zeros(L, D);
grad_log_q = zeros(L, D);
epsilon_store = zeros(L, D);

valid_count = 0;
for l = 1:L
    % Reparameterization sampling
    epsilon = randn(D, 1);
    theta = m + s .* epsilon;
    
    % Compute joint probability and variational probability
    [log_joint_val, grad_joint, success_joint] = compute_log_joint(theta, params);
    [log_q_val, grad_q] = compute_log_q(theta, m, log_s);
    
    if success_joint && isfinite(log_joint_val) && isfinite(log_q_val)
        valid_count = valid_count + 1;
        log_joint(valid_count) = log_joint_val;
        log_q(valid_count) = log_q_val;
        grad_log_joint(valid_count, :) = grad_joint';
        grad_log_q(valid_count, :) = grad_q';
        epsilon_store(valid_count, :) = epsilon';
    end
end

if valid_count == 0
    elbo = -1e6;
    grad_m = zeros(D, 1);
    grad_log_s = zeros(D, 1);
    success = false;
    return;
end

% Trim arrays
log_joint = log_joint(1:valid_count);
log_q = log_q(1:valid_count);
grad_log_joint = grad_log_joint(1:valid_count, :);
grad_log_q = grad_log_q(1:valid_count, :);
epsilon_store = epsilon_store(1:valid_count, :);

% Compute ELBO
elbo = mean(log_joint - log_q);

% Compute gradients
if nargout > 1
    grad_total = grad_log_joint - grad_log_q;
    grad_m = mean(grad_total, 1)';
    
    % Gradient for log_s
    s_matrix = repmat(s', valid_count, 1);
    grad_log_s = mean(grad_total .* epsilon_store .* s_matrix, 1)';
    
    % Gradient clipping
    grad_m = clip_gradient(grad_m, 5.0);
    grad_log_s = clip_gradient(grad_log_s, 5.0);
    
    success = true;
end
end

function [log_joint, grad, success] = compute_log_joint(theta, params)
% Compute joint probability (likelihood + prior)

theta_orig = transform_parameters(theta);

% Compute log-likelihood
[log_lik, grad_lik, success_lik] = compute_log_likelihood_optimized(theta_orig, params);
if ~success_lik
    log_joint = -1e10;
    grad = zeros(size(theta));
    success = false;
    return;
end

% Compute prior
[log_prior, grad_prior] = compute_log_prior_optimized(theta);

% Consider Jacobian of parameter transformation
jacobian = compute_jacobian(theta);

log_joint = log_lik + log_prior + jacobian;
grad = transform_gradient(grad_lik + grad_prior, theta);

success = true;
end

function [log_lik, grad, success] = compute_log_likelihood_optimized(theta, params)
% Optimized log-likelihood computation

mu = theta(1);
omega = theta(2);
alpha = theta(3);
beta = theta(4);
y = params.y;  % Should now be accessible normally
T = params.T;
eps = params.epsilon;

% Parameter boundary check
if ~are_parameters_valid(omega, alpha, beta)
    log_lik = -1e10;
    grad = zeros(4, 1);
    success = false;
    return;
end

% Initialization
log_sigma = zeros(T, 1);
log_sigma(1) = omega / (1 - beta); % Stationary initialization

% Recursive computation
for t = 2:T
    epsilon_prev = (y(t-1) - mu) / exp(log_sigma(t-1));
    u_prev = abs(epsilon_prev) - params.sqrt_2_pi;
    
    log_sigma(t) = omega + alpha * u_prev + beta * log_sigma(t-1);
    
    % Numerical stability
    log_sigma(t) = max(min(log_sigma(t), 10), -10);
end

sigma2 = exp(2 * log_sigma);

% Compute log-likelihood
log_lik = -0.5 * T * log(2*pi) - sum(log_sigma) - 0.5 * sum(((y - mu) ./ exp(log_sigma)).^2);

% Numerical gradient (can be replaced with analytical gradient)
grad = compute_numerical_gradient(@(th) compute_log_likelihood_standalone(th, params), theta);

success = isfinite(log_lik);
end

function [log_prior, grad] = compute_log_prior_optimized(theta)
% Corrected prior computation - using correct prior ranges

mu = theta(1);
omega = theta(2);
alpha = theta(3);
tilde_beta = theta(4);
beta = tanh(tilde_beta);  % Convert back to beta for prior check

% Flat prior
log_prior_mu = 0;

% Uniform prior - corrected to required ranges
if omega >= -2 && omega <= 2
    log_prior_omega = -log(4);  % Density of Uniform(-2,2) is 1/4
else
    log_prior_omega = -1e10;
end

if alpha >= -2 && alpha <= 2
    log_prior_alpha = -log(4);  % Density of Uniform(-2,2) is 1/4
else
    log_prior_alpha = -1e10;
end

% Beta prior + Jacobian - corrected to required ranges
if beta >= -1 && beta <= 1
    % Density of Uniform(-1,1) is 1/2, plus Jacobian term
    log_prior_beta = -log(2) + log(1 - beta^2);
else
    log_prior_beta = -1e10;
end

log_prior = log_prior_mu + log_prior_omega + log_prior_alpha + log_prior_beta;

% Numerical gradient
grad = compute_numerical_gradient(@compute_log_prior_standalone, theta);
end

function [log_q, grad] = compute_log_q(theta, m, log_s)
% Compute variational distribution density
s = exp(log_s);
D = length(m);

diff = (theta - m) ./ s;
log_q = -0.5 * D * log(2*pi) - sum(log_s) - 0.5 * sum(diff.^2);

% Analytical gradient
grad = -diff ./ s;
end

%% =========================================================================
% Utility Functions
% =========================================================================

function [x_new, m_new, v_new] = adam_update(x, grad, m, v, iter, optim)
% Adam update
m_new = optim.beta1 * m + (1 - optim.beta1) * grad;
v_new = optim.beta2 * v + (1 - optim.beta2) * (grad.^2);

m_hat = m_new / (1 - optim.beta1^iter);
v_hat = v_new / (1 - optim.beta2^iter);

x_new = x + optim.learning_rate * m_hat ./ (sqrt(v_hat) + optim.epsilon);
end

function grad_clipped = clip_gradient(grad, threshold)
% Gradient clipping
grad_norm = norm(grad);
if grad_norm > threshold
    grad_clipped = grad * (threshold / grad_norm);
else
    grad_clipped = grad;
end
end

function valid = are_parameters_valid(omega, alpha, beta)
% Corrected parameter validity check - using required prior ranges
valid = (omega >= -2 && omega <= 2) && ...
        (alpha >= -2 && alpha <= 2) && ...
        (beta >= -1 && beta <= 1);
end

function theta_orig = transform_parameters(theta)
% Corrected parameter transformation - using proper tanh transformation
theta_orig = theta;
theta_orig(4) = tanh(theta(4));  % β = tanh(tilde_beta)
end

function jacobian = compute_jacobian(theta)
% Compute transformation Jacobian - corrected for tanh transformation Jacobian
tilde_beta = theta(4);
beta = tanh(tilde_beta);
jacobian = log(1 - beta^2);  % Jacobian of tanh transformation
end

function grad_transformed = transform_gradient(grad, theta)
% Transform gradient - corrected for tanh transformation gradient
grad_transformed = grad;
tilde_beta = theta(4);
grad_transformed(4) = grad(4) * (1 - tanh(tilde_beta)^2);  % Derivative of tanh
end

function grad = compute_numerical_gradient(func, x)
% Numerical gradient computation
h = 1e-7;
grad = zeros(size(x));
f0 = func(x);

for i = 1:length(x)
    x_temp = x;
    x_temp(i) = x_temp(i) + h;
    f1 = func(x_temp);
    grad(i) = (f1 - f0) / h;
end
end

%% =========================================================================
% Standalone Functions (Avoid Recursive Calls)
% =========================================================================

function log_lik = compute_log_likelihood_standalone(theta, params)
% Standalone log-likelihood computation, avoiding recursion

mu = theta(1);
omega = theta(2);
alpha = theta(3);
beta = theta(4);
y = params.y;
T = params.T;

% Parameter boundary check - using corrected ranges
if ~(omega >= -2 && omega <= 2 && alpha >= -2 && alpha <= 2 && beta >= -1 && beta <= 1)
    log_lik = -1e10;
    return;
end

try
    % Initialization
    log_sigma = zeros(T, 1);
    log_sigma(1) = omega / (1 - beta); % Stationary initialization

    % Recursive computation
    for t = 2:T
        epsilon_prev = (y(t-1) - mu) / exp(log_sigma(t-1));
        u_prev = abs(epsilon_prev) - params.sqrt_2_pi;
        
        log_sigma(t) = omega + alpha * u_prev + beta * log_sigma(t-1);
        
        % Numerical stability
        log_sigma(t) = max(min(log_sigma(t), 10), -10);
    end

    % Compute log-likelihood
    log_lik = -0.5 * T * log(2*pi) - sum(log_sigma) - 0.5 * sum(((y - mu) ./ exp(log_sigma)).^2);
    
    if ~isfinite(log_lik)
        log_lik = -1e10;
    end
catch
    log_lik = -1e10;
end
end

function log_prior = compute_log_prior_standalone(theta)
% Standalone prior computation, avoiding recursion - corrected for prior ranges

mu = theta(1);
omega = theta(2);
alpha = theta(3);
tilde_beta = theta(4);
beta = tanh(tilde_beta);  % Convert back to beta

% Flat prior
log_prior_mu = 0;

% Uniform prior - corrected to required ranges
if omega >= -2 && omega <= 2
    log_prior_omega = -log(4);  % Density of Uniform(-2,2) is 1/4
else
    log_prior_omega = -1e10;
end

if alpha >= -2 && alpha <= 2
    log_prior_alpha = -log(4);  % Density of Uniform(-2,2) is 1/4
else
    log_prior_alpha = -1e10;
end

% Beta prior + Jacobian - corrected to required ranges
if beta >= -1 && beta <= 1
    % Density of Uniform(-1,1) is 1/2, plus Jacobian term
    log_prior_beta = -log(2) + log(1 - beta^2);
else
    log_prior_beta = -1e10;
end

log_prior = log_prior_mu + log_prior_omega + log_prior_alpha + log_prior_beta;

if ~isfinite(log_prior)
    log_prior = -1e10;
end
end

%% =========================================================================
% Posterior Analysis Functions
% =========================================================================

function results = perform_posterior_analysis(variational, config)
% Posterior analysis

% Sampling
samples = sample_from_variational_optimized(variational, config.N_samples);

% Transform parameters - corrected to proper tanh transformation
samples.beta = tanh(samples.tilde_beta);

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

samples_raw = m + s .* randn(D, N);

samples.mu = samples_raw(1, :)';
samples.omega = samples_raw(2, :)';
samples.alpha = samples_raw(3, :)';
samples.tilde_beta = samples_raw(4, :)';
end

function stats = compute_posterior_statistics_optimized(samples)
% Optimized posterior statistics computation

param_names = {'mu', 'omega', 'alpha', 'beta'};
stats = struct();

for i = 1:length(param_names)
    param = param_names{i};
    if strcmp(param, 'beta')
        data = samples.beta;
    else
        data = samples.(param);
    end
    
    stats.([param '_mean']) = mean(data);
    stats.([param '_std']) = std(data);
    stats.([param '_median']) = median(data);
    stats.([param '_ci']) = prctile(data, [2.5, 97.5]);
end
end

function display_optimized_results(results, variational, elbo_history)
% Optimized results display

fprintf('\n=== FFVB Estimation Results ===\n');
if ~isempty(elbo_history)
    fprintf('Final ELBO: %.4f\n', elbo_history(end));
end

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
% Volatility Forecasting Function (NEW)
% =========================================================================

function [vol1, vol2] = forecast_volatility_ffvb(stats, params)
% Forecast volatility using FFVB posterior means
% 1-step: uses actual epsilon from last observation
% 2-step: uses E[|epsilon|] = sqrt(2/pi) => u = 0

% Extract posterior mean parameters (in original scale)
mu = stats.mu_mean;
omega = stats.omega_mean;
alpha = stats.alpha_mean;
beta_val = stats.beta_mean;  % already transformed back via tanh

y = params.y;
T = params.T;
sqrt_2_pi = params.sqrt_2_pi;

% Reconstruct log(sigma_t) series up to time T
log_sigma = zeros(T, 1);
% Initialize with stationary value or sample variance
log_sigma(1) = omega / (1 - beta_val + eps); % avoid division by zero

for t = 2:T
    epsilon_prev = (y(t-1) - mu) / exp(log_sigma(t-1));
    u_prev = abs(epsilon_prev) - sqrt_2_pi;
    log_sigma(t) = omega + alpha * u_prev + beta_val * log_sigma(t-1);
    % Numerical stability
    log_sigma(t) = max(min(log_sigma(t), 10), -10);
end

% 1-step ahead forecast (15-Oct-2025)
epsilon_T = (y(T) - mu) / exp(log_sigma(T));
u_T = abs(epsilon_T) - sqrt_2_pi;
log_sigma_1step = omega + alpha * u_T + beta_val * log_sigma(T);
vol1 = exp(log_sigma_1step);

% 2-step ahead forecast (16-Oct-2025)
% E[|epsilon|] = sqrt(2/pi) => u = E[|epsilon|] - sqrt(2/pi) = 0
log_sigma_2step = omega + alpha * 0 + beta_val * log_sigma_1step;
vol2 = exp(log_sigma_2step);

end

function create_optimized_plots(elbo_history, m_history, results)
% Optimized plotting

figure('Position', [100, 100, 1400, 900]);

% ELBO convergence
subplot(2, 3, 1);
plot(elbo_history, 'LineWidth', 2, 'Color', [0, 0.4470, 0.7410]);
xlabel('Iteration');
ylabel('ELBO');
title('ELBO Convergence History');
grid on;

% Parameter trajectories
subplot(2, 3, 2);
colors = lines(4);
for i = 1:4
    plot(m_history(:, i), 'LineWidth', 1.5, 'Color', colors(i, :));
    hold on;
end
xlabel('Iteration');
ylabel('Parameter Value');
title('Variational Mean Convergence Trajectory');
legend({'\mu', '\omega', '\alpha', 'tilde{\beta}'}, 'Location', 'best');
grid on;

% Posterior distributions
param_names = {'mu', 'omega', 'alpha', 'beta'};
param_labels = {'\mu', '\omega', '\alpha', '\beta'};

for i = 1:4
    subplot(2, 3, 2+i);
    param = param_names{i};
    
    if strcmp(param, 'beta')
        data = results.samples.beta;
    else
        data = results.samples.(param);
    end
    
    histogram(data, 40, 'Normalization', 'pdf', ...
              'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'none', ...
              'FaceAlpha', 0.7);
    hold on;
    
    % Add prior information - corrected for prior boundaries
    if i == 2 || i == 3 % omega and alpha
        xline(-2, '--r', 'LineWidth', 1.5, 'Alpha', 0.7);
        xline(2, '--r', 'LineWidth', 1.5, 'Alpha', 0.7);
    elseif i == 4 % beta
        xline(-1, '--r', 'LineWidth', 1.5, 'Alpha', 0.7);
        xline(1, '--r', 'LineWidth', 1.5, 'Alpha', 0.7);
    end
    
    xlabel(param_labels{i});
    ylabel('Density');
    title([param_labels{i} ' Posterior Distribution']);
    grid on;
end

sgtitle('FFVB for Log-GARCH(1,1) - Optimized Results', 'FontSize', 14, 'FontWeight', 'bold');
end