%%%%%%%%%%%%%%%%%%%%%%%%
%  RWM Sampler (Star-Space version) for log-GARCH(1,1)

clc; clear; close all;

%%  Load & subset data 
data = readtable('ASX200_Cleaned_Last10Years.csv');

% Adjust variable names if needed
if any(strcmp(data.Properties.VariableNames,'ret_asx'))
    data.Properties.VariableNames = {'Return','Date'};
end

y = data.Return;
dates = datetime(data.Date);

fprintf('Loaded cleaned dataset: %d observations (%s → %s)\n', ...
    numel(y), string(min(dates)), string(max(dates)));

%%  Hyperparameters 
burnin      = 6500;          % burn-in iterations
nkeep       = 30000;         % kept draws
step_star   = [0.075, 0.0094, 0.05, 0.38];   % proposal std dev in star-space
adapt_every = 200;           % adapt Σ every K iterations during burn-ina
target_acc  = 0.25;          % target acceptance rate

%%  Run RWM (star-space) 
fprintf('\n=== Running RWM (star-space) for log-GARCH(1,1) ===\n');
res = rwm_star_loggarch(y, burnin, nkeep, step_star, adapt_every, target_acc, 12345);

%%  Posterior summary 
theta_names = {'\mu','\omega','\alpha','\beta'}';
Tsum = table(theta_names, res.mean', res.sd', res.ci(1,:)', res.ci(2,:)', ...
    'VariableNames', {'Parameter','PostMean','PostSD','CI_2p5','CI_97p5'});
disp(Tsum);
fprintf('Acceptance (kept phase): %.2f%%\n', 100*res.acc_rate_kept);

%%  Trace plots 
plot_traces(res.draws, {'\mu','\omega','\alpha','\beta'}, 'RWM (Star-space) Trace Plots: log-GARCH(1,1)');

%%  Volatility Forecasts 
fcast = forecast_vol_loggarch(y, res.draws);
fprintf('\n1-day vol: %.4f [%.4f, %.4f]\n', fcast.mean1, fcast.ci1(1), fcast.ci1(2));
fprintf('2-day vol: %.4f [%.4f, %.4f]\n', fcast.mean2, fcast.ci2(1), fcast.ci2(2));

%%  FUNCTIONS 

function out = rwm_star_loggarch(y, burnin, nkeep, step, adapt_every, target_acc, seed)
    if nargin < 7, seed = 123; end
    rng(seed);
    T = numel(y);

    % Transformations
    unpack_star = @(th) deal( ...
        th(1), ...                                 % mu
        2*tanh(th(2)), ...                         % omega
        2*tanh(th(3)), ...                         % alpha
        tanh(th(4)) );                             % beta

    % Log-likelihood in original parameter space
    function ll = log_kernel(y, mu, omega, alpha, beta)
        if abs(beta) >= 1 || abs(alpha) >= 2 || abs(omega) >= 2
            ll = -Inf; return;
        end
        sig = zeros(T,1); eps = zeros(T,1);
        sig(1) = std(y);
        eps(1) = (y(1)-mu)/sig(1);
        for t = 2:T
            logsig = omega + alpha*(abs(eps(t-1))-sqrt(2/pi)) + beta*log(sig(t-1));
            sig(t) = exp(logsig);
            if sig(t) <= 0 || ~isfinite(sig(t))
                ll = -Inf; return;
            end
            eps(t) = (y(t)-mu)/sig(t);
        end
        ll = -sum(log(sig)) - 0.5*sum(eps.^2);
    end

    % Initialise
    th0 = [mean(y), atanh(0/2), atanh(0.1/2), atanh(0.9)];
    Sigma = diag(step.^2);
    total = burnin + nkeep;
    draws = zeros(nkeep,4);
    acc = 0; acc_bi = 0; lp_curr = -Inf;

    [mu,omega,alpha,beta] = unpack_star(th0);
    lp_curr = log_kernel(y, mu,omega,alpha,beta);

    acc_path = []; checks = 0;

    % RWM loop
    for i = 1:total
        prop_star = th0 + mvnrnd(zeros(1,4), Sigma);
        [mu_p,omega_p,alpha_p,beta_p] = unpack_star(prop_star);
        lp_prop = log_kernel(y, mu_p,omega_p,alpha_p,beta_p);

        if log(rand) < (lp_prop - lp_curr)
            th0 = prop_star;
            lp_curr = lp_prop;
            acc = acc + 1;
            if i <= burnin, acc_bi = acc_bi + 1; end
        end

        % Adaptation (same as your Assignment 2)
        if i <= burnin && mod(i,adapt_every)==0
            checks = checks + 1;
            ar = acc_bi / i;
            acc_path(checks,1) = ar;
            scale = exp(0.5*(ar - target_acc));
            Sigma = Sigma * scale;
        end

        if i > burnin
            [mu,omega,alpha,beta] = unpack_star(th0);
            draws(i-burnin,:) = [mu,omega,alpha,beta];
        end
    end

    pm = mean(draws); psd = std(draws); pci = quantile(draws,[0.025 0.975]);
    out.draws = draws;
    out.mean = pm; out.sd = psd; out.ci = pci;
    out.acc_rate_kept = (acc - acc_bi)/(total - burnin);
    out.acc_path = acc_path;
end

function plot_traces(draws, names, figtitle)
    if nargin < 2, names = {'\mu','\omega','\alpha','\beta'}; end
    figure('Color','w','Name','RWM Star-space Trace Plots');
    for j = 1:4
        subplot(2,2,j);
        plot(draws(:,j),'LineWidth',0.7,'Color',[0.1 0.4 0.8]);
        title(['Trace: ', names{j}]);
        xlabel('Iteration'); ylabel(names{j});
        grid on; box on;
    end
    sgtitle(figtitle,'FontWeight','bold','FontSize',13);
end

function f = forecast_vol_loggarch(y, draws)
    T = numel(y);
    mu_m = mean(draws(:,1)); omega_m = mean(draws(:,2));
    alpha_m = mean(draws(:,3)); beta_m  = mean(draws(:,4));
    sig = zeros(T,1); eps = zeros(T,1);
    sig(1) = std(y); eps(1) = (y(1)-mu_m)/sig(1);
    for t = 2:T
        logsig = omega_m + alpha_m*(abs(eps(t-1))-sqrt(2/pi)) + beta_m*log(sig(t-1));
        sig(t) = exp(logsig); eps(t) = (y(t)-mu_m)/sig(t);
    end
    sigma_T = sig(end); eps_T = eps(end);
    M = size(draws,1);
    sigma1 = zeros(M,1); sigma2 = zeros(M,1);
    for m = 1:M
        omega = draws(m,2); alpha = draws(m,3); beta = draws(m,4);
        logsig1 = omega + alpha*(abs(eps_T)-sqrt(2/pi)) + beta*log(sigma_T);
        s1 = exp(logsig1);
        sigma1(m) = s1;
        logsig2 = omega + beta*log(s1);
        sigma2(m) = exp(logsig2);
    end
    f.mean1 = mean(sigma1); f.ci1 = quantile(sigma1,[0.025 0.975]);
    f.mean2 = mean(sigma2); f.ci2 = quantile(sigma2,[0.025 0.975]);
end

