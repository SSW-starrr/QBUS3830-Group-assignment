function results = rwm_loggarch_baseline(filename, burnin, nkeep, seed)
    if nargin < 4, seed = 123; end
    rng(seed);

    if nargin < 1 || isempty(filename)
        filename = 'ASX200_Cleaned_Last10Years.csv';
    end
    fprintf('Loading data from %s...\n', filename);
    data = readtable(filename);

    % === Create synthetic dates for plotting ===
    startDate = datetime(2000,1,4);
    data.Date = startDate + caldays(data.Index - 1);

    % === Subset to 2015–2025 ===
    data = data(year(data.Date) >= 2015, :);

    y = data.ret_asx;  % use subsetted returns
    T = numel(y);
    fprintf('Using %d observations from 2015 to 2025.\n', T);
    
    %% ======== Transformations & Jacobian ========
    toRaw = @(th) [th(1), 2*tanh(th(2)), 2*tanh(th(3)), tanh(th(4))];
    jacTerm = @(raw) log(1 - (raw(2)/2).^2) + log(1 - (raw(3)/2).^2) + log(1 - raw(4).^2);

    %% ======== Log-posterior definition ========
    function lp = logpost(th_tilde)
        raw = toRaw(th_tilde);
        mu = raw(1); omega = raw(2); alpha = raw(3); beta = raw(4);
        sig = zeros(T,1); eps = zeros(T,1);

        sig(1) = std(y); eps(1) = (y(1) - mu)/sig(1);
        for t = 2:T
            logsig = omega + alpha*(abs(eps(t-1)) - sqrt(2/pi)) + beta*log(sig(t-1));
            sig(t) = exp(logsig);
            eps(t) = (y(t) - mu)/sig(t);
            if ~isfinite(sig(t)) || sig(t)<=0
                lp = -Inf; return;
            end
        end
        ll = -sum(log(sig)) - 0.5*sum(eps.^2);
        lp = ll + jacTerm(raw); % uniform priors -> constant
    end

    %% ======== Initialisation ========
    th0_raw = [mean(y), 0, 0.1, 0.9];
    th0 = [th0_raw(1), atanh(th0_raw(2)/2), atanh(th0_raw(3)/2), atanh(th0_raw(4))];
    Sigma = diag([0.05, 0.1, 0.05, 0.2].^2);

    total = burnin + nkeep;
    draws = zeros(nkeep,4);
    acc = 0;
    th = th0;
    lp_th = logpost(th);

    % Track chain on transformed scale (like MWG)
    chain_tilde = zeros(total,4);
    chain_tilde(1,:) = th';

    fprintf('Running Random Walk Metropolis (block) for %d iterations (%d burn-in, %d kept)...\n', total, burnin, nkeep);

    %% ======== Main RWM loop ========
    for i = 1:total
        th_prop = th + mvnrnd(zeros(1,4), Sigma, 1);
        lp_prop = logpost(th_prop);

        if isfinite(lp_prop) && (log(rand) < (lp_prop - lp_th))
            th = th_prop;
            lp_th = lp_prop;
            acc = acc + 1;
        end

        % Store transformed chain (for diagnostics)
        chain_tilde(i,:) = th;

        if i <= burnin && mod(i,200)==0
            ar = acc/i;
            Sigma = Sigma * exp(ar - 0.25); % tune towards 25%
        end

        if i > burnin
            draws(i-burnin,:) = toRaw(th);
        end
    end

    %% Step: Volatility Forecast for 1-step and 2-step ahead
    fprintf('\nForecasting volatility...\n');
    [vol_1step, vol_2step] = forecast_volatility(draws, y);
    fprintf('1-step forecast (15-Oct-2025): %.4f\n', mean(vol_1step));
    fprintf('2-step forecast (16-Oct-2025): %.4f\n', mean(vol_2step));

    % --- Posterior summary (on raw scale) ---
    theta = {'mu','omega','alpha','beta'}';
    mean_post = mean(draws);
    sd_post   = std(draws);
    ci_post   = quantile(draws,[0.025 0.975]);

    % Round like MWG
    mean_post_r = round(mean_post, 4);
    sd_post_r   = round(sd_post, 4);
    ci_post_r   = round(ci_post, 4);

    fprintf('\nPosterior Estimates (4 decimal places):\n');
    for ii = 1:4
        fprintf('%s: mean=%.4f, std=%.4f, 95%% CI=[%.4f, %.4f]\n', ...
            theta{ii}, mean_post_r(ii), sd_post_r(ii), ci_post_r(1,ii), ci_post_r(2,ii));
    end

    acc_rate = acc / total;
    fprintf('Acceptance rate (block RWM): %.2f%%\n', 100*acc_rate);

    % Trace plots on kept (raw) draws — similar to MWG
    post_burn = draws; % kept draws are already raw
    theta_names = {'\mu','\omega','\alpha','\beta'};
    figure('Position', [100, 100, 800, 600]);
    for j = 1:4
        subplot(2,2,j);
        plot(post_burn(:,j), 'Color', [0.1 0.4 0.8]);
        title(['Trace plot: ', theta{j}]);
        xlabel('Iteration');
        ylabel(theta_names{j});
        grid on; box on;
    end
    sgtitle('Trace Plots (after burn-in)');
    saveas(gcf, 'rwm_trace_plots.png');

    %% ======== Return results in MWG-like structure ========
    results = struct();
    results.post_mean = mean_post;
    results.post_std  = sd_post;
    results.post_CI   = ci_post;
    results.theta_samp = draws;        % kept draws in raw scale
    results.post_burn  = post_burn;    % same as theta_samp here
    results.chain = chain_tilde;       % full transformed chain
    results.sigma_proposal = Sigma;
    results.accept_rate = acc_rate;
    results.vol_1step = vol_1step;
    results.vol_2step = vol_2step;
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