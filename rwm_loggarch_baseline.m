function results = rwm_loggarch(y, burnin, nkeep, seed)
    if nargin < 4, seed = 123; end
    rng(seed);

    T = numel(y);

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
    Sigma = diag([0.05 0.05 0.05 0.05].^2);

    total = burnin + nkeep;
    draws = zeros(nkeep,4);
    acc = 0;
    th = th0;
    lp_th = logpost(th);

    %% ======== Main RWM loop ========
    for i = 1:total
        th_prop = th + mvnrnd(zeros(1,4), Sigma, 1);
        lp_prop = logpost(th_prop);

        if log(rand) < (lp_prop - lp_th)
            th = th_prop;
            lp_th = lp_prop;
            acc = acc + 1;
        end

        if i <= burnin && mod(i,200)==0
            ar = acc/i;
            Sigma = Sigma * exp(ar - 0.25); % tune towards 25%
        end

        if i > burnin
            draws(i-burnin,:) = toRaw(th);
        end
    end

    %% ======== Posterior summaries ========
    mean_post = mean(draws);
    sd_post   = std(draws);
    ci_post   = quantile(draws,[0.025 0.975]);

    results.draws = draws;
    results.mean  = mean_post;
    results.sd    = sd_post;
    results.ci    = ci_post;
    results.acc_rate = acc / total;
end

% === Load data ===
data = readtable('ASX_2000_2025.csv');
data.Properties.VariableNames = {'Index','Return'};

% === Create synthetic dates for plotting ===
startDate = datetime(2000,1,4);
data.Date = startDate + caldays(data.Index - 1);

% === Subset to 2015–2025 ===
data = data(year(data.Date) >= 2015, :);
y = data.Return;

y = data.Return;  % your ASX200 returns vector
res = rwm_loggarch(y, 10000, 20000, 123);

% --- Display posterior summary ---
theta = {'mu','omega','alpha','beta'}';
Tsum = table(theta, res.mean', res.sd', res.ci(1,:)', res.ci(2,:)', ...
    'VariableNames', {'Parameter','PostMean','PostSD','CI_2p5','CI_97p5'});
disp(Tsum)
fprintf('Acceptance rate: %.2f%%\n', 100*res.acc_rate);

function plot_traces(draws, theta_names)
    if nargin < 2
        theta_names = {'\mu','\omega','\alpha','\beta'};
    end

    n_params = size(draws,2);

    figure('Name','MCMC Trace Plots','Color','w');
    for j = 1:n_params
        subplot(2,2,j);
        plot(draws(:,j), 'Color', [0.1 0.4 0.8]);
        xlabel('Iteration');
        ylabel(['',theta_names{j},'']);
        title(['Trace Plot of ', theta_names{j}],'FontWeight','bold');
        grid on; box on;
    end
end

theta_names = {'\mu','\omega','\alpha','\beta'};
plot_traces(res.draws, theta_names);
