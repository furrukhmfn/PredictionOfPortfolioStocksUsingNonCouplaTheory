% Code Creation
clear;
clc;

load Data_GlobalIdx1
returns = price2ret(Data);    % Logarithmic returns
T       = size(returns,1);    % # of returns (i.e., historical sample size)
index = 1;  % 1 = Canada, 2 = France, 3 = Germany, 4 = Japan, 5 = UK, 6 = US
figure
plot(dates(2:end), returns(:,index))
datetick('x')
xlabel('Date')
ylabel('Return')
title('Daily Logarithmic Returns')
figure
autocorr(returns(:,index))
title('Sample ACF of Returns')
figure
autocorr(returns(:,index).^2)
title('Sample ACF of Squared Returns')
model     = arima('AR', NaN, 'Distribution', 't', 'Variance', gjr(1,1));
nIndices  = size(Data,2);        % # of indices
residuals = NaN(T, nIndices);    % preallocate storage
variances = NaN(T, nIndices);
fit       = cell(nIndices,1);

options   = optimoptions(@fmincon, 'Display', 'off', ...
    'Diagnostics', 'off', 'Algorithm', 'sqp', 'TolCon', 1e-7);

for i = 1:nIndices
    fit{i} = estimate(model, returns(:,i), 'Display', 'off', 'Options', options);
    [residuals(:,i), variances(:,i)] = infer(fit{i}, returns(:,i));
end
figure
subplot(2,1,1)
plot(dates(2:end), residuals(:,index))
datetick('x')
xlabel('Date')
ylabel('Residual')
title ('Filtered Residuals')
subplot(2,1,2)
plot(dates(2:end), sqrt(variances(:,index)))
datetick('x')
xlabel('Date')
ylabel('Volatility')
title ('Filtered Conditional Standard Deviations')
residuals = residuals ./ sqrt(variances);
figure
autocorr(residuals(:,index))
title('Sample ACF of Standardized Residuals')
figure
autocorr(residuals(:,index).^2)
title('Sample ACF of Squared Standardized Residuals')
nPoints      = 200;      % # of sampled points in each region of the CDF
tailFraction = 0.1;      % Decimal fraction of residuals allocated to each tail
tails = cell(nIndices,1);  % Cell array of Pareto tail objects
for i = 1:nIndices
    tails{i} = paretotails(residuals(:,i), tailFraction, 1 - tailFraction, 'kernel');
end
figure
hold on
grid on
minProbability = cdf(tails{index}, (min(residuals(:,index))));
maxProbability = cdf(tails{index}, (max(residuals(:,index))));
pLowerTail = linspace(minProbability  , tailFraction    , nPoints); % sample lower tail
pUpperTail = linspace(1 - tailFraction, maxProbability  , nPoints); % sample upper tail
pInterior  = linspace(tailFraction    , 1 - tailFraction, nPoints); % sample interior
plot(icdf(tails{index}, pLowerTail), pLowerTail, 'red'  , 'LineWidth', 2)
plot(icdf(tails{index}, pInterior) , pInterior , 'black', 'LineWidth', 2)
plot(icdf(tails{index}, pUpperTail), pUpperTail, 'blue' , 'LineWidth', 2)
xlabel('Standardized Residual')
ylabel('Probability')
title('Empirical CDF')
legend({'Pareto Lower Tail' 'Kernel Smoothed Interior' 'Pareto Upper Tail'}, 'Location', 'NorthWest')
figure
[P,Q] = boundary(tails{index});  % cumulative probabilities & quantiles at boundaries
y     = sort(residuals(residuals(:,index) > Q(2), index) - Q(2)); % sort exceedances
plot(y, (cdf(tails{index}, y + Q(2)) - P(2))/P(1))
[F,x] = ecdf(y);                 % empirical CDF
hold on
stairs(x, F, 'r')
grid on
legend('Fitted Generalized Pareto CDF','Empirical CDF','Location','SouthEast');
xlabel('Exceedance')
ylabel('Probability')
title('Upper Tail of Standardized Residuals')
U = zeros(size(residuals));
for i = 1:nIndices
    U(:,i) = cdf(tails{i}, residuals(:,i)); % transform margin to uniform
end
[R, DoF] = copulafit('t', U, 'Method', 'approximateml'); % fit the copula
s = RandStream.getGlobalStream();
reset(s)
nTrials = 2000;                                   % # of independent random trials
horizon = 22;                                     % VaR forecast horizon
Z = zeros(horizon, nTrials, nIndices);            % standardized residuals array
U = copularnd('t', R, DoF, horizon * nTrials);    % t copula simulation
for j = 1:nIndices
    Z(:,:,j) = reshape(icdf(tails{j}, U(:,j)), horizon, nTrials);
end
Y0 = returns(end,:);                    % presample returns
Z0 = residuals(end,:);                  % presample standardized residuals
V0 = variances(end,:);                  % presample variances
simulatedReturns = zeros(horizon, nTrials, nIndices);
for i = 1:nIndices
    simulatedReturns(:,:,i) = filter(fit{i}, Z(:,:,i), ...
                                    'Y0', Y0(i), 'Z0', Z0(i), 'V0', V0(i));
dat
simulatedReturns = permute(simulatedReturns, [1 3 2]);
cumulativeReturns = zeros(nTrials, 1);
weights           = repmat(1/nIndices, nIndices, 1); % equally weighted portfolio
for j=1:horizon
    VaRForcast(j) = quantile(simulatedReturns(j,:,:), [0.05]');
end

for i = 1:nTrials
    cumulativeReturns(i) = sum(log(1 + (exp(simulatedReturns(:,:,i)) - 1) * weights));
end
VaR = 100 * quantile(cumulativeReturns, [0.10 0.05 0.01]');
disp(' ')
fprintf('Maximum Simulated Loss: %8.4f%s\n'   , -100*min(cumulativeReturns), '%')
fprintf('Maximum Simulated Gain: %8.4f%s\n\n' ,  100*max(cumulativeReturns), '%')
fprintf('     Simulated 90%% VaR: %8.4f%s\n'  ,  VaR(1), '%')
fprintf('     Simulated 95%% VaR: %8.4f%s\n'  ,  VaR(2), '%')
fprintf('     Simulated 99%% VaR: %8.4f%s\n\n',  VaR(3), '%')
figure
h = cdfplot(cumulativeReturns);
h.Color = 'Red';
xlabel('Logarithmic Return')
ylabel('Probability')
title ('Simulated One-Month Global Portfolio Returns CDF')
% Addition of the VAR Backtesting Model
Returns = returns(:,index);
DateReturns = dates(2:end);
SampleSize = length(Returns);
TestWindowStart      = find(year(DateReturns)==1996,1);
TestWindow           = TestWindowStart : SampleSize;
EstimationWindowSize = 5;
pVaR = [0.05 0.01];
Zscore   = norminv(pVaR);
Normal95 = zeros(length(TestWindow),1);
Normal99 = zeros(length(TestWindow),1);
for t = TestWindow
    i = t - TestWindowStart + 1;
    EstimationWindow = t-EstimationWindowSize:t;
    Sigma = std(Returns(EstimationWindow));
    Normal95(i) = -Zscore(1)*Sigma;
    Normal99(i) = -Zscore(2)*Sigma;
end
figure;
plot(DateReturns(TestWindow),[Normal95 Normal99])
datetick('x')
xlabel('Date')
ylabel('VaR')
legend({'95% Confidence Level','99% Confidence Level'},'Location','Best')
title('VaR Estimation Using the Normal Distribution Method')
Historical95 = zeros(length(TestWindow),1);
Historical99 = zeros(length(TestWindow),1);
for t = TestWindow
    i = t - TestWindowStart + 1;
    EstimationWindow = t-EstimationWindowSize:t;
    X = Returns(EstimationWindow);
    Historical95(i) = -quantile(X,pVaR(1));
    Historical99(i) = -quantile(X,pVaR(2));
end

figure;
plot(DateReturns(TestWindow),[Historical95 Historical99])
datetick('x')
ylabel('VaR')
xlabel('Date')
legend({'95% Confidence Level','99% Confidence Level'},'Location','Best')
title('VaR Estimation Using the Historical Simulation Method')
Lambda = 0.94;
Sigma2     = zeros(length(Returns),1);
Sigma2(1)  = Returns(1)^2;
for i = 2 : (TestWindowStart)
    Sigma2(i) = (1-Lambda) * Returns(i-1)^2 + Lambda * Sigma2(i-1);
end
Zscore = norminv(pVaR);
EWMA95 = zeros(length(TestWindow),1);
EWMA99 = zeros(length(TestWindow),1);

for t = TestWindow
    k     = t - TestWindowStart + 1;
    Sigma2(t) = (1-Lambda) * Returns(t-1)^2 + Lambda * Sigma2(t-1);
    Sigma = sqrt(Sigma2(t));
    EWMA95(k) = -Zscore(1)*Sigma;
    EWMA99(k) = -Zscore(2)*Sigma;
end

figure;
plot(DateReturns(TestWindow),[EWMA95 EWMA99])
datetick('x')
ylabel('VaR')
xlabel('Date')
legend({'95% Confidence Level','99% Confidence Level'},'Location','Best')
title('VaR Estimation Using the EWMA Method')
ReturnsTest = Returns(TestWindow);
DatesTest   = DateReturns(TestWindow);
figure;
plot(DatesTest,[ReturnsTest -Normal95 -Historical95 -EWMA95])
datetick('x')
ylabel('VaR')
xlabel('Date')
legend({'Returns','Normal','Historical','EWMA'},'Location','Best')
title('Comparison of returns and VaR at 95% for different models')


formatOut = 'mm-dd-yyyy';
str = datestr(DateReturns(TestWindow),formatOut,'local');
datetimevar = datetime(str,'InputFormat','MM-dd-yyyy');
ZoomInd   = (DateReturns(TestWindow) >= datenum(2003,6,12)) & (DateReturns(TestWindow) <= datenum(2003,7,14));
SimulatedDataVar = SimulatedDataVar';
VaRData   = [-Normal95(ZoomInd) -Historical95(ZoomInd) -EWMA95(ZoomInd)];
VaRFormat = {'-','--','-.','-'};
D = DatesTest(ZoomInd);
R = ReturnsTest(ZoomInd);
S = SimulatedDataVar(ZoomInd);
N = Normal95(ZoomInd);
H = Historical95(ZoomInd);
E = EWMA95(ZoomInd);
IndN95    = (R < -N);
IndHS95   = (R < -H);
IndEWMA95 = (R < -E);
IndSim = (R < S);
figure;
bar(D,R,0.5,'FaceColor',[0.7 0.7 0.7]);
hold on
for i = 1 : size(VaRData,2)
    stairs(D-0.5,VaRData(:,i),VaRFormat{i});
end
ylabel('VaR')
datetick('x')
xlabel('Date')
plot(D(IndN95),-N(IndN95),'o',D(IndHS95),-H(IndHS95),'o',...
   D(IndEWMA95),-E(IndEWMA95),'o','MarkerSize',8,'LineWidth',1.5)
legend({'Returns','Normal','Historical','EWMA','Simulated Data'},'Location','Best','AutoUpdate','Off')
title('95% VaR violations for different models')
ax = gca;
ax.ColorOrderIndex = 1;
formatOut = 'mm-dd-yyyy';
% Datetime1 = datestr(D(IndN95),formatOut,'local');
% Datetime2 = datestr(D(IndHS95),formatOut,'local');
% Datetime3 = datestr(D(IndEWMA95),formatOut,'local');

xlim([D(1)-1, D(end)+1])
hold off;

vbt = varbacktest(ReturnsTest,[Normal95 Normal99],'PortfolioID','S&P','VaRID',...
    {'Normal95','Normal99'},'VaRLevel',[0.95 0.99]);
summary(vbt)