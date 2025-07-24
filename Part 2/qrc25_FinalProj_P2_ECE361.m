%% Clear
close all, clc, clear

%% Load in file
filename = 'Carrel-Project-S2324.xls';
split_str = split(filename, '-');
lastname = split_str(1);
lastname = lastname{:}; % No longer need to do lastname{:}

%% Initialize data sets
data = readmatrix(filename); % Make sure your CWD has this xls in it
sortedData = sort(data);
NT = sort(data(1:70));% Top 70 samples represent TARGET ABSENT (NT)
T = sort(data(71:end));% Remaining 60 represent TARGET PRESENT (T)

%% Make Chart of Data -- Not really necessary but nice option to have
figure(1);
xlim([0,5]);
ylim([0,5]);
axis off;
dats=reshape(NT,[10,7]); % create the data as 20 x 5 size
text(.5,2.2,num2str(dats, ' %.3f '))
title('Target Absent (sorted)')

figure(2);
xlim([0,5]);
ylim([0,5]);
axis off;
dats=reshape(T,[10,6]); % create the data as 20 x 5 size
text(.5,2.2,num2str(dats, ' %.3f '))
title('Target Present (sorted)')

%% Histogram and pdf over given values
figure(3);
histogram(NT, 'Normalization', 'pdf', EdgeColor='k', FaceColor='none'); % Plot the pdf histogram for target absent
hold on
histogram(T, 'Normalization', 'pdf', EdgeColor='k') % Plot the pdf histogram for target present

x_range = 0:0.01:max(T); % Create an array of x values from zero to the maximum value measured for target present with a step of 0.01


fx0 = ksdensity(NT, x_range); % Tet absent, corresponds to H_0
plot(x_range, fx0, '-k', 'LineWidth', 1.5) % Plot the densities over the xrange

fx1 = ksdensity(T, x_range); % Tet present, corresponds to H_1
plot(x_range, fx1, '--r', 'LineWidth', 1.5) % Plot the densities over the xrange 

NT_Max_idx = find(fx0==max(fx0)); % Get the index of the maximum value in the density return
fx0max = x_range(NT_Max_idx);
plot(x_range(NT_Max_idx),fx0(NT_Max_idx), 'bo') % Plot the point using the index above for the max

T_Max_idx = find(fx1==max(fx1)); % Get the index of the maximum value in the density return
fx1max = x_range(T_Max_idx);
plot(x_range(T_Max_idx),fx1(T_Max_idx), 'bs') % Plot the point using the index above for the max

[~, index] = min(abs(NT - fx0max));
fx0max_real = NT(index);

[~, index] = min(abs(T - fx1max));
fx1max_real = T(index);

legend("H_0", "H_1", "Fit (H_0)", "Fit (H_1)", sprintf("max(H_0)"), sprintf("max(H_1)"));
ylabel("Estimated density")
xlabel("Values")
title(sprintf("%s",lastname))
%ylim([0, 0.85]) % Change this for YOUR data
%xlim([0, 10]) % Change this for YOUR data

%% ROC Curve - Neyman Pearson
first_fail = (fx0max_real + fx1max_real)/2; % Get the first failing point
[~, index] = min(abs(data-first_fail)); % Gets the index of the closest data point in All Data
midpoint = data(index)

mean_T = mean(T); % Get the mean of targets present
var_T = var(T); % Get the variance of the targets present
mean_NT = mean(NT); % Get the mean of the targets present
var_NT = var(NT); % Get the variance of the targets present

Perf_IDX = abs(mean_NT - mean_T) / sqrt(var_T + var_NT);

max_val = round(NT(70)*100);
min_val = round(T(1)*100);
temp = abs(fx0(min_val:max_val) - fx1(min_val:max_val));
[~, index] = min(temp);

x_int = x_range(index+min_val);
[~, index] = min(abs(data-x_int));
intersection = data(index)

[NT_parG] = fitdist(NT, 'gamma');
NTG_a=NT_parG.a;
NTG_b=NT_parG.b;

[T_parG] = fitdist(T, 'gamma');
TG_a=T_parG.a;
TG_b=T_parG.b;

PF_bigma = 1-cdf('gamma', 0:0.01:100, NTG_a, NTG_b);
PD_bigma = 1-cdf('gamma', 0:0.01:100, TG_a, TG_b);
AUC_bigma=0.5+polyarea(PF_bigma,PD_bigma);

N0 = 70;
N1 = 60;
zerosColumn = zeros(N0, 1);
onesColumn = ones(N1, 1);
resultColumn = [zerosColumn; onesColumn];

NPT_Array = [data resultColumn]; % Neyman Pearson Threshold array
Sorted_NPT_Array = flipud(sortrows(NPT_Array, 1));

NC = 0;
NF = 0;
Final_NPT_Array = [];
for row1 = 1:size(Sorted_NPT_Array, 1)
    currentRow = Sorted_NPT_Array(row1, :);
    PD = NC / N1;
    PF = NF / N0;
    distance = sqrt(PF^2 + (1-PD)^2);
    PD_PF = PD-PF;
    newRow = [currentRow NC NF PD PF distance PD_PF];
    if currentRow(2) == 1
        NC = 1 + NC;
    else
        NF = 1 + NF;
    end
    Final_NPT_Array = [Final_NPT_Array ; newRow];
end
Final_NPT_Array = [Final_NPT_Array; 0 0 60 70 1 1 1 0];

[~, index] = min(Final_NPT_Array(:, 7));
NPT = Final_NPT_Array(index, 1);
PF =  Final_NPT_Array(index, 6);
PD =  Final_NPT_Array(index, 5);
Distance = Final_NPT_Array(index, 7);

[~, ~, ~, AUC] = perfcurve(resultColumn, data, 1);

figure(4);
stairs(Final_NPT_Array(:, 6), Final_NPT_Array(:, 5), 'k', LineWidth=2)
hold on
plot(PF_bigma, PD_bigma, 'm:', LineWidth=2)
scatter(PF, PD, 30, 'r', 'filled')
%plot(PF, PD, 'ro', 'MarkerFaceColor', 'r')

plot([0 PF], [1 PD], 'b')
plot([PF PF], [PD PF], 'r')
plot([0 1], [0 1], 'g--')

title(sprintf("Optimal threshold value (data) = %.4f", NPT), color='b')
xlabel("P_F = [1-Specificity]")
ylabel("Sensitivity = [1-P_M]")

legend(sprintf("Empirical ROC: AUC = %.3f", AUC), sprintf("Bigamma fit: AUC = %.3f", AUC_bigma), sprintf("OOP (Neyman Pearson)"), Location="southeast");
text(PF-.05, PD+.1, sprintf("d = %.3f", Distance))
text(PF+.025, PD-.025, sprintf("[P_F=%.3f, P_D=%.3f]",PF, PD))
text(PF+.01, PD/2, sprintf("←(P_D-P_F)=%.3f",PD-PF))

figure(5)

[~, index] = max(Final_NPT_Array(:, 8));
YPT = Final_NPT_Array(index, 1);
PF =  Final_NPT_Array(index, 6);
PD =  Final_NPT_Array(index, 5);
Distance = Final_NPT_Array(index, 7);
Index_Y = Final_NPT_Array(index, 8);

stairs(Final_NPT_Array(:, 6), Final_NPT_Array(:, 5), 'k', LineWidth=2)
hold on
plot(PF_bigma, PD_bigma, 'm:', LineWidth=2)
scatter(PF, PD, 30, 'r', 'filled')

plot([0 PF], [1 PD], 'b')
plot([PF PF], [PD PF], 'r')
plot([0 1], [0 1], 'g--')

title(sprintf("Youden's index threshold (data) = %.4f", YPT), color='b')
xlabel("P_F = [1-Specificity]")
ylabel("Sensitivity = [1-P_M]")

legend(sprintf("Empirical ROC: AUC = %.3f", AUC), sprintf("Bigamma fit: AUC = %.3f", AUC_bigma), sprintf("OOP (Youden's index)=%.3f", Index_Y), Location="southeast");
text(PF-.05, PD+.12, sprintf("d = %.3f", Distance))
text(PF+.025, PD-.025, sprintf("[P_F=%.3f, P_D=%.3f]",PF, PD))
text(PF+.01, PD/2, sprintf("←(P_D-P_F)=%.3f",PD-PF))

figure(6);
plot(x_range, fx0, 'k')
hold on
plot(x_range, fx1, '--r')
plot([YPT YPT], [0 max(fx0)+.1], 'b')
plot([YPT], [max(fx0)+.1], 'b^')

roundedYPT = round(YPT*100)/100;
YPT_X_INDEX = find(abs(x_range - roundedYPT) < 1e-6);
fill([x_range(YPT_X_INDEX:end), fliplr(x_range(YPT_X_INDEX:end))], [fx0(YPT_X_INDEX:end), zeros(size(fx0(YPT_X_INDEX:end)))], 'g', 'FaceAlpha', 0.5)
% fill(x_range(YPT_X_INDEX:end), fliplr(x_range(YPT_X_INDEX:end)), 'g', 'FaceAlpha', 0.5)
fill([x_range(1:YPT_X_INDEX), fliplr(x_range(1:YPT_X_INDEX))], [fx1(1:YPT_X_INDEX), zeros(size(fx1(1:YPT_X_INDEX)))], 'b', 'FaceAlpha', 0.3)
xlabel("Values (v)")
ylabel("Estimated pdf")
title("Density fit")
legend("f_v(v|H_0)", "f_x(v|H_1)")
text(YPT-.5, 0.03, sprintf("P_M"))
text(YPT+.2, 0.03, sprintf("P_F"))
text(YPT+.1, max(fx0)+.05, sprintf("v_T (Youden's index) = %.4f", YPT))

%% ---------- PART 2 STUFF ----------

%% Chi^2 Stuffs
% Lower case a means absent (Taken from Keith)

% Distributions for target absent
parWa = fitdist(NT, 'Weibull');
parNa = fitdist(NT, 'Nakagami');
parGa = fitdist(NT, 'Gamma');
parRa = fitdist(NT, 'Rician');
parRay = fitdist(NT, 'Rayleigh');
parLog = fitdist(NT, 'Lognormal');

%h values, p stats, and other stats for the target absent distrubutions
[hWa, pWa, statsWa] = chi2gof(NT, 'CDF',parWa, 'nBins', 7, 'Emin', 2);
[hNa, pNa, statsNa] = chi2gof(NT, 'CDF',parNa, 'nBins', 7, 'Emin', 2);
[hGa, pGa, statsGa] = chi2gof(NT, 'CDF',parGa, 'nBins', 7, 'Emin', 2);
[hRa, pRa, statsRa] = chi2gof(NT, 'CDF',parRa, 'nBins', 7, 'Emin', 2);
[hRay, pRay, statsRay] = chi2gof(NT, 'CDF',parRay, 'nBins', 7, 'Emin', 2);
[hLog, pLog, statsLog] = chi2gof(NT, 'CDF',parLog, 'nBins', 7, 'Emin', 2);

%X_RW = statsWa.chi2stat / statsWa.df;
%X_Gn = statsNa.chi2stat / statsNa.df;
%X_Rg = statsGa.chi2stat / statsGa.df;
%X_Rr = statsRa.chi2stat / statsRa.df;
%X_Ray = statsRay.chi2stat / statsRay.df;
%X_Log = statsLog.chi2stat / statsLog.df;

density = ["Weibull"; "Nakagami"; "Gamma"; "Rician"; "Rayleigh"; "Lognormal"]; % array of density names

h = [hWa; hNa; hGa; hRa; hRay; hLog];
DoF = [statsWa.df; statsNa.df; statsGa.df; statsRa.df; statsRay.df; statsLog.df];
chi_squared_stat = [statsWa.chi2stat; statsNa.chi2stat; statsGa.chi2stat; statsRa.chi2stat; statsRay.chi2stat; statsLog.chi2stat];
p_valuea = [pWa; pNa; pGa; pRa; pRay; pLog];
T_1 = table(h, DoF, chi_squared_stat, p_valuea, 'RowNames', density);

empirical = ksdensity(sortedData,x_range);

% Distributions for target present
parW = fitdist(T, 'Weibull');
parN = fitdist(T, 'Nakagami');
parG = fitdist(T, 'Gamma');
parR1 = fitdist(T, 'Rician');
parRy = fitdist(T, 'Rayleigh');
parLN = fitdist(T, 'Lognormal');

[hW, pW, statsW] = chi2gof(T, 'CDF',parW, 'nBins', 7, 'Emin', 2);
[hN, pN, statsN] = chi2gof(T, 'CDF',parN, 'nBins', 7, 'Emin', 2);
[hG, pG, statsG] = chi2gof(T, 'CDF',parG, 'nBins', 7, 'Emin', 2);
[hR, pR, statsR] = chi2gof(T, 'CDF',parR1, 'nBins', 7, 'Emin', 2);
[hRy, pRy, statsRy] = chi2gof(T, 'CDF',parRy, 'nBins', 7, 'Emin', 2);
[hLN, pLN, statsLN] = chi2gof(T, 'CDF',parLN, 'nBins', 7, 'Emin', 2);

%X_RW = statsW.chi2stat / statsW.df;
%X_Gn = statsN.chi2stat / statsN.df;
%X_Rg = statsG.chi2stat / statsG.df;
%X_Rr = statsR.chi2stat / statsR.df;
%X_Ry = statsRy.chi2stat / statsRy.df;
%X_LN = statsLN.chi2stat / statsLN.df;

h = [hW; hN; hG; hR; hRy; hLN]; % Array of H values for Target Present

DoF = [statsW.df;statsN.df;statsG.df;statsR.df;statsRy.df;statsLN.df];
chi_squared_stat = [statsW.chi2stat;statsN.chi2stat;statsG.chi2stat;statsR.chi2stat;statsRy.chi2stat;statsLN.chi2stat];
p_value = [pW;pN;pG;pR;pRy;pLN];
T_2 = table(h, DoF, chi_squared_stat, p_value, 'RowNames', density);

BF_NT = density(find(p_valuea == max(p_valuea)));
BF_NT_index = find(density == BF_NT);

BF_T = density(find(p_value == max(p_value)));
BF_index = find(density == BF_T);

% Display table of target absent Chi^2 results
fprintf("\t\t\t\t\t\tTarget Absent\n")
disp(T_1)
disp("Best fit " + BF_NT)

% Display table of target present Chi^2 results
fprintf("\t\t\t\t\t\tTarget Present\n")
disp(T_2)
disp("Best fit " + BF_T)

%% Plot Best Fits
figure(7)
hNT=histogram(NT,'normalization', 'pdf');
hold on
hNT.FaceColor="none";
hNT.LineWidth=1;

if BF_NT_index == 5 % If this is rayleigh
    parNT=fitdist(NT,BF_NT);
    FNT= pdf(BF_NT,x_range,parNT.ParameterValues(1));
    plot(x_range,FNT,'r','LineWidth', 1.5)
    legendNT = sprintf("theoretical fit (Target Absent): %s (%.3f)", BF_NT, parNT.ParameterValues(1));
else % If the density is non-rayleigh
    parNT=fitdist(NT,BF_NT);
    FNT= pdf(BF_NT,x_range,parNT.ParameterValues(1),parNT.ParameterValues(2));
    plot(x_range,FNT,'r','LineWidth', 2)
    legendNT = sprintf("theoretical fit (Target Absent): %s (%.3f,%.3f)", BF_NT, parNT.ParameterValues(1),parNT.ParameterValues(2));
end

hT=histogram(T,'normalization', 'pdf');
hT.FaceColor="#3A3B3C";
hT.LineWidth=1;

if BF_index == 5 % If this is rayleigh
    parT=fitdist(T,BF_T);
    FT= pdf(BF_T,x_range,parT.ParameterValues(1));
    plot(x_range,FT,'--b','LineWidth', 1.5)
    legendT = sprintf("theoretical fit (Target Present): %s (%.3f)", BF_T, parT.ParameterValues(1));
else % If the density is non-rayleigh
    parT=fitdist(T,BF_T);
    FT= pdf(BF_T,x_range,parT.ParameterValues(1),parT.ParameterValues(2));
    plot(x_range,FT,'--b','LineWidth', 1.5)
    legendT = sprintf("theoretical fit (Target Present): %s (%.3f,%.3f)", BF_T, parT.ParameterValues(1),parT.ParameterValues(2));
end

xlabel("values")
ylabel("histogram or pdf fit")
legend("data (Target Absent)", legendNT, "data (Target Present)", legendT)

%% 4 Bootstrapping Graphs

figure(8);

thr = 0:0.01:100;
dat=[NT; T];
resp = [zeros(N0, 1); ones(N1, 1)];
[pf,pd,~,AUC,opt]=perfcurve(resultColumn,dat,1);

if BF_NT_index == 5
    F_t = cdf(BF_NT,x_range,parNT.ParameterValues(1));
else
    F_t = cdf(BF_NT,x_range,parNT.ParameterValues(1), parNT.ParameterValues(2));
end

if BF_index == 5
    F_t2 = cdf(BF_T,x_range,parT.ParameterValues(1));
else
    F_t2 = cdf(BF_T,x_range,parT.ParameterValues(1),parT.ParameterValues(2));
end

PF_t = 1 - F_t;
PD_t = 1 - F_t2;

nb=5000;% number boot samples

boot_AUC = zeros(nb,1);

nonpar_roc_curves = cell(nb, 2);

for i=1:nb
    absent_index_array = randi(70,[70,1]);
    present_index_array = randi(60,[60,1]);
    rand_absent = NT(absent_index_array);
    rand_present = T(present_index_array);
    rand_dat = [rand_absent;rand_present];
    [pf_temp, pd_temp, ~, auc_i] = perfcurve(resp, rand_dat, 1);
    boot_AUC(i) = auc_i;
    nonpar_roc_curves{i}{1} = pf_temp;
    nonpar_roc_curves{i}{2} = pd_temp;
end

y25=prctile(boot_AUC,2.5); % 2.5%
y975=prctile(boot_AUC,97.5); % 97.5%

subplot(2, 2, 1)
stairs(pf, pd, 'k', LineWidth=2)
hold on
for i=1:nb
    if boot_AUC(i) >= y25 && boot_AUC(i) <= y975
        plot(nonpar_roc_curves{i}{1}, nonpar_roc_curves{i}{2}, 'b:')
    end
end
plot([0 1], [0 1], 'g--')
legend("empirical", "95% CI", Location="southeast")
xlabel("1-Specificity")
ylabel("Sensitivity")
title("Nonparametric Bootstrapping")

subplot(2,2,2)
histogram(boot_AUC, 20);
xlim([0.5 1])
xlabel('AUC values'),ylabel('frequency')
title(sprintf("Non-parametric\nµ = %.3f σ = %.4f \n95%% CI of µ = [%.3f, %.3f]", mean(boot_AUC), std(boot_AUC), y25, y975))

subplot(2,2,3)
para_AUC = zeros(1, nb);
par_roc_curves = cell(nb, 2);

for i = 1:nb
    if BF_NT_index == 5
        NT_R = random(BF_NT,parNT.ParameterValues(1),70,1);
    else
        NT_R = random(BF_NT,parNT.ParameterValues(1), parNT.ParameterValues(2), 70, 1);
    end
    
    if BF_index == 5
        T_R = random(BF_T,parT.ParameterValues(1),60,1);
    else
        T_R = random(BF_T,parT.ParameterValues(1), parT.ParameterValues(2), 60, 1);
    end

    dat_r = [NT_R; T_R];
    
    [pf_temp,pd_temp,~,para_AUC(i)]=perfcurve(resultColumn,dat_r,1);
    par_roc_curves{i}{1} = pf_temp;
    par_roc_curves{i}{2} = pd_temp;
end

y25=prctile(para_AUC,2.5); % 2.5%
y975=prctile(para_AUC,97.5); % 97.5%

plot(PF_t, PD_t, 'r--', "linewidth", 2)
hold on
for i=1:nb
    if para_AUC(i) >= y25 && para_AUC(i) <= y975
        plot(par_roc_curves{i}{1}, par_roc_curves{i}{2}, 'b:')
    end
end

plot([0 1], [0 1], 'g--')
legend("best fit", "95% CI", Location="southeast")
xlabel("1-Specificity")
ylabel("Sensitivity")
title("Parametric Bootstrapping")

subplot(2,2,4)
histogram(para_AUC, 20);
xlim([0.5 1])
xlabel('AUC values'),ylabel('frequency')
title(sprintf("Parametric\nµ = %.3f σ = %.4f \n95%% CI of µ = [%.3f, %.3f]", mean(para_AUC), std(para_AUC), y25, y975))

%% ROC Curve Stuffs

figure(9)
A1 = AUC/(2-AUC); % Why is this 2 and not 1?
A2 = 2 * AUC^2/(1+AUC);
Std = sqrt((AUC*(1-AUC) + (N1-1)*(A1-AUC^2) + (N0-1)*(A2-AUC^2))/(N0*N1));

areaROC = 0.5 + polyarea(pf,pd);

tArea = 0.5 + polyarea(PF_t, PD_t);

hold on
plot(PF_t, PD_t, 'r--', "linewidth", 1.5)
stairs(pf, pd, 'k', 'LineWidth', 1.5) % black line
plot(PF_bigma, PD_bigma,'-.b', 'LineWidth', 1.5)

plot([0 1], [0 1], 'b')

xlabel("Prob. of False Alarm (1-Specificity)")
ylabel("Prob. of Detection (Sensitivity)")

legend(sprintf("Theoretical fit: AUC = %.3f", tArea),...
    sprintf("Empirical fit: AUC = %.3f", AUC), ...
   sprintf("Bigamma fit: AUC = %.3f", AUC_bigma))

Text = {sprintf("AUC(empirical) = %.3f, σ = %.4f", AUC, Std),...
    sprintf("AUC(mean) = %.3f, σ = %.4f (nonparametric bootstrapping)", mean(boot_AUC), std(boot_AUC)),...
    sprintf("AUC(mean) = %.3f, σ = %.4f (parametric bootstrapping)", mean(para_AUC), std(para_AUC))};
text(0.15,0.1,Text)
box on; % This SHOULD not have to be on if plotted correctly?

%% Functions

function area=calculateROCarea(resp,dat)
    [~,~,~,AUC] = perfcurve(resp,dat,1);
    area = AUC;
end