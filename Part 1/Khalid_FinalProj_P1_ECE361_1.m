close all;
clear;
 
[status,sheets] = xlsfinfo('Khalid-Project-F2425.xls'); 
[A,names,raw] =xlsread('Khalid-Project-F2425.xls',1); 
 

A; 
Abs=A(1:70);
N0=length(Abs);
Prs=A(71:end); 
N1=length(Prs);
dat=[Abs;Prs];
resp =[zeros(N0,1);ones(N1,1)];
[pf,pd,~,AUC,opt]=perfcurve(resp,dat,1);
parg0=fitdist(Abs,'gamma');
parg1=fitdist(Prs,'gamma');
thr = 0:0.01:100;
aprs=parg1.a;
aabs=parg0.a;
bprs=parg1.b;
babs=parg0.b;
PF=1-cdf('gamma',thr,aabs,babs);
PD=1-cdf('gamma',thr,aprs,bprs);
AUCG=0.5 + polyarea(PF,PD);

xx=0:0.001:1.2*max(dat);

%{
figure(5)
histogram(Abs, 'normalization', 'pdf')
hold on
histogram(Prs, 'normalization', 'pdf')

plot (xx, fx1,'-r', 'linewidth',1.5)
plot (xx, fx2,'--g', 'linewidth',1.5)
intersect=find(round(fx1,2)==round(fx2,2))
finds max & median
peaks_fx1 = findpeaks(fx1);
peaks_fy1 = findpeaks(fx2);
plot(1.063, peaks_fx1, "red", "Marker","o", "LineWidth", 2)
plot(2.56, peaks_fy1, '*k', "Marker", "square", "LineWidth", 2)
%}
fx1=ksdensity(Abs,xx);
fx2=ksdensity(Prs,xx);

% Plot histograms with normalization to PDF
figure(1);
histogram(Abs, 'Normalization', 'pdf', 'FaceColor', 'none', 'EdgeColor', 'k'); % H0 outlined in black with no fill
hold on;
histogram(Prs, 'Normalization', 'pdf', 'FaceColor', [0.850, 0.325, 0.098], 'EdgeColor', 'k'); % H1 outlined in black with darker orange fill

% Define the range for density estimation
xx = 0 : 0.25 : 1.2 * max(A);

% KDE for NoTarget data
fx1 = ksdensity(Abs, xx);
plot(xx, fx1, 'k', 'LineWidth', 2.5); % H0 solid black line

% KDE for PresentTarget data
fx2 = ksdensity(Prs, xx);
plot(xx, fx2, '--r', 'LineWidth', 2.5); % H1 red dashed line

% Find and plot the maximum values of the KDEs
[maxFx1, idxMaxFx1] = max(fx1);
[maxFx2, idxMaxFx2] = max(fx2);
plot(xx(idxMaxFx1), maxFx1, 'bo', 'LineWidth', 1, 'MarkerSize', 9); % max H0 blue circle
plot(xx(idxMaxFx2), maxFx2, 'bs', 'LineWidth', 1, 'MarkerSize', 9); % max H1 blue square

% Find intersection points of the KDEs
intersectionPoints = [];
for i = 1:length(xx)-1
    if (fx1(i) - fx2(i)) * (fx1(i+1) - fx2(i+1)) < 0
        x_intersect = fzero(@(x) interp1(xx, fx1, x) - interp1(xx, fx2, x), [xx(i) xx(i+1)]);
        intersectionPoints = [intersectionPoints, x_intersect];
    end
end

% Plot the intersection points
%for i = 1:length(intersectionPoints)
 %   plot(intersectionPoints(i), interp1(xx, fx1, intersectionPoints(i)), 'c*', 'LineWidth', 1.25, 'MarkerSize', 9); % intersection point magenta
%end

% Label axes
xlabel('Values', 'FontWeight', 'bold', 'FontSize', 12);
ylabel('Estimated Density', 'FontWeight', 'bold', 'FontSize', 12);

% Add legend
legend({'H_0', 'H_1', 'fit(H_0)', 'fit(H_1)', 'max(H_0)', 'max(H_1)'}, ...
    'FontSize', 12);

% Adjust the plot aesthetics
ax = gca;
ax.FontSize = 16;
hold off;

display(intersectionPoints)

% The given value you want to find the closest sample to
mp_calc_int = intersectionPoints(1);

% Calculate the absolute difference between each data sample and the given value
differences_int = abs(A - mp_calc_int);

% Find the index of the minimum difference
[~, minIndex_int] = min(differences_int);

% Find the closest data sample
closestSample_int = A(minIndex_int);


[fNoTarget,xNoTarget] = ksdensity(Abs,xx);
[fTarget,xTarget] =ksdensity(Prs,xx);

idmax=find(fNoTarget==max(fNoTarget));
idm1=round(median(idmax));
idmax=find(fTarget==max(fTarget));
idm2=round(median(idmax));

% The given value you want to find the closest sample to
mp_calc = ((xTarget(idm2))+xNoTarget(idm1))/2;

% Calculate the absolute difference between each data sample and the given value
differences = abs(A - mp_calc);

% Find the index of the minimum difference
[~, minIndex] = min(differences);

% Find the closest data sample
closestSample = A(minIndex);

%Use Midpoint as Threshold
vt=closestSample;
CorrectTargetDet = Prs > vt;
NcM = sum(CorrectTargetDet);
Miss = Abs > vt;
NfM = sum(Miss);
N=N0+N1;
%Confusion Matrix
confusionMM = [(N0 - NfM), (NfM); (N1 -NcM) (NcM)];
errorrateM = (NfM + (N1 - NcM))/N;
ppvM = NcM/(NfM + NcM);
pmM = NfM/130;
pfM = NcM/130;
tranisitionmM = [(1 - pfM), pmM; pfM, (1-pmM)];


%intersection threshold
vt1 = closestSample_int;
CorrectTargetDetI = Prs > vt1;
NcI = sum(CorrectTargetDetI);
MissI = Abs > vt1;
NfI = sum(MissI);
%Confusion Matrix
confusionMI = [(N0 - NfI), (NfI); (N1 -NcI) (NcI)];
errorrateI = (NfI + (N1 - NcI))/N;
ppvI = NcI/(NfI + NcI);
pmI = NfI/130;
pfI = NcI/130;
tranisitionmI = [(1 - pfI), pmI; pfI, (1-pmI)];


%% Data Table
figure(2)
xlim([0, 10])
ylim([0, 8])  % Increased ylim to provide more vertical space
axis off

% Read and process data
B = readmatrix('Khalid-Project-F2425', 'Sheet', 1);
Abs_Sorted = sort(B(1:70));
Prs_Sorted = sort(B(71:end)); 

meanAbs = mean(Abs_Sorted);
meanPrs = mean(Prs_Sorted);
varAbs = var(Abs_Sorted);
varPrs = var(Prs_Sorted);
pindex = abs(meanAbs - meanPrs) / sqrt(varAbs + varPrs);

% Reshape data for display
dats_Absent = reshape(Abs_Sorted, [10, 7]);
dats_Present = reshape(Prs_Sorted, [10, 6]);

% **Display Target Absent Data**

% Title for Target Absent
text(1.5, 7, 'Target Absent (sorted)', 'FontWeight', 'bold', 'FontSize', 12)

% Display data in a grid
for row = 1:size(dats_Absent, 1)
    for col = 1:size(dats_Absent, 2)
        x_pos = 0.3 + (col - 1) * 0.5;  % Adjust horizontal spacing
        y_pos = 6 - (row - 1) * 0.3;    % Adjust vertical spacing
        text(x_pos, y_pos, num2str(dats_Absent(row, col)), 'FontSize', 10)
    end
end

% Mean and Variance for Target Absent
text(1, 2.5, ['mean \mu_1 = ', num2str(meanAbs)], 'Color', 'b', 'FontWeight', 'bold', 'FontSize', 11)
text(1, 2.0, ['var \sigma_1^2 = ', num2str(varAbs)], 'Color', 'b', 'FontWeight', 'bold', 'FontSize', 11)

% Mean and Variance for Target Present
text(5, 2.5, ['mean \mu_2 = ', num2str(meanPrs)], 'Color', 'b', 'FontWeight', 'bold', 'FontSize', 11)
text(5, 2.0, ['var \sigma_2^2 = ', num2str(varPrs)], 'Color', 'b', 'FontWeight', 'bold', 'FontSize', 11)

% Display data in a grid
for row = 1:size(dats_Present, 1)
    for col = 1:size(dats_Present, 2)
        x_pos = 5.5 + (col - 1) * 0.5;  % Adjust horizontal spacing
        y_pos = 6 - (row - 1) * 0.3;    % Adjust vertical spacing
        text(x_pos, y_pos, num2str(dats_Present(row, col)), 'FontSize', 10)
    end
end


% **Add Annotation Line**

% Adjusted annotation line positions
ann = annotation('line', [0.45 0.45], [0.4 0.85]);
ann.Color = 'r';
ann.LineWidth = 1.5;

% **Title and Performance Index**

title('Khalid', 'FontSize', 14)
set(get(gca, 'title'), 'Position', [5, 7.5, 1.00011])  % Adjust title position
text(3.5, 1.5, ['Performance Index PI = ', num2str(pindex)], 'Color', 'b', 'FontWeight', 'bold', 'FontSize', 11)

%% figure 3

dat=[Abs;Prs];
unsorted = [zeros(size(Abs));ones(size(Prs))];
unsorted = [unsorted, dat];
sorted = sortrows(unsorted, 2, 'descend');
sorted = [sorted; 0,0];
T = sorted(:, 2);
counts = zeros(size(sorted));
for i = 1:size(counts)
    Nc = 0;
    Nf = 0;
    if i == 0
        continue;
    else
        for j = 1:(i-1)
            if sorted(j, 1) == 1
                Nc = Nc + 1;
            else
                Nf = Nf + 1;
            end
        end
        counts(i, 1) = Nc;
        counts(i, 2) = Nf;
    end
end
P = zeros(size(counts));
for i = 1:111
    P(i, 1) = counts(i, 1)/N1;
    P(i, 2) = counts(i, 2)/N0;
end
dist = zeros(111,1);
for i = 1:111
    dist(i, 1) = pdist([0,1;P(i, 2), P(i, 1)],'euclidean');
end

[opt_dist, I] = min(dist);

opt_thresh = T(I);
%Confusion Matrix
CorrectTargetDeto = Prs > opt_thresh;
Nco = sum(CorrectTargetDeto);
Misso = Abs > opt_thresh;
Nfo = sum(Misso);
confusiono = [(N0 - Nfo), (Nfo); (N1 -Nco) (Nco)];
errorrateo = (Nfo + (N1 - Nco))/N;
ppvo = Nco/(Nfo + Nco);
pmo = Nfo/130;
pfo = Nco/130;
tranisitionmo = [(1 - pfo), pmo; pfo, (1-pmo)];



figure(3);
xlim([0,1])
ylim([0,1])
plot(pf, pd,'k','LineWidth',1.5)
hold on

plot(PF,PD,':m','LineWidth',2)
plot(opt(1),opt(2), ".r", markersize=15)
plot([0,1],[0,1], '--g','LineWidth',1.5)
plot([0,opt(1)],[1,opt(2)],'-b','linewidth',1.25)
text(0.15,0.85,['d_o_p_t = ',num2str(opt_dist)])
text(opt(1)*1.1,opt(2)*.95,['[P_F = ', num2str(opt(1)),', P_D = ', num2str(opt(2)),']'])
text(0.80,0.46,'Not Detected    Detected','Color','r','fontweight','bold')
text(0.78,0.40,{sprintf('C_x = [%d,                    %d',(N0-Nfo),Nfo)
 sprintf('         %d,                      %d]',(N1-Nco),Nco)},'fontweight','bold')
text(0.80,0.30,{sprintf('error rate = %d/%d',Nfo+(N1-Nco),N)
 sprintf('PPV = %d/%d',Nco,(Nfo+Nco))},'Color','r','fontweight','bold')
text(0.78,0.20,{sprintf('T_x = [%.4f,           %.4f',(1-pfo),pmo)
 sprintf('         %.4f,           %.4f]',pfo,(1-pmo))},'fontweight','bold')
xlabel('P_F = [1-Specificity]'),ylabel('Sensitivity = [1-P_M]')
strAUC=['Empirical ROC: AUC = ',num2str(AUC)];
strAUCG=['Bigamma fit: AUC = ',num2str(AUCG)];

legend(strAUC,strAUCG,'OOP (Neyman Pearson)','Location', 'Southeast')
title(['Optimal Threshold Value (data) = ',num2str(opt_thresh)],'Color','b')

%% Youden ROC
youdenIndex=opt(2)-opt(1);


figure(4);
xlim([0,1])
ylim([0,1])
plot(pf, pd,'k','LineWidth',1.5)
hold on;
% Plot line from optimal point to diagonal
plot([0, opt(1)], [1, opt(2)], '-b', 'LineWidth', 1.25);

hold on

plot(PF,PD,':m','LineWidth',2)
plot(opt(1),opt(2), ".r", markersize=15)
plot([0,1],[0,1], '--g','LineWidth',1.5)
plot([0,opt(1)],[1,opt(2)],'-b','linewidth',1.25)
text(0.15,0.85,['d_o_p_t = ',num2str(opt_dist)])
text(opt(1)*1.1,opt(2)*.95,['[P_F = ', num2str(opt(1)),', P_D = ', num2str(opt(2)),']'])
text(0.80,0.46,'Not Detected    Detected','Color','r','fontweight','bold')
text(0.78,0.40,{sprintf('C_x = [%d,                    %d',(N0-Nfo),Nfo)
 sprintf('         %d,                      %d]',(N1-Nco),Nco)},'fontweight','bold')
text(0.80,0.30,{sprintf('error rate = %d/%d',Nfo+(N1-Nco),N)
 sprintf('PPV = %d/%d',Nco,(Nfo+Nco))},'Color','r','fontweight','bold')
text(0.78,0.20,{sprintf('T_x = [%.4f,           %.4f',(1-pfo),pmo)
 sprintf('         %.4f,           %.4f]',pfo,(1-pmo))},'fontweight','bold')
xlabel('P_F = [1-Specificity]'),ylabel('Sensitivity = [1-P_M]')
strAUC=['Empirical ROC: AUC = ',num2str(AUC)];
strAUCG=['Bigamma fit: AUC = ',num2str(AUCG)];

legend(strAUC,strAUCG,'OOP (Youden''s index) = 0.612','Location', 'Southeast')
title(['Youden''s Index Threshold (data) = ',num2str(opt_thresh)], 'Color','b')


% Figure 5: Density Fit with Intersection Threshold
figure(5)

xx = 0:0.0001:1.2 * max(A);
fx1 = ksdensity(Abs, xx);
fx2 = ksdensity(Prs, xx);

% Plot the density estimates
plot(xx, fx1, 'k', 'linewidth', 1.5)
hold on
plot(xx, fx2, '--r', 'linewidth', 1.5)
xlim([0, 5])
xlabel('Values (v)', 'FontWeight', 'bold', 'FontSize', 12)
ylabel('Estimated PDF', 'FontWeight', 'bold', 'FontSize', 12)
title('Density Fit with Intersection Threshold', 'FontSize', 14)

% Plot vertical line at intersection threshold vt1
line([vt1, vt1], [0, max([fx1, fx2])], 'Color', 'm', 'LineStyle', '-.', 'LineWidth', 2);
text(vt1 + 0.1, max([fx1, fx2]) * 0.9, ['v_T (Intersection) = ', num2str(vt1)], 'Color', 'm', 'FontWeight', 'bold')

% Shade areas for P_M and P_F using vt1
idx_1 = find(xx >= vt1);
X1 = [xx(idx_1), fliplr(xx(idx_1))];
Y1 = [fx1(idx_1), zeros(size(fx1(idx_1)))];
h1 = fill(X1, Y1, 'green');
set(h1, 'facealpha', .4)

idx_2 = find(xx <= vt1);
X2 = [xx(idx_2), fliplr(xx(idx_2))];
Y2 = [fx2(idx_2), zeros(size(fx2(idx_2)))];
h2 = fill(X2, Y2, 'blue');
set(h2, 'facealpha', .4)

% Add legend
legend('f_v(v|H_0)', 'f_v(v|H_1)', 'Threshold at Intersection', 'P_M Region', 'P_F Region', 'Location', 'Best')

% Adjust plot aesthetics
set(gca, 'FontSize', 12)
hold off

%% Confusion and Transition Matrices for Intersection Threshold
disp(['Threshold (vt1) = ', num2str(vt1), ' (intersection point)'])
disp(table(confusionMI, tranisitionmI, 'VariableNames', {'Cx', 'Tx'}))
disp(['Error rate = ', num2str(errorrateI), ', ', 'PPV = ', num2str(ppvI)])

disp(['Threshold (vt) = ',num2str(vt),' (midpoint)'])
disp(table(confusionMM,tranisitionmM,'VariableNames',{'Cx','Tx'}))
disp(['error rate = ', num2str(errorrateM),', ','PPV = ', num2str(ppvM)])

fprintf('\n')

disp(table(confusionMI,tranisitionmI,'VariableNames',{'Cx','Tx'}))
disp(['error rate = ', num2str(errorrateI),', ','PPV = ', num2str(ppvI)])
