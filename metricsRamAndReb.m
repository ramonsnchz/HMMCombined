df = [];
for i = 1:20
    if i == 1
        newDir = '/home/ramon/Documents/HMMRam+Reb/Result_0';
    else
        newDir = sprintf('/home/ramon/Documents/HMMRam+Reb/Result_%d', i-1);
    end
    cd(newDir)
    filename = sprintf('performance_matrics.csv');
    datafile = csvread(filename,2,0);
    df = [df; datafile];
    CCR = df(:,1);
    CCRmean = mean(CCR);
    CCRstd = std(CCR);
    Precision = df(:,2);
    Pmean = mean(Precision);
    Pstd = std(Precision);
    Recall = df(:,3);
    Rmean = mean(Recall);
    Rstd = std(Recall);
    F1 = df(:,4);
    F1mean = mean(F1);
    F1std = std(F1);
    CCR1 = [CCRmean;CCRstd];
    Recall1 = [Rmean;Rstd];
    Precision1 = [Pmean;Pstd];
    F11 = [F1mean;F1std];
    sample = [ CCR1 Recall1 Precision1 F11];
    rowNames = {'Mean','Std'};
    colNames = {'CCR','Precision','Recall', 'F1'};
    sTable = array2table(sample,'RowNames',rowNames,'VariableNames',colNames)
    T = table(CCR, Precision, Recall, F1)
%     writetable(T,'performances.csv','Delimiter',',','QuoteStrings',true);
%     type 'performances.csv';
%     writetable(sTable,'performances2.csv','Delimiter',',','QuoteStrings',true);
%     type 'performances2.csv';
    
end