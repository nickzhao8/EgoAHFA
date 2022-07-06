cd 'M:\Wearable Hand Monitoring\CODE AND DOCUMENTATION\Nick Z\Code\GRASSP Annotation\Results\slowfast\Metrics'
files = dir('M:\Wearable Hand Monitoring\CODE AND DOCUMENTATION\Nick Z\Code\GRASSP Annotation\Results\slowfast\Metrics');
files = files(~ismember({files.name},{'.','..'}));

result_table = table;

allFileNames = {files(:).name};
for k = 1 : length(allFileNames)
    filename = allFileNames{k};
    fid = fopen(filename); % Opening the file
    raw = fread(fid,inf); % Reading the contents
    str = char(raw'); % Transformation
    fclose(fid); % Closing the file
    data = jsondecode(str); % Using the jsondecode function to parse JSON from string
    data = struct2table(data);
    result_table = [result_table;data];
end

size_table = size(result_table);

data = [];
for k = 1: size_table(2)
    cross_val_avg = mean(table2array(result_table(:,k)));
    result_table(18,k) = table(cross_val_avg);
end

acc = result_table(:,2);
macro_precision = result_table(:,5);
macro_recall = result_table(:,7);
macro_f1 = result_table(:,9);

graph_table = [acc macro_precision macro_recall macro_f1];
graph_array = table2array(graph_table);

figure
bar(graph_array)
title('SlowFast Performance by Participant')
xlabel('Participant')
xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
xticklabels({'1','2','3','4','5','6','7','8','9','10','11','12',...
    '13','14','15','16','17','Avg'})
legend({'Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1 Score'}, 'Location', 'southeast')


