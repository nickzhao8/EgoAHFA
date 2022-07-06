file = ('M:\Wearable Hand Monitoring\CODE AND DOCUMENTATION\Nick Z\Code\GRASSP Annotation\Results\slowfast\task_metrics.json');

fid = fopen(file); % Opening the file
raw = fread(fid,inf); % Reading the contents
str = char(raw'); % Transformation
fclose(fid); % Closing the file
data = jsondecode(str); % Using the jsondecode function to parse JSON from string

tasks = fieldnames(data);

result_table = [];
for i = 1:numel(tasks)
    row = struct2table(data.(tasks{i}));
    result_table = [result_table;row];
end

% average
size_table = size(result_table);
for k = 1: size_table(2)
    cross_val_avg = mean(table2array(result_table(:,k)));
    result_table(39,k) = table(cross_val_avg);
end

tasks{39} = 'Average';

%%

acc = result_table(:,7);
macro_precision = result_table(:,1);
macro_recall = result_table(:,2);
macro_f1 = result_table(:,3);

graph_table = [acc macro_f1];
graph_array = table2array(graph_table);

figure
bar(graph_array)
title('SlowFast Performance by Task')
xlabel('Task')
xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20, ...
    21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39])
xticklabels(tasks)
xtickangle(60)
legend({'Accuracy', 'Macro F1 Score'}, 'Location', 'northeast')


