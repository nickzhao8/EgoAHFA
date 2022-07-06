cd 'M:\Wearable Hand Monitoring\CODE AND DOCUMENTATION\Nick Z\Code\GRASSP Annotation\Results\slowfast\Raw'
files = dir('.');
files = files(~ismember({files.name},{'.','..'}));

result_table = table;
all_results = struct;

allFileNames = {files(:).name};
for k = 1 : length(allFileNames)
    filename = allFileNames{k};
    fid = fopen(filename); % Opening the file
    raw = fread(fid,inf); % Reading the contents
    str = char(raw'); % Transformation
    fclose(fid); % Closing the file
    data = jsondecode(str); % Using the jsondecode function to parse JSON from string

    
    
end