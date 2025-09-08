% test.m
% This script generates a file named 'output.txt' with sample data.

% Define the data to write
data = {'Hello, MATLAB on Linux Server!', 'This is a test file.', datestr(now)};

% Open (or create) the file for writing
fileID = fopen('output.txt', 'w');

if fileID == -1
    error('Cannot open file for writing.');
end

% Write each line of data to the file
for i = 1:length(data)
    fprintf(fileID, '%s\n', data{i});
end

% Close the file
fclose(fileID);

% Display a message indicating success
disp('output.txt has been successfully created.');
