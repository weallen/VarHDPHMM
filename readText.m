function string = readText(filename)
%readText    Reads text characters from file and processes word boundaries

  % read all words from file, ignoring whitespace
  fid = fopen(filename, 'r');
  words = textscan(fid, '%s');
  words = words{1};
  fclose(fid);

  % combine words in single string, separated by single underscore characters
  string = [];
  for ww = 1:length(words)
    string = strcat(string, words{ww}, '_');
  end
