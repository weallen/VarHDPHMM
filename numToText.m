

function string = numToText(numbers)
%numToText    Inverse of textToNum: converts integer encoding back to text string

  underscoreInd = find(numbers == 27);
  numbers(underscoreInd) = -1;
  starInd = find(numbers == 28);
  numbers(starInd) = -54;
  string = char(numbers+96);

