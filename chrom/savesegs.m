function  savesegs(filename, states )
fid = fopen(filename,'w');
T = length(states);
currstate = states(1);
start = 0;
stop = 200;
for i=1:T
    stop = (i-1)*200;
    if states(i) ~= currstate
        fprintf(fid, 'chr11\t%d\t%d\t%s\n', start, stop, strcat('E',num2str(currstate)));
        start = (i-1)*200;
        currstate = states(i);
    end           
end
fclose(fid);
end

