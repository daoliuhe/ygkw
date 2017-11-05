function out = filterData(b,a,in,limit)

out = [];
for ii = 1:100
    temp = in(in.Unit==ii,:);
    temp{:,3:end} = filter(b,a,temp{:,3:end});
    if nargin==4
        out = [out;temp(5:limit,:)];
    else
        out = [out;temp(5:end,:)];
    end
end
    