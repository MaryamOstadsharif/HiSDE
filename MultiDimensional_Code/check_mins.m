[N,K]=size(Z);
mins = zeros(N,1);
for ns=1:N
    if sum(Z(ns,:))>1
        mins(ns)= min(diff(find(Z(ns,:))));
    else
        mins(ns) = 1000;
    end
end