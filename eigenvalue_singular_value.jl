function neural_net(n,m,L,f)
    X = randn(n, m)    
    W = randn(n, n) / sqrt(n)
    Y = f.(W*X)    
    for i = 1:(L-1)
        Y = f.(randn(n,n)*Y)
    end
    return Y/sqrt(n) 
end

n = 200
m = 200
num_layers = 1

n0 = 200
m = 200
num_layers = 1
l=1
#activation function chose f3 from earlier
bins = 0:0.05:2.2
h = Histogram(bins, :left);
for t = 1:4000
     append!(h, svdvals(neural_net(n0,m,num_layers,f3)))
end
w = h.weights
plot(collect(bins)[1:end-1], w/sum(w)/bins.step.hi, label="Layers = $(num_layers)", lw=3,title="For L = 1 the difference is small for large value of n0")
plot!(0:0.05:2, sqrt.(4 - collect(0:0.05:2).^2) / pi, label="Singular Values - XX^T",color="green")


#n0 = 10
#m = 10
#num_layers = 1
#l=1
#activation function chose f3 from earlier
#bins = 0:0.05:2.2
#h = Histogram(bins, :left);
#for t = 1:4000
#     append!(h, svdvals(neural_net(n0,m,num_layers,f3)))
#end
#w = h.weights
#plot(collect(bins)[1:end-1], w/sum(w)/bins.step.hi, label="Layers = $(num_layers)", lw=3,title="For L = 1 the difference is large for small value of n0")
#plot!(0:0.05:2, sqrt.(4 - collect(0:0.05:2).^2) / pi, label="Singular Values - XX^T")
