using StatsBase, Plots

forward_difference(f, x0, h) = (f(x0 + h) - f(x0))/h
Df(f; h=1e-8) = x -> forward_difference(f, x, h)

m = -4:0.1:4
f1(x) = (-1+sqrt.(5)*exp.(-2*x.^2))
f2(x) = (sin.(2x)+cos.(3x/2)-2*exp.(-2)*x-exp.(-9/8))
f3(x) = (abs.(x)-sqrt.(2/pi))./sqrt(1 - 2/pi)
f4(x) = (1-(4/sqrt.(3))*exp.(-(x.^2)/2))
plot(m,f1(m),color="red",title="Activation Functions tau = 0 and eta = 1")
plot!(m,f2(m),color="blue")
plot!(m,f3(m),color="green")
plot!(m,f4(m),color="yellow")

# As you can see the mean of derivate of these functions = 0  
plot(Df(f1)(m),color="red",title="f' of the activation functions shown above")
plot!(Df(f2)(m),color="blue")
plot!(Df(f3)(m),color="green")
plot!(Df(f4)(m),color="yellow")
