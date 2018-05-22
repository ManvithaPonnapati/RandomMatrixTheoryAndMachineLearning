using Interact
using Gadfly

N = 3 #Number of eigenvalues
x = ones(N) # Set initial errors for all components to be one
eig_val_1 = 0.01
eig_val_2 = 0.1
eig_val_3 = 0.5
eig_val_4 = 1.0
k = 1:1:50
alpha_value = 2.0
@manipulate for alpha_value=0:0.1:2
    v1 = ((1-alpha_value*eig_val_1).^(2*k))*eig_val_1
    v2 = ((1-alpha_value*eig_val_2).^(2*k))*eig_val_2
    v3 = ((1-alpha_value*eig_val_3).^(2*k))*eig_val_3
    v4 = ((1-alpha_value*eig_val_4).^(2*k))*eig_val_4
    Gadfly.plot(layer(x=k, y=v4, Geom.point, Geom.line,Theme(default_color="blue")),layer(x=k, y=v1, Geom.point, Geom.line,Theme(default_color="orange")),layer(x=k, y=v2, Geom.point, Geom.line,Theme(default_color="purple")),layer(x=k, y=v3, Geom.point, Geom.line,Theme(default_color="green"),),Guide.ylabel("E(W)-E(W<sup>*)"),Guide.xlabel("Distance of weight matrix from the optimum at each iteration k"),Guide.manual_color_key("Legend", [string(eig_val_1), string(eig_val_2), string(eig_val_3),string(eig_val_4)], ["orange", "purple", "green","blue"]))
end
