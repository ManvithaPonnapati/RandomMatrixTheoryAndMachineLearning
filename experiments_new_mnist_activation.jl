# Take a look at some random images and labels in data
using Knet,Images
include(Knet.dir("data","mnist.jl"))
xtrn,ytrn,xtst,ytst = mnist()
rp = randperm(10000)
for i=1:3; display(mnistview(xtst,rp[i])); end
display(ytst[rp[1:3]])

global accuracy_values = []
for p in ("Knet","ArgParse")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
include(Pkg.dir("Knet","data","mnist.jl"))
module MLP
using Knet,ArgParse
f3(x) = (abs.(x)-sqrt.(2/pi))./sqrt(1 - 2/pi)
function predict(w,x)
    for i=1:2:length(w)
        x = w[i]*mat(x) .+ w[i+1]
        if i<length(w)-1
            x = relu.(x) # max(0,x)
        end
    end
    return x
end

loss(w,x,ygold) = nll(predict(w,x),ygold)

lossgradient = grad(loss)

function train(w, dtrn; lr=.5, epochs=10)
    for epoch=1:epochs
        for (x,y) in dtrn
            g = lossgradient(w, x, y)
            update!(w,g;lr=lr)
        end
    end
    return w
end

function weights(h...; atype=Array{Float32}, winit=0.1)
    w = Any[]
    x = 28*28
    y = 128
    push!(w, convert(atype, winit*randn(y,x)))
    push!(w, convert(atype, zeros(y, 1)))
    x = y
    
    return w
end

function main(args="")
    accuracy_values = []
    s = ArgParseSettings()
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--batchsize"; arg_type=Int; default=50; help="minibatch size")
        ("--epochs"; arg_type=Int; default=1000; help="number of epochs for training")
        ("--hidden"; nargs='*'; arg_type=Int; help="sizes of hidden layers, e.g. --hidden 128 64 for a net with two hidden layers")
        ("--lr"; arg_type=Float64; default=0.5; help="learning rate")
        ("--winit"; arg_type=Float64; default=0.1; help="w initialized with winit*randn()")
        ("--fast"; action=:store_true; help="skip loss printing for faster run")
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"); help="array type: Array for cpu, KnetArray for gpu")
        ("--gcheck"; arg_type=Int; default=0; help="check N random gradients per parameter")
        # These are to experiment with sparse arrays
        # ("--xtype"; help="input array type: defaults to atype")
        # ("--ytype"; help="output array type: defaults to atype")
    end
    isa(args, AbstractString) && (args=split(args))
    if in("--help", args) || in("-h", args)
        ArgParse.show_help(s; exit_when_done=false)
        return
    end
    o = parse_args(args, s; as_symbols=true)
    if !o[:fast]
        println("opts=",[(k,v) for (k,v) in o]...)
    end
    o[:seed] > 0 && srand(o[:seed])
    atype = eval(parse(o[:atype]))
    w = weights(o[:hidden]...; atype=atype, winit=o[:winit])
    xtrn,ytrn,xtst,ytst = Main.mnist()
    global dtrn = minibatch(xtrn, ytrn, o[:batchsize]; xtype=atype)
    global dtst = minibatch(xtst, ytst, o[:batchsize]; xtype=atype)
    report(epoch)=println((:epoch,epoch,:trn,accuracy(w,dtrn,predict),:tst,accuracy(w,dtst,predict)))
    if o[:fast]
        (train(w, dtrn; lr=o[:lr], epochs=o[:epochs]); gpu()>=0 && Knet.cudaDeviceSynchronize())
    else
        report(0)
        @time for epoch=1:o[:epochs]
            print("Epoch $(epoch)")
            train(w, dtrn; lr=o[:lr], epochs=1)
            report(epoch)
            train_accuracy = accuracy(w,dtrn,predict)
            append!(accuracy_values,train_accuracy)
            if o[:gcheck] > 0
                gradcheck(loss, w, first(dtrn)...; gcheck=o[:gcheck], verbose=true)
            end
        end
    end
    return w,accuracy_values
end
PROGRAM_FILE == "mlp.jl" && main(ARGS)

end 
