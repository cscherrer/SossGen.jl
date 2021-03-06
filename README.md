[![Build Status](https://travis-ci.com/cscherrer/SossGen.jl.svg?branch=master)](https://travis-ci.com/cscherrer/SossGen.jl)
[![Codecov](https://codecov.io/gh/cscherrer/SossGen.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/cscherrer/SossGen.jl)

````julia
using Revise, SossGen
import Gen
````



````julia
m = @model σ begin
    μ ~ Normal()
    y ~ Normal(μ, σ) |> iid(100000)
end;
````



````julia
args = (σ=1.1,);

truth = rand(m(args));
pairs(truth)
````


````
pairs(::NamedTuple) with 3 entries:
  :σ => 1.1
  :μ => 0.398088
  :y => [-0.849513, 0.777053, -0.800264, 0.105648, -1.50177, 1.26158, -0.43
2465…
````



````julia
g1 = SossGenerativeFunction(m);
g2 = SossGenerativeFunction(m,codegen);
````



````julia
# tr = simulate(g, (args,)) # generate a full trace

tr1, w1 = generate(g1, (args,), choicemap(:y => truth.y)); # generate a trace with y = 3
tr2, w2 = generate(g2, (args,), choicemap(:y => truth.y)); # generate a trace with y = 3
````



````julia
@gen (static) function q(current, args)
    @trace(normal(current[:μ], 0.2), :μ)
end

Gen.load_generated_functions();
````



````julia
using BenchmarkTools
@btime mh($tr1, $q, $(args,));
````


````
891.681 μs (147 allocations: 8.64 KiB)
````



````julia
@btime mh($tr2, $q, $(args,));
````


````
24.929 μs (148 allocations: 8.67 KiB)
````


