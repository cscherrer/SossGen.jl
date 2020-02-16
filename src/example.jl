using Soss, SossGen
using Gen

const logpdf = Soss.logpdf

m = @model σ begin
    μ ~ Normal()
    y ~ Normal(μ, σ) |> iid(1000)
end;

truth = rand(m(σ=1))

g1 = SossGenerativeFunction(m)
g2 = SossGenerativeFunction(m,codegen)

args = (σ=1,)

# tr = simulate(g, (args,)) # generate a full trace
tr1, w1 = generate(g1, (args,), choicemap(:y => truth.y)) # generate a trace with y = 3

tr2, w2 = generate(g2, (args,), choicemap(:y => truth.y)) # generate a trace with y = 3

@gen function q(current, args)
    @trace(normal(current[:μ], 0.2), :μ)
end

using BenchmarkTools

q(tr1, (args,))


@btime mh(tr1, q, (args,));
@btime mh(tr2, q, (args,));


function run_mh(tr)
  for i=1:1000
    tr, = mh(tr, q, ())
  end
  tr
end

@btime get_choices(run_mh(tr1))

# for i=1:1000
#     mh(tr, Gen.select(:x))
#     mh(tr, Gen.select(:y))
# end

using Revise, SossGen
import Gen

m = @model σ begin
    μ ~ Normal()
    y ~ Normal(μ, σ) |> iid(100000)
end;

args = (σ=1.1,);

truth = rand(m(args));
pairs(truth)

# pairs(::NamedTuple) with 3 entries:
#   :σ => 1.1
#   :μ => 0.398088
#   :y => [-0.849513, 0.777053, -0.800264, 0.105648, -1.50177, 1.26158, -0.43
# 2465…

g1 = SossGenerativeFunction(m);
g2 = SossGenerativeFunction(m,codegen);

# tr = simulate(g, (args,)) # generate a full trace

tr1, w1 = generate(g1, (args,), choicemap(:y => truth.y)); # generate a trace with y = 3
tr2, w2 = generate(g2, (args,), choicemap(:y => truth.y)); # generate a trace with y = 3

@gen (static) function q(current, args)
    @trace(normal(current[:μ], 0.2), :μ)
end

Gen.load_generated_functions();

using BenchmarkTools
@btime mh($tr1, $q, $(args,));

# 891.681 μs (147 allocations: 8.64 KiB)

@btime mh($tr2, $q, $(args,));

# 24.929 μs (148 allocations: 8.67 KiB)
