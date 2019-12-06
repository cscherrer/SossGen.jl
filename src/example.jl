using Revise, Soss, SossGen
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

@btime get_choices(run_mh(tr))

# for i=1:1000
#     mh(tr, Gen.select(:x))
#     mh(tr, Gen.select(:y))
# end