module SossGen

using Soss
import Gen
import NamedTupleTools: namedtuple

struct SossTrace <: Gen.Trace
  gen_fn
  choices
  logprob
  args
end

struct SossGenerativeFunction <: Gen.GenerativeFunction{NamedTuple,SossTrace}
  model
end

function Gen.simulate(gen_fn::SossGenerativeFunction, args::Tuple)
    choices = rand(gen_fn.model(args...))
    score = logpdf(gen_fn.model(args...), choices)
    SossTrace(gen_fn, choices, score, args)
end

function Gen.generate(gen_fn::SossGenerativeFunction, args::Tuple, constraints::Gen.ChoiceMap)
  kvs = Gen.get_values_shallow(constraints)
  data = namedtuple(Dict{Symbol, Any}(kvs))
  weight, choices = weightedSample(gen_fn.model(args...), data)
  logprob = logpdf(gen_fn.model(args...), choices)
  SossTrace(gen_fn, choices, logprob, args), weight
end

Gen.project(t::SossTrace, ::Gen.EmptySelection) = 0.0

function Gen.get_retval(t::SossTrace)
  t.choices
end

function Gen.get_args(t::SossTrace)
  t.args
end

function Gen.get_score(t::SossTrace)
  t.logprob
end

function Gen.get_choices(t::SossTrace)
  c = Gen.choicemap()
  for (k, v) in pairs(t.choices)
    c[k] = v
  end
  c
end


end 
