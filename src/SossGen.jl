module SossGen

using Soss
import Gen
import NamedTupleTools: namedtuple

struct SossTrace <: Gen.Trace
  gen_fn :: Gen.GenerativeFunction{NamedTuple,SossTrace}
  choices :: NamedTuple
  logprob 
  args
end

export SossGenerativeFunction
struct SossGenerativeFunction <: Gen.GenerativeFunction{NamedTuple,SossTrace}
    model  :: Model
    logpdf :: Function
end

function SossGenerativeFunction(m::Model, ::typeof(codegen))
    ℓ(args, data) = logpdf(m(args...), data, codegen)
    SossGenerativeFunction(m, ℓ)
end

function SossGenerativeFunction(m::Model)
    ℓ(args, data) = logpdf(m(args...), data)
    SossGenerativeFunction(m, ℓ)
end

function Gen.simulate(gen_fn::SossGenerativeFunction, args::Tuple)

    choices = rand(gen_fn.model(args...))
    score = gen_fn.logpdf(args, choices)
    SossTrace(gen_fn, choices, score, args)
end

function Gen.generate(gen_fn::SossGenerativeFunction, args::Tuple, constraints::Gen.ChoiceMap)
  kvs = Gen.get_values_shallow(constraints)
  data = namedtuple(Dict{Symbol, Any}(kvs))
  weight, choices = weightedSample(gen_fn.model(args...), data)
  logprob = gen_fn.logpdf(args, choices)
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

function Gen.update(t::SossTrace, new_args::Tuple, argdiffs::Tuple, constraints::Gen.ChoiceMap)
    retdiff = ifelse(isempty(constraints), Gen.NoChange(), Gen.UnknownChange())
    new_choices = merge(t.choices, namedtuple(Dict{Symbol}(Gen.get_values_shallow(constraints))))
    new_logprob = t.gen_fn.logpdf(new_args, new_choices)
    new_trace = SossTrace(t.gen_fn, new_choices, new_logprob, new_args)
    weight = new_logprob - t.logprob
    discard = Gen.choicemap()
    for (k, v) in Gen.get_values_shallow(constraints)
      discard[k] = t.choices[k]
    end
    return (new_trace, weight, retdiff, discard)
end

end 
