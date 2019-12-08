module SossGen

using Reexport

@reexport using Soss
@reexport using Gen 
const logpdf = Soss.logpdf
const ifelse = Base.ifelse

import Gen
import NamedTupleTools: namedtuple

getN(nt::NamedTuple{N,T}) where {N,T} = N
getN(::Type{NamedTuple{N,T}}) where {N,T} = N

getT(nt::NamedTuple{N,T}) where {N,T} = T
getT(::Type{NamedTuple{N,T}}) where {N,T} = tuple(T.types...)

NTtypes(NT::Type{NamedTuple{N,T}}) where {N,T} = namedtuple(getN(NT),getT(NT))

struct SossTrace{F, NT} <: Gen.Trace
  gen_fn :: F
  choices :: NT
  logprob :: Float64
  args
end

function variables(m::Model{A,B,M}) where {A,B,M}
    tuple(Soss.variables(m)...)
end

function variables(::Type{Model{A,B,M}}) where {A,B,M}
    variables(Soss.type2model(Model{A,B,M}))
end

export SossGenerativeFunction
struct SossGenerativeFunction{M,F,V}  <: Gen.GenerativeFunction{V,SossTrace{SossGenerativeFunction}} 
    model  :: M
    logpdf :: F
end

function SossGenerativeFunction(m::Model{A,B,M}, ::typeof(codegen)) where {A,B,M}
    ℓ(args, data) = logpdf(m(args...), data, codegen)::Float64
    SossGenerativeFunction{Model{A,B,M},typeof(ℓ), typeof(variables(m))}(m, ℓ)
end

function SossGenerativeFunction(m::Model{A,B,M}) where {A,B,M}
    ℓ(args, data) = logpdf(m(args...), data)::Float64
    SossGenerativeFunction{Model{A,B,M},typeof(ℓ), typeof(variables(m))}(m, ℓ)
end

function Gen.simulate(gen_fn::SossGenerativeFunction{M,F,V}, args::Tuple) where {M,F,V}
    choices = rand(gen_fn.model(args...))
    score = gen_fn.logpdf(args, choices)
    SossTrace{SossGenerativeFunction,NT}(gen_fn, choices, score, args)
end

function Gen.generate(gen_fn::SossGenerativeFunction{M,F,V}, args::Tuple, constraints::Gen.ChoiceMap) where {M,F,V}
  kvs = Gen.get_values_shallow(constraints)
  data = namedtuple(Dict{Symbol, Any}(kvs))
  weight, choices = weightedSample(gen_fn.model(args...), data)
  logprob = gen_fn.logpdf(args, choices)
  SossTrace{SossGenerativeFunction,typeof(choices)}(gen_fn, choices, logprob, args), weight
end

Gen.project(t::SossTrace{SossGenerativeFunction,NT}, ::Gen.EmptySelection) where {NT} = 0.0

function Gen.get_retval(t::SossTrace{SossGenerativeFunction,NT}) where {NT}
  t.choices
end

function Gen.get_args(t::SossTrace{SossGenerativeFunction,NT}) where {NT}
  t.args
end

function Gen.get_score(t::SossTrace{SossGenerativeFunction,NT}) where {NT}
  t.logprob
end


function Gen.get_choices(t::SossTrace{SossGenerativeFunction,NT}) where {NT}
  c = Gen.choicemap()
  Ts = NTtypes(NT)
  for (k::Symbol, v) in pairs(t.choices)
    c[k] = v :: Ts[k]
  end
  c
end

function Gen.update(t::SossTrace{SossGenerativeFunction,NT}, new_args::Tuple, argdiffs::Tuple, constraints::Gen.ChoiceMap) where {NT}
    retdiff = ifelse(isempty(constraints), Gen.NoChange(), Gen.UnknownChange())
    new_choices = NT(merge(t.choices, namedtuple(Dict{Symbol,Any}(Gen.get_values_shallow(constraints))))) # :: NT
    new_logprob = t.gen_fn.logpdf(new_args, new_choices) :: Float64
    new_trace = SossTrace{SossGenerativeFunction,NT}(t.gen_fn, new_choices, new_logprob, new_args)
    weight = new_logprob - t.logprob
    discard = Gen.choicemap()
    for (k::Symbol, v) in Gen.get_values_shallow(constraints)
      discard[k] = t.choices[k]
    end
    return (new_trace, weight, retdiff, discard)
end

end 
