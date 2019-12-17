import NamedTupleTools: namedtuple

getN(nt::NamedTuple{N,T}) where {N,T} = N
getN(::Type{NamedTuple{N,T}}) where {N,T} = N

getT(nt::NamedTuple{N,T}) where {N,T} = T
getT(::Type{NamedTuple{N,T}}) where {N,T} = tuple(T.types...)

NTtypes(NT::Type{NamedTuple{N,T}}) where {N,T} = namedtuple(getN(NT),getT(NT))


function variables(m::Model{A,B,M}) where {A,B,M}
    tuple(Soss.variables(m)...)
end

function variables(::Type{Model{A,B,M}}) where {A,B,M}
    variables(Soss.type2model(Model{A,B,M}))
end
