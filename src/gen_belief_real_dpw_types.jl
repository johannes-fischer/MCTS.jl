"""
MCTS solver with RealDPW

Fields:

    depth::Int64
        Maximum rollout horizon and tree depth.
        default: 10

    exploration_constant::Float64
        Specified how much the solver should explore.
        In the UCB equation, Q + c*sqrt(log(t/N)), c is the exploration constant.
        default: 1.0

    n_iterations::Int64
        Number of iterations during each action() call.
        default: 100

    max_time::Float64
        Maximum amount of CPU time spent iterating through simulations.
        default: Inf

    k_action::Float64
    alpha_action::Float64
    k_state::Float64
    alpha_state::Float64
        These constants control the double progressive widening. A new state
        or action will be added if the number of children is less than or equal to kN^alpha.
        defaults: k:10, alpha:0.5

    keep_tree::Bool
        If true, store the tree in the planner for reuse at the next timestep (and every time it is used in the future). There is a computational cost for maintaining the state dictionary necessary for this.
        default: false

    enable_action_pw::Bool
        If true, enable progressive widening on the action space; if false just use the whole action space.
        default: true

    enable_state_pw::Bool
        If true, enable progressive widening on the state space; if false just use the single next state (for deterministic problems).
        default: true

    check_repeat_state::Bool
    check_repeat_action::Bool
        When constructing the tree, check whether a state or action has been seen before (there is a computational cost to maintaining the dictionaries necessary for this)
        default: true

    tree_in_info::Bool
        If true, return the tree in the info dict when action_info is called. False by default because it can use a lot of memory if histories are being saved.
        default: false

    rng::AbstractRNG
        Random number generator

    estimate_value::Any (rollout policy)
        Function, object, or number used to estimate the value at the leaf nodes.
        If this is a function `f`, `f(mdp, s, depth)` will be called to estimate the value.
        If this is an object `o`, `estimate_value(o, mdp, s, depth)` will be called.
        If this is a number, the value will be set to that number.
        default: RolloutEstimator(RandomSolver(rng))

    init_Q::Any
        Function, object, or number used to set the initial Q(s,a) value at a new node.
        If this is a function `f`, `f(mdp, s, a)` will be called to set the value.
        If this is an object `o`, `init_Q(o, mdp, s, a)` will be called.
        If this is a number, Q will always be set to that number.
        default: 0.0

    init_N::Any
        Function, object, or number used to set the initial N(s,a) value at a new node.
        If this is a function `f`, `f(mdp, s, a)` will be called to set the value.
        If this is an object `o`, `init_N(o, mdp, s, a)` will be called.
        If this is a number, N will always be set to that number.
        default: 0

    next_action::Any
        Function or object used to choose the next action to be considered for progressive widening.
        The next action is determined based on the MDP, the state, `s`, and the current `GenBeliefRealDPWStateNode`, `snode`.
        If this is a function `f`, `f(mdp, s, snode)` will be called to set the value.
        If this is an object `o`, `next_action(o, mdp, s, snode)` will be called.
        default: RandomActionGenerator(rng)

    default_action::Any
        Function, action, or Policy used to determine the action if POMCP fails with exception `ex`.
        If this is a Function `f`, `f(pomdp, belief, ex)` will be called.
        If this is a Policy `p`, `action(p, belief)` will be called.
        If it is an object `a`, `default_action(a, pomdp, belief, ex)` will be called, and if this method is not implemented, `a` will be returned directly.
        default: `ExceptionRethrow()`

    reset_callback::Function
        Function used to reset/reinitialize the MDP to a given state `s`.
        Useful when the simulator state is not truly separate from the MDP state.
        `f(mdp, s)` will be called.
        default: `(mdp, s)->false` (optimized out)

    show_progress::Bool
        Show progress bar during simulation.
        default: false

    timer::Function:
        Timekeeping method. Search iterations ended when `timer() - start_time ≥ max_time`.
"""
mutable struct GenBeliefRealDPWSolver <: AbstractMCTSSolver
    updater::Updater
    depth::Int
    ucb::Any
    criterion::Any
    n_iterations::Int
    max_time::Float64
    k_action::Float64
    alpha_action::Float64
    k_state::Float64
    alpha_state::Float64
    keep_tree::Bool
    enable_action_pw::Bool
    enable_state_pw::Bool
    check_repeat_state::Bool
    check_repeat_action::Bool
    sample_pw_belief::Bool
    tree_in_info::Bool
    rng::AbstractRNG
    estimate_value::Any
    init_Q::Any
    init_N::Any
    next_action::Any
    default_action::Any
    reset_callback::Function
    show_progress::Bool
    timer::Function
end

"""
    GenBeliefRealDPWSolver()

Use keyword arguments to specify values for the fields
"""
function GenBeliefRealDPWSolver(;
    updater::Updater,
    depth::Int=10,
    ucb::Any=MaxUCB(1.0),
    criterion::Any=MaxQ(),
    n_iterations::Int=100,
    max_time::Float64=Inf,
    k_action::Float64=10.0,
    alpha_action::Float64=0.5,
    k_state::Float64=10.0,
    alpha_state::Float64=0.5,
    keep_tree::Bool=false,
    enable_action_pw::Bool=true,
    enable_state_pw::Bool=true,
    check_repeat_state::Bool=true,
    check_repeat_action::Bool=true,
    sample_pw_belief::Bool=true,
    tree_in_info::Bool=false,
    rng::AbstractRNG=Random.GLOBAL_RNG,
    estimate_value::Any=(args...) -> error("estimate_value has no default value"),
    init_Q::Any=0.0,
    init_N::Any=0,
    next_action::Any=RandomActionGenerator(rng),
    default_action::Any=ExceptionRethrow(),
    reset_callback::Function=(mdp, s) -> false,
    show_progress::Bool=false,
    timer=() -> 1e-9 * time_ns())
    GenBeliefRealDPWSolver(
        updater,
        depth,
        ucb,
        criterion,
        n_iterations,
        max_time,
        k_action,
        alpha_action,
        k_state,
        alpha_state,
        keep_tree,
        enable_action_pw,
        enable_state_pw,
        check_repeat_state,
        check_repeat_action,
        sample_pw_belief,
        tree_in_info,
        rng,
        estimate_value,
        init_Q,
        init_N,
        next_action,
        default_action,
        reset_callback,
        show_progress,
        timer,
    )
end

#=
mutable struct StateActionStateNode
    N::Int
    R::Float64
    StateActionStateNode() = new(0,0)
end

mutable struct RealDPWStateActionNode{S}
    V::Dict{S,StateActionStateNode}
    N::Int
    Q::Float64
    RealDPWStateActionNode(N,Q) = new(Dict{S,StateActionStateNode}(), N, Q)
end

mutable struct RealDPWStateNode{S,A} <: AbstractStateNode
    A::Dict{A,RealDPWStateActionNode{S}}
    N::Int
    RealDPWStateNode{S,A}() where {S,A} = new(Dict{A,RealDPWStateActionNode{S}}(),0)
end
=#

mutable struct GenBeliefRealDPWTree{B, A}
    # for each state node
    total_n::Vector{Int}
    children::Vector{Vector{Int}}
    b_labels::Vector{B}
    b_lookup::Dict{B, Int}

    # for each state-action node
    n::Vector{Int}
    q::Vector{Float64}
    prior::Vector{Float64}
    q_init::Vector{Float64}
    transitions::Vector{Vector{Tuple{Int, Float64}}}
    a_labels::Vector{A}
    a_lookup::Dict{Tuple{Int, A}, Int}

    # for tracking transitions
    n_a_children::Vector{Int}
    unique_transitions::Set{Tuple{Int, Int}}

    function GenBeliefRealDPWTree{B, A}(sz::Int=1000) where {B, A}
        sz = min(sz, 100_000)
        return new(sizehint!(Int[], sz),
            sizehint!(Vector{Int}[], sz),
            sizehint!(B[], sz),
            Dict{B, Int}(),
            sizehint!(Int[], sz),
            sizehint!(Float64[], sz),
            sizehint!(Float64[], sz),
            sizehint!(Float64[], sz),
            sizehint!(Vector{Tuple{Int, Float64}}[], sz),
            sizehint!(A[], sz),
            Dict{Tuple{Int, A}, Int}(), sizehint!(Int[], sz),
            Set{Tuple{Int, Int}}(),
        )
    end
end

function insert_belief_node!(
    tree::GenBeliefRealDPWTree{B, A},
    b::B,
    maintain_s_lookup=true,
) where {B, A}
    push!(tree.total_n, 0)
    push!(tree.children, Int[])
    push!(tree.b_labels, b)
    snode = length(tree.total_n)
    if maintain_s_lookup
        tree.b_lookup[b] = snode
    end
    return snode
end

function insert_action_node!(
    tree::GenBeliefRealDPWTree{B, A},
    snode::Int,
    a::A,
    n0::Int,
    q0::Float64,
    p0::Float64,
    maintain_a_lookup=true,
) where {B, A}
    push!(tree.n, n0)
    push!(tree.q, q0)
    push!(tree.prior, p0)
    push!(tree.q_init, q0)
    push!(tree.a_labels, a)
    push!(tree.transitions, Vector{Tuple{Int, Float64}}[])
    banode = length(tree.n)
    push!(tree.children[snode], banode)
    push!(tree.n_a_children, 0)
    if maintain_a_lookup
        tree.a_lookup[(snode, a)] = banode
    end
    return banode
end

Base.isempty(tree::GenBeliefRealDPWTree) = isempty(tree.n) && isempty(tree.q)

# struct RealDPWBeliefNode{S,A} <: AbstractStateNode
#     tree::GenBeliefRealDPWTree{S,A}
#     index::Int
# end

# children(n::RealDPWBeliefNode) = n.tree.children[n.index]
# n_children(n::RealDPWBeliefNode) = length(children(n))
# isroot(n::RealDPWBeliefNode) = n.index == 1

mutable struct GenBeliefRealDPWPlanner{P <: POMDP, UP, B, A, SE, NA, RCB, RNG, UCB, CRIT} <:
               AbstractMCTSPlanner{P}
    solver::GenBeliefRealDPWSolver
    updater::UP
    pomdp::P
    ucb::UCB
    criterion::CRIT
    tree::Union{Nothing, GenBeliefRealDPWTree{B, A}}
    solved_estimate::SE
    next_action::NA
    reset_callback::RCB
    rng::RNG
end

function GenBeliefRealDPWPlanner(solver::GenBeliefRealDPWSolver, pomdp::P) where {P <: POMDP}
    se = convert_estimator(solver.estimate_value, solver, pomdp)
    B = typeof(initialize_belief(solver.updater, initialstate(pomdp)))
    return GenBeliefRealDPWPlanner{
        P,
        typeof(solver.updater),
        B,
        actiontype(P),
        typeof(se),
        typeof(solver.next_action),
        typeof(solver.reset_callback),
        typeof(solver.rng),
        typeof(solver.ucb),
        typeof(solver.criterion)
    }(solver,
        solver.updater,
        pomdp,
        solver.ucb,
        solver.criterion,
        nothing,
        se,
        solver.next_action,
        solver.reset_callback,
        solver.rng,
    )
end

Random.seed!(p::GenBeliefRealDPWPlanner, seed) = Random.seed!(p.rng, seed)
