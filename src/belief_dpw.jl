# TODO
# 2. Implement updater choice TrueStateMergingUpdater
# 3. Implement sample_pw_belief=false

POMDPs.solve(solver::BeliefDPWSolver, mdp::Union{POMDP, MDP}) =
    BeliefDPWPlanner(solver, mdp)

"""
Delete existing decision tree.
"""
function clear_tree!(p::BeliefDPWPlanner)
    p.tree = nothing
end

"""
Construct an MCTSBeliefDPW tree and choose the best action.
"""
POMDPs.action(p::BeliefDPWPlanner, b) = first(action_info(p, b))

"""
Construct an MCTSBeliefDPW tree and choose the best action. Also output some information.
"""
function POMDPModelTools.action_info(
    p::BeliefDPWPlanner{P, UP, B, A},
    b;
    tree_in_info=false,
) where {P, UP, B, A}
    local a::actiontype(p.pomdp)
    info = Dict{Symbol, Any}()
    try
        if isterminal(p.pomdp, b)
            error("""
                  MCTS cannot handle terminal states. action was called with
                  b = $b
                  """)
        end

        # if p.solver.keep_tree && p.tree != nothing
        #     tree = p.tree
        #     if haskey(tree.s_lookup, b)
        #         snode = tree.s_lookup[b]
        #     else
        #         snode = insert_belief_node!(tree, b, true)
        #     end
        # else
        tree = BeliefDPWTree{B, A}(p.solver.n_iterations)
        p.tree = tree
        bnode = insert_belief_node!(tree, b, p.solver.check_repeat_state)
        # end

        timer = p.solver.timer
        p.solver.show_progress ? progress = Progress(p.solver.n_iterations) : nothing
        nquery = 0
        start_s = timer()
        for i in 1:p.solver.n_iterations
            nquery += 1
            s = rand(p.rng, b)
            simulate(p, bnode, p.solver.depth, s, b) # (not 100% sure we need to make a copy of the state here)
            p.solver.show_progress ? next!(progress) : nothing
            if timer() - start_s >= p.solver.max_time
                p.solver.show_progress ? finish!(progress) : nothing
                break
            end
        end
        p.reset_callback(p.pomdp, b) # Optional: leave the MDP in the current state.
        info[:search_time] = timer() - start_s
        info[:tree_queries] = nquery
        if p.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end

        sanode = best_sanode(tree, bnode)
        a = tree.a_labels[sanode] # choose action with highest approximate value
    catch ex
        a = convert(
            actiontype(p.pomdp),
            default_action(p.solver.default_action, p.pomdp, b, ex),
        )
        info[:exception] = ex
    end

    return a, info
end

"""
Return the reward for one iteration of MCTSBeliefDPW.
"""
function simulate(dpw::BeliefDPWPlanner, snode::Int, d::Int, s, b)
    # S = statetype(dpw.mdp)
    # A = actiontype(dpw.mdp)
    sol = dpw.solver
    tree = dpw.tree
    # b = tree.b_labels[snode]
    # dpw.reset_callback(dpw.mdp, b) # Optional: used to reset/reinitialize MDP to a given state.
    if isterminal(dpw.pomdp, b)
        return 0.0
    elseif d == 0
        return maximum(MCTS.estimate_q_value(dpw.solved_estimate, dpw.pomdp, b, d))
    end

    # action progressive widening
    # if dpw.solver.enable_action_pw
    #     if length(tree.children[snode]) <= sol.k_action*tree.total_n[snode]^sol.alpha_action # criterion for new action generation
    #         a = next_action(dpw.next_action, dpw.mdp, b, DPWBeliefNode(tree, snode)) # action generation step
    #         if !sol.check_repeat_action || !haskey(tree.a_lookup, (snode, a))
    #             n0 = init_N(sol.init_N, dpw.mdp, b, a)
    #             insert_action_node!(tree, snode, a, n0,
    #                                 init_Q(sol.init_Q, dpw.mdp, b, a),
    #                                 sol.check_repeat_action
    #                                )
    #             tree.total_n[snode] += n0
    #         end
    #     end
    # else
    if isempty(tree.children[snode])
        q_vals = MCTS.estimate_q_value(dpw.solved_estimate, dpw.pomdp, b, d)
        if d  >= dpw.solver.depth - 1
            @show q_vals
        end
        @assert length(q_vals) == length(actions(dpw.pomdp, b))
        prior = exp.(q_vals)
        prior /= sum(prior)

        for (a, q0, p0) in zip(actions(dpw.pomdp, b), q_vals, prior)
            # n0 = init_N(sol.init_N, dpw.mdp, b, a)
            n0 = 1
            insert_action_node!(tree, snode, a, n0,
                q0,
                # init_Q(sol.init_Q, dpw.mdp, b, a),
                p0,
                false)
            tree.total_n[snode] += n0
        end
    end

    banode = best_sanode_UCB(tree, snode, sol.exploration_constant)
    a = tree.a_labels[banode]

    # new
    sp, r, o = @gen(:sp, :r, :o)(dpw.pomdp, s, a, dpw.rng)
    bp = update(dpw.updater, b, a, o)

    # state progressive widening
    new_node = false
    if (
        dpw.solver.enable_state_pw &&
        tree.n_a_children[banode] <= sol.k_state * tree.n[banode]^sol.alpha_state
    ) || tree.n_a_children[banode] == 0
        # sp, r = @gen(:sp, :r)(dpw.pomdp, b, a, dpw.rng)

        # if sol.check_repeat_state && haskey(tree.b_lookup, sp)
        #     spnode = tree.b_lookup[sp]
        # else
        bpnode = insert_belief_node!(tree, bp, sol.keep_tree || sol.check_repeat_state)
        new_node = true
        # end

        push!(tree.transitions[banode], (bpnode, r))

        if !sol.check_repeat_state
            tree.n_a_children[banode] += 1
        elseif !((banode, bpnode) in tree.unique_transitions)
            push!(tree.unique_transitions, (banode, bpnode))
            tree.n_a_children[banode] += 1
        end
    else
        bpnode, r = rand(dpw.rng, tree.transitions[banode])
    end

    if new_node
        q = r + discount(dpw.pomdp)*maximum(MCTS.estimate_q_value(dpw.solved_estimate, dpw.pomdp, bp, d-1))
    else
        q = r + discount(dpw.pomdp)*simulate(dpw, bpnode, d-1, sp, bp)
    end

    tree.n[banode] += 1
    tree.total_n[snode] += 1
    tree.q[banode] += (q - tree.q[banode]) / tree.n[banode]

    return q
end

"""
Return the best action.

Some publications say to choose action that has been visited the most
e.g., Continuous Upper Confidence Trees by Couëtoux et al.
"""
function best_sanode(tree::BeliefDPWTree, snode::Int)
    best_Q = -Inf
    sanode = 0
    for child in tree.children[snode]
        if tree.q[child] > best_Q
            best_Q = tree.q[child]
            sanode = child
        end
    end
    return sanode
end

"""
Return the best action node based on an alternative UCB score used in AlphaZero with exploration constant c
"""
function best_sanode_UCB(tree::BeliefDPWTree, snode::Int, c::Float64)
    best_UCB = -Inf
    sanode = 0
    sqrtN = sqrt(tree.total_n[snode])
    for child in tree.children[snode]
        n = tree.n[child]
        q = tree.q[child]
        p = tree.prior[child]
        if c == 0.0
            UCB = q
        else
            UCB = q + c * p * sqrtN / (1 + n)
        end
        @assert !isnan(UCB) "UCB was NaN (q=$q, c=$c, p=$p, ltn=$sqrtN, n=$n)"
        @assert !isequal(UCB, -Inf)
        if UCB > best_UCB
            best_UCB = UCB
            sanode = child
        end
    end
    return sanode
end
