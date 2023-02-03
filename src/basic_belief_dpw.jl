# TODO
# 2. Implement updater choice TrueStateMergingUpdater
# 3. Implement sample_pw_belief=false

POMDPs.solve(solver::BasicBeliefDPWSolver, mdp::Union{POMDP, MDP}) =
    BasicBeliefDPWPlanner(solver, mdp)

"""
Delete existing decision tree.
"""
function clear_tree!(p::BasicBeliefDPWPlanner)
    p.tree = nothing
end

"""
Construct an MCTSBasicBeliefDPW tree and choose the best action.
"""
POMDPs.action(p::BasicBeliefDPWPlanner, b) = first(action_info(p, b))

"""
Construct an MCTSBasicBeliefDPW tree and choose the best action. Also output some information.
"""
function POMDPTools.action_info(
    p::BasicBeliefDPWPlanner{P, UP, B, A},
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
        tree = BasicBeliefDPWTree{B, A}(p.solver.n_iterations)
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

        sanode = best_sanode(p.criterion, tree, bnode)
        # end
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
Return the reward for one iteration of MCTSBasicBeliefDPW.
"""
function simulate(dpw::BasicBeliefDPWPlanner, snode::Int, d::Int, s, b)
    # S = statetype(dpw.pomdp)
    # A = actiontype(dpw.pomdp)
    sol = dpw.solver
    tree = dpw.tree
    # b = tree.b_labels[snode]
    # dpw.reset_callback(dpw.pomdp, b) # Optional: used to reset/reinitialize MDP to a given state.
    if isterminal(dpw.pomdp, s)
        return 0.0
    elseif d == 0
        return estimate_value(
            dpw.solved_estimate,
            GenerativeBeliefMDP(dpw.pomdp, dpw.updater),
            b,
            d,
        )
    end

    # action progressive widening
    if dpw.solver.enable_action_pw
        if length(tree.children[snode]) <= sol.k_action * tree.total_n[snode]^sol.alpha_action # criterion for new action generation
            a = next_action(dpw.next_action, dpw.pomdp, s, DPWStateNode(tree, snode)) # action generation step
            if !sol.check_repeat_action || !haskey(tree.a_lookup, (snode, a))
                n0 = init_N(sol.init_N, dpw.pomdp, s, a)
                insert_action_node!(tree, snode, a, n0,
                    init_Q(sol.init_Q, dpw.pomdp, s, a),
                    sol.check_repeat_action,
                )
                tree.total_n[snode] += n0
            end
        end
    elseif isempty(tree.children[snode])
        for a in actions(dpw.pomdp, s)
            n0 = init_N(sol.init_N, dpw.pomdp, s, a)
            insert_action_node!(tree, snode, a, n0,
                init_Q(sol.init_Q, dpw.pomdp, s, a),
                false)
            tree.total_n[snode] += n0
        end
    end

    banode = best_sanode(dpw.ucb, tree, snode)
    a = tree.a_labels[banode]

    # state progressive widening
    new_node = false
    if (
        dpw.solver.enable_state_pw &&
        tree.n_a_children[banode] <= sol.k_state * tree.n[banode]^sol.alpha_state
    ) || tree.n_a_children[banode] == 0
        # new
        sp, r, o = @gen(:sp, :r, :o)(dpw.pomdp, s, a, dpw.rng)
        bp = update(dpw.updater, b, a, o)

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
        bp = tree.b_labels[bpnode]
        sp = rand(dpw.rng, bp)
    end

    if new_node
        future =
            discount(dpw.pomdp) * estimate_value(
                dpw.solved_estimate,
                GenerativeBeliefMDP(dpw.pomdp, dpw.updater),
                bp,
                d - 1,
            )
    else
        future = discount(dpw.pomdp) * simulate(dpw, bpnode, d - 1, sp, bp)
    end
    q = r + future

    tree.n[banode] += 1
    tree.total_n[snode] += 1
    tree.q[banode] += (q - tree.q[banode]) / tree.n[banode]

    return q
end
