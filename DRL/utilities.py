from jax import numpy as jnp

def loss_tp_tn(prev_state, prev_action, prev_results, current_state):
    weight = 0.7
    acc_frac = weight * prev_results.tn_frac + prev_results.tp_frac
    return 1 - jnp.sum(prev_state.pr_G * acc_frac)

def loss_tp_frac(prev_state, prev_action, prev_results, current_state):
    return 1 - jnp.sum(prev_state.pr_G * prev_results.tp_frac)

def loss_qr(prev_state, prev_action, prev_results, current_state):
    """
    Negative total classifier accuracy after action taken.
    """
    return 1-jnp.sum(prev_state.pr_G * current_state.pr_Y1)

def loss_qr_ar(prev_state, prev_action, prev_results, current_state):
    weight = 0.5
    qr = jnp.sum(prev_state.pr_G * current_state.pr_Y1)
    ar = jnp.sum(prev_state.pr_G * prev_results.accept_rate)
    return weight * (1 - qr) + (1 - weight) * (1 - ar)

def tpr_loss(prev_state, prev_action, prev_results, current_state):
    weight = 0.5
    qr = jnp.sum(prev_state.pr_G * current_state.pr_Y1)
    ar = jnp.sum(prev_state.pr_G * prev_results.accept_rate)
    return weight * (1 - qr) + (1 - weight) * (1 - ar)

def equal_opportunity_disparity(prev_state, prev_action, prev_results, current_state):
    v = prev_results.tp_rate
    return jnp.sum((v[:, jnp.newaxis] - v) ** 2) / 2

def equalized_odds_disparity(prev_state, prev_action, prev_results, current_state):
    v1 = prev_results.tp_rate
    v2 = prev_results.tn_rate
    dis_v1 = jnp.sum((v1[:, jnp.newaxis] - v1) ** 2) / 2
    dis_v2 = jnp.sum((v2[:, jnp.newaxis] - v2) ** 2) / 2
    return 0.5 * dis_v1 + 0.5 * dis_v2

# tpf + fnf + tnf + fpf

# fn + fp error
# tp + tn acc
