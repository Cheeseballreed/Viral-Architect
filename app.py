import numpy as np
import streamlit as st

st.set_page_config(page_title="Viral Architect – Absorbing Markov Lab", layout="wide")

st.title("The Viral Architect – Absorbing Markov Chain Dashboard")

st.markdown(
    """
This app models a 6-state social media funnel with absorbing states using an absorbing Markov chain.
States (by index):

1. Newbie  
2. Casual  
3. Power User  
4. Community Leader  
5. Verified Legend (absorbing)  
6. Deleted Account (absorbing)
"""
)

st.subheader("Step 1 – Base Transition Matrix P (6×6)")

st.markdown(
    """
Enter a valid stochastic matrix (each row sums to 1).  
Default values match the scenario in the prompt.
"""
)

default_P = np.array([
    [0.40, 0.20, 0.00, 0.00, 0.00, 0.40],  # S1: Newbie
    [0.00, 0.30, 0.30, 0.00, 0.00, 0.40],  # S2: Casual
    [0.00, 0.00, 0.30, 0.50, 0.00, 0.20],  # S3: Power
    [0.00, 0.00, 0.00, 0.30, 0.60, 0.10],  # S4: Community
    [0.00, 0.00, 0.00, 0.00, 1.00, 0.00],  # S5: Legend (absorbing)
    [0.00, 0.00, 0.00, 0.00, 0.00, 1.00],  # S6: Deleted (absorbing)
])

state_labels = [
    "S1 Newbie",
    "S2 Casual",
    "S3 Power",
    "S4 Community",
    "S5 Legend",
    "S6 Deleted",
]

# Editable grid: one line per row with comma-separated probabilities
edited_rows = []
for i in range(6):
    row_str = st.text_input(
        f"Row {i+1} – {state_labels[i]} (comma-separated 6 numbers)",
        value=", ".join(f"{x:.2f}" for x in default_P[i]),
        key=f"row_{i}",
    )
    try:
        row_vals = [float(x.strip()) for x in row_str.split(",")]
        if len(row_vals) != 6:
            st.error(f"Row {i+1} must have exactly 6 entries.")
        edited_rows.append(row_vals)
    except Exception:
        st.error(f"Invalid numeric values in row {i+1}.")
        edited_rows.append(default_P[i].tolist())

P = np.array(edited_rows, dtype=float)

# Normalize small numerical drift
row_sums = P.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
P = P / row_sums

st.markdown("**Current P matrix**")
st.dataframe(
    P,
    use_container_width=True
)

# Identify absorbing vs transient
absorbing = np.isclose(P[np.arange(6), np.arange(6)], 1.0) & np.all(
    np.isclose(P - np.eye(6), 0.0), axis=1
)

transient_states = [i for i, a in enumerate(absorbing) if not a]
absorbing_states = [i for i, a in enumerate(absorbing) if a]

st.subheader("Step 2 – State Types")
st.write("**Transient states (indices are 0-based in numpy):**",
         [f"{i+1}: {state_labels[i]}" for i in transient_states])
st.write("**Absorbing states (indices are 0-based in numpy):**",
         [f"{i+1}: {state_labels[i]}" for i in absorbing_states])

if len(absorbing_states) == 0:
    st.error("No absorbing states detected. At least one absorbing state is required.")
    st.stop()

t = len(transient_states)
r = len(absorbing_states)

# Reorder P so all transient states come first, absorbing last
order = transient_states + absorbing_states
P_reordered = P[np.ix_(order, order)]

Q = P_reordered[:t, :t]
R = P_reordered[:t, t:]

st.subheader("Step 3 – Q and R matrices")
st.markdown("**Q – transient-to-transient (top-left)**")
st.dataframe(Q, use_container_width=True)
st.markdown("**R – transient-to-absorbing (top-right)**")
st.dataframe(R, use_container_width=True)

st.subheader("Step 4 – Fundamental Matrix F and Absorption Matrix B")

I = np.eye(t)
try:
    F = np.linalg.inv(I - Q)
except np.linalg.LinAlgError:
    st.error("Matrix (I - Q) is not invertible; cannot compute F.")
    st.stop()

B = F @ R

st.markdown("**F = (I − Q)⁻¹ (expected total visits among transient states)**")
st.dataframe(F, use_container_width=True)

st.markdown("**B = F × R (absorption probabilities)**")
st.dataframe(B, use_container_width=True)

# Map back to original indices for interpretation
# Find index of S1 Newbie and S5 Legend in the reordered system
try:
    idx_newbie = order.index(0)      # original index 0
except ValueError:
    idx_newbie = None

try:
    idx_legend_abs = absorbing_states.index(4)  # original index 4
    legend_col = idx_legend_abs                 # in R/B columns
except ValueError:
    legend_col = None

success_prob = None
if (idx_newbie is not None) and (legend_col is not None) and (idx_newbie < t):
    success_prob = B[idx_newbie, legend_col]
    st.markdown(
        f"### Success metric – P(Newbie → Legend)\n"
        f"**Estimated probability:** `{success_prob:.4f}`"
    )
else:
    st.warning("Could not identify Newbie or Legend in the current configuration to compute the success metric.")

# Life expectancy (starting from Newbie)
if idx_newbie is not None and idx_newbie < t:
    life_expectancy = F[idx_newbie, :].sum()
    st.markdown(
        f"**Life expectancy (expected steps before absorption for Newbie):** `{life_expectancy:.4f}`"
    )

# Bottleneck: state with largest F entry from Newbie
if idx_newbie is not None and idx_newbie < t:
    bottleneck_idx = np.argmax(F[idx_newbie, :])
    bottleneck_state_original = order[bottleneck_idx]
    st.markdown(
        f"**Bottleneck (most time spent, starting from Newbie):** "
        f"{state_labels[bottleneck_state_original]} "
        f"(index {bottleneck_state_original+1}), "
        f"expected visits ≈ `{F[idx_newbie, bottleneck_idx]:.4f}`"
    )

st.subheader("Step 5 – What-if: reduce Newbie dropout")

st.markdown(
    """
Adjust the slider to *reduce* the probability that Newbies drop out in one step.  
The freed probability mass is added to “progress to Casual” (S1 → S2).
"""
)

delta = st.slider(
    "Reduction in S1→Deleted probability (absolute amount)",
    min_value=0.0,
    max_value=float(P[0, 5]),
    value=0.1,
    step=0.01,
)

P_whatif = P.copy()
original_drop = P_whatif[0, 5]
P_whatif[0, 5] = original_drop - delta
P_whatif[0, 1] += delta  # push more Newbies toward Casual

st.markdown("**What-if P matrix (Newbie dropout reduced):**")
st.dataframe(P_whatif, use_container_width=True)

# Recompute Q,R,F,B for what-if
P_reordered_w = P_whatif[np.ix_(order, order)]
Q_w = P_reordered_w[:t, :t]
R_w = P_reordered_w[:t, t:]

try:
    F_w = np.linalg.inv(np.eye(t) - Q_w)
    B_w = F_w @ R_w
except np.linalg.LinAlgError:
    st.error("For the what-if scenario, (I - Q) is not invertible.")
    st.stop()

if (idx_newbie is not None) and (legend_col is not None) and (idx_newbie < t):
    success_prob_w = B_w[idx_newbie, legend_col]
    st.markdown(
        f"**What-if success metric – P(Newbie → Legend) with reduced dropout:** "
        f"`{success_prob_w:.4f}` (baseline `{success_prob:.4f}`)"
    )
    if success_prob is not None:
        st.markdown(
            f"**Incremental Legends per cohort of 1000 Newbies:** "
            f"`{(success_prob_w - success_prob)*1000:.1f}`"
        )
