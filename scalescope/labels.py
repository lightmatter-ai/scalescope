"""Common metric and display labels used across reporting and plotting.

These are UI-facing strings intended for tables, JSON reports, and plots.
"""

tp_node_label = "TP_Node"
dp_node_label = "DP_Node"
tp_xpu_count_label = "TP_XPU_Count"
dp_ranks_label = "DP_Ranks"
high_bandwidth_xpu_count_label = "High_Bandwidth_XPU_Count"
total_xpu_count_label = "Total_XPU_Count"
low_bandwidth_ranks_label = "Low_Bandwidth_Ranks"
batch_size_per_dp_label = "Batch_Size_per_DP"
shared_expert_count_label = "Shared_Expert_Count"
regular_expert_count_label = "Routed_Expert_Count"
shared_experts_per_dp_rank_label = "Shared_Experts_Per_DP_Rank"
regular_experts_per_dp_rank_label = "Routed_Experts_Per_DP_Rank"
num_selected_experts_label = "Num_Selected_Experts_Per_Token"
num_experts_per_dp_rank_label = "Num_Experts_Per_DP_Rank"

num_dp_ranks_per_complete_expert_set_label = "Num_DP_Ranks_Per_Complete_Expert_Set"
num_complete_expert_sets_label = "Num_Complete_Expert_Sets"
num_dp_ranks_per_expert_group_label = "Num_DP_Ranks_Per_Expert_Group"

optimizer_label = "Optmizer"
seq_len_label = "Seq_Length"
n_layers_label = "Num_Layers_Per_PP_Stage"
mlp_ratio_label = "MLP_Ratio"
pp_stages_label = "Num_PP_Stages"
model_hidden_dim_label = "Model_Hidden_Dim"
float_size_label = "Float_Size"
topology_label = "Topology"
topology_dims_label = "Topology_Dims"
topology_can_reallocate_bandwidth_label = "Topology_Can_Reallocate_Bandwidth"
tensor_parallel_variant_label = "TP_Variant"

flops_forward_pass_label = "Flops_Forward_Pass"
flops_backward_pass_label = "Flops_Backward_Pass"
compute_time_forward_pass_label = "Compute_Time_Forward_Pass"
compute_time_backward_pass_label = "Compute_Time_Backward_Pass"
mem_forward_pass_label = "Mem_Forward_Pass"
mem_backward_pass_label = "Mem_Backward_Pass"
mem_time_forward_pass_label = "Mem_Time_Forward_Pass"
mem_time_backward_pass_label = "Mem_Time_Backward_Pass"

comm_vol_forward_pass_label = "Comm_Volume_Forward_Pass"
comm_vol_backward_pass_label = "Comm_Volume_Backward_Pass"
comm_time_forward_pass_label = "Comm_Time_Forward_Pass"
comm_time_backward_pass_label = "Comm_Time_Backward_Pass"

# Detailed Comm labels
comm_vol_tp_attn_fwd_label = "Comm_Volume_TP_Attn_Fwd"
comm_vol_tp_moe_fwd_label = "Comm_Volume_TP_MoE_Fwd"
comm_vol_tp_attn_bwd_label = "Comm_Volume_TP_Attn_Bwd"
comm_vol_tp_moe_bwd_label = "Comm_Volume_TP_MoE_Bwd"
comm_vol_dp_attn_fwd_label = "Comm_Volume_DP_Attn_Fwd"
comm_vol_dp_moe_fwd_label = "Comm_Volume_DP_MoE_Fwd"
comm_vol_dp_attn_bwd_label = "Comm_Volume_DP_Attn_Bwd"
comm_vol_dp_moe_bwd_label = "Comm_Volume_DP_MoE_Bwd"

comm_time_tp_attn_fwd_label = "Comm_Time_TP_Attn_Fwd"
comm_time_tp_moe_fwd_label = "Comm_Time_TP_MoE_Fwd"
comm_time_tp_attn_bwd_label = "Comm_Time_TP_Attn_Bwd"
comm_time_tp_moe_bwd_label = "Comm_Time_TP_MoE_Bwd"
comm_time_dp_attn_fwd_label = "Comm_Time_DP_Attn_Fwd"
comm_time_dp_moe_fwd_label = "Comm_Time_DP_MoE_Fwd"
comm_time_dp_attn_bwd_label = "Comm_Time_DP_Attn_Bwd"
comm_time_dp_moe_bwd_label = "Comm_Time_DP_MoE_Bwd"

# TP-attention forward phase breakdown (projection vs core attention)
comm_time_tp_attn_fwd_phase1_label = "Comm_Time_TP_Attn_Fwd_Phase1"
comm_time_tp_attn_fwd_phase2_label = "Comm_Time_TP_Attn_Fwd_Phase2"
comm_time_tp_attn_fwd_phase3_label = "Comm_Time_TP_Attn_Fwd_Phase3"
comm_vol_tp_attn_fwd_phase1_label = "Comm_Volume_TP_Attn_Fwd_Phase1"
comm_vol_tp_attn_fwd_phase2_label = "Comm_Volume_TP_Attn_Fwd_Phase2"
comm_vol_tp_attn_fwd_phase3_label = "Comm_Volume_TP_Attn_Fwd_Phase3"

# Pipeline-parallel (PP) boundary transfers (activation / activation-grad) across stages.
comm_vol_pp_fwd_label = "Comm_Volume_PP_Fwd"
comm_vol_pp_bwd_label = "Comm_Volume_PP_Bwd"
comm_time_pp_fwd_label = "Comm_Time_PP_Fwd"
comm_time_pp_bwd_label = "Comm_Time_PP_Bwd"

moe_to_experts_vol_label = "MoE_To_Experts_Volume"
moe_from_experts_vol_label = "MoE_From_Experts_Volume"
moe_to_experts_time_label = "MoE_To_Experts_Time"
moe_from_experts_time_label = "moe_from_experts_time"

# Backward MoE A2A labels
moe_alltoall_bwd_time_label = "MoE_AllToAll_Bwd_Time"
moe_alltoall_bwd_vol_label = "MoE_AllToAll_Bwd_Volume"

model_param_count_per_dp_rank_label = "Model_Param_Count_Per_DP_Rank"
time_to_train_sec_label = "Time_To_Train_Sec"
time_to_train_day_label = "Time_To_Train_Day"
tokens_per_second_label = "Tokens_Per_Sec"
tokens_per_second_per_xpu_label = "Tokens_Per_Sec_Per_XPU"
# Decode-only TPOT variants:
# - System TPOT: wall-time per *system-level* output token (across the whole batch).
# - User TPOT: wall-time per *user-visible* output token (per sequence). This is typically
#   `TPOT_System_Sec * global_batch_size` and therefore depends on batch size.
tpot_system_sec_label = "TPOT_System_Sec"
tpot_user_sec_label = "TPOT_User_Sec"
ttft_forward_sec_label = "TTFT_Forward_Sec"

compute_time_label = "Compute_Time"
memory_time_label = "Memory_Time"
comm_time_label = "Comm_Time"
comm_compute_ratio_label = "Comm_Compute_Ratio"
memory_compute_ratio_label = "Memory_Compute_Ratio"
roi_label = "ROI"

# Ratios (duplicates kept for compatibility)
comm_compute_ratio_label = "Comm_Compute_Ratio"
memory_compute_ratio_label = "Memory_Compute_Ratio"

# Detailed Compute and Memory labels
compute_time_attn_fwd_label = "Compute_Time_Attn_Fwd"
compute_time_moe_fwd_label = "Compute_Time_MoE_Fwd"
compute_time_attn_bwd_label = "Compute_Time_Attn_Bwd"
compute_time_moe_bwd_label = "Compute_Time_MoE_Bwd"
flops_attn_fwd_label = "FLOPs_Attn_Fwd"
flops_moe_fwd_label = "FLOPs_MoE_Fwd"
flops_attn_bwd_label = "FLOPs_Attn_Bwd"
flops_moe_bwd_label = "FLOPs_MoE_Bwd"

mem_time_attn_fwd_label = "Mem_Time_Attn_Fwd"
mem_time_moe_fwd_label = "Mem_Time_MoE_Fwd"
mem_time_attn_bwd_label = "Mem_Time_Attn_Bwd"
mem_time_moe_bwd_label = "Mem_Time_MoE_Bwd"
mem_vol_attn_fwd_label = "Mem_Volume_Attn_Fwd"
mem_vol_moe_fwd_label = "Mem_Volume_MoE_Fwd"
mem_vol_attn_bwd_label = "Mem_Volume_Attn_Bwd"
mem_vol_moe_bwd_label = "Mem_Volume_MoE_Bwd"

# Embedding/Unembedding detailed labels
compute_time_embedding_fwd_label = "Compute_Time_Embedding_Fwd"
compute_time_unembedding_fwd_label = "Compute_Time_Unembedding_Fwd"
flops_embedding_fwd_label = "FLOPs_Embedding_Fwd"
flops_unembedding_fwd_label = "FLOPs_Unembedding_Fwd"

embedding_mem_time_fwd_label = "Embedding_Mem_Time_Fwd"
embedding_mem_vol_fwd_label = "Embedding_Mem_Volume_Fwd"
embedding_mem_time_bwd_label = "Embedding_Mem_Time_Bwd"
embedding_mem_vol_bwd_label = "Embedding_Mem_Volume_Bwd"
unembedding_mem_time_fwd_label = "Unembedding_Mem_Time_Fwd"
unembedding_mem_vol_fwd_label = "Unembedding_Mem_Volume_Fwd"
unembedding_mem_time_bwd_label = "Unembedding_Mem_Time_Bwd"
unembedding_mem_vol_bwd_label = "Unembedding_Mem_Volume_Bwd"

comm_time_unembedding_ag_label = "Comm_Time_Unembedding_AllGather"
comm_vol_unembedding_ag_label = "Comm_Volume_Unembedding_AllGather"
