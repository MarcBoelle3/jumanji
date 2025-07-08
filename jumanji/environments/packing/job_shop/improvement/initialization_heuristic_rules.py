import jax
import jax.numpy as jnp
from typing import Tuple

def get_job_and_op_id(op_id: int, max_num_ops: int) -> Tuple[int, int]:
    """Extracts the job ID and the operation's index within that job."""
    job_id = op_id // max_num_ops
    op_id_in_job = op_id % max_num_ops
    return job_id, op_id_in_job

def _find_earliest_permissible_insertion_point(
    op_duration: float,
    job_pred_end_time: float,
    machine_schedule: jnp.ndarray,
    machine_start_times: jnp.ndarray,
    machine_end_times: jnp.ndarray
) -> int:
    """
    Finds the first available slot on a machine's schedule where an operation can be inserted.

    A slot is "permissible" if it respects both job and machine constraints and is long enough.
    """
    # Find the end time of the operation scheduled just before each potential slot.
    machine_predecessor_end_times = jnp.concatenate([
        jnp.array([0.0]), machine_end_times[:-1]
    ])
    
    # The earliest an op can start is constrained by both its job and machine predecessors.
    earliest_possible_starts = jnp.maximum(job_pred_end_time, machine_predecessor_end_times)

    # Calculate the time gap available at each slot.
    available_gaps = machine_start_times - earliest_possible_starts

    # Treat unoccupied slots (marked as -1) as having an infinitely large gap.
    permissible_gaps = jnp.where(machine_schedule == -1, jnp.inf, available_gaps)

    # A gap is sufficient if its size is greater than or equal to the op's duration.
    is_gap_sufficient = permissible_gaps >= op_duration

    # Find the index of the first sufficient gap.
    insertion_point = jnp.argmax(is_gap_sufficient)
    
    return insertion_point

def schedule_op_in_earliest_slot(
    op_id: int,
    ops_durations: jnp.ndarray,
    machine_schedule: jnp.ndarray,
    machine_start_times: jnp.ndarray,
    machine_end_times: jnp.ndarray,
    job_predecessor_end_times: jnp.ndarray,
    max_num_ops: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Schedules an operation into the earliest possible permissible slot on its machine.
    Adapted from "permissibleLeftShift" from https://github.com/zcaicaros/L2S/blob/main/env/permissible_LS.py .
    """
    # 1. Get Operation Details
    job_id, op_idx_in_job = get_job_and_op_id(op_id, max_num_ops)
    op_duration = ops_durations[job_id, op_idx_in_job]
    job_pred_end_time = job_predecessor_end_times[job_id]

    # 2. Find the earliest permissible insertion point on the machine
    insertion_point = _find_earliest_permissible_insertion_point(
        op_duration, job_pred_end_time, machine_schedule, machine_start_times, machine_end_times
    )

    # 3. Calculate the new operation's start and end times
    # Get the end time of the op physically preceding it on the machine, or 0 if it's the first.
    machine_pred_end_time = jax.lax.cond(
        insertion_point == 0,
        lambda: 0.0,
        lambda: machine_end_times[insertion_point - 1]
    )
    
    new_op_start_time = jnp.maximum(machine_pred_end_time, job_pred_end_time)
    new_op_end_time = new_op_start_time + op_duration
    
    # 4. Create the updated schedules by inserting the new operation
    updated_schedule = jnp.insert(machine_schedule, insertion_point, op_id)[:-1]
    updated_start_times = jnp.insert(machine_start_times, insertion_point, new_op_start_time)[:-1]
    updated_end_times = jnp.insert(machine_end_times, insertion_point, new_op_end_time)[:-1]

    # 5. Update the predecessor end time for the current job
    updated_job_preds = job_predecessor_end_times.at[job_id].set(new_op_end_time)

    return updated_schedule, updated_start_times, updated_end_times, updated_job_preds