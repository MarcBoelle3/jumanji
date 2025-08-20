import jax
from jumanji.environments.packing.job_shop.improvement.types import BestSolution, ImprovementState

def improvement_update(new_state: ImprovementState):
            """Update best solution and reset counter when improved."""

            best_solution = BestSolution(
                scheduled_times=new_state.scheduled_times,
                adj_mat_pc=new_state.adj_mat_pc,
                adj_mat_mc=new_state.adj_mat_mc,
                is_on_critical_path=new_state.is_on_critical_path,
                action_mask=new_state.action_mask,
                operation_pairs_mask=new_state.operation_pairs_mask,
                critical_block_info=new_state.critical_block_info,
                gap_left_right=new_state.gap_left_right,
                est=new_state.est,
                lst=new_state.lst,
            )

            return new_state.replace(
                best_solution_so_far=best_solution,
                step_since_best=0
            )
        
def restart_needed(new_state: ImprovementState):
    """Restart from best solution when threshold reached."""
    best_sol = new_state.best_solution_so_far
    return new_state.replace(
        scheduled_times=best_sol.scheduled_times,
        adj_mat_pc=best_sol.adj_mat_pc,
        adj_mat_mc=best_sol.adj_mat_mc,
        is_on_critical_path=best_sol.is_on_critical_path,
        action_mask=best_sol.action_mask,
        operation_pairs_mask=best_sol.operation_pairs_mask,
        critical_block_info=best_sol.critical_block_info,
        gap_left_right=best_sol.gap_left_right,
        est=best_sol.est,
        lst=best_sol.lst,
        makespan=new_state.incumbent_makespan,
        step_since_best=0,
    )

def increase_step_since_best(new_state: ImprovementState):
    return new_state.replace(
        step_since_best=new_state.step_since_best + 1
    )