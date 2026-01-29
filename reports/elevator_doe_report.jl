# ============================================================================
# elevator_doe_report.jl - Report generation for elevator DOE
# ============================================================================

module ElevatorDOEReportModule

export ElevatorDOEReport, generate_elevator_doe_report, export_elevator_report_md

using Statistics

"""
    ElevatorDOEReport

Aggregated results from an elevator DOE campaign.
"""
struct ElevatorDOEReport
    n_total::Int
    n_pass::Int
    n_fail::Int
    mode_a_results::Vector
    mode_b_results::Vector
    mode_c_results::Vector
    worst_cases::Vector
    poisoning_results::Vector
end

"""
    generate_elevator_doe_report(results, poisoning_results) -> ElevatorDOEReport

Aggregate DOE results into a report structure.
"""
function generate_elevator_doe_report(results::Vector, poisoning_results::Vector = [])
    n_total = length(results)
    n_pass = count(r -> r.pass, results)
    n_fail = n_total - n_pass

    # Split by mode (using run.mode field)
    mode_a = filter(r -> Int(r.run.mode) == 1, results)
    mode_b = filter(r -> Int(r.run.mode) == 2, results)
    mode_c = filter(r -> Int(r.run.mode) == 3, results)

    # Worst cases: failed runs sorted by P90 error descending
    failed = filter(r -> !r.pass, results)
    worst = sort(failed, by = r -> r.metrics.p90_error, rev=true)
    worst_cases = length(worst) > 10 ? worst[1:10] : worst

    return ElevatorDOEReport(n_total, n_pass, n_fail,
                             mode_a, mode_b, mode_c,
                             worst_cases, poisoning_results)
end

"""
    _mode_summary(results, mode_name) -> String

Generate a summary row for one mode.
"""
function _mode_summary(results::Vector, mode_name::String)
    isempty(results) && return "| $mode_name | - | - | - | - | - | - |\n"

    rmses = [r.metrics.rmse for r in results]
    p90s = [r.metrics.p90_error for r in results]
    dnhs = [r.metrics.do_no_harm_ratio for r in results]
    pass_rate = count(r -> r.pass, results) / length(results) * 100.0

    return "| $mode_name | $(length(results)) | $(round(mean(rmses), digits=3)) | " *
           "$(round(mean(p90s), digits=3)) | " *
           "$(round(maximum(p90s), digits=3)) | $(round(mean(dnhs), digits=3)) | " *
           "$(round(pass_rate, digits=1))% |\n"
end

"""
    export_elevator_report_md(report::ElevatorDOEReport, path::String)

Write the elevator DOE report as a markdown file.
"""
function export_elevator_report_md(report::ElevatorDOEReport, path::String)
    lines = String[]

    push!(lines, "# Elevator DOE Report")
    push!(lines, "")
    push!(lines, "## Executive Summary")
    push!(lines, "")
    push!(lines, "- **Total runs**: $(report.n_total)")
    push!(lines, "- **Pass**: $(report.n_pass) ($(round(100.0 * report.n_pass / max(1, report.n_total), digits=1))%)")
    push!(lines, "- **Fail**: $(report.n_fail)")
    push!(lines, "")

    # Mode comparison table
    push!(lines, "## Mode Comparison (A vs B vs C)")
    push!(lines, "")
    push!(lines, "| Mode | Runs | Mean RMSE (m) | Mean P90 (m) | Max P90 (m) | Mean DNH Ratio | Pass Rate |")
    push!(lines, "|------|------|--------------|-------------|-------------|----------------|-----------|")
    push!(lines, _mode_summary(report.mode_a_results, "A (Baseline)"))
    push!(lines, _mode_summary(report.mode_b_results, "B (Robust Ignore)"))
    push!(lines, _mode_summary(report.mode_c_results, "C (Source-Aware)"))
    push!(lines, "")

    # Do-no-harm compliance
    push!(lines, "## Do-No-Harm Compliance (≤ 1.10)")
    push!(lines, "")
    dnh_fail_b = count(r -> r.metrics.do_no_harm_ratio > 1.10, report.mode_b_results)
    dnh_fail_c = count(r -> r.metrics.do_no_harm_ratio > 1.10, report.mode_c_results)
    push!(lines, "- Mode B violations: $dnh_fail_b / $(length(report.mode_b_results))")
    push!(lines, "- Mode C violations: $dnh_fail_c / $(length(report.mode_c_results))")
    push!(lines, "")

    # Mode C benefit
    if !isempty(report.mode_b_results) && !isempty(report.mode_c_results)
        push!(lines, "## Mode C Near-Shaft Benefit vs Mode B")
        push!(lines, "")
        b_near_p90 = mean([r.segment_metrics.near_shaft.p90_error for r in report.mode_b_results])
        c_near_p90 = mean([r.segment_metrics.near_shaft.p90_error for r in report.mode_c_results])
        reduction = b_near_p90 > 0 ? (1.0 - c_near_p90 / b_near_p90) * 100.0 : 0.0
        gate = reduction >= 20.0 ? "PASS" : "FAIL"
        push!(lines, "- Mode B near-shaft P90: $(round(b_near_p90, digits=3)) m")
        push!(lines, "- Mode C near-shaft P90: $(round(c_near_p90, digits=3)) m")
        push!(lines, "- Reduction: $(round(reduction, digits=1))% (gate ≥20%: **$gate**)")
        push!(lines, "")
    end

    # Map poisoning
    if !isempty(report.poisoning_results)
        push!(lines, "## Map Poisoning Results (≤ 1.05)")
        push!(lines, "")
        push!(lines, "| Test | M2 Base P90 | M2 Learned P90 | Ratio | Pass |")
        push!(lines, "|------|-------------|----------------|-------|------|")
        for (i, pr) in enumerate(report.poisoning_results)
            pass_str = pr.passes ? "PASS" : "FAIL"
            push!(lines, "| $i | $(round(pr.m2_baseline_p90, digits=3)) | " *
                         "$(round(pr.m2_learned_p90, digits=3)) | " *
                         "$(round(pr.contamination_ratio, digits=3)) | $pass_str |")
        end
        push!(lines, "")
    end

    # Worst cases
    if !isempty(report.worst_cases)
        push!(lines, "## Worst Case Failures (Top 10)")
        push!(lines, "")
        push!(lines, "| Run | Mode | Archetype | Approach | Speed | P90 (m) | DNH | Reasons |")
        push!(lines, "|-----|------|-----------|----------|-------|---------|-----|---------|")
        for r in report.worst_cases
            mode_str = Int(r.run.mode) == 1 ? "A" : Int(r.run.mode) == 2 ? "B" : "C"
            push!(lines, "| $(r.run.run_id) | $mode_str | $(r.run.point.archetype) | " *
                         "$(r.run.point.closest_approach) | $(r.run.point.elevator_speed) | " *
                         "$(round(r.metrics.p90_error, digits=3)) | " *
                         "$(round(r.metrics.do_no_harm_ratio, digits=3)) | " *
                         "$(join(r.failure_reasons, "; ")) |")
        end
        push!(lines, "")
    end

    # Acceptance criteria summary
    push!(lines, "## Acceptance Criteria Summary")
    push!(lines, "")
    push!(lines, "| Gate | Requirement | Status |")
    push!(lines, "|------|------------|--------|")

    # Do-no-harm
    dnh_pass = dnh_fail_b == 0
    push!(lines, "| Do-no-harm (Mode B) | P90 ratio ≤ 1.10 | $(dnh_pass ? "PASS" : "FAIL") |")

    # Map poisoning
    if !isempty(report.poisoning_results)
        poison_pass = all(pr -> pr.passes, report.poisoning_results)
        push!(lines, "| Map poisoning | P90 ratio ≤ 1.05 | $(poison_pass ? "PASS" : "FAIL") |")
    end

    # Mode C benefit
    if !isempty(report.mode_b_results) && !isempty(report.mode_c_results)
        b_near = mean([r.segment_metrics.near_shaft.p90_error for r in report.mode_b_results])
        c_near = mean([r.segment_metrics.near_shaft.p90_error for r in report.mode_c_results])
        benefit = b_near > 0 ? (1.0 - c_near / b_near) * 100.0 : 0.0
        push!(lines, "| Mode C benefit | ≥20% P90 reduction near shaft | $(benefit >= 20.0 ? "PASS" : "FAIL") |")
    end

    push!(lines, "")

    report_text = join(lines, "\n")

    mkpath(dirname(path))
    open(path, "w") do f
        write(f, report_text)
    end

    return report_text
end

end # module ElevatorDOEReportModule
