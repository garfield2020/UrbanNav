module UrbanDOEReport

export generate_report, DOEReportConfig

struct DOEReportConfig
    title::String
    output_path::String
    include_plots::Bool
end

function DOEReportConfig(;
    title::String = "UrbanNav Qualification DOE Report",
    output_path::String = "results/doe_report.md",
    include_plots::Bool = false
)
    DOEReportConfig(title, output_path, include_plots)
end

function generate_report(config::DOEReportConfig, results::Vector)
    lines = String[]
    push!(lines, "# $(config.title)")
    push!(lines, "")
    push!(lines, "## Summary")
    push!(lines, "- Total scenarios: $(length(results))")
    push!(lines, "- Pass rate: TBD")
    push!(lines, "")
    push!(lines, "## Acceptance Criteria")
    push!(lines, "| Metric | Threshold | Result | Pass |")
    push!(lines, "|--------|-----------|--------|------|")
    push!(lines, "| Horizontal RMSE | ≤ 2.0 m | TBD | TBD |")
    push!(lines, "| Floor detection | ≥ 95% | TBD | TBD |")
    push!(lines, "| Elevator detection latency | ≤ 30 s | TBD | TBD |")
    push!(lines, "")

    report = join(lines, "\n")

    if !isempty(config.output_path)
        mkpath(dirname(config.output_path))
        open(config.output_path, "w") do f
            write(f, report)
        end
    end

    return report
end

end # module
