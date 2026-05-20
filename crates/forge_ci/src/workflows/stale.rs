use gh_workflow::generate::Generate;
use gh_workflow::*;

/// Generate a disabled stale-issue workflow for this fork.
pub fn generate_stale_workflow() {
    let disabled_job = Job::new("Stale automation disabled")
        .permissions(Permissions::default().contents(Level::Read))
        .add_step(Step::new("Explain disabled stale automation").run(
            "echo 'Stale issue and PR automation is disabled in Stranmor/oven to avoid unintended issue or PR mutations.'",
        ));

    let workflow = Workflow::default()
        .name("Stale Automation Disabled")
        .on(Event {
            workflow_dispatch: Some(WorkflowDispatch::default()),
            ..Event::default()
        })
        .permissions(Permissions::default().contents(Level::Read))
        .add_job("stale_automation_disabled", disabled_job);

    Generate::new(workflow)
        .name("stale.yml")
        .generate()
        .unwrap();
}
