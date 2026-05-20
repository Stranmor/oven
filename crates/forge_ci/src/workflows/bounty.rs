use gh_workflow::generate::Generate;
use gh_workflow::*;

/// Generate a disabled bounty workflow for this fork.
pub fn generate_bounty_workflow() {
    let disabled_job = Job::new("Bounty automation disabled")
        .permissions(Permissions::default().contents(Level::Read))
        .add_step(Step::new("Explain disabled bounty automation").run(
            "echo 'Bounty automation is disabled in Stranmor/oven to avoid unintended issue or PR mutations.'",
        ));

    let workflow = Workflow::default()
        .name("Bounty Automation Disabled")
        .on(Event {
            workflow_dispatch: Some(WorkflowDispatch::default()),
            ..Event::default()
        })
        .permissions(Permissions::default().contents(Level::Read))
        .add_job("bounty_automation_disabled", disabled_job);

    Generate::new(workflow)
        .name("bounty.yml")
        .generate()
        .unwrap();
}
