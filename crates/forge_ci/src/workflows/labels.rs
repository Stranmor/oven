use gh_workflow::generate::Generate;
use gh_workflow::*;

/// Generate a disabled label-sync workflow for this fork.
pub fn generate_labels_workflow() {
    let disabled_job = Job::new("Label sync disabled")
        .permissions(Permissions::default().contents(Level::Read))
        .add_step(Step::new("Explain disabled label sync").run(
            "echo 'Automatic label sync is disabled in Stranmor/oven to avoid unintended repository metadata mutations.'",
        ));

    let workflow = Workflow::default()
        .name("Label Sync Disabled")
        .on(Event {
            workflow_dispatch: Some(WorkflowDispatch::default()),
            ..Event::default()
        })
        .permissions(Permissions::default().contents(Level::Read))
        .add_job("label_sync_disabled", disabled_job);

    Generate::new(workflow)
        .name("labels.yml")
        .generate()
        .expect("disabled labels workflow should be generated");
}
