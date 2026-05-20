use gh_workflow::generate::Generate;
use gh_workflow::*;

/// Generate a disabled release workflow for this fork.
pub fn release_publish() {
    let disabled_job = Job::new("Release publishing disabled")
        .permissions(Permissions::default().contents(Level::Read))
        .add_step(Step::new("Explain disabled release publishing").run(
            "echo 'Release publishing is disabled in Stranmor/oven to prevent accidental upstream package or Homebrew mutations.'",
        ));

    let workflow = Workflow::default()
        .name("Release Publishing Disabled")
        .on(Event {
            workflow_dispatch: Some(WorkflowDispatch::default()),
            ..Event::default()
        })
        .permissions(Permissions::default().contents(Level::Read))
        .add_job("release_publishing_disabled", disabled_job);

    Generate::new(workflow)
        .name("release.yml")
        .generate()
        .expect("disabled release workflow should be generated");
}
