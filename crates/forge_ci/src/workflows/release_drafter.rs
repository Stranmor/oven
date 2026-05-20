use gh_workflow::generate::Generate;
use gh_workflow::*;

/// Generate a disabled release planning workflow for this fork.
pub fn generate_release_drafter_workflow() {
    let disabled_job = Job::new("Release drafting disabled")
        .permissions(Permissions::default().contents(Level::Read))
        .add_step(Step::new("Explain disabled release drafting").run(
            "echo 'Release drafting is disabled in Stranmor/oven until fork-owned release policy is configured.'",
        ));

    let workflow = Workflow::default()
        .name("Release Drafting Disabled")
        .on(Event {
            workflow_dispatch: Some(WorkflowDispatch::default()),
            ..Event::default()
        })
        .permissions(Permissions::default().contents(Level::Read))
        .add_job("release_drafting_disabled", disabled_job);

    Generate::new(workflow)
        .name("release-drafter.yml")
        .generate()
        .expect("disabled release planning workflow should be generated");
}
