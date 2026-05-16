use ort::ep::ExecutionProviderDispatch;

pub fn register() -> Vec<ExecutionProviderDispatch> {
    tracing::debug!("configuring CPU execution provider (default)");
    Vec::new()
}
