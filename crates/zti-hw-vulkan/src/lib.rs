use ort::ep::ExecutionProviderDispatch;

pub fn register() -> Vec<ExecutionProviderDispatch> {
    tracing::debug!("vulkan backend stub — not yet implemented");
    Vec::new()
}
