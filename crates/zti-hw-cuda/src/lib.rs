use ort::ep::{CUDA, ExecutionProviderDispatch};

pub fn register() -> Vec<ExecutionProviderDispatch> {
    tracing::debug!("configuring CUDA execution provider");
    vec![CUDA::default().build()]
}
