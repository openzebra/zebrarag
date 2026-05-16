use ort::ep::{CoreML, ExecutionProviderDispatch};

pub fn register() -> Vec<ExecutionProviderDispatch> {
    tracing::debug!("configuring CoreML execution provider");
    vec![CoreML::default().build()]
}
