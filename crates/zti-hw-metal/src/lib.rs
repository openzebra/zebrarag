use ort::ep::coreml::{ComputeUnits, ModelFormat, SpecializationStrategy};
use ort::ep::{CoreML, ExecutionProviderDispatch};

fn resolve_compute_units() -> ComputeUnits {
    match std::env::var("ZTI_COREML_COMPUTE_UNITS").ok().as_deref() {
        Some("cpu_and_gpu") => ComputeUnits::CPUAndGPU,
        Some("cpu_and_ane") => ComputeUnits::CPUAndNeuralEngine,
        Some("cpu_only") => ComputeUnits::CPUOnly,
        _ => ComputeUnits::All,
    }
}

pub fn register() -> Vec<ExecutionProviderDispatch> {
    let units = resolve_compute_units();
    let profile_plan = std::env::var("ZTI_COREML_PROFILE").is_ok();

    let mut ep = CoreML::default()
        .with_compute_units(units)
        .with_model_format(ModelFormat::MLProgram)
        .with_specialization_strategy(SpecializationStrategy::FastPrediction)
        .with_static_input_shapes(true)
        .with_subgraphs(true)
        .with_low_precision_accumulation_on_gpu(false);

    if profile_plan {
        ep = ep.with_profile_compute_plan(true);
    }

    match zti_common::paths::models_dir() {
        Ok(dir) => {
            let cache = dir.join("coreml_cache");
            match std::fs::create_dir_all(&cache) {
                Ok(()) => match cache.to_str() {
                    Some(path) => {
                        ep = ep.with_model_cache_dir(path);
                        tracing::debug!(cache = path, "coreml cache enabled");
                    }
                    None => tracing::warn!(
                        "coreml cache path is not valid UTF-8; running without cache"
                    ),
                },
                Err(e) => tracing::warn!(
                    error = %e,
                    "coreml cache dir create failed; running without cache"
                ),
            }
        }
        Err(e) => tracing::warn!(error = %e, "models_dir() failed; coreml cache disabled"),
    }

    tracing::debug!(
        ?units,
        profile_plan,
        "configuring CoreML execution provider"
    );
    vec![ep.build()]
}
