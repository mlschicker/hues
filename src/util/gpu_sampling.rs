//! GPU-based random sampling utilities for HUBO objectives.

use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::coeff::Coeff;
use crate::{domain::{VarDomain, VarType}, instance::HuboInstance};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuParams {
    n_vars: u32,
    n_terms: u32,
    sample_count: u32,
    var_type: u32,
    sample_base_lo: u32,
    sample_base_hi: u32,
    offset: f32,
    _pad0: u32,
}

/// Params for the histogram shader.  Mirrors `GpuParams` but replaces the
/// per-sample output buffer with atomic histogram bin counters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuHistParams {
    n_vars: u32,
    n_terms: u32,
    sample_count: u32,
    var_type: u32,
    sample_base_lo: u32,
    sample_base_hi: u32,
    offset: f32,
    n_bins: u32,
    hist_lo: f32,
    hist_hi: f32,
    _pad0: u32,
    _pad1: u32,
}

const DEFAULT_CHUNK_SIZE: u32 = 1_000_000;

/// Sample many random assignments on GPU and print an ASCII histogram.
///
/// Returns the sampled objective values as `f64`.
pub fn sample_and_print_distribution_gpu<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    sample_count: u64,
    bins: usize,
) -> Result<Vec<f64>, String> {
    // Take a small pilot sample to determine the objective range, then run the
    // full histogram entirely on the GPU using atomic bin counters so no
    // per-sample data needs to be transferred back to the CPU.
    let pilot_count = sample_count.min(500_000);
    let pilot = pollster::block_on(sample_objectives_gpu_chunked(
        instance,
        pilot_count,
        DEFAULT_CHUNK_SIZE,
    ))?;
    let min_val = pilot.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = pilot.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // Small margin so extreme values don't fall exactly on the boundary.
    let margin = (max_val - min_val).abs() * 0.02 + 1e-6;
    let hist_lo = (min_val - margin) as f32;
    let hist_hi = (max_val + margin) as f32;

    let n_bins = bins.max(1) as u32;
    let (counts, total) = pollster::block_on(sample_histogram_gpu(
        instance,
        sample_count,
        hist_lo,
        hist_hi,
        n_bins,
    ))?;

    print_histogram_from_bins(&counts, total, hist_lo as f64, hist_hi as f64, 48);
    Ok(Vec::new())
}

/// Sample objective values for random assignments using GPU compute.
pub async fn sample_objectives_gpu<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    sample_count: u32,
) -> Result<Vec<f64>, String> {
    sample_objectives_gpu_chunked(instance, sample_count as u64, sample_count).await
}

/// Sample objective values for random assignments using GPU compute in chunks.
///
/// This allows arbitrarily large total sample counts while keeping each GPU
/// dispatch bounded by `chunk_size`.
pub async fn sample_objectives_gpu_chunked<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    sample_count: u64,
    chunk_size: u32,
) -> Result<Vec<f64>, String> {
    if sample_count == 0 {
        return Ok(Vec::new());
    }

    if sample_count > usize::MAX as u64 {
        return Err("sample_count exceeds addressable host memory on this platform".to_string());
    }

    if chunk_size == 0 {
        return Err("chunk_size must be > 0".to_string());
    }

    let max_chunk = chunk_size.min(u32::MAX - 1);

    let base_params = GpuParams {
        n_vars: instance.n_vars() as u32,
        n_terms: instance.terms.len() as u32,
        sample_count: 0,
        var_type: match V::VAR_TYPE {
            VarType::Bin => 0,
            VarType::Spin => 1,
        },
        sample_base_lo: 0,
        sample_base_hi: 0,
        offset: instance.offset.to_f64() as f32,
        _pad0: 0,
    };

    let mut term_starts: Vec<u32> = Vec::with_capacity(instance.terms.len() + 1);
    let mut term_indices: Vec<u32> = Vec::new();
    let mut term_coeffs: Vec<f32> = Vec::with_capacity(instance.terms.len());

    term_starts.push(0);
    for term in &instance.terms {
        for &idx in &term.indices {
            term_indices.push(idx as u32);
        }
        term_starts.push(term_indices.len() as u32);
        term_coeffs.push(term.coeff.to_f64() as f32);
    }

    let instance_gpu = wgpu::Instance::default();
    let adapter = instance_gpu
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|e| format!("GPU adapter request failed: {e}"))?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("hues-gpu-sampler-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        })
        .await
        .map_err(|e| format!("GPU device request failed: {e}"))?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hues-gpu-sampler-shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hues-gpu-sampler-params"),
        contents: bytemuck::bytes_of(&base_params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let term_starts_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hues-gpu-sampler-term-starts"),
        contents: bytemuck::cast_slice(&term_starts),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let term_indices_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hues-gpu-sampler-term-indices"),
        contents: bytemuck::cast_slice(&term_indices),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let term_coeffs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hues-gpu-sampler-term-coeffs"),
        contents: bytemuck::cast_slice(&term_coeffs),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_size = max_chunk as u64 * std::mem::size_of::<f32>() as u64;
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hues-gpu-sampler-output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hues-gpu-sampler-readback"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hues-gpu-sampler-bind-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hues-gpu-sampler-bind-group"),
        layout: &bind_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: term_starts_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: term_indices_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: term_coeffs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_buf.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hues-gpu-sampler-pipeline-layout"),
        bind_group_layouts: &[Some(&bind_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("hues-gpu-sampler-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    let mut out: Vec<f64> = Vec::with_capacity(sample_count as usize);
    let mut produced: u64 = 0;

    while produced < sample_count {
        let remaining = sample_count - produced;
        let this_chunk = remaining.min(max_chunk as u64) as u32;

        let params = GpuParams {
            sample_count: this_chunk,
            sample_base_lo: produced as u32,
            sample_base_hi: (produced >> 32) as u32,
            ..base_params
        };
        queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hues-gpu-sampler-encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hues-gpu-sampler-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = this_chunk.div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buf, 0, &readback_buf, 0, output_size);
        queue.submit(Some(encoder.finish()));

        let slice = readback_buf.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });

        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        rx.recv()
            .map_err(|e| format!("readback map channel error: {e}"))?
            .map_err(|e| format!("readback map error: {e}"))?;

        let data = slice.get_mapped_range();
        let values_f32: &[f32] = bytemuck::cast_slice(&data);
        out.extend(
            values_f32
                .iter()
                .take(this_chunk as usize)
                .copied()
                .map(f64::from),
        );
        drop(data);
        readback_buf.unmap();

        produced += this_chunk as u64;
    }

    Ok(out)
}

/// Sample `sample_count` random assignments on the GPU and accumulate their
/// objective values into `n_bins` histogram bins entirely on-device.
///
/// All dispatches are submitted to the GPU queue without any intermediate CPU
/// readbacks, so the GPU stays fully occupied for the whole run.  Only the
/// tiny histogram buffer (n_bins × 4 bytes) is copied back at the very end.
///
/// Returns `(bin_counts, total_samples)`.
async fn sample_histogram_gpu<C: Coeff, V: VarDomain>(
    instance: &HuboInstance<C, V>,
    sample_count: u64,
    hist_lo: f32,
    hist_hi: f32,
    n_bins: u32,
) -> Result<(Vec<u32>, u64), String> {
    if sample_count == 0 || n_bins == 0 {
        return Ok((vec![0; n_bins as usize], 0));
    }

    let base_params = GpuHistParams {
        n_vars: instance.n_vars() as u32,
        n_terms: instance.terms.len() as u32,
        sample_count: 0,
        var_type: match V::VAR_TYPE {
            VarType::Bin => 0,
            VarType::Spin => 1,
        },
        sample_base_lo: 0,
        sample_base_hi: 0,
        offset: instance.offset.to_f64() as f32,
        n_bins,
        hist_lo,
        hist_hi,
        _pad0: 0,
        _pad1: 0,
    };

    let mut term_starts: Vec<u32> = Vec::with_capacity(instance.terms.len() + 1);
    let mut term_indices: Vec<u32> = Vec::new();
    let mut term_coeffs: Vec<f32> = Vec::with_capacity(instance.terms.len());
    term_starts.push(0);
    for term in &instance.terms {
        for &idx in &term.indices {
            term_indices.push(idx as u32);
        }
        term_starts.push(term_indices.len() as u32);
        term_coeffs.push(term.coeff.to_f64() as f32);
    }

    let instance_gpu = wgpu::Instance::default();
    let adapter = instance_gpu
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|e| format!("GPU adapter request failed: {e}"))?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("hues-gpu-hist-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        })
        .await
        .map_err(|e| format!("GPU device request failed: {e}"))?;

    // Maximum samples per dispatch: fill every available workgroup slot in X.
    let max_wg = device.limits().max_compute_workgroups_per_dimension;
    let max_batch: u64 = max_wg as u64 * 64;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hues-gpu-hist-shader"),
        source: wgpu::ShaderSource::Wgsl(HISTOGRAM_SHADER.into()),
    });

    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hues-gpu-hist-params"),
        contents: bytemuck::bytes_of(&base_params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let term_starts_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hues-gpu-hist-term-starts"),
        contents: bytemuck::cast_slice(&term_starts),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let term_indices_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hues-gpu-hist-term-indices"),
        contents: bytemuck::cast_slice(&term_indices),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let term_coeffs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("hues-gpu-hist-term-coeffs"),
        contents: bytemuck::cast_slice(&term_coeffs),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Histogram bins — zero-initialised by the GPU spec.
    let hist_size = n_bins as u64 * std::mem::size_of::<u32>() as u64;
    let hist_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hues-gpu-hist-bins"),
        size: hist_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hues-gpu-hist-readback"),
        size: hist_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hues-gpu-hist-bind-layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hues-gpu-hist-bind-group"),
        layout: &bind_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: term_starts_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: term_indices_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: term_coeffs_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: hist_buf.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hues-gpu-hist-pipeline-layout"),
        bind_group_layouts: &[Some(&bind_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("hues-gpu-hist-pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    // Submit all dispatches to the GPU queue without any intermediate CPU
    // readbacks.  `queue.write_buffer` is ordered before the subsequent
    // `queue.submit`, so each batch picks up the correct sample_base offset.
    let mut produced: u64 = 0;
    while produced < sample_count {
        let remaining = sample_count - produced;
        let this_batch = remaining.min(max_batch) as u32;

        let params = GpuHistParams {
            sample_count: this_batch,
            sample_base_lo: produced as u32,
            sample_base_hi: (produced >> 32) as u32,
            ..base_params
        };
        queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("hues-gpu-hist-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hues-gpu-hist-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = this_batch.div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        queue.submit(Some(encoder.finish()));

        produced += this_batch as u64;
    }

    // Single copy + readback of just the histogram bins.
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("hues-gpu-hist-copy-encoder"),
    });
    encoder.copy_buffer_to_buffer(&hist_buf, 0, &readback_buf, 0, hist_size);
    queue.submit(Some(encoder.finish()));

    let slice = readback_buf.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    rx.recv()
        .map_err(|e| format!("readback map channel error: {e}"))?
        .map_err(|e| format!("readback map error: {e}"))?;

    let data = slice.get_mapped_range();
    let counts: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&data).to_vec();
    drop(data);
    readback_buf.unmap();

    Ok((counts, produced))
}

/// Print an ASCII histogram for sampled objective values.
pub fn print_histogram(values: &[f64], bins: usize, bar_width: usize) {
    if values.is_empty() {
        println!("no samples");
        return;
    }

    let min = values
        .iter()
        .copied()
        .fold(f64::INFINITY, |a, b| if b < a { b } else { a });
    let max = values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a });

    println!("samples: {}", values.len());
    println!("min: {:.6}", min);
    println!("max: {:.6}", max);

    if (max - min).abs() < f64::EPSILON {
        println!("all sampled objective values are identical");
        return;
    }

    let step = (max - min) / bins as f64;
    let mut counts = vec![0usize; bins];
    for &v in values {
        let mut idx = ((v - min) / step).floor() as usize;
        if idx >= bins {
            idx = bins - 1;
        }
        counts[idx] += 1;
    }

    let max_count = *counts.iter().max().unwrap_or(&1);
    println!("distribution:");
    for (i, &count) in counts.iter().enumerate() {
        let lo = min + step * i as f64;
        let hi = if i + 1 == bins { max } else { lo + step };
        let filled = (count * bar_width).checked_div(max_count).unwrap_or(0);
        let bar = "#".repeat(filled);
        println!("[{lo:>10.4}, {hi:>10.4}]  {:>8}  {}", count, bar);
    }
}

/// Print an ASCII histogram from pre-computed GPU histogram bin counts.
fn print_histogram_from_bins(
    counts: &[u32],
    total: u64,
    hist_lo: f64,
    hist_hi: f64,
    bar_width: usize,
) {
    let n_bins = counts.len();
    if n_bins == 0 || total == 0 {
        println!("no samples");
        return;
    }

    let step = (hist_hi - hist_lo) / n_bins as f64;
    // Approximate min/max from first/last occupied bin.
    let first = counts.iter().position(|&c| c > 0).unwrap_or(0);
    let last = counts.iter().rposition(|&c| c > 0).unwrap_or(n_bins - 1);
    let min_approx = hist_lo + step * first as f64;
    let max_approx = hist_lo + step * (last + 1) as f64;

    println!("samples: {total}");
    println!("min (approx): {:.6}", min_approx);
    println!("max (approx): {:.6}", max_approx);

    let max_count = counts.iter().copied().max().unwrap_or(1) as usize;
    println!("distribution:");
    for (i, &count) in counts.iter().enumerate() {
        let lo = hist_lo + step * i as f64;
        let hi = if i + 1 == n_bins { hist_hi } else { lo + step };
        let filled = (count as usize * bar_width)
            .checked_div(max_count)
            .unwrap_or(0);
        let bar = "#".repeat(filled);
        println!("[{lo:>10.4}, {hi:>10.4}]  {:>8}  {}", count, bar);
    }
}

const SHADER: &str = r#"
struct Params {
    n_vars: u32,
    n_terms: u32,
    sample_count: u32,
    var_type: u32,
    sample_base_lo: u32,
    sample_base_hi: u32,
    offset: f32,
    _pad0: u32,
};

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read> term_starts: array<u32>;

@group(0) @binding(2)
var<storage, read> term_indices: array<u32>;

@group(0) @binding(3)
var<storage, read> term_coeffs: array<f32>;

@group(0) @binding(4)
var<storage, read_write> outputs: array<f32>;

fn hash_u32(x: u32) -> u32 {
    var x_mut = x;
    x_mut = x_mut ^ (x_mut >> 16u);
    x_mut = x_mut * 0x7feb352du;
    x_mut = x_mut ^ (x_mut >> 15u);
    x_mut = x_mut * 0x846ca68bu;
    x_mut = x_mut ^ (x_mut >> 16u);
    return x_mut;
}

fn sample_var(sample_id: u32, var_idx: u32) -> f32 {
    let global_lo = params.sample_base_lo + sample_id;
    var carry = 0u;
    if global_lo < params.sample_base_lo {
        carry = 1u;
    }
    let global_hi = params.sample_base_hi + carry;

    let seed = global_lo ^ hash_u32(global_hi ^ 0x517cc1b7u);
    let h = hash_u32(seed ^ (var_idx * 0x9e3779b9u + 0x85ebca6bu));
    let bit = (h >> 31u) & 1u;

    if params.var_type == 0u {
        return f32(bit);
    }
    if bit == 0u {
        return -1.0;
    }
    return 1.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sample_id = gid.x;
    if sample_id >= params.sample_count {
        return;
    }

    var value = params.offset;

    for (var t: u32 = 0u; t < params.n_terms; t = t + 1u) {
        let lo = term_starts[t];
        let hi = term_starts[t + 1u];
        var prod = 1.0;

        for (var p: u32 = lo; p < hi; p = p + 1u) {
            let vidx = term_indices[p];
            prod = prod * sample_var(sample_id, vidx);
        }

        value = value + term_coeffs[t] * prod;
    }

    outputs[sample_id] = value;
}
"#;

const HISTOGRAM_SHADER: &str = r#"
struct Params {
    n_vars: u32,
    n_terms: u32,
    sample_count: u32,
    var_type: u32,
    sample_base_lo: u32,
    sample_base_hi: u32,
    offset: f32,
    n_bins: u32,
    hist_lo: f32,
    hist_hi: f32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read> term_starts: array<u32>;

@group(0) @binding(2)
var<storage, read> term_indices: array<u32>;

@group(0) @binding(3)
var<storage, read> term_coeffs: array<f32>;

@group(0) @binding(4)
var<storage, read_write> hist_bins: array<atomic<u32>>;

fn hash_u32(x: u32) -> u32 {
    var x_mut = x;
    x_mut = x_mut ^ (x_mut >> 16u);
    x_mut = x_mut * 0x7feb352du;
    x_mut = x_mut ^ (x_mut >> 15u);
    x_mut = x_mut * 0x846ca68bu;
    x_mut = x_mut ^ (x_mut >> 16u);
    return x_mut;
}

fn sample_var(sample_id: u32, var_idx: u32) -> f32 {
    let global_lo = params.sample_base_lo + sample_id;
    var carry = 0u;
    if global_lo < params.sample_base_lo {
        carry = 1u;
    }
    let global_hi = params.sample_base_hi + carry;

    let seed = global_lo ^ hash_u32(global_hi ^ 0x517cc1b7u);
    let h = hash_u32(seed ^ (var_idx * 0x9e3779b9u + 0x85ebca6bu));
    let bit = (h >> 31u) & 1u;

    if params.var_type == 0u {
        return f32(bit);
    }
    if bit == 0u {
        return -1.0;
    }
    return 1.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sample_id = gid.x;
    if sample_id >= params.sample_count {
        return;
    }

    var value = params.offset;

    for (var t: u32 = 0u; t < params.n_terms; t = t + 1u) {
        let lo = term_starts[t];
        let hi = term_starts[t + 1u];
        var prod = 1.0;

        for (var p: u32 = lo; p < hi; p = p + 1u) {
            let vidx = term_indices[p];
            prod = prod * sample_var(sample_id, vidx);
        }

        value = value + term_coeffs[t] * prod;
    }

    let range = params.hist_hi - params.hist_lo;
    var bin_idx = u32((value - params.hist_lo) / range * f32(params.n_bins));
    if bin_idx >= params.n_bins {
        bin_idx = params.n_bins - 1u;
    }
    atomicAdd(&hist_bins[bin_idx], 1u);
}
"#;
