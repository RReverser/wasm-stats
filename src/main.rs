use anyhow::{anyhow, Context, Result};
use indicatif::ProgressBar;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    io::{BufRead, BufReader, Seek, SeekFrom, Write},
    path::Path,
};
use wasmbin::{
    builtins::Blob,
    sections::{ExportDesc, FuncBody, ImportDesc, Section},
    types::ValueType,
    visit::Visit,
};
use written_size::WrittenSize;

#[derive(Default, Debug, Serialize)]
struct ProposalStats {
    atomics: usize,
    ref_types: usize,
    simd: usize,
    tail_calls: usize,
    bulk: usize,
    multi_value: usize,
    non_trapping_conv: usize,
    sign_extend: usize,
    mutable_externals: usize,
    bigint_externals: usize,
}

#[derive(Default, Debug, Serialize)]
struct InstructionCategoryStats {
    load_store: usize,
    local_var: usize,
    global_var: usize,
    table: usize,
    memory: usize,
    control_flow: usize,
    direct_calls: usize,
    indirect_calls: usize,
    constants: usize,
    wait_notify: usize,
    other: usize,
}

#[derive(Default, Debug, Serialize)]
struct InstructionStats {
    total: usize,
    proposals: ProposalStats,
    categories: InstructionCategoryStats,
}

#[derive(Default, Debug, Serialize)]
struct SizeStats {
    code: usize,
    init: usize,
    externals: usize,
    types: usize,
    custom: usize,
    descriptors: usize,
    total: usize,
    total_br: usize,
    total_strip: usize,
    total_strip_br: usize,
    total_opt: usize,
    total_opt_br: usize,
}

#[derive(Default, Debug, Serialize)]
struct ExternalStats {
    funcs: usize,
    memories: usize,
    globals: usize,
    tables: usize,
}

#[derive(Default, Debug, Serialize)]
struct Stats<'a> {
    filename: &'a str,
    funcs: usize,
    instr: InstructionStats,
    size: SizeStats,
    imports: ExternalStats,
    exports: ExternalStats,
    custom_sections: Vec<String>,
    has_start: bool,
}

fn calc_size(wasm: &impl wasmbin::io::Encode) -> Result<usize> {
    let mut written_size = WrittenSize::new();
    wasm.encode(&mut written_size)?;
    Ok(written_size.size() as usize)
}

fn get_instruction_stats(funcs: &[Blob<FuncBody>]) -> Result<InstructionStats> {
    use wasmbin::instructions::{simd::SIMD, Instruction as I, Misc as M};

    let mut stats = InstructionStats::default();
    for func in funcs {
        let func = &func.try_contents()?.expr;
        stats.total += func.len();
        for i in func {
            match i {
                I::BlockStart(_)
                | I::LoopStart(_)
                | I::IfStart(_)
                | I::IfElse
                | I::End
                | I::Unreachable
                | I::Br(_)
                | I::BrIf(_)
                | I::BrTable { .. }
                | I::Return
                | I::Select
                | I::SelectWithTypes(_)
                | I::Nop
                | I::Drop => stats.categories.control_flow += 1,
                I::SIMD(i) => {
                    stats.proposals.simd += 1;
                    match i {
                        SIMD::V128Load(_)
                        | SIMD::V128Load8x8S(_)
                        | SIMD::V128Load8x8U(_)
                        | SIMD::V128Load16x4S(_)
                        | SIMD::V128Load16x4U(_)
                        | SIMD::V128Load32x2S(_)
                        | SIMD::V128Load32x2U(_)
                        | SIMD::V128Load8Splat(_)
                        | SIMD::V128Load16Splat(_)
                        | SIMD::V128Load32Splat(_)
                        | SIMD::V128Load64Splat(_)
                        | SIMD::V128Store(_)
                        | SIMD::V128Load8Lane(_, _)
                        | SIMD::V128Load16Lane(_, _)
                        | SIMD::V128Load32Lane(_, _)
                        | SIMD::V128Load64Lane(_, _)
                        | SIMD::V128Store8Lane(_, _)
                        | SIMD::V128Store16Lane(_, _)
                        | SIMD::V128Store32Lane(_, _)
                        | SIMD::V128Store64Lane(_, _) => stats.categories.load_store += 1,
                        SIMD::V128Const(_) => stats.categories.constants += 1,
                        _ => stats.categories.other += 1,
                    }
                }
                I::Atomic(i) => {
                    stats.proposals.atomics += 1;
                    match i {
                        wasmbin::instructions::Atomic::Wake(_)
                        | wasmbin::instructions::Atomic::I32Wait(_)
                        | wasmbin::instructions::Atomic::I64Wait(_) => {
                            stats.categories.wait_notify += 1;
                        }
                        wasmbin::instructions::Atomic::I32Load(_)
                        | wasmbin::instructions::Atomic::I64Load(_)
                        | wasmbin::instructions::Atomic::I32Load8U(_)
                        | wasmbin::instructions::Atomic::I32Load16U(_)
                        | wasmbin::instructions::Atomic::I64Load8U(_)
                        | wasmbin::instructions::Atomic::I64Load16U(_)
                        | wasmbin::instructions::Atomic::I64Load32U(_)
                        | wasmbin::instructions::Atomic::I32Store(_)
                        | wasmbin::instructions::Atomic::I64Store(_)
                        | wasmbin::instructions::Atomic::I32Store8(_)
                        | wasmbin::instructions::Atomic::I32Store16(_)
                        | wasmbin::instructions::Atomic::I64Store8(_)
                        | wasmbin::instructions::Atomic::I64Store16(_)
                        | wasmbin::instructions::Atomic::I64Store32(_) => {
                            stats.categories.load_store += 1;
                        }
                        _ => stats.categories.other += 1,
                    }
                }
                I::RefFunc(_) | I::RefIsNull | I::RefNull(_) => {
                    stats.proposals.ref_types += 1;
                    match i {
                        I::RefIsNull => stats.categories.other += 1,
                        _ => stats.categories.constants += 1,
                    }
                }
                I::Misc(i) => match i {
                    M::MemoryInit { .. }
                    | M::MemoryCopy { .. }
                    | M::MemoryFill(_)
                    | M::DataDrop(_) => {
                        stats.proposals.bulk += 1;
                        stats.categories.memory += 1;
                    }
                    M::TableInit { .. }
                    | M::TableCopy { .. }
                    | M::TableFill(_)
                    | M::ElemDrop(_) => {
                        stats.proposals.bulk += 1;
                        stats.categories.table += 1;
                    }
                    M::TableGrow(_) | M::TableSize(_) => {
                        stats.proposals.ref_types += 1;
                        stats.categories.table += 1;
                    }
                    M::I32TruncSatF32S
                    | M::I32TruncSatF32U
                    | M::I32TruncSatF64S
                    | M::I32TruncSatF64U
                    | M::I64TruncSatF32S
                    | M::I64TruncSatF32U
                    | M::I64TruncSatF64S
                    | M::I64TruncSatF64U => {
                        stats.proposals.non_trapping_conv += 1;
                        stats.categories.other += 1;
                    }
                },
                I::Call(_) => stats.categories.direct_calls += 1,
                I::CallIndirect(_) => stats.categories.indirect_calls += 1,
                I::ReturnCall(_) => {
                    stats.categories.control_flow += 1;
                    stats.categories.direct_calls += 1;
                    stats.proposals.tail_calls += 1;
                }
                I::ReturnCallIndirect(_) => {
                    stats.categories.control_flow += 1;
                    stats.categories.indirect_calls += 1;
                    stats.proposals.tail_calls += 1;
                }
                I::I32Const(_) | I::I64Const(_) | I::F32Const(_) | I::F64Const(_) => {
                    stats.categories.constants += 1
                }
                I::LocalGet(_) | I::LocalSet(_) | I::LocalTee(_) => {
                    stats.categories.local_var += 1;
                }
                I::GlobalGet(_) | I::GlobalSet(_) => {
                    stats.categories.global_var += 1;
                }
                I::TableGet(_) | I::TableSet(_) => {
                    stats.categories.table += 1;
                }
                I::I32Load(_)
                | I::I64Load(_)
                | I::F32Load(_)
                | I::F64Load(_)
                | I::I32Load8S(_)
                | I::I32Load8U(_)
                | I::I32Load16S(_)
                | I::I32Load16U(_)
                | I::I64Load8S(_)
                | I::I64Load8U(_)
                | I::I64Load16S(_)
                | I::I64Load16U(_)
                | I::I64Load32S(_)
                | I::I64Load32U(_)
                | I::I32Store(_)
                | I::I64Store(_)
                | I::F32Store(_)
                | I::F64Store(_)
                | I::I32Store8(_)
                | I::I32Store16(_)
                | I::I64Store8(_)
                | I::I64Store16(_)
                | I::I64Store32(_) => {
                    stats.categories.load_store += 1;
                }
                I::MemorySize(_) | I::MemoryGrow(_) => {
                    stats.categories.memory += 1;
                }
                I::I64ExtendI32U
                | I::I32Extend8S
                | I::I32Extend16S
                | I::I64Extend8S
                | I::I64Extend16S
                | I::I64Extend32S => {
                    stats.proposals.sign_extend += 1;
                    stats.categories.other += 1;
                }
                _ => {
                    stats.categories.other += 1;
                }
            }
        }
    }
    Ok(stats)
}

macro_rules! get_external_stats {
    ($section:expr, $ns:path) => {{
        use $ns::*;

        let mut stats = ExternalStats::default();

        for external in $section {
            match external.desc {
                Func(_) => stats.funcs += 1,
                Global(_) => stats.globals += 1,
                Mem(_) => stats.memories += 1,
                Table(_) => stats.tables += 1,
            }
        }

        stats
    }};
}

struct MaybeExternal<T> {
    pub value: T,
    pub is_external: bool,
}

impl<T> MaybeExternal<T> {
    fn external(self) -> Option<T> {
        if self.is_external {
            Some(self.value)
        } else {
            None
        }
    }
}

fn get_stats(wasm: &[u8]) -> Result<Stats> {
    let m = wasmbin::Module::decode_from(wasm)?;
    let mut stats = Stats {
        size: SizeStats {
            total: wasm.len(),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut global_types = Vec::new();
    let mut func_types = Vec::new();
    let mut types = &[] as &[_];
    for section in &m.sections {
        match section {
            Section::Custom(section) => {
                stats.size.custom += calc_size(section)?;
                stats
                    .custom_sections
                    .push(section.try_contents()?.name().to_owned());
            }
            Section::Type(section) => {
                stats.size.types += calc_size(section)?;
                types = section.try_contents()?;
                for ty in types {
                    if ty.results.len() > 1 {
                        stats.instr.proposals.multi_value += 1;
                    }
                }
            }
            Section::Import(section) => {
                stats.size.externals += calc_size(section)?;
                let section = section.try_contents()?;
                stats.imports = get_external_stats!(section, ImportDesc);
                for item in section {
                    match &item.desc {
                        ImportDesc::Global(ty) => {
                            global_types.push(MaybeExternal {
                                value: ty.clone(),
                                is_external: true,
                            });
                        }
                        ImportDesc::Func(type_id) => {
                            func_types.push(MaybeExternal {
                                value: *type_id,
                                is_external: true,
                            });
                        }
                        _ => {}
                    }
                }
            }
            Section::Function(section) => {
                stats.size.descriptors += calc_size(section)?;
                func_types.extend(section.try_contents()?.iter().map(|type_id| MaybeExternal {
                    value: *type_id,
                    is_external: false,
                }));
            }
            Section::Table(section) => {
                stats.size.descriptors += calc_size(section)?;
            }
            Section::Memory(section) => {
                stats.size.descriptors += calc_size(section)?;
                for ty in section.try_contents()? {
                    if ty.is_shared {
                        stats.instr.proposals.atomics += 1;
                    }
                }
            }
            Section::Global(section) => {
                stats.size.descriptors += calc_size(section)?;
                global_types.extend(section.try_contents()?.iter().map(|global| MaybeExternal {
                    value: global.ty.clone(),
                    is_external: false,
                }));
            }
            Section::Export(section) => {
                stats.size.externals += calc_size(section)?;
                let section = section.try_contents()?;
                stats.exports = get_external_stats!(section, ExportDesc);
                for item in section {
                    match item.desc {
                        ExportDesc::Global(global_id) => {
                            global_types[global_id.index as usize].is_external = true;
                        }
                        ExportDesc::Func(func_id) => {
                            func_types[func_id.index as usize].is_external = true;
                        }
                        _ => {}
                    }
                }
            }
            Section::Start(_) => {
                stats.has_start = true;
            }
            Section::Element(section) => {
                stats.size.init += calc_size(section)?;
            }
            Section::DataCount(_) => {
                stats.instr.proposals.bulk += 1;
            }
            Section::Code(section) => {
                stats.size.code = calc_size(section)?;
                let funcs = section.try_contents()?;
                stats.funcs = funcs.len();
                stats.instr = get_instruction_stats(funcs)?;
            }
            Section::Data(section) => {
                stats.size.init += calc_size(section)?;
            }
        }
    }
    global_types
        .into_iter()
        .filter_map(MaybeExternal::external)
        .for_each(|ty| {
            if ty.mutable {
                stats.instr.proposals.mutable_externals += 1;
            }
            if let ValueType::I64 = ty.value_type {
                stats.instr.proposals.bigint_externals += 1;
            }
        });
    func_types
        .into_iter()
        .filter_map(MaybeExternal::external)
        .try_for_each(|type_id| {
            types[type_id.index as usize].visit(|ty: &ValueType| {
                if let ValueType::I64 = ty {
                    stats.instr.proposals.bigint_externals += 1;
                }
            })
        })?;
    Ok(stats)
}

fn path_to_filename(path: &Path) -> Result<&str> {
    path.file_name()
        .context("Missing filename")?
        .to_str()
        .context("Invalid filename")
}

fn file_size(path: impl AsRef<Path>) -> Result<usize> {
    let path = path.as_ref();
    std::fs::metadata(path)
        .map(|metadata| metadata.len() as _)
        .with_context(move || path_to_filename(path).unwrap_or("(no filename)").to_owned())
}

fn main() -> Result<()> {
    let dir = std::env::args_os()
        .nth(1)
        .ok_or_else(|| anyhow!("Please provide directory"))?;

    let mut output = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(Path::new(&dir).join("stats.json"))?;

    output.seek(SeekFrom::Start(0))?;

    let prev_files = BufReader::new(&mut output)
        .lines()
        .map(|line| -> Result<_> {
            #[derive(Deserialize)]
            struct QuickStats {
                filename: String,
            }

            Ok(serde_json::from_str::<QuickStats>(&line?)?.filename)
        })
        .collect::<Result<HashSet<_>>>()?;

    let new_files = std::fs::read_dir(&dir)?
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().and_then(|s| s.to_str()) == Some("wasm"))
        .filter(|path| !prev_files.contains(path_to_filename(path).unwrap()))
        .collect::<Vec<_>>();

    let pb = ProgressBar::new((prev_files.len() + new_files.len()) as _);

    pb.inc(prev_files.len() as _);

    let output = std::sync::Mutex::new(std::io::BufWriter::new(output));

    let errors = new_files
        .into_par_iter()
        .filter_map(|path| {
            let filename = path_to_filename(&path).unwrap().to_owned();
            let handler = || -> Result<()> {
                let wasm = std::fs::read(&path)?;
                let mut stats = get_stats(&wasm)?;
                stats.filename = &filename;
                stats.size.total_br = file_size(path.with_extension("wasm.br"))?;
                stats.size.total_strip = file_size(path.with_extension("wasm.strip"))?;
                stats.size.total_strip_br = file_size(path.with_extension("wasm.strip.br"))?;
                stats.size.total_opt = file_size(path.with_extension("wasm.opt"))?;
                stats.size.total_opt_br = file_size(path.with_extension("wasm.opt.br"))?;
                // Serialize to string to ensure atomic write of an entire line.
                let serialized = serde_json::to_string(&stats)? + "\n";
                output.lock().unwrap().write_all(serialized.as_bytes())?;
                Ok(())
            };
            let result = handler();
            pb.inc(1);
            Some(result.err()?.context(filename))
        })
        .collect::<Vec<_>>();

    if !errors.is_empty() {
        anyhow::bail!("Multiple errors: {:#?}", errors);
    }

    Ok(())
}
