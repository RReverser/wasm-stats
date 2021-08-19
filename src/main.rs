use anyhow::Result;
use wasmbin::{
    builtins::Blob,
    indices::{GlobalId, LocalId, MemId, TableId},
    sections::{payload, FuncBody},
    visit::Visit,
};
use written_size::WrittenSize;

#[derive(Default, Debug)]
struct RefStats {
    global: usize,
    local: usize,
    table: usize,
    mem: usize,
}

#[derive(Default, Debug)]
struct ProposalStats {
    atomics: usize,
    ref_types: usize,
    simd: usize,
    tail_calls: usize,
    bulk: usize,
}

#[derive(Default, Debug)]
struct InstructionCategoryStats {
    load_store: usize,
    control_flow: usize,
    direct_calls: usize,
    indirect_calls: usize,
    constants: usize,
}

#[derive(Default, Debug)]
struct InstructionStats {
    total: usize,
    refs: RefStats,
    proposals: ProposalStats,
    categories: InstructionCategoryStats,
}

#[derive(Default, Debug)]
struct SizeStats {
    code: usize,
    init: usize,
    externals: usize,
    total: usize,
}

#[derive(Default, Debug)]
struct ExternalStats {
    funcs: usize,
    memories: usize,
    globals: usize,
    tables: usize,
}

#[derive(Default, Debug)]
struct Stats {
    funcs: usize,
    instructions: InstructionStats,
    size: SizeStats,
    imports: ExternalStats,
    exports: ExternalStats,
}

fn calc_size(wasm: &impl wasmbin::io::Encode) -> Result<usize> {
    let mut written_size = WrittenSize::new();
    wasm.encode(&mut written_size)?;
    Ok(written_size.size() as usize)
}

fn get_instruction_stats(funcs: &[Blob<FuncBody>]) -> Result<InstructionStats> {
    use wasmbin::instructions::{simd::SIMD, Instruction as I, MemArg, Misc as M};

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
                | I::SelectWithTypes(_) => stats.categories.control_flow += 1,
                I::SIMD(i) => {
                    stats.proposals.simd += 1;
                    if let SIMD::V128Const(_) = i {
                        stats.categories.constants += 1
                    }
                }
                I::Atomic(_) => stats.proposals.atomics += 1,
                I::RefFunc(_) | I::RefIsNull | I::RefNull(_) => stats.proposals.ref_types += 1,
                I::Misc(i) => match i {
                    M::MemoryInit { .. }
                    | M::MemoryCopy { .. }
                    | M::MemoryFill(_)
                    | M::DataDrop(_)
                    | M::TableInit { .. }
                    | M::TableCopy { .. }
                    | M::TableFill(_)
                    | M::ElemDrop(_) => {
                        stats.proposals.bulk += 1;
                    }
                    _ => {}
                },
                I::Call(_) => stats.categories.direct_calls += 1,
                I::CallIndirect(_) => stats.categories.indirect_calls += 1,
                I::ReturnCall(_) => {
                    stats.categories.direct_calls += 1;
                    stats.proposals.tail_calls += 1;
                }
                I::ReturnCallIndirect(_) => {
                    stats.categories.indirect_calls += 1;
                    stats.proposals.tail_calls += 1;
                }
                I::I32Const(_) | I::I64Const(_) | I::F32Const(_) | I::F64Const(_) => {
                    stats.categories.constants += 1
                }
                _ => {}
            }
        }
        // Covers all load and store instruction - those are the only ones containing MemArg,
        // and only one MemArg.
        func.visit(|_: &MemArg| {
            stats.categories.load_store += 1;
            // Those don't technically reference MemId because they assume single memory,
            // but count them as memory refs anyway.
            stats.refs.mem += 1;
        })?;
        // Now count all MemId references (normally in `memory.*` instructions).
        func.visit(|_: &MemId| {
            stats.refs.mem += 1;
        })?;
        // Same for ops on locals.
        func.visit(|_: &LocalId| {
            stats.refs.local += 1;
        })?;
        // Same for ops on globals.
        func.visit(|_: &GlobalId| {
            stats.refs.global += 1;
        })?;
        func.visit(|_: &TableId| {
            stats.refs.table += 1;
        })?;
    }
    Ok(stats)
}

macro_rules! get_external_stats {
    ($section:expr, $ns:path) => {{
        use $ns::*;

        let mut stats = ExternalStats::default();

        for external in $section.try_contents()? {
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

fn get_stats(wasm: &[u8]) -> Result<Stats> {
    let mut stats = Stats {
        size: SizeStats {
            total: wasm.len(),
            ..Default::default()
        },
        ..Default::default()
    };
    let m = wasmbin::Module::decode_from(wasm)?;
    if let Some(code) = m.find_std_section::<payload::Code>() {
        stats.size.code = calc_size(code)?;
        let funcs = code.try_contents()?;
        stats.funcs = funcs.len();
        stats.instructions = get_instruction_stats(funcs)?;
    }
    if let Some(section) = m.find_std_section::<payload::Data>() {
        stats.size.init += calc_size(section)?;
    }
    if let Some(section) = m.find_std_section::<payload::Element>() {
        stats.size.init += calc_size(section)?;
    }
    if let Some(section) = m.find_std_section::<payload::Import>() {
        stats.size.externals += calc_size(section)?;
        stats.imports = get_external_stats!(section, wasmbin::sections::ImportDesc);
    }
    if let Some(section) = m.find_std_section::<payload::Export>() {
        stats.size.externals += calc_size(section)?;
        stats.exports = get_external_stats!(section, wasmbin::sections::ExportDesc);
    }
    Ok(stats)
}

fn main() -> Result<()> {
    let filename = std::env::args_os()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("Please provide filename"))?;
    let file = std::fs::read(filename)?;
    println!("{:#?}", get_stats(&file)?);
    Ok(())
}
