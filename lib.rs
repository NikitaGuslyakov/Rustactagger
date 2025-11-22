use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::Bound;
use aho_corasick::{AhoCorasick, AhoCorasickBuilder, PatternID};

/// Простой враппер Aho-Corasick для Python.
/// Принимает список шаблонов (forms), возвращает все вхождения как (start_char, end_char, pattern_idx).
#[pyclass]
pub struct RustAC {
    ac: AhoCorasick,
}

#[pymethods]
impl RustAC {
    /// Создание автомата по списку нормализованных форм.
    #[new]
    pub fn new(patterns: Vec<String>) -> PyResult<Self> {
        if patterns.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "patterns list is empty",
            ));
        }

        let ac = AhoCorasickBuilder::new()
            .ascii_case_insensitive(false)
            .build(&patterns)
            .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("build error: {e}"))
            })?;

        Ok(RustAC { ac })
    }

    /// Найти все вхождения.
    ///
    /// Возвращает список трёхэлем. кортежей:
    /// (start_char, end_char, pattern_index),
    /// где индексы — в символах (char), а не в байтах.
    pub fn find_all(&self, text: &str) -> Vec<(usize, usize, usize)> {
        // Маппинг байтовых индексов → индексы по char
        let mut byte_to_char = vec![0usize; text.len() + 1];
        let mut current_char = 0usize;
        for (b_idx, _ch) in text.char_indices() {
            byte_to_char[b_idx] = current_char;
            current_char += 1;
        }
        // Конец строки
        byte_to_char[text.len()] = current_char;

        let mut out = Vec::new();
        for mat in self.ac.find_iter(text) {
            let s_byte = mat.start();
            let e_byte = mat.end();
            if s_byte > text.len() || e_byte > text.len() {
                continue;
            }
            let s_char = byte_to_char[s_byte];
            let e_char = byte_to_char[e_byte];
            let pid: PatternID = mat.pattern();
            out.push((s_char, e_char, pid.as_usize()));
        }
        out
    }
}

#[pymodule]
fn rust_ac_tagger(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<RustAC>()?;
    Ok(())
}


