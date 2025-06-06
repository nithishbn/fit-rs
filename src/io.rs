use crate::DoseResponse;
use anyhow::{Context, Result};
use csv::Reader;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct CsvRecord {
    concentration: f64,
    response: f64,
}

/// Load dose-response data from CSV file with "concentration" and "response" columns
pub fn load_csv<P: AsRef<Path>>(file_path: P) -> Result<Vec<DoseResponse>> {
    let mut reader = Reader::from_path(&file_path)
        .with_context(|| format!("Failed to open CSV file: {}", file_path.as_ref().display()))?;

    let mut data = Vec::new();

    for result in reader.deserialize() {
        let record: CsvRecord = result.context("Failed to parse CSV row")?;

        data.push(DoseResponse {
            concentration: record.concentration,
            response: record.response,
        });
    }

    Ok(data)
}

// Convenience method for BayesianEC50Fitter
impl crate::BayesianEC50Fitter {
    /// Create fitter from CSV file
    pub fn from_csv<P: AsRef<Path>>(file_path: P) -> Result<Self> {
        let data = load_csv(file_path)?;
        Ok(Self::new(data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_csv() -> Result<()> {
        // Create test CSV
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "concentration,response")?;
        writeln!(temp_file, "1e-9,10.5")?;
        writeln!(temp_file, "1e-8,25.3")?;
        writeln!(temp_file, "1e-7,45.1")?;
        temp_file.flush()?;

        // Test loading
        let data = load_csv(temp_file.path())?;
        assert_eq!(data.len(), 3);
        assert_eq!(data[0].concentration, 1e-9);
        assert_eq!(data[0].response, 10.5);

        Ok(())
    }
}
