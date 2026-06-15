use std::borrow::Cow;
use std::fmt::Write;
use std::sync::Arc;

use anyhow::Result;
use lance::session::Session;
use lance_io::object_store::ObjectStoreRegistry;
use lancedb::connect;

use crate::chunks_table::ChunksTable;
use crate::files_table::FilesTable;
use crate::projects_table::{ProjectRow, ProjectsTable};

#[derive(Clone)]
pub struct Db {
    db: lancedb::Connection,
}

impl Db {
    pub async fn open(project_id: &[u8; 32]) -> Result<Self> {
        let root = zti_common::paths::project_dir(project_id)?;
        let lance_dir = root.join("lance");
        std::fs::create_dir_all(&lance_dir)?;

        let session = Arc::new(Session::new(
            16 * 1024 * 1024,
            64 * 1024 * 1024,
            Arc::new(ObjectStoreRegistry::default()),
        ));

        let db = connect(
            lance_dir
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("invalid path"))?,
        )
        .session(session)
        .execute()
        .await?;

        Ok(Self { db })
    }

    pub async fn open_global() -> Result<Self> {
        let data = zti_common::paths::data_dir()?;
        let registry = data.join("registry");
        std::fs::create_dir_all(&registry)?;

        let db = connect(
            registry
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("invalid path"))?,
        )
        .execute()
        .await?;

        Ok(Self { db })
    }

    pub fn connection(&self) -> &lancedb::Connection {
        &self.db
    }

    pub async fn chunks_table(&self, dim: usize) -> Result<ChunksTable> {
        ChunksTable::open(&self.db, dim).await
    }

    pub async fn files_table(&self) -> Result<FilesTable> {
        FilesTable::open(&self.db).await
    }

    pub async fn projects_table(&self) -> Result<ProjectsTable> {
        ProjectsTable::open(&self.db).await
    }
}

pub async fn list_projects() -> Result<Vec<ProjectRow>> {
    let data = zti_common::paths::data_dir()?;
    let projects_dir = data.join("projects");
    if !projects_dir.is_dir() {
        return Ok(Vec::new());
    }

    let dir_entries: Vec<_> = std::fs::read_dir(&projects_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .collect();

    let mut entries = Vec::with_capacity(dir_entries.len());
    for entry in dir_entries {
        let lance_dir = entry.path().join("lance");
        if !lance_dir.is_dir() {
            continue;
        }

        let db = connect(
            lance_dir
                .to_str()
                .ok_or_else(|| anyhow::anyhow!("invalid path"))?,
        )
        .execute()
        .await?;

        let table_names = db.table_names().execute().await?;
        if !table_names.iter().any(|n| n == "projects") {
            continue;
        }

        let pt = ProjectsTable::open(&db).await?;
        let rows = pt.list().await?;
        entries.extend(rows);
    }

    entries.sort_by(|a, b| a.root_path.cmp(&b.root_path));
    Ok(entries)
}

/// Find a project in a list by reference string. Pure — no I/O.
///
/// Matching order:
/// 1. Exact `root_path` match
/// 2. Canonicalized path match
/// 3. Numeric index (1-based)
/// 4. Project name (directory basename)
pub fn find_project<'a>(projects: &'a [ProjectRow], project_ref: &str) -> Option<&'a ProjectRow> {
    // 1. Exact root_path match
    if let Some(p) = projects.iter().find(|p| p.root_path == project_ref) {
        return Some(p);
    }

    // 2. Canonicalized path match
    if let Ok(canonical) = std::path::Path::new(project_ref).canonicalize() {
        let s = canonical.to_string_lossy();
        if let Some(p) = projects.iter().find(|p| p.root_path == s.as_ref()) {
            return Some(p);
        }
    }

    // 3. Numeric index (1-based)
    if let Ok(idx) = project_ref.parse::<usize>()
        && idx > 0
        && idx <= projects.len()
    {
        return Some(&projects[idx - 1]);
    }

    // 4. Project name (directory basename)
    projects.iter().find(|p| {
        std::path::Path::new(&p.root_path)
            .file_name()
            .map(|f| f == project_ref)
            .unwrap_or(false)
    })
}

/// Resolve a project reference to its canonical `root_path`.
///
/// Accepts:
/// - Full root path: `/home/user/project`
/// - Project name: `myproject` (matches directory basename)
/// - Project index: `1`, `2` (1-based index from `zebraindex projects`)
/// - `None`: auto-resolve via CWD match or single-project fallback
pub async fn resolve_project(project_ref: Option<&str>) -> Result<String> {
    let projects = list_projects().await?;

    let Some(r) = project_ref else {
        return resolve_auto(&projects);
    };

    find_project(&projects, r)
        .map(|p| p.root_path.clone())
        .ok_or_else(|| project_list_error(&projects, Some(r)))
}

fn resolve_auto(projects: &[ProjectRow]) -> Result<String> {
    if let Ok(cwd) = std::env::current_dir().and_then(|c| c.canonicalize()) {
        let cwd_str = cwd.to_string_lossy();
        for p in projects {
            if cwd_str.starts_with(&p.root_path) {
                return Ok(p.root_path.clone());
            }
        }
    }

    match projects.len() {
        0 => Err(anyhow::anyhow!(
            "No indexed projects. Index a project first."
        )),
        1 => Ok(projects[0].root_path.clone()),
        _ => Err(project_list_error(projects, None)),
    }
}

pub(crate) fn project_list_error(projects: &[ProjectRow], query: Option<&str>) -> anyhow::Error {
    let mut msg = String::with_capacity(64 + projects.len() * 88);
    match query {
        Some(q) => {
            let _ = writeln!(msg, "Project '{q}' not found in index. Available projects:");
        }
        None => {
            msg.push_str("Multiple projects found. Use --root or `project`:\n");
        }
    }
    for (i, p) in projects.iter().enumerate() {
        let name = std::path::Path::new(&p.root_path)
            .file_name()
            .map(|s| s.to_string_lossy())
            .unwrap_or(Cow::Borrowed(&p.root_path));
        let _ = writeln!(msg, "  {}. {}  ({})", i + 1, name, p.root_path);
    }
    anyhow::anyhow!(msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proj(root_path: &str) -> ProjectRow {
        ProjectRow {
            project_id: vec![],
            root_path: root_path.to_string(),
            languages: vec![],
            model_id: String::new(),
            model_dim: 0,
            total_chunks: 0,
            total_files: 0,
            last_indexed_ns: 0,
            created_at_ns: 0,
            index_version: crate::projects_table::INDEX_FORMAT_VERSION,
            search_method: None,
            search_params: None,
        }
    }

    #[test]
    fn find_by_exact_root_path() {
        let projects = vec![proj("/a/b"), proj("/c/d")];
        assert_eq!(
            find_project(&projects, "/c/d").map(|p| p.root_path.as_str()),
            Some("/c/d")
        );
    }

    #[test]
    fn find_by_numeric_index_first() {
        let projects = vec![proj("/first"), proj("/second")];
        assert_eq!(
            find_project(&projects, "1").map(|p| p.root_path.as_str()),
            Some("/first")
        );
    }

    #[test]
    fn find_by_numeric_index_last() {
        let projects = vec![proj("/first"), proj("/second")];
        assert_eq!(
            find_project(&projects, "2").map(|p| p.root_path.as_str()),
            Some("/second")
        );
    }

    #[test]
    fn find_by_numeric_index_zero_returns_none() {
        let projects = vec![proj("/a")];
        assert!(find_project(&projects, "0").is_none());
    }

    #[test]
    fn find_by_numeric_index_out_of_range_returns_none() {
        let projects = vec![proj("/a")];
        assert!(find_project(&projects, "2").is_none());
    }

    #[test]
    fn find_by_project_name() {
        let projects = vec![
            proj("/home/user/my-project"),
            proj("/other/path/another-one"),
        ];
        assert_eq!(
            find_project(&projects, "my-project").map(|p| p.root_path.as_str()),
            Some("/home/user/my-project")
        );
    }

    #[test]
    fn find_by_project_name_case_sensitive() {
        let projects = vec![proj("/path/MyProject")];
        assert!(find_project(&projects, "myproject").is_none());
    }

    #[test]
    fn find_no_match_returns_none() {
        let projects = vec![proj("/a/b")];
        assert!(find_project(&projects, "nonexistent").is_none());
    }

    #[test]
    fn find_empty_list_returns_none() {
        assert!(find_project(&[], "anything").is_none());
    }

    #[test]
    fn find_by_canonicalized_path() {
        let cwd = std::env::current_dir().unwrap();
        let cwd_str = cwd.to_string_lossy();
        let projects = vec![proj(&cwd_str)];

        let parent = cwd.parent().unwrap();
        let name = cwd.file_name().unwrap();
        let indirect = parent.join(name);
        let indirect_str = indirect.to_string_lossy();

        assert_eq!(
            find_project(&projects, &indirect_str).map(|p| p.root_path.as_str()),
            Some(cwd_str.as_ref())
        );
    }

    #[test]
    fn find_prefers_exact_match_over_index() {
        let projects = vec![proj("1"), proj("/other")];
        assert_eq!(
            find_project(&projects, "1").map(|p| p.root_path.as_str()),
            Some("1")
        );
    }

    #[test]
    fn project_list_error_format() {
        let projects = vec![proj("/alpha/beta"), proj("/gamma/delta")];
        let err = project_list_error(&projects, None);
        let msg = format!("{err:#}");
        assert!(msg.contains("1."));
        assert!(msg.contains("2."));
        assert!(msg.contains("beta"));
        assert!(msg.contains("/alpha/beta"));
        assert!(msg.contains("delta"));
        assert!(msg.contains("/gamma/delta"));
        assert!(msg.contains("--root"));
    }

    #[test]
    fn project_not_found_error_format() {
        let projects = vec![proj("/a/b"), proj("/c/d")];
        let err = project_list_error(&projects, Some("foobar"));
        let msg = format!("{err:#}");
        assert!(msg.contains("foobar"));
        assert!(msg.contains("not found in index"));
        assert!(msg.contains("a"));
        assert!(msg.contains("b"));
        assert!(msg.contains("c"));
        assert!(msg.contains("d"));
    }
}
