const DELETE_FILTER_MAX_BYTES: usize = 8 * 1024;

pub(crate) fn file_path_delete_filters(paths: &[&str]) -> Vec<String> {
    let mut filters = Vec::with_capacity(1);
    let mut current = String::with_capacity(DELETE_FILTER_MAX_BYTES);
    for path in paths {
        let clause_len = "file_path = ''".len() + path.len() + " OR ".len();
        if !current.is_empty()
            && current.len().saturating_add(clause_len) > DELETE_FILTER_MAX_BYTES
        {
            filters.push(std::mem::take(&mut current));
            current.reserve(DELETE_FILTER_MAX_BYTES);
        }
        if !current.is_empty() {
            current.push_str(" OR ");
        }
        current.push_str("file_path = '");
        push_escaped_sql_string(path, &mut current);
        current.push('\'');
    }
    if !current.is_empty() {
        filters.push(current);
    }
    filters
}

fn push_escaped_sql_string(value: &str, out: &mut String) {
    value.chars().for_each(|ch| {
        if ch == '\'' {
            out.push('\'');
        }
        out.push(ch);
    });
}
