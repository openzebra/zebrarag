pub(crate) struct OutputPos {
    pub line: u32,
}

pub(crate) struct BytePos {
    pub byte_offset: usize,
    pub output: Option<OutputPos>,
}

impl BytePos {
    pub fn new(byte_offset: usize) -> Self {
        Self {
            byte_offset,
            output: None,
        }
    }
}

pub(crate) fn compute_positions(text: &str, mut positions: Vec<&mut BytePos>) {
    positions.sort_by_key(|p| p.byte_offset);

    let mut iter = positions.into_iter();
    let Some(mut next) = iter.next() else {
        return;
    };

    let mut line = 1u32;

    for (byte_off, _) in text.char_indices() {
        while next.byte_offset == byte_off {
            next.output = Some(OutputPos { line });
            match iter.next() {
                Some(p) => next = p,
                None => return,
            }
        }
        if text.as_bytes()[byte_off] == b'\n' {
            line += 1;
        }
    }

    loop {
        next.output = Some(OutputPos { line });
        match iter.next() {
            Some(p) => next = p,
            None => return,
        }
    }
}
